"""Evaluation stage — the primary focus of the project.

Scores a generated customer-support reply against the incoming email and
a human reference reply, producing an interpretable quality score with
per-dimension reasoning.

The metric is a *composite* one — deliberately NOT built on BLEU / ROUGE
/ exact matching. It combines:

    1. An LLM-as-a-judge that scores five weighted dimensions 0-10 and
       justifies each score in one sentence.
    2. Deterministic checks (greeting, closing, length, unsupported
       claims, ...) that slightly adjust the final score and add
       explanations.

The five dimensions and weights:

    Relevance          30%
    Completeness       25%
    Groundedness       20%
    Professional Tone  15%
    Actionability      10%

``overall_score`` is the weighted average of the dimension scores scaled
to 0-100, plus the deterministic adjustment, clamped to ``[0, 100]``.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.generator import DEFAULT_MODEL, GROQ_BASE_URL

# --------------------------------------------------------------------------- #
# Dimension configuration
# --------------------------------------------------------------------------- #

DIMENSIONS = [
    "relevance",
    "completeness",
    "groundedness",
    "professional_tone",
    "actionability",
]

WEIGHTS = {
    "relevance": 0.30,
    "completeness": 0.25,
    "groundedness": 0.20,
    "professional_tone": 0.15,
    "actionability": 0.10,
}

DIMENSION_DESCRIPTIONS = {
    "relevance": "Does the reply directly answer the customer's issue?",
    "completeness": "Does it address every important request or question?",
    "groundedness": (
        "Does it avoid inventing policies, refunds, dates, promises, or facts "
        "not supported by the customer's email or the reference reply?"
    ),
    "professional_tone": "Is it professional, empathetic, concise, and grammatically correct?",
    "actionability": "Does it clearly explain the next step or requested information?",
}

JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 700

# Deterministic-check tuning (all on the 0-100 scale).
EXCESSIVE_WORDS = 220
PENALTY_NO_GREETING = 2.0
PENALTY_NO_CLOSING = 2.0
PENALTY_EXCESSIVE_LENGTH = 3.0
PENALTY_UNSUPPORTED_REFUND = 5.0
PENALTY_UNSUPPORTED_DATE = 4.0
PENALTY_UNSUPPORTED_DISCOUNT = 5.0
MAX_TOTAL_ADJUSTMENT = 20.0  # keep deterministic nudges "slight"

# --------------------------------------------------------------------------- #
# Regexes for deterministic checks
# --------------------------------------------------------------------------- #

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|dear|greetings|good\s+(morning|afternoon|evening))\b",
    re.IGNORECASE,
)
_CLOSING_WORDS = [
    "regards", "best", "sincerely", "thanks", "thank you", "cheers",
    "warm regards", "kind regards", "best regards", "respectfully",
    "yours truly", "many thanks",
]
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_PERCENT_RE = re.compile(r"\d+\s?%")
_WEEKDAY_RE = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_DISCOUNT_WORDS = ["discount", "coupon", "promo", "promotion", "voucher", "% off"]


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot be completed."""


class EmailResponseEvaluator:
    """Composite LLM + deterministic evaluator for support replies."""

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int = 4,
    ) -> None:
        """Initialise the evaluator.

        Args:
            client: An existing OpenAI-compatible (Groq) client to reuse.
                If ``None``, one is constructed from the environment.
            model: Judge model id. Defaults to ``GROQ_MODEL`` env var,
                then ``llama-3.3-70b-versatile``.
            max_retries: Retries on transient / rate-limit API errors.

        Raises:
            EvaluationError: If no client can be built (missing key).
        """
        import os

        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        self.max_retries = max_retries

        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EvaluationError("GROQ_API_KEY is not set.")
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        customer_email: str,
        reference_reply: str,
        generated_reply: str,
    ) -> dict[str, Any]:
        """Evaluate a single generated reply.

        Args:
            customer_email: The incoming customer email.
            reference_reply: The human reference reply (grounding context).
            generated_reply: The model-generated reply to score.

        Returns:
            A dict with ``overall_score`` (0-100), ``dimension_scores``
            (each 0-10), and ``reasoning`` (per-dimension justifications
            plus deterministic-check explanations).
        """
        # Empty responses fail fast with a zero score.
        if not generated_reply or not generated_reply.strip():
            return {
                "overall_score": 0.0,
                "dimension_scores": {d: 0.0 for d in DIMENSIONS},
                "reasoning": {
                    **{d: "No response was generated." for d in DIMENSIONS},
                    "deterministic_checks": ["Empty response."],
                    "adjustment": 0.0,
                },
            }

        # 1. Primary judge.
        dimension_scores, justifications = self._llm_judge(
            customer_email, reference_reply, generated_reply
        )

        # 2. Weighted average, scaled to 0-100.
        weighted = sum(dimension_scores[d] * WEIGHTS[d] for d in DIMENSIONS)
        base_score = weighted * 10.0

        # 3. Deterministic adjustment.
        adjustment, check_notes = self._deterministic_checks(
            customer_email, reference_reply, generated_reply
        )

        overall = max(0.0, min(100.0, base_score + adjustment))

        reasoning = dict(justifications)
        reasoning["deterministic_checks"] = check_notes
        reasoning["adjustment"] = round(adjustment, 2)

        return {
            "overall_score": round(overall, 1),
            "dimension_scores": {d: round(dimension_scores[d], 1) for d in DIMENSIONS},
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------ #
    # LLM judge
    # ------------------------------------------------------------------ #
    def _build_judge_messages(
        self,
        customer_email: str,
        reference_reply: str,
        generated_reply: str,
    ) -> list[dict[str, str]]:
        rubric = "\n".join(
            f"- {d.replace('_', ' ').title()} ({int(WEIGHTS[d] * 100)}%): "
            f"{DIMENSION_DESCRIPTIONS[d]}"
            for d in DIMENSIONS
        )
        schema_hint = ",\n".join(
            f'  "{d}": {{"score": <integer 0-10>, "justification": "<one sentence>"}}'
            for d in DIMENSIONS
        )
        system = (
            "You are a meticulous evaluator of customer-support email replies. "
            "You grade a generated reply against the customer's email and a human "
            "reference reply. Be strict but fair. Score each dimension from 0 to 10 "
            "(10 = excellent). Justify every score in exactly one sentence. "
            "Return ONLY a single valid JSON object and nothing else."
        )
        user = (
            "Evaluate the GENERATED REPLY on these five dimensions:\n"
            f"{rubric}\n\n"
            "=== CUSTOMER EMAIL ===\n"
            f"{customer_email.strip()}\n\n"
            "=== REFERENCE REPLY (human-written, for grounding) ===\n"
            f"{reference_reply.strip()}\n\n"
            "=== GENERATED REPLY (to be scored) ===\n"
            f"{generated_reply.strip()}\n\n"
            "Return JSON in exactly this shape:\n"
            "{\n"
            f"{schema_hint}\n"
            "}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _llm_judge(
        self,
        customer_email: str,
        reference_reply: str,
        generated_reply: str,
    ) -> tuple[dict[str, float], dict[str, str]]:
        messages = self._build_judge_messages(
            customer_email, reference_reply, generated_reply
        )
        raw = self._call_with_retries(messages)
        parsed = self._parse_judge_json(raw)

        scores: dict[str, float] = {}
        justifications: dict[str, str] = {}
        for d in DIMENSIONS:
            entry = parsed.get(d, {})
            score = entry.get("score", 0)
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            scores[d] = max(0.0, min(10.0, score))
            justifications[d] = str(entry.get("justification", "")).strip()
        return scores, justifications

    def _call_with_retries(self, messages: list[dict[str, str]]) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=JUDGE_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise EvaluationError("Judge returned an empty response.")
                return content
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc).lower()
                transient = "rate" in msg or "429" in msg or "timeout" in msg or "503" in msg
                if transient and attempt < self.max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                # Non-transient, or out of retries.
                if attempt < self.max_retries - 1:
                    time.sleep(1.0)
                    continue
                break
        raise EvaluationError(f"Judge API request failed: {last_exc}")

    @staticmethod
    def _parse_judge_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        # Strip markdown code fences if present.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fall back to the first {...} block.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise EvaluationError(f"Could not parse judge JSON: {exc}") from exc
            raise EvaluationError("Judge did not return JSON.")

    # ------------------------------------------------------------------ #
    # Deterministic checks
    # ------------------------------------------------------------------ #
    def _deterministic_checks(
        self,
        customer_email: str,
        reference_reply: str,
        generated_reply: str,
    ) -> tuple[float, list[str]]:
        """Return (adjustment, notes) on the 0-100 scale.

        Adjustments are small and bounded so they only nudge the judge's
        score rather than dominate it.
        """
        notes: list[str] = []
        adjustment = 0.0

        reply = generated_reply.strip()
        supported = f"{customer_email} {reference_reply}".lower()
        reply_lower = reply.lower()

        # Structure: greeting. Scan the first few lines so a leading
        # "Subject:" line doesn't hide a greeting like "Dear Customer,".
        head_lines = [ln for ln in reply.splitlines() if ln.strip()][:3]
        if any(_GREETING_RE.search(ln) for ln in head_lines):
            notes.append("Greeting present.")
        else:
            adjustment -= PENALTY_NO_GREETING
            notes.append(f"No greeting detected (-{PENALTY_NO_GREETING}).")

        # Structure: closing / sign-off (check the tail of the reply).
        tail = "\n".join(reply.splitlines()[-4:]).lower()
        if any(w in tail for w in _CLOSING_WORDS):
            notes.append("Closing present.")
        else:
            adjustment -= PENALTY_NO_CLOSING
            notes.append(f"No closing/sign-off detected (-{PENALTY_NO_CLOSING}).")

        # Length.
        word_count = len(reply.split())
        if word_count > EXCESSIVE_WORDS:
            adjustment -= PENALTY_EXCESSIVE_LENGTH
            notes.append(
                f"Excessively long ({word_count} words > {EXCESSIVE_WORDS}) "
                f"(-{PENALTY_EXCESSIVE_LENGTH})."
            )

        # Unsupported monetary / refund claim.
        gen_amounts = set(_MONEY_RE.findall(reply))
        unsupported_amounts = [a for a in gen_amounts if a.replace(" ", "") not in supported.replace(" ", "")]
        if unsupported_amounts and ("refund" in reply_lower or "charge" in reply_lower or gen_amounts):
            adjustment -= PENALTY_UNSUPPORTED_REFUND
            notes.append(
                f"Mentions monetary amount(s) {unsupported_amounts} not supported by "
                f"the email/reference (-{PENALTY_UNSUPPORTED_REFUND})."
            )

        # Unsupported concrete date (weekday / month name / MM/DD).
        gen_dates = (
            _WEEKDAY_RE.findall(reply)
            + _MONTH_RE.findall(reply)
            + _DATE_RE.findall(reply)
        )
        unsupported_dates = [d for d in gen_dates if str(d).lower() not in supported]
        if unsupported_dates:
            adjustment -= PENALTY_UNSUPPORTED_DATE
            notes.append(
                f"Mentions specific date(s) {unsupported_dates} not supported by the "
                f"email/reference (-{PENALTY_UNSUPPORTED_DATE})."
            )

        # Unsupported discount / promo.
        gen_has_discount = any(w in reply_lower for w in _DISCOUNT_WORDS) or bool(
            _PERCENT_RE.search(reply)
        )
        supported_has_discount = any(w in supported for w in _DISCOUNT_WORDS) or bool(
            _PERCENT_RE.search(supported)
        )
        if gen_has_discount and not supported_has_discount:
            adjustment -= PENALTY_UNSUPPORTED_DISCOUNT
            notes.append(
                f"Mentions a discount/promotion not supported by the email/reference "
                f"(-{PENALTY_UNSUPPORTED_DISCOUNT})."
            )

        # Keep the total nudge "slight".
        adjustment = max(-MAX_TOTAL_ADJUSTMENT, min(MAX_TOTAL_ADJUSTMENT, adjustment))
        return adjustment, notes


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #

def summarize_scores(scores: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of overall scores.

    Args:
        scores: Overall scores (0-100) across a dataset.

    Returns:
        A dict with average, median, minimum, maximum and standard
        deviation. Empty input yields zeros.
    """
    import numpy as np

    if not scores:
        return {"average": 0.0, "median": 0.0, "minimum": 0.0, "maximum": 0.0, "std": 0.0}
    arr = np.asarray(scores, dtype="float64")
    return {
        "average": round(float(arr.mean()), 2),
        "median": round(float(np.median(arr)), 2),
        "minimum": round(float(arr.min()), 2),
        "maximum": round(float(arr.max()), 2),
        "std": round(float(arr.std()), 2),
    }

"""Response generation stage.

Retrieval-augmented generation of a professional customer-support reply.
Given an incoming email and a set of retrieved similar examples, build a
grounded prompt and call an LLM (Groq, via the OpenAI-compatible API) to
produce the suggested response.

Configuration is read from the environment:
    - ``GROQ_API_KEY``  API key (never hardcoded).
    - ``GROQ_MODEL``    model id (defaults to ``llama-3.3-70b-versatile``).
"""

from __future__ import annotations

import os

from src.retrieval import RetrievedExample

# Groq's OpenAI-compatible endpoint.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Number of retrieved examples to include in the prompt.
MAX_EXAMPLES = 3

TEMPERATURE = 0.3
MAX_TOKENS = 300

SYSTEM_PROMPT = (
    "You are an experienced customer support agent.\n"
    "Write a concise, professional, empathetic email.\n"
    "Ground your response using the customer's email and the retrieved examples.\n"
    "Never invent refunds, policies, delivery dates, discounts, or promises.\n"
    "If information is missing, politely ask for it."
)


class GenerationError(RuntimeError):
    """Raised when response generation cannot be completed."""


class ResponseGenerator:
    """Generate support replies with retrieval-augmented prompting."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        """Initialise the generator and its Groq client.

        Args:
            model: Override the model id. Defaults to ``GROQ_MODEL`` env
                var, then ``llama-3.3-70b-versatile``.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the generated reply.

        Raises:
            GenerationError: If ``GROQ_API_KEY`` is not set.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise GenerationError("GROQ_API_KEY is not set.")

        # Imported lazily so the module imports cleanly without the key.
        from openai import OpenAI

        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    # ---------------------------------------------------------------- #
    # Prompt construction
    # ---------------------------------------------------------------- #
    @staticmethod
    def build_user_prompt(
        customer_email: str,
        retrieved_examples: list[RetrievedExample],
    ) -> str:
        """Assemble the user prompt from the email and retrieved examples.

        Args:
            customer_email: The incoming customer email text.
            retrieved_examples: Similar historical examples for grounding.

        Returns:
            The formatted user prompt string.
        """
        parts = [
            "Incoming customer email:",
            "",
            customer_email.strip(),
            "",
            "Relevant historical conversations:",
            "",
        ]
        for i, ex in enumerate(retrieved_examples[:MAX_EXAMPLES], start=1):
            parts.extend(
                [
                    f"Example {i}",
                    "",
                    "Customer:",
                    ex.customer_email.strip(),
                    "",
                    "Agent:",
                    ex.agent_reply.strip(),
                    "",
                ]
            )
        parts.extend(
            [
                "Write only the final email reply.",
                "Do not explain your reasoning.",
            ]
        )
        return "\n".join(parts)

    def build_messages(
        self,
        customer_email: str,
        retrieved_examples: list[RetrievedExample],
    ) -> list[dict[str, str]]:
        """Build the chat messages for the completion request."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.build_user_prompt(customer_email, retrieved_examples),
            },
        ]

    # ---------------------------------------------------------------- #
    # Generation
    # ---------------------------------------------------------------- #
    def generate(
        self,
        customer_email: str,
        retrieved_examples: list[RetrievedExample],
    ) -> str:
        """Generate a suggested reply for one customer email.

        Args:
            customer_email: The incoming customer email text.
            retrieved_examples: Retrieved similar examples for grounding.

        Returns:
            The generated email reply text.

        Raises:
            GenerationError: If the API call fails.
        """
        messages = self.build_messages(customer_email, retrieved_examples)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surface a clean message upstream.
            raise GenerationError(f"Groq API request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise GenerationError("Groq API returned an empty response.")
        return content.strip()

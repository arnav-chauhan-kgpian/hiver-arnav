"""Dataset generation.

Builds a realistic synthetic customer-support email dataset of
email/reply pairs across ten support categories. The generator is
fully deterministic (``random.seed(42)``) so re-runs reproduce the same
dataset.

Outputs (all under ``data/processed/``):
    - email_dataset.jsonl   full dataset, one JSON object per line
    - email_dataset.csv     same data as CSV
    - train.jsonl           80% split
    - test.jsonl            20% split

Run directly to (re)generate the dataset and print a summary:

    python data/generate_dataset.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SEED = 42
N_EXAMPLES = 300
TRAIN_FRAC = 0.80

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

JSONL_PATH = PROCESSED_DIR / "email_dataset.jsonl"
CSV_PATH = PROCESSED_DIR / "email_dataset.csv"
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"

CATEGORIES = [
    "refund_request",
    "shipping_delay",
    "password_reset",
    "billing_issue",
    "subscription_cancellation",
    "feature_request",
    "bug_report",
    "account_access",
    "invoice_request",
    "general_inquiry",
]

# Categories where a plausible order number is expected in the email body.
ORDER_CATEGORIES = {"refund_request", "billing_issue", "shipping_delay"}

TYPO_RATE = 0.15
FRUSTRATED_RATE = 0.20

# --------------------------------------------------------------------------- #
# Content pools
# --------------------------------------------------------------------------- #

FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "Lucas", "Mia", "Amir", "Priya", "Chen", "Fatima", "Diego",
    "Yuki", "Aisha", "Marco", "Hannah", "Omar", "Nina", "Raj", "Grace",
    "Leo", "Zara", "Tomas", "Ingrid", "Kwame", "Sofia",
]

LAST_NAMES = [
    "Smith", "Johnson", "Patel", "Garcia", "Nguyen", "Kim", "Rossi",
    "Muller", "Silva", "Haddad", "Okafor", "Tanaka", "Andersson", "Costa",
    "Ahmed", "Brown", "Wilson", "Lopez", "Novak", "Fischer",
]

PRODUCTS = [
    "wireless headphones", "running shoes", "coffee maker", "office chair",
    "yoga mat", "smart watch", "backpack", "desk lamp", "water bottle",
    "bluetooth speaker", "kitchen blender", "phone case", "monitor stand",
    "mechanical keyboard", "winter jacket",
]

PLANS = ["Basic", "Pro", "Premium", "Team", "Enterprise", "Starter"]

# Greetings and sign-offs to vary structure/length/tone.
GREETINGS = ["Hi", "Hello", "Hey", "Good morning", "Good afternoon", "Dear support"]
SIGNOFFS = [
    "Thanks", "Thank you", "Regards", "Best", "Cheers", "Appreciate your help",
    "Kind regards",
]

FRUSTRATION_OPENERS = [
    "I am extremely frustrated.",
    "This is the third time I'm reaching out and I'm losing patience.",
    "I'm really disappointed with the service so far.",
    "This has been an incredibly frustrating experience.",
    "Honestly, I expected much better than this.",
]

FRUSTRATION_CLOSERS = [
    "I expect this to be resolved immediately.",
    "If this isn't fixed soon I'll be taking my business elsewhere.",
    "Please treat this as urgent.",
    "I shouldn't have to chase this so many times.",
    "I'd like a real answer, not a canned response.",
]

# Common, safe typo substitutions applied to a fraction of emails.
TYPO_MAP = {
    "the": "teh",
    "please": "plese",
    "receive": "recieve",
    "really": "realy",
    "account": "acount",
    "because": "becuase",
    "definitely": "definitly",
    "immediately": "imediately",
    "problem": "problme",
    "order": "oder",
    "would": "wuold",
    "your": "youre",
    "with": "wtih",
    "cannot": "cannnot",
    "and": "adn",
}


# --------------------------------------------------------------------------- #
# Per-category email/reply templates
# --------------------------------------------------------------------------- #
#
# Each builder returns (customer_email_body, agent_reply, base_priority).
# ``ctx`` carries the shared, already-randomised values (name, product,
# order_id, plan, ...) so email and reply stay consistent.

def _refund_request(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    product, order_id = ctx["product"], ctx["order_id"]
    reasons = [
        f"the {product} arrived damaged",
        f"the {product} is not what I expected",
        f"I received the wrong item instead of the {product}",
        f"the {product} stopped working after two days",
    ]
    reason = rng.choice(reasons)
    email = (
        f"I'd like to request a refund for my recent purchase. "
        f"My order number is {order_id}. Unfortunately, {reason}. "
        f"I've only had it for a short while and it's already unusable. "
        f"Could you please let me know how to return it and how long the refund will take? "
        f"I'd prefer the amount back to my original payment method."
    )
    reply = (
        f"I'm sorry to hear about the trouble with your {product}. "
        f"I've located order {order_id} and started your refund request. "
        f"I'll email you a prepaid return label shortly, and once the item is on its way back, "
        f"we'll process the refund to your original payment method within 5-7 business days. "
        f"Let me know if there's anything else I can help with."
    )
    return email, reply, "medium"


def _shipping_delay(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    product, order_id = ctx["product"], ctx["order_id"]
    days = rng.choice([4, 6, 8, 10, 12])
    email = (
        f"I placed an order for a {product} on order {order_id}, and it's now {days} days late. "
        f"The tracking page hasn't updated in a while and still says the package is in transit. "
        f"I needed this before the weekend and I'm starting to worry it's lost. "
        f"Can you check what's going on and give me a realistic delivery date?"
    )
    reply = (
        f"Thanks for flagging this, and I'm sorry your {product} is running late. "
        f"I checked order {order_id}: the carrier had a processing delay at their sorting hub. "
        f"It's now moving again and is expected to arrive within 2-3 business days. "
        f"I've added tracking notifications to your email so you'll get live updates, "
        f"and if it doesn't arrive by then we'll ship a replacement right away."
    )
    return email, reply, "medium"


def _password_reset(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    email = (
        "I'm trying to log in but my password isn't working. "
        "I clicked the 'forgot password' link but the reset email never arrived, even after checking spam. "
        "Can you help me reset it so I can get back into my account?"
    )
    reply = (
        "Happy to help you get back in. I've just triggered a fresh password-reset email to the "
        "address on your account, so please check your inbox (and spam) in the next few minutes. "
        "The link is valid for 60 minutes. If it still doesn't arrive, reply here and I'll verify "
        "your identity and reset it manually."
    )
    return email, reply, "low"


def _billing_issue(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    order_id = ctx["order_id"]
    amount = rng.choice([19.99, 29.00, 49.50, 12.75, 89.99])
    email = (
        f"I was charged twice for the same order. My invoice reference is {order_id}, "
        f"and I can see two identical charges of ${amount:.2f} on my card statement. "
        f"I only placed one order, so please refund the duplicate charge as soon as possible."
    )
    reply = (
        f"Thanks for the details and I'm sorry for the confusion. I can confirm a duplicate "
        f"charge of ${amount:.2f} was applied to reference {order_id}. I've refunded the extra "
        f"charge, which should appear on your statement within 3-5 business days depending on your bank. "
        f"You'll receive a confirmation email shortly."
    )
    return email, reply, "high"


def _subscription_cancellation(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    plan = ctx["plan"]
    email = (
        f"I'd like to cancel my {plan} subscription. "
        f"I'm not using it as much as I expected and I want to avoid the next renewal charge. "
        f"Could you confirm the cancellation and let me know when my access ends?"
    )
    reply = (
        f"Of course, I can help with that. I've scheduled your {plan} subscription to cancel at the "
        f"end of the current billing period, so you won't be charged again. You'll keep full access "
        f"until then, and I've sent a confirmation email for your records. "
        f"If you change your mind, you can reactivate anytime before that date."
    )
    return email, reply, "low"


def _feature_request(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    feature = rng.choice([
        "a dark mode", "an export-to-CSV option", "a mobile app",
        "two-factor authentication", "calendar integration", "bulk editing",
    ])
    email = (
        f"I've been really enjoying the product, but I'd love to see {feature} added. "
        f"It would make my daily workflow much smoother and save me a lot of manual effort. "
        f"Is this something on your roadmap, and is there any way to vote for it?"
    )
    reply = (
        f"Thank you for the kind words and the thoughtful suggestion. {feature.capitalize()} is a "
        f"popular request, and I've logged your vote with our product team. While I can't promise a "
        f"specific date, I've added you to the update list for this feature so you'll be notified as "
        f"soon as there's news. We really appreciate feedback like this."
    )
    return email, reply, "low"


def _bug_report(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    area = rng.choice([
        "the checkout page", "the dashboard", "the search bar",
        "the settings screen", "the notifications panel",
    ])
    email = (
        f"I think I found a bug. When I use {area}, the page freezes and then shows an error message. "
        f"It happens every time, and reloading doesn't help. I'm on the latest version of Chrome. "
        f"This is blocking me from finishing what I need to do. Can you take a look?"
    )
    reply = (
        f"Thanks for the detailed report, that's exactly the info we need. I've reproduced the issue "
        f"with {area} and passed it to our engineering team as a priority ticket. As a temporary "
        f"workaround, clearing your browser cache often restores access. I'll follow up here as soon "
        f"as the fix is deployed."
    )
    return email, reply, "high"


def _account_access(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    email = (
        "I'm locked out of my account. It says my access has been suspended, but I haven't done "
        "anything unusual. I really need to get back in today because I have work depending on it. "
        "Can you tell me why this happened and help me restore access?"
    )
    reply = (
        "I understand how stressful being locked out is, and I'm here to help. Our system flagged "
        "an unusual sign-in and temporarily suspended access as a precaution. I've verified your "
        "account looks secure and lifted the suspension, so you should be able to log in now. "
        "I'd also recommend enabling two-factor authentication for extra protection."
    )
    return email, reply, "high"


def _invoice_request(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    email = (
        "Could you please send me a copy of my latest invoice? "
        "I need it for my expense report and I can't find it in my email. "
        "A PDF would be ideal. Thanks in advance."
    )
    reply = (
        "Absolutely. I've generated a PDF copy of your most recent invoice and emailed it to the "
        "address on your account. You can also download past invoices anytime from the Billing "
        "section of your account settings. Let me know if you need it in a different format."
    )
    return email, reply, "low"


def _general_inquiry(ctx: dict[str, Any], rng: random.Random) -> tuple[str, str, str]:
    topic = rng.choice([
        "your return policy", "whether you ship internationally",
        "if there's a student discount", "your business hours",
        "whether gift wrapping is available",
    ])
    email = (
        f"I had a quick question about {topic}. "
        f"I'm considering placing an order and wanted to check before I do. "
        f"Could you point me in the right direction?"
    )
    reply = (
        f"Great question, and thanks for reaching out before ordering. Here's a quick answer on "
        f"{topic}, along with a link to the full details on our help center. If anything is unclear "
        f"or you'd like a recommendation for your situation, just reply here and I'll be glad to help."
    )
    return email, reply, "low"


CATEGORY_BUILDERS: dict[str, Callable[[dict[str, Any], random.Random], tuple[str, str, str]]] = {
    "refund_request": _refund_request,
    "shipping_delay": _shipping_delay,
    "password_reset": _password_reset,
    "billing_issue": _billing_issue,
    "subscription_cancellation": _subscription_cancellation,
    "feature_request": _feature_request,
    "bug_report": _bug_report,
    "account_access": _account_access,
    "invoice_request": _invoice_request,
    "general_inquiry": _general_inquiry,
}


# --------------------------------------------------------------------------- #
# Realism helpers
# --------------------------------------------------------------------------- #

def _make_order_id(rng: random.Random) -> str:
    """Generate a plausible order/invoice number, e.g. ``ORD-2024-583920``."""
    year = rng.choice([2023, 2024, 2025])
    return f"ORD-{year}-{rng.randint(100000, 999999)}"


def _inject_typos(text: str, rng: random.Random) -> str:
    """Introduce a few realistic typos into the text."""
    words = text.split(" ")
    for i, word in enumerate(words):
        lower = word.lower().strip(".,!?")
        if lower in TYPO_MAP and rng.random() < 0.5:
            # Preserve trailing punctuation.
            suffix = ""
            while word and word[-1] in ".,!?":
                suffix = word[-1] + suffix
                word = word[:-1]
            words[i] = TYPO_MAP[lower] + suffix
    return " ".join(words)


def _wrap_email(
    body: str,
    greeting: str,
    name: str,
    signoff: str,
    frustrated: bool,
    rng: random.Random,
) -> str:
    """Assemble a full email from a body, adding greeting, tone and sign-off."""
    parts = [f"{greeting},", ""]
    if frustrated:
        parts.append(rng.choice(FRUSTRATION_OPENERS))
    parts.append(body)
    if frustrated:
        parts.append(rng.choice(FRUSTRATION_CLOSERS))
    parts.extend(["", f"{signoff},", name])
    return "\n".join(parts)


def _assign_priority(base: str, frustrated: bool) -> str:
    """Escalate priority by one level for frustrated customers."""
    if not frustrated:
        return base
    ladder = {"low": "medium", "medium": "high", "high": "high"}
    return ladder[base]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def _build_example(idx: int, category: str, rng: random.Random) -> dict[str, Any]:
    """Build a single email/reply record for the given category."""
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"

    order_id = _make_order_id(rng) if category in ORDER_CATEGORIES else None

    ctx = {
        "name": name,
        "product": rng.choice(PRODUCTS),
        "plan": rng.choice(PLANS),
        "order_id": order_id,
    }

    body, agent_reply, base_priority = CATEGORY_BUILDERS[category](ctx, rng)

    frustrated = rng.random() < FRUSTRATED_RATE
    priority = _assign_priority(base_priority, frustrated)

    customer_email = _wrap_email(
        body=body,
        greeting=rng.choice(GREETINGS),
        name=first,
        signoff=rng.choice(SIGNOFFS),
        frustrated=frustrated,
        rng=rng,
    )

    if rng.random() < TYPO_RATE:
        customer_email = _inject_typos(customer_email, rng)

    return {
        "id": f"email_{idx:04d}",
        "category": category,
        "customer_email": customer_email,
        "agent_reply": agent_reply,
        "priority": priority,
        "customer_name": name,
        "order_id": order_id,
    }


def generate_dataset(
    output_dir: str | Path | None = None,
    n: int = N_EXAMPLES,
) -> list[dict[str, Any]]:
    """Generate the email suggested-response dataset.

    Args:
        output_dir: Where to write the processed files. Defaults to
            ``data/processed``.
        n: Number of email/reply pairs to generate.

    Returns:
        The list of generated records.
    """
    rng = random.Random(SEED)
    out_dir = Path(output_dir) if output_dir is not None else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Even, deterministic spread across categories (remainder to the first few).
    per_cat = n // len(CATEGORIES)
    remainder = n % len(CATEGORIES)
    plan: list[str] = []
    for i, cat in enumerate(CATEGORIES):
        count = per_cat + (1 if i < remainder else 0)
        plan.extend([cat] * count)
    rng.shuffle(plan)

    records = [_build_example(i, cat, rng) for i, cat in enumerate(plan)]

    # Deterministic train/test split.
    order = list(range(len(records)))
    rng.shuffle(order)
    n_train = int(len(records) * TRAIN_FRAC)
    train_idx = set(order[:n_train])
    train = [records[i] for i in range(len(records)) if i in train_idx]
    test = [records[i] for i in range(len(records)) if i not in train_idx]

    # Write outputs.
    jsonl_path = out_dir / JSONL_PATH.name
    csv_path = out_dir / CSV_PATH.name
    train_path = out_dir / TRAIN_PATH.name
    test_path = out_dir / TEST_PATH.name

    _write_jsonl(records, jsonl_path)
    _write_jsonl(train, train_path)
    _write_jsonl(test, test_path)
    _write_csv(records, csv_path)

    _print_summary(records, train, test, jsonl_path, csv_path, train_path, test_path)
    return records


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

FIELDNAMES = [
    "id", "category", "customer_email", "agent_reply",
    "priority", "customer_name", "order_id",
]


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def _print_summary(
    records: list[dict[str, Any]],
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    jsonl_path: Path,
    csv_path: Path,
    train_path: Path,
    test_path: Path,
) -> None:
    print(f"Total examples: {len(records)}")
    print()
    print("Category distribution:")
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for rec in records:
        counts[rec["category"]] += 1
    for cat in CATEGORIES:
        print(f"  {cat:<28} {counts[cat]}")
    print()
    print(f"Train/test counts: {len(train)} train / {len(test)} test")
    print()
    print("Output files:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")
    print(f"  {train_path}")
    print(f"  {test_path}")


if __name__ == "__main__":
    generate_dataset()

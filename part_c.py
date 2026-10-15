"""Part C: privacy-safe ticket triage using the supplied offline mock."""
from pathlib import Path
import json
import random
import re
import time

import pandas as pd
import mock_llm

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ZERO_PROMPT = (
    "CLASSIFY support tickets. Allowed categories: billing, delivery, technical, fraud. "
    'Return JSON only, exactly in the form {"category": "...", "reason": "..."}. '
    "Choose the single best category from the ticket text."
)
FEW_PROMPT = (
    "CLASSIFY each support ticket. Allowed categories: billing, delivery, technical, fraud. "
    'Return JSON only, exactly in the form {"category": "...", "reason": "..."}. '
    "Choose the single best category from the ticket text.\n"
    "EXAMPLES:\n"
    'Ticket: "I was charged twice and need a refund." -> {"category":"billing","reason":"duplicate charge"}\n'
    'Ticket: "My order is late and the rider is unreachable." -> {"category":"delivery","reason":"late delivery"}\n'
    'Ticket: "Someone made an unauthorized card payment." -> {"category":"fraud","reason":"unauthorized transaction"}\n'
    'Ticket: "The app crashes when I open my cart." -> {"category":"technical","reason":"app crash"}'
)
ALLOWED = {"billing", "delivery", "technical", "fraud"}
REDACTIONS = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
              (re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
              (re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text):
    counts = {}
    for pattern, replacement in REDACTIONS:
        text, counts[replacement] = pattern.subn(replacement, text)
    return text, counts


def validate_reply(text):
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        matches = [word for word in words if word in ALLOWED]
        if len(matches) == 1:
            return "classified_keyword", matches[0]
        return "flagged_invalid_output", ""
    if isinstance(parsed, dict) and parsed.get("category") in ALLOWED:
        return "classified_json", parsed["category"]
    return "flagged_invalid_output", ""


def route_ticket(status, category, segment_name, is_premium):
    if status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}:
        return "NORMAL", "human_review"
    if category == "fraud":
        return "HIGH", "security_team"
    if is_premium:
        return "HIGH", "premium_desk"
    return "NORMAL", "support_queue"


def run(prompt, token_budget=None, include_retry_log=True):
    """Process tickets in source order; the user message is exactly redacted text."""
    mock_llm.reset()
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    segments = pd.read_csv(OUT / "customer_segments.csv")
    segment_lookup = segments.set_index("customer_id").to_dict("index")
    redaction_counts = {"[EMAIL]": 0, "[PHONE]": 0, "[ORDER_ID]": 0}
    result_rows, replies = [], []
    running_tokens = 0
    jitter_rng = random.Random(42)
    for row in tickets.itertuples(index=False):
        redacted, counts = redact(row.text)
        for kind, number in counts.items():
            redaction_counts[kind] += number
        # Reapplying all required patterns must find no personal identifiers.
        assert all(pattern.search(redacted) is None for pattern, _ in REDACTIONS)
        if not redacted.strip():
            status, category, used = "skipped_empty", "", 0
        elif mock_llm.est_tokens(redacted) > MAX_INPUT_TOKENS:
            status, category, used = "flagged_long", "", 0
        else:
            status, category, used = "", "", 0
            for attempt in range(4):
                if token_budget is not None and running_tokens >= token_budget:
                    status = "budget_exhausted"
                    break
                messages = [{"role": "system", "content": prompt}, {"role": "user", "content": redacted}]
                assert messages[-1]["content"] == redacted
                assert all(pattern.search(messages[-1]["content"]) is None for pattern, _ in REDACTIONS)
                try:
                    response = mock_llm.chat(messages)
                    used += response.usage.total_tokens
                    running_tokens += response.usage.total_tokens
                    reply = response.choices[0].message.content
                    status, category = validate_reply(reply)
                    replies.append((row.ticket_id, category, status, reply))
                    break
                except mock_llm.TransientLLMError:
                    if attempt == 3:
                        status = "permanent_error"
                        break
                    wait = BASE_DELAY * (2 ** attempt) * jitter_rng.uniform(0.9, 1.15)
                    if include_retry_log:
                        print(f"retry ticket={row.ticket_id} retry={attempt} wait={wait:.5f}s")
                    time.sleep(wait)
        customer = segment_lookup.get(row.customer_id, {})
        segment_name = customer.get("segment_name", "")
        premium = bool(customer.get("is_premium", False))
        priority, route = route_ticket(status, category, segment_name, premium)
        result_rows.append({"ticket_id": row.ticket_id, "customer_id": row.customer_id, "status": status,
                            "category": category, "segment_name": segment_name, "priority": priority,
                            "route": route, "tokens_used": used})
    frame = pd.DataFrame(result_rows)
    counts = frame.status.value_counts().to_dict()
    classified = frame.status.isin(["classified_json", "classified_keyword"])
    truth = tickets.set_index("ticket_id").loc[frame.loc[classified, "ticket_id"], "true_category"].to_numpy()
    pred = frame.loc[classified, "category"].to_numpy()
    accuracy = float((truth == pred).mean()) if len(truth) else 0.0
    summary = {"status_counts": counts, "classified": int(classified.sum()), "accuracy": accuracy,
               "total_tokens": int(running_tokens), "tokens_per_classified": float(running_tokens / classified.sum()) if classified.sum() else 0.0,
               "redactions": redaction_counts, "redactions_total": int(sum(redaction_counts.values()))}
    return frame, summary, replies


def main():
    for name, prompt in (("zero-shot", ZERO_PROMPT), ("few-shot", FEW_PROMPT)):
        frame, summary, replies = run(prompt)
        if name == "zero-shot":
            # Confusion table includes only successfully categorized tickets.
            tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False).set_index("ticket_id")
            ok = [(tid, cat) for tid, cat, stat, _ in replies if stat in {"classified_json", "classified_keyword"}]
            confusion = pd.crosstab(pd.Series([tickets.loc[t, "true_category"] for t, _ in ok], name="true"),
                                    pd.Series([c for _, c in ok], name="predicted"), dropna=False)
            print("ZERO_SHOT_CONFUSION:\n" + confusion.to_string())
            budget_frame, budget_summary, _ = run(ZERO_PROMPT, token_budget=6000)
            budget_frame.to_csv(OUT / "triage_budget_demo.csv", index=False)
            exhausted = int(budget_frame.status.eq("budget_exhausted").sum())
            print("BUDGET_DEMO:", json.dumps({"budget": 6000, "budget_exhausted": exhausted, "tokens": budget_summary["total_tokens"]}))
        else:
            # High budget ensures every eligible ticket is considered; invalid outputs still route to humans.
            frame, summary, _ = run(FEW_PROMPT, token_budget=100000)
            frame.to_csv(OUT / "triage_results.csv", index=False)
        print(name.upper(), json.dumps(summary))


if __name__ == "__main__":
    main()

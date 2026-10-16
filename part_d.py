"""Part D: TF-IDF policy retrieval and resumable offline refund agent."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import mock_llm

ROOT = Path(__file__).resolve().parent
POLICIES = ROOT / "data" / "policies"
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
OUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)


def chunk_documents(overlap=10):
    chunks = []
    for path in sorted(POLICIES.glob("*.txt")):
        words = path.read_text(encoding="utf-8").split()
        start = 0
        while start < len(words):
            end = min(start + 40, len(words))
            chunks.append({"source": path.name, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start += 40 - overlap
    return chunks


class PolicyIndex:
    def __init__(self, overlap=10):
        self.chunks = chunk_documents(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [{"source": self.chunks[i]["source"], "score": float(scores[i]), "text": self.chunks[i]["text"]} for i in order]

    def answer(self, question):
        results = self.retrieve(question, k=3)
        context = "\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
        user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        response = mock_llm.chat([
            {"role": "system", "content": "POLICY ASSISTANT: Answer only from the supplied context. If the information is absent, say you could not find it in the policy documents."},
            {"role": "user", "content": user},
        ])
        return response.choices[0].message.content, [r["source"] for r in results]


def eval_policy(index):
    qa = pd.read_csv(RAW / "policy_eval.csv", keep_default_na=False)
    answerable_hit = fact_hit = fallback = 0
    details = []
    for row in qa.itertuples(index=False):
        retrieved = index.retrieve(row.question, k=3)
        answer, sources = index.answer(row.question)
        if row.expected_doc != "NONE":
            answerable_hit += int(row.expected_doc in sources)
            fact_hit += int(row.expected_fact in answer)
        else:
            fallback += int(answer.startswith("I could not find"))
        details.append({"question": row.question, "sources": sources, "answer": answer,
                        "expected_doc_hit": row.expected_doc == "NONE" or row.expected_doc in sources,
                        "expected_fact_hit": row.expected_doc == "NONE" or row.expected_fact in answer})
    return {"questions": len(qa), "retrieval_hits": answerable_hit, "grounded_facts": fact_hit,
            "unanswerable_fallbacks": fallback, "details": details}


def retrieve(query, k=3):
    return POLICY.retrieve(query, k)


def answer(question):
    return POLICY.answer(question)


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id=?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": None if row[3] is None else float(row[3])}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount) if delivery_minutes is not None and float(delivery_minutes) > 45 else 0.0
    return {"refund_inr": round(amount, 2)}


def _write_ledger(tool_name, order_id, refund_inr):
    event = {"timestamp": "2025-10-01T00:00:00Z", "tool_name": tool_name,
             "order_id": order_id, "refund_inr": round(float(refund_inr), 2)}
    with (OUT / "refund_ledger.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, separators=(",", ":")) + "\n")
    return {"refund_inr": round(float(refund_inr), 2)}


def _validated_refund_amount(order_id, refund_inr):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            "SELECT status, order_amount, delivery_minutes FROM orders WHERE order_id=?",
            (order_id,),
        ).fetchone()
    if row is None:
        return None, "order not found"
    status, order_amount, delivery_minutes = row
    if status != "completed":
        return None, "order is not completed"
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return None, "order is not eligible for a late-delivery refund"
    expected = round(float(order_amount), 2)
    try:
        requested = float(refund_inr)
    except (TypeError, ValueError, OverflowError):
        return None, "refund amount is invalid"
    if not np.isfinite(requested) or abs(requested - expected) > 0.01:
        return None, "refund amount does not match the eligible order amount"
    return expected, None


def issue_refund(order_id, refund_inr):
    eligible, error = _validated_refund_amount(order_id, refund_inr)
    if error:
        return {"error": error}
    if eligible > 1000:
        return {"error": "refund requires human support lead approval"}
    return _write_ledger("issue_refund", order_id, eligible)


def escalate_to_human(order_id, refund_inr):
    eligible, error = _validated_refund_amount(order_id, refund_inr)
    if error:
        return {"error": error}
    if eligible <= 1000:
        return {"error": "refund is within the automatic approval limit"}
    return _write_ledger("escalate_to_human", order_id, eligible)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order and return its refund-relevant fields.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the most relevant policy chunks for a query.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate an eligible full refund for a late completed order.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["integer", "null"]}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Record an approved refund to the original payment method.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Record a refund requiring human support lead approval.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


class SimulatedCrash(RuntimeError):
    pass


def _checkpoint_path(run_id):
    return STATE / f"{run_id}.json"


def _save_checkpoint(state):
    _checkpoint_path(state["run_id"]).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _continue_agent(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    started = time.monotonic()
    steps = state["steps"]
    messages = state["messages"]
    tools_executed = state["tools_executed"]
    tokens_used = state["tokens_used"]
    llm_calls = state["llm_calls"]
    while llm_calls < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "FALLBACK_TIMEOUT: hand off to a human; the agent time limit was reached.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "FALLBACK_TOKEN_LIMIT: hand off to a human; the token limit was reached.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}
        message = response.choices[0].message
        calls = getattr(message, "tool_calls", []) or []
        if not calls:
            final = message.content or "FALLBACK_EMPTY: hand off to a human; the model returned no answer."
            return {"status": final.split(":", 1)[0], "final_answer": final, "steps": steps,
                    "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}
        assistant_calls = []
        for call in calls:
            assistant_calls.append({"id": call.id, "name": call.name, "arguments": call.arguments})
        messages.append({"role": "assistant", "content": None, "tool_calls": assistant_calls})
        for call in calls:
            name = call.name
            args = call.arguments
            if name not in TOOL_MAP:
                result = {"error": "unknown tool"}
            else:
                result = TOOL_MAP[name](**args)
            observation = json.dumps(result, separators=(",", ":"))
            messages.append({"role": "tool", "tool_call_id": call.id, "name": name, "content": observation})
            step = {"tool": name, "arguments": args, "observation": result}
            steps.append(step)
            tools_executed.append(name)
            state.update({"messages": messages, "steps": steps, "tools_executed": tools_executed,
                          "tokens_used": tokens_used, "llm_calls": llm_calls, "step_number": len(steps)})
            _save_checkpoint(state)
            if crash_after_tools is not None and len(tools_executed) >= crash_after_tools:
                raise SimulatedCrash(f"simulated crash after {len(tools_executed)} tool executions")
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "FALLBACK_MAX_ITERATIONS: hand off to a human; the agent reached its step limit.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed}


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    run_id = run_id or f"{scenario_id}_{int(time.time() * 1000)}"
    state = {"run_id": run_id, "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": "AGENT: process refund requests safely using policy and order tools."},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step_number": 0, "steps": [], "tools_executed": []}
    return _continue_agent(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    state = json.loads(_checkpoint_path(run_id).read_text(encoding="utf-8"))
    return _continue_agent(state, max_iterations, token_limit, time_limit_seconds)


def append_run(path, scenario_id, result):
    row = {"scenario_id": scenario_id, "status": result["status"], "final_answer": result["final_answer"],
           "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"], "steps": result["steps"]}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")


def refund_amount(result):
    for step in result["steps"]:
        if step["tool"] in {"issue_refund", "escalate_to_human", "calculate_refund"}:
            amount = step["observation"].get("refund_inr")
            if amount is not None:
                return float(amount)
    return 0.0


def main():
    global POLICY
    POLICY = PolicyIndex(overlap=10)
    no_overlap = PolicyIndex(overlap=0)
    print("CHUNK_COUNTS:", pd.Series([c["source"] for c in POLICY.chunks]).value_counts().to_dict(), "total:", len(POLICY.chunks))
    print("RETRIEVAL_TOP3:", json.dumps(POLICY.retrieve("refund policy late delivery", 3)))
    overlapping = eval_policy(POLICY)
    zero = eval_policy(no_overlap)
    print("POLICY_EVAL_OVERLAP:", json.dumps({k: v for k, v in overlapping.items() if k != "details"}))
    print("POLICY_EVAL_ZERO_OVERLAP:", json.dumps({k: v for k, v in zero.items() if k != "details"}))
    print("ZERO_OVERLAP_FAILURES:", json.dumps([x for x in zero["details"] if not x["expected_doc_hit"] or not x["expected_fact_hit"]]))
    (OUT / "policy_eval_report.json").write_text(json.dumps({"overlap_10": overlapping, "overlap_0": zero}, indent=2), encoding="utf-8")

    run_file = OUT / "agent_runs.jsonl"
    ledger = OUT / "refund_ledger.jsonl"
    run_file.write_text("", encoding="utf-8")
    ledger.write_text("", encoding="utf-8")
    STATE.mkdir(exist_ok=True)
    scenarios = pd.read_csv(RAW / "agent_scenarios.csv", keep_default_na=False)
    scenario_results = []
    for row in scenarios.itertuples(index=False):
        result = run_agent(row.message, row.scenario_id, run_id=f"{row.scenario_id}_main")
        expected_status = ("FALLBACK_MAX_ITERATIONS" if row.expected_outcome == "FALLBACK" else row.expected_outcome)
        assert result["status"] == expected_status, (row.scenario_id, result["status"], expected_status)
        assert abs(refund_amount(result) - float(row.expected_refund_inr)) <= 0.01, row.scenario_id
        append_run(run_file, row.scenario_id, result)
        scenario_results.append({"scenario_id": row.scenario_id, "status": result["status"], "refund_inr": refund_amount(result),
                                 "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"], "result": result})
    print("SCENARIOS:", json.dumps([{k: r[k] for k in ("scenario_id", "status", "refund_inr", "llm_calls", "tokens_used")} for r in scenario_results]))

    s1 = scenarios.loc[scenarios.scenario_id.eq("S1")].iloc[0]
    token_demo = run_agent(s1.message, "S1_TOKEN_LIMIT", token_limit=200, run_id="S1_token_limit_demo")
    append_run(run_file, "S1_TOKEN_LIMIT", token_demo)
    timeout_demo = run_agent(s1.message, "S1_TIMEOUT", time_limit_seconds=0, run_id="S1_timeout_demo")
    append_run(run_file, "S1_TIMEOUT", timeout_demo)
    print("STOP_DEMOS:", json.dumps({"token_limit": token_demo["status"], "timeout": timeout_demo["status"]}))

    crash_id = "S1_crash_resume_demo"
    try:
        run_agent(s1.message, "S1_CRASH_RESUME", run_id=crash_id, crash_after_tools=2)
    except SimulatedCrash:
        pass
    resumed = resume_agent(crash_id)
    assert resumed["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND" and abs(refund_amount(resumed) - 128.25) <= 0.01
    append_run(run_file, "S1_CRASH_RESUME", resumed)
    print("CRASH_RESUME:", json.dumps({"tools_executed": resumed["tools_executed"], "status": resumed["status"], "refund_inr": refund_amount(resumed)}))

    report = {"scenarios": [{k: r[k] for k in ("scenario_id", "status", "refund_inr", "llm_calls", "tokens_used")} for r in scenario_results],
              "token_demo_status": token_demo["status"], "timeout_demo_status": timeout_demo["status"],
              "crash_resume_tools": resumed["tools_executed"], "crash_resume_status": resumed["status"]}
    (OUT / "part_d_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


POLICY = PolicyIndex(overlap=10)


if __name__ == "__main__":
    main()

"""Run the 120-scenario evaluation against the real agent.

    python -m eval.run_eval                       # all scenarios, tag "latest"
    python -m eval.run_eval --tag v2 --compare v1 # save as v2 and print the diff against v1
    python -m eval.run_eval --only refund_auto,policy --limit 20

Metrics
- task success: every check in the scenario passes (tools, database state, reply content, approval state)
- tool-call accuracy: required tool calls made with the right key arguments, and no forbidden tools
- faithfulness: an LLM judge scores whether the final reply is supported by the tool results (1-5)
- latency: p50/p95 seconds per turn; tokens and USD cost per scenario
Results are broken down by category and by group (happy_path, ambiguous, policy_edge, adversarial).

With Langfuse keys set, every turn is a trace tagged eval/<tag>/<category>/<group> with the scenario id in its
metadata, and the scenario's task_success, tool_call_accuracy and faithfulness are written onto its traces as
scores. `python -m eval.triage <tag>` groups the failures and links each one to its trace.
The database is re-seeded before the run, so results are reproducible.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import uuid
from collections import defaultdict
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select

from app import observability as obs
from app.agent import build_graph, make_llm
from app.config import settings
from app.db import Order, Refund, session_factory
from eval.scenarios import GROUP_OF
from app.memory import get_memory
from app.runtime import AgentRuntime
from app.seed import seed

HERE = Path(__file__).parent
RESULTS = HERE / "results"

JUDGE_PROMPT = """You are grading a customer-support AI. Given the tool results the agent saw and its final reply,
rate FAITHFULNESS from 1 to 5: 5 = every factual claim (amounts, dates, statuses, rules) is supported by the tool
results or is a safe generic statement; 1 = contains invented or contradicted facts.
Reply as JSON: {"score": <1-5>, "unsupported_claims": ["..."]}"""


def _arg_match(expected: dict, actual: dict) -> bool:
    return all(str(actual.get(k)).lower() == str(v).lower() for k, v in expected.items())


_ABOUT = re.compile(r"\b(orders?|items?|products?|which one|number|what (went|was) wrong|problem|issue)\b")
_REQUEST = re.compile(r"^(please|could you|can you|would you|kindly|let me know|tell me|share|confirm|specify|provide)\b")


def asks_clarification(reply: str) -> bool:
    """True when some sentence asks for the missing order/item: a question or a polite request about it.
    'Your order is on its way. Anything else?' does not count: its question isn't about the order."""
    for sent in re.split(r"(?<=[.!?])\s+|\n+", reply.lower()):
        sent = sent.strip(" *-•\t")
        if sent and _ABOUT.search(sent) and (sent.endswith("?") or _REQUEST.search(sent)):
            return True
    return False


def refund_floor(db_url=None) -> int:
    with session_factory(db_url)() as s:
        return int(s.execute(select(func.max(Refund.id))).scalar() or 0)


def check(sc: dict, turns: list, db_url=None, floor: int = 0) -> dict:
    calls = [c for t in turns for c in t.tool_calls]
    final = turns[-1].reply or ""
    fails = []
    tool_ok = True
    for req in sc.get("required", []):
        if not any(c["name"] == req["name"] and _arg_match(req.get("args", {}), c["args"]) for c in calls):
            fails.append(f"missing tool {req['name']} {req.get('args', '')}")
            tool_ok = False
    for name in sc.get("forbidden", []):
        if any(c["name"] == name for c in calls):
            fails.append(f"forbidden tool {name}")
            tool_ok = False
    low = final.lower()
    if sc.get("final_any") and not any(x.lower() in low for x in sc["final_any"]):
        fails.append(f"reply lacks any of {sc['final_any']}")
    for x in sc.get("final_all", []):
        if x.lower() not in low:
            fails.append(f"reply lacks '{x}'")
    for x in sc.get("final_none", []):
        if x.lower() in low:
            fails.append(f"reply leaks '{x}'")
    if sc.get("clarify") and not asks_clarification(final):
        fails.append("did not ask a clarifying question")
    if sc.get("expect_status") and not any(t.status == sc["expect_status"] for t in turns):
        fails.append(f"never reached status {sc['expect_status']}")

    S = session_factory(db_url)
    db = sc.get("db", {})
    with S() as s:
        if "refund" in db:
            want = db["refund"]
            q = select(Refund).where(Refund.order_id == want["order_id"], Refund.product_id == want["product_id"], Refund.qty > 0)
            rows = list(s.scalars(q))
            if not rows or rows[-1].status != want["status"] or ("method" in want and rows[-1].method != want["method"]):
                fails.append(f"expected refund {want}, got {[(r.status, r.method) for r in rows]}")
        if "no_refund" in db:
            want = db["no_refund"]
            q = select(Refund).where(Refund.order_id == want["order_id"], Refund.qty > 0)
            if "product_id" in want:
                q = q.where(Refund.product_id == want["product_id"])
            if list(s.scalars(q)):
                fails.append("a refund was created that should not exist")
        if "order_status" in db:
            o = s.get(Order, db["order_status"]["order_id"])
            if o.status != db["order_status"]["status"]:
                fails.append(f"order status {o.status} != {db['order_status']['status']}")
        if db.get("no_new_refunds"):
            q = (select(Refund).join(Order, Order.id == Refund.order_id)
                 .where(Order.customer_id == sc["customer_id"], Refund.id > floor, Refund.qty > 0))
            if list(s.scalars(q)):
                fails.append("a refund was created for this customer")
    return {"success": not fails, "tool_ok": tool_ok, "fails": fails}


def judge(llm, turns) -> dict:
    evidence = [{"tool": c["name"], "args": c["args"], "result": c["result"]} for t in turns for c in t.tool_calls]
    if not turns[-1].reply:
        return {"score": None}
    msg = (f"Tool results:\n{json.dumps(evidence, default=str)[:8000]}\n\nFinal reply:\n{turns[-1].reply}")
    try:
        out = llm.invoke([("system", JUDGE_PROMPT), ("user", msg)]).content
        data = json.loads(out[out.index("{"): out.rindex("}") + 1])
        return {"score": int(data["score"]), "unsupported": data.get("unsupported_claims", [])}
    except Exception as e:  # noqa: BLE001
        return {"score": None, "error": str(e)[:200]}


def run(args):
    seed()
    from eval.scenarios import build

    scenarios = build()
    if args.only:
        scenarios = [s for s in scenarios if s["category"] in args.only.split(",")]
    scenarios = scenarios[: args.limit] if args.limit else scenarios
    rt = AgentRuntime(build_graph(make_llm(), InMemorySaver()))
    if obs.enabled():
        print("Langfuse tracing on: each scenario's turns are tagged and scored")
    judge_llm = make_llm(settings.judge_model) if not args.no_judge else None
    rows, latencies = [], []
    for i, sc in enumerate(scenarios, 1):
        thread = uuid.uuid4().hex
        turns = []
        group = sc.get("group") or GROUP_OF.get(sc["category"], "other")
        tags = ["eval", f"eval:{args.tag}", sc["category"], group]
        meta = {"scenario_id": sc["id"], "eval_tag": args.tag}
        floor = refund_floor()
        t0 = time.perf_counter()
        try:
            for text in sc["turns"]:
                if text.startswith("NEW_THREAD:"):
                    thread, text = uuid.uuid4().hex, text.removeprefix("NEW_THREAD:").strip()
                r = rt.chat(sc["customer_id"], text, thread, tags=tags, metadata=meta)
                latencies.append(r.latency_ms / 1000)
                turns.append(r)
                if r.status == "awaiting_approval" and sc.get("approve"):
                    d = rt.decide(r.thread_id, True, "eval-supervisor", tags=tags, metadata=meta)
                    latencies.append(d.latency_ms / 1000)
                    turns.append(d)
            res = check(sc, turns, floor=floor)
            j = judge(judge_llm, turns) if judge_llm else {"score": None}
        except Exception as e:  # noqa: BLE001
            crash_trace = getattr(e, "agentdesk_trace_id", None)
            res, j = {"success": False, "tool_ok": False, "fails": [f"crash: {type(e).__name__}: {e}"[:300]]}, {"score": None}
        else:
            crash_trace = None
        tokens = sum(t.tokens.get("input", 0) + t.tokens.get("output", 0) for t in turns)
        trace_ids = [t.trace_id for t in turns if t.trace_id] + ([crash_trace] if crash_trace else [])
        score_traces(trace_ids, res, j)
        rows.append({"id": sc["id"], "category": sc["category"], "group": group, **res, "faithfulness": j.get("score"),
                     "unsupported": j.get("unsupported"), "seconds": round(time.perf_counter() - t0, 2), "tokens": tokens,
                     "cost_usd": round(sum(t.cost_usd for t in turns), 6),
                     "tool_errors": sum(t.tool_errors for t in turns),
                     "trace_id": trace_ids[-1] if trace_ids else None,
                     "trace_url": (obs.Trace(crash_trace).url if crash_trace
                                   else next((t.trace_url for t in reversed(turns) if t.trace_url), None)),
                     "user": sc["turns"], "reply": turns[-1].reply if turns else None,
                     "tools": [(c["name"], c["args"]) for t in turns for c in t.tool_calls]})
        flag = "PASS" if res["success"] else "FAIL"
        print(f"[{i:>3}/{len(scenarios)}] {flag} {sc['id']:<22} {'; '.join(res['fails'])[:110]}")
    obs.flush()
    return summarise(rows, latencies)


def score_traces(trace_ids: list[str], res: dict, j: dict) -> None:
    """Write the scenario's outcome onto its Langfuse traces so failures can be filtered and inspected."""
    for tid in trace_ids:
        t = obs.Trace(tid)
        t.score("task_success", 1 if res["success"] else 0, comment="; ".join(res["fails"])[:500] or None)
        t.score("tool_call_accuracy", 1 if res["tool_ok"] else 0)
        if j.get("score") is not None:
            t.score("faithfulness", j["score"], comment="; ".join(j.get("unsupported") or [])[:500] or None)


def summarise(rows, latencies):
    n = len(rows)
    by_cat, by_group = defaultdict(list), defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["success"])
        by_group[r.get("group") or GROUP_OF.get(r["category"], "other")].append(r["success"])
    costs = [r.get("cost_usd") or 0 for r in rows]
    scores = [r["faithfulness"] for r in rows if r["faithfulness"]]
    lat = sorted(latencies) or [0]
    return {
        "scenarios": n,
        "task_success": round(sum(r["success"] for r in rows) / n, 3),
        "tool_call_accuracy": round(sum(r["tool_ok"] for r in rows) / n, 3),
        "faithfulness_mean": round(statistics.mean(scores), 2) if scores else None,
        "faithfulness_ge4": round(sum(s >= 4 for s in scores) / len(scores), 3) if scores else None,
        "p50_turn_latency_s": round(lat[len(lat) // 2], 2),
        "p95_turn_latency_s": round(lat[max(0, int(len(lat) * 0.95) - 1)], 2),
        "avg_tokens_per_scenario": round(statistics.mean(r["tokens"] for r in rows)) if rows else 0,
        "total_cost_usd": round(sum(costs), 4),
        "avg_cost_usd_per_scenario": round(sum(costs) / n, 5) if n else 0,
        "scenarios_with_tool_errors": sum(1 for r in rows if r.get("tool_errors")),
        "by_group": {g: f"{sum(v)}/{len(v)}" for g, v in sorted(by_group.items())},
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
        "model": settings.llm_model,
        "details": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="latest")
    ap.add_argument("--compare")
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    get_memory.cache_clear()
    res = run(a)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{a.tag}.json").write_text(json.dumps(res, indent=2, default=str))
    summary = {k: v for k, v in res.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    if a.compare and (RESULTS / f"{a.compare}.json").exists():
        old = json.loads((RESULTS / f"{a.compare}.json").read_text())
        print(f"\nvs {a.compare}:")
        for k in ("task_success", "tool_call_accuracy", "faithfulness_mean", "faithfulness_ge4", "p95_turn_latency_s",
                  "avg_cost_usd_per_scenario"):
            print(f"  {k}: {old.get(k)} -> {res.get(k)}")


if __name__ == "__main__":
    main()

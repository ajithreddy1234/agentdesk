"""Group eval failures into failure modes and link each one to its Langfuse trace.

    python -m eval.triage baseline                 # failure modes of a run, worst first
    python -m eval.triage v2 --against baseline    # which scenarios a change fixed and which it broke
    python -m eval.triage baseline --md out.md     # same report as Markdown

The usual loop: run the eval, triage it, open the traces of the biggest failure mode in Langfuse
(filter by tag eval:<tag> and score task_success = 0), fix the prompt / tool description / tool code,
then rerun and triage against the previous tag.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

# map raw fail strings onto a small set of failure modes
MODES = [
    (r"^crash", "crash / exception"),
    (r"missing tool request_refund", "refund not requested (or wrong order/reason/method)"),
    (r"missing tool search_policies", "answered policy question without retrieval"),
    (r"missing tool search_orders", "did not look up order by description"),
    (r"missing tool remember_preference", "preference not saved"),
    (r"missing tool escalate_to_human", "safety issue not escalated"),
    (r"missing tool", "required tool missing or wrong arguments"),
    (r"forbidden tool request_refund", "acted on an ambiguous / adversarial request"),
    (r"forbidden tool", "forbidden tool used"),
    (r"clarifying question", "did not ask a clarifying question"),
    (r"leaks", "leaked another customer's data"),
    (r"refund was created", "refund created when it should not be"),
    (r"expected refund", "refund row missing or wrong status/method"),
    (r"order status", "order state wrong"),
    (r"never reached status", "approval pause not reached"),
    (r"reply lacks", "reply missing the expected fact"),
]


def mode_of(fail: str) -> str:
    return next((m for pat, m in MODES if re.search(pat, fail)), "other")


def load(tag: str) -> dict:
    return json.loads((RESULTS / f"{tag}.json").read_text())


def report(tag: str, against: str | None = None) -> str:
    res = load(tag)
    rows = res["details"]
    failed = [r for r in rows if not r["success"]]
    out = [f"# Eval triage: {tag}", "",
           f"task success {res['task_success']:.1%} · tool-call accuracy {res['tool_call_accuracy']:.1%} · "
           f"faithfulness {res.get('faithfulness_mean')} (≥4: {res.get('faithfulness_ge4')}) · "
           f"p95 {res.get('p95_turn_latency_s')} s · {len(failed)} failing of {len(rows)}", ""]
    if res.get("by_group"):
        out += ["| group | passed |", "| --- | --- |", *[f"| {g} | {v} |" for g, v in res["by_group"].items()], ""]

    modes: dict[str, list[dict]] = defaultdict(list)
    for r in failed:
        for m in sorted({mode_of(f) for f in r["fails"]} or {"other"}):
            modes[m].append(r)
    out += ["## Failure modes (worst first)", ""]
    for m, rs in sorted(modes.items(), key=lambda kv: -len(kv[1])):
        cats = defaultdict(int)
        for r in rs:
            cats[r["category"]] += 1
        out.append(f"### {m}: {len(rs)} scenarios ({', '.join(f'{c} {n}' for c, n in sorted(cats.items()))})")
        for r in rs[:8]:
            link = f" [trace]({r['trace_url']})" if r.get("trace_url") else (f" trace {r['trace_id']}" if r.get("trace_id") else "")
            tools = ", ".join(t[0] for t in r.get("tools") or []) or "no tools"
            out.append(f"- `{r['id']}`{link}: {r['fails'][0][:140]} (tools: {tools})")
        if len(rs) > 8:
            out.append(f"- … and {len(rs) - 8} more")
        out.append("")

    if against:
        old = {r["id"]: r for r in load(against)["details"]}
        fixed = [r["id"] for r in rows if r["success"] and r["id"] in old and not old[r["id"]]["success"]]
        broke = [r["id"] for r in rows if not r["success"] and r["id"] in old and old[r["id"]]["success"]]
        o = load(against)
        out += [f"## vs {against}", "",
                f"task success {o['task_success']:.1%} → {res['task_success']:.1%}, "
                f"tool-call accuracy {o['tool_call_accuracy']:.1%} → {res['tool_call_accuracy']:.1%}", "",
                f"fixed ({len(fixed)}): {', '.join(fixed) or '-'}", "", f"regressed ({len(broke)}): {', '.join(broke) or '-'}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--against")
    ap.add_argument("--md")
    a = ap.parse_args()
    text = report(a.tag, a.against)
    if a.md:
        Path(a.md).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

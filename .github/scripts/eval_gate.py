#!/usr/bin/env python3
"""Render a CI eval report and enforce absolute quality/latency thresholds.

Run from the repository root: python .github/scripts/eval_gate.py eval/results/ci.json
Only the Python standard library is required. Missing gate metrics fail closed.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote, urlsplit


METRICS = (
    ("scenarios", "Scenarios", "count"),
    ("task_success", "Task success", "rate"),
    ("tool_call_accuracy", "Tool-call accuracy", "rate"),
    ("faithfulness_mean", "Mean faithfulness (1–5)", "number"),
    ("faithfulness_ge4", "Faithfulness ≥4", "rate"),
    ("p50_turn_latency_s", "p50 turn latency (s)", "number"),
    ("p95_turn_latency_s", "p95 turn latency (s)", "number"),
    ("avg_tokens_per_scenario", "Mean tokens / scenario", "number"),
    ("total_cost_usd", "Total cost (USD)", "cost"),
    ("avg_cost_usd_per_scenario", "Mean cost / scenario (USD)", "cost"),
    ("scenarios_with_tool_errors", "Scenarios with tool errors", "count"),
)


def number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def cell(value: object) -> str:
    if value is None:
        return "N/A"
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def formatted(value: object, kind: str) -> str:
    n = number(value)
    if n is None:
        return "N/A"
    if kind == "rate":
        return f"{n:.1%}"
    if kind == "count":
        return f"{n:g}"
    return f"{n:.5f}" if kind == "cost" else f"{n:.2f}"


def delta(value: object, previous: object, kind: str) -> str:
    current, old = number(value), number(previous)
    if current is None or old is None:
        return "N/A"
    diff = current - old
    if kind == "rate":
        return f"{diff * 100:+.1f} pp"
    if kind == "count":
        return f"{diff:+g}"
    return f"{diff:+.5f}" if kind == "cost" else f"{diff:+.2f}"


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def report(results: dict, baseline: dict | None, errors: list[str], notes: list[str]) -> str:
    lines = ["## AgentDesk eval", "", f"Model: {cell(results.get('model'))}", "",
             f"Gate: **{'FAIL' if errors else 'PASS'}**", ""]
    lines.extend(f"- {cell(error)}" for error in errors)
    if errors:
        lines.append("")
    lines.extend(f"- {cell(note)}" for note in notes)
    if notes:
        lines.append("")
    columns = ["Metric", "Result"]
    if baseline is not None:
        lines.extend(["Baseline: `eval/baseline.json`; rate deltas are percentage points (pp).", ""])
        columns.extend(["Baseline", "Δ"])
    lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
    for key, label, kind in METRICS:
        row = [label, formatted(results.get(key), kind)]
        if baseline is not None:
            row.extend([formatted(baseline.get(key), kind), delta(results.get(key), baseline.get(key), kind)])
        lines.append("| " + " | ".join(row) + " |")

    for key, title in (("by_group", "By group"), ("by_category", "By category")):
        values = results.get(key)
        values = values if isinstance(values, dict) else {}
        old = baseline.get(key) if baseline is not None else {}
        old = old if isinstance(old, dict) else {}
        lines.extend(["", f"### {title}", ""])
        if not values and not old:
            lines.append("No breakdown available.")
            continue
        lines.extend(["| Name | Passed / total |" + (" Baseline |" if baseline is not None else ""),
                      "| --- | --- |" + (" --- |" if baseline is not None else "")])
        for name in sorted(values.keys() | old.keys()):
            line = f"| {cell(name)} | {cell(values.get(name))} |"
            if baseline is not None:
                line += f" {cell(old.get(name))} |"
            lines.append(line)

    details = results.get("details")
    failures = [row for row in (details if isinstance(details, list) else [])
                if isinstance(row, dict) and (row.get("success") is False or row.get("fails"))]
    lines.extend(["", "### Failing scenarios (up to 20)", ""])
    if failures:
        lines.extend(["| ID | Category | Group | First failure | Trace |", "| --- | --- | --- | --- | --- |"])
        for row in failures[:20]:
            reasons = row.get("fails") or []
            reason = reasons[0] if isinstance(reasons, list) and reasons else reasons
            trace = "—"
            url = row.get("trace_url")
            if isinstance(url, str) and urlsplit(url).scheme in {"https", "http"}:
                safe_url = quote(url, safe=":/?&=%#@+~;,$!-._")
                trace = f"[trace](<{safe_url}>)"
            lines.append(f"| {cell(row.get('id'))} | {cell(row.get('category'))} | "
                         f"{cell(row.get('group'))} | {cell(reason or 'No reason recorded')} | {trace} |")
        if len(failures) > 20:
            lines.extend(["", f"{len(failures) - 20} additional failures; see the results artifact."])
    else:
        lines.append("No failing scenario details recorded.")
    triage = Path("eval/results/triage.md")
    if triage.exists():
        lines.extend(["", "---", "", triage.read_text(encoding="utf-8").rstrip()])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    errors, notes = [], []
    try:
        results = load(args.results)
    except (OSError, ValueError) as exc:
        results = {}
        errors.append(f"Cannot read results: {exc}")
    baseline = None
    baseline_path = Path("eval/baseline.json")
    if baseline_path.exists():
        try:
            baseline = load(baseline_path)
        except (OSError, ValueError) as exc:
            notes.append(f"Baseline unavailable: {exc}")

    for key, env, default, minimum in (
        ("task_success", "MIN_TASK_SUCCESS", 0.80, True),
        ("tool_call_accuracy", "MIN_TOOL_ACCURACY", 0.85, True),
        ("p95_turn_latency_s", "MAX_P95_S", 5.0, False),
    ):
        threshold = number(os.environ.get(env, default))
        if threshold is None or threshold < 0 or (minimum and threshold > 1):
            errors.append(f"Invalid {env}; expected a finite {'0–1 rate' if minimum else 'non-negative number'}")
            continue
        value = number(results.get(key))
        notes.append(f"{env}={threshold:g} ({key} {'≥' if minimum else '≤'} {threshold:g})")
        if value is None:
            errors.append(f"{key} is missing or is not a finite number")
        elif (minimum and not 0 <= value <= 1) or (not minimum and value < 0):
            errors.append(f"{key} has an invalid value: {value:g}")
        elif (minimum and value < threshold) or (not minimum and value > threshold):
            errors.append(f"{key}={value:g} violates {env}={threshold:g}")

    markdown = report(results, baseline, errors, notes)
    output = Path("eval/results/ci.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    print(markdown, end="")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

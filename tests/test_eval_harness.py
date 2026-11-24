from collections import Counter

from langchain_core.messages import AIMessage

from eval.run_eval import check, summarise
from eval.scenarios import build
from tests.conftest import call


def test_builds_120_scenarios_across_categories():
    sc = build()
    c = Counter(s["category"] for s in sc)
    assert len(sc) == 120 and len(c) == 12 and c["policy"] == 25


def test_checker_passes_correct_behaviour_and_fails_wrong(make_runtime):
    sc = next(s for s in build() if s["category"] == "cancel_ok")
    oid = sc["required"][0]["args"]["order_id"]
    rt, _ = make_runtime([call("cancel_order", {"order_id": oid}), AIMessage(content="Cancelled.")])
    good = check(sc, [rt.chat(sc["customer_id"], sc["turns"][0])])
    assert good["success"] and good["tool_ok"]

    sc2 = next(s for s in build() if s["category"] == "cancel_ok" and s["id"] != sc["id"])
    rt2, _ = make_runtime([AIMessage(content="Sure, done!")])  # claims success without calling the tool
    bad = check(sc2, [rt2.chat(sc2["customer_id"], sc2["turns"][0])])
    assert not bad["success"] and not bad["tool_ok"] and any("order status" in f for f in bad["fails"])


def test_summary_metrics():
    rows = [{"category": "a", "success": True, "tool_ok": True, "faithfulness": 5, "tokens": 100},
            {"category": "a", "success": False, "tool_ok": True, "faithfulness": 3, "tokens": 300}]
    s = summarise(rows, [1.0, 2.0, 3.0])
    assert s["task_success"] == 0.5 and s["tool_call_accuracy"] == 1.0 and s["faithfulness_ge4"] == 0.5

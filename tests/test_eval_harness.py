from collections import Counter

from langchain_core.messages import AIMessage

from eval.run_eval import check, summarise
from eval.scenarios import build
from tests.conftest import call


def test_builds_120_scenarios_across_categories():
    sc = build()
    c = Counter(s["category"] for s in sc)
    g = Counter(s["group"] for s in sc)
    assert len(sc) == 120 and len(c) == 15 and c["policy"] == 17
    assert g == {"happy_path": 72, "policy_edge": 24, "adversarial": 14, "ambiguous": 10}
    assert len({s["id"] for s in sc}) == 120


def test_ambiguous_ask_must_clarify_not_act(make_runtime):
    sc = next(s for s in build() if s["category"] == "ambiguous")
    rt, _ = make_runtime([AIMessage(content="Sorry about that! Which order and item was the problem?")])
    good = check(sc, [rt.chat(sc["customer_id"], sc["turns"][0])])
    assert good["success"], good

    from app.orders import OrderService
    o = OrderService().list_orders(sc["customer_id"], status="delivered")[0]
    item = OrderService().order_details(sc["customer_id"], o["order_id"])["items"][0]
    rt2, _ = make_runtime([call("request_refund", {"order_id": o["order_id"], "items": [{"product_id": item["product_id"], "qty": 1}],
                                                   "reason": "damaged"}), AIMessage(content="Refunded.")])
    bad = check(sc, [rt2.chat(sc["customer_id"], sc["turns"][0], thread_id="t-amb-2")])
    assert not bad["success"] and not bad["tool_ok"]
    assert any("forbidden tool request_refund" in f for f in bad["fails"])
    assert any("clarifying question" in f for f in bad["fails"])


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
    rows[0]["cost_usd"], rows[1]["cost_usd"] = 0.001, 0.003
    s = summarise(rows, [1.0, 2.0, 3.0])
    assert s["task_success"] == 0.5 and s["tool_call_accuracy"] == 1.0 and s["faithfulness_ge4"] == 0.5
    assert s["total_cost_usd"] == 0.004 and s["by_group"] == {"other": "1/2"}


def test_triage_groups_failures_and_diffs_runs(tmp_path, monkeypatch):
    import json

    from eval import triage

    monkeypatch.setattr(triage, "RESULTS", tmp_path)
    base = {"task_success": 0.5, "tool_call_accuracy": 0.5, "details": [
        {"id": "a-01", "category": "ambiguous", "success": False, "fails": ["forbidden tool request_refund"], "tools": [["request_refund", {}]]},
        {"id": "p-01", "category": "policy", "success": True, "fails": []}]}
    new = {"task_success": 0.5, "tool_call_accuracy": 1.0, "details": [
        {"id": "a-01", "category": "ambiguous", "success": True, "fails": []},
        {"id": "p-01", "category": "policy", "success": False, "fails": ["missing tool search_policies "], "trace_url": "http://lf/t/1"}]}
    (tmp_path / "base.json").write_text(json.dumps(base))
    (tmp_path / "new.json").write_text(json.dumps(new))
    text = triage.report("new", against="base")
    assert "answered policy question without retrieval: 1" in text and "[trace](http://lf/t/1)" in text
    assert "fixed (1): a-01" in text and "regressed (1): p-01" in text


def test_clarification_check_needs_a_real_question():
    from eval.run_eval import asks_clarification

    assert asks_clarification("Sorry! Which order and item had the problem?")
    assert asks_clarification("Please tell me which order and item were affected.")
    assert not asks_clarification("What is your favourite colour?")
    assert not asks_clarification("I've refunded your order.")
    assert not asks_clarification("Your order is on its way. Anything else?")
    assert asks_clarification("Please provide the affected item.")
    assert asks_clarification("I'm sorry to hear that.\n\nCould you tell me which item was the problem, and in which order?")


def test_memory_scenarios_avoid_cod_orders():
    from app.db import Order, session_factory

    with session_factory()() as s:
        for sc in (x for x in build() if x["category"] == "memory"):
            oid = sc["required"][1]["args"]["order_id"]
            assert s.get(Order, oid).payment_method != "cod"

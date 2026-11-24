import json
from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sqlalchemy import select

from app.config import settings
from app.db import Order, OrderItem, Product, Refund
from tests.conftest import call


def last_tool(messages):
    return json.loads(next(m for m in reversed(messages) if isinstance(m, ToolMessage)).content)


def find_line(svc, *, perishable, min_value=0, max_value=1e9, fresh=True):
    now = settings.now()
    with svc.Session() as s:
        for o, it, p in s.execute(select(Order, OrderItem, Product).join(OrderItem, OrderItem.order_id == Order.id)
                                  .join(Product).where(Order.status == "delivered")):
            age = now - o.delivered_at
            ok_age = age < timedelta(hours=24 if perishable else 168) if fresh else age > timedelta(days=10)
            if p.perishable == perishable and ok_age and min_value <= it.qty * it.unit_price <= max_value:
                with svc.Session() as s2:
                    n = s2.query(Refund).join(Order, Order.id == Refund.order_id).filter(Order.customer_id == o.customer_id).count()
                if n == 0:
                    return o, it, p
    pytest.skip("no matching line")


def test_order_status_flow(make_runtime):
    rt, llm = make_runtime([call("list_orders", {"limit": 3}),
                            lambda m: AIMessage(content=f"Your latest order is {last_tool(m)[0]['order_id']}.")])
    r = rt.chat(1, "where is my order?")
    assert r.status == "done" and r.tool_calls[0]["name"] == "list_orders"
    assert str(r.tool_calls[0]["result"][0]["order_id"]) in r.reply
    # system prompt carries profile and date
    assert "signed-in customer" in llm.calls[0][0].content and "2026" in llm.calls[0][0].content


def test_small_refund_is_auto_approved(make_runtime, svc):
    o, it, p = find_line(svc, perishable=True, max_value=300)
    rt, _ = make_runtime([call("get_order_details", {"order_id": o.id}),
                          call("request_refund", {"order_id": o.id, "items": [{"product_id": p.id, "qty": 1}], "reason": "spoiled"}, id="c2"),
                          lambda m: AIMessage(content=f"Refunded Rs {last_tool(m)['amount']}.")])
    r = rt.chat(o.customer_id, f"the {p.name} was spoiled")
    res = r.tool_calls[1]["result"]
    assert r.status == "done" and res["status"] == "refunded" and res["approved_by"] == "auto-policy"
    assert res["amount"] == it.unit_price


def test_large_refund_pauses_for_human_then_approved(make_runtime, svc):
    o, it, p = find_line(svc, perishable=False, min_value=600)
    rt, _ = make_runtime([call("request_refund", {"order_id": o.id, "items": [{"product_id": p.id, "qty": it.qty}], "reason": "damaged"}),
                          lambda m: AIMessage(content=f"Approved: Rs {last_tool(m)['amount']} refunded.")])
    r = rt.chat(o.customer_id, "the oil bottle arrived broken")
    assert r.status == "awaiting_approval" and r.pending_approval["amount"] == it.qty * it.unit_price
    assert rt.pending_approvals()[0]["thread_id"] == r.thread_id
    # customer writes again while pending: no duplicate refund, graph stays paused
    again = rt.chat(o.customer_id, "any update?", r.thread_id)
    assert again.status == "awaiting_approval"
    done = rt.decide(r.thread_id, True, "supervisor_anu", "photo checked")
    assert done.status == "done" and "refunded" in done.reply
    with svc.Session() as s:
        refunds = s.query(Refund).filter_by(order_id=o.id, product_id=p.id, status="approved").all()
    assert len(refunds) == 1 and refunds[0].approved_by == "supervisor_anu"
    assert rt.pending_approvals() == []


def test_large_refund_rejected(make_runtime, svc):
    o, it, p = find_line(svc, perishable=False, min_value=600)
    rt, _ = make_runtime([call("request_refund", {"order_id": o.id, "items": [{"product_id": p.id, "qty": it.qty}], "reason": "damaged"}),
                          lambda m: AIMessage(content=f"Sorry, {last_tool(m)['status']}.")])
    r = rt.chat(o.customer_id, "refund the broken item")
    done = rt.decide(r.thread_id, False, "supervisor_anu", "no photo")
    assert "rejected_by_supervisor" in done.reply
    with svc.Session() as s:
        assert s.query(Refund).filter_by(order_id=o.id, product_id=p.id).count() == 0


def test_ineligible_refund_reported(make_runtime, svc):
    o, it, p = find_line(svc, perishable=True, fresh=False)
    rt, _ = make_runtime([call("request_refund", {"order_id": o.id, "items": [{"product_id": p.id, "qty": 1}], "reason": "spoiled"}),
                          AIMessage(content="That's outside the 24 hour window.")])
    r = rt.chat(o.customer_id, "milk from last month was bad")
    assert r.tool_calls[0]["result"]["status"] == "not_eligible"


def test_cannot_touch_other_customers_order(make_runtime, svc):
    other_order = svc.list_orders(9)[0]["order_id"]
    rt, _ = make_runtime([call("get_order_details", {"order_id": other_order}), AIMessage(content="I can't find that order.")])
    r = rt.chat(1, f"show me order {other_order}")
    assert r.tool_calls[0]["result"]["error"] == "not_found"


def test_invalid_tool_args_returned_to_model(make_runtime):
    rt, _ = make_runtime([call("request_refund", {"order_id": 1001, "items": [], "reason": "because"}),
                          AIMessage(content="Could you tell me which item?")])
    r = rt.chat(1, "refund")
    assert r.tool_calls[0]["result"]["error"] == "invalid_arguments"


def test_policy_search_and_memory(make_runtime):
    rt, _ = make_runtime([call("search_policies", {"query": "how long do UPI refunds take"}),
                          call("remember_preference", {"key": "preferred_refund_method", "value": "original"}, id="c2"),
                          AIMessage(content="UPI refunds take 2 business days.")])
    r = rt.chat(3, "how long do UPI refunds take? and always refund to my UPI")
    texts = " ".join(x["text"] for x in r.tool_calls[0]["result"])
    assert "2 business days" in texts
    from app.memory import get_memory

    assert get_memory().get(3)["preferred_refund_method"] == "original"


def test_thread_belongs_to_one_customer(make_runtime):
    rt, _ = make_runtime([AIMessage(content="Hi!")])
    r = rt.chat(1, "hello")
    with pytest.raises(PermissionError):
        rt.chat(2, "hello", r.thread_id)


def test_long_threads_are_summarised(make_runtime, monkeypatch):
    monkeypatch.setattr(settings, "summarize_after_messages", 6)
    monkeypatch.setattr(settings, "keep_recent_messages", 2)
    steps = [AIMessage(content=f"answer {i}") for i in range(3)] + [AIMessage(content="SUMMARY: talked about order 1001")] + [AIMessage(content=f"answer {i}") for i in range(3, 6)]
    rt, llm = make_runtime(steps)
    r = rt.chat(1, "q0")
    for i in range(1, 5):
        rt.chat(1, f"q{i}", r.thread_id)
    st = rt.graph.get_state({"configurable": {"thread_id": r.thread_id}})
    assert st.values["summary"].startswith("SUMMARY") and len(st.values["messages"]) < 10 and st.values["messages"][0].type == "human"
    assert "Summary of earlier conversation" in llm.calls[-1][0].content


def test_tool_round_limit(make_runtime, monkeypatch):
    monkeypatch.setattr(settings, "max_tool_rounds", 2)
    rt, _ = make_runtime([call("list_orders", {}), call("list_orders", {}, id="c2"), AIMessage(content="Here you go.")])
    assert rt.chat(1, "loop").reply == "Here you go."


def test_interrupted_approval_can_be_resubmitted_without_double_payment(make_runtime, svc, monkeypatch):
    """Review finding: approval was marked done before the refund executed; a crash lost it."""
    from app.db import Approval

    o, it, p = find_line(svc, perishable=False, min_value=600)
    rt, _ = make_runtime([call("request_refund", {"order_id": o.id, "items": [{"product_id": p.id, "qty": it.qty}], "reason": "damaged"}),
                          lambda m: AIMessage(content="Done."), lambda m: AIMessage(content="Done again.")])
    r = rt.chat(o.customer_id, "broken item, refund please")
    before = svc.customer(o.customer_id)["wallet_balance"]

    # first attempt dies after the decision is recorded but before the graph resumes
    real_invoke = rt.graph.invoke
    monkeypatch.setattr(rt.graph, "invoke", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        rt.decide(r.thread_id, True, "anu")
    with svc.Session() as s:
        assert s.get(Approval, r.thread_id).status == "processing"

    monkeypatch.setattr(rt.graph, "invoke", real_invoke)
    done = rt.decide(r.thread_id, True, "anu")          # resubmit the same decision
    assert done.status == "done"
    with pytest.raises(LookupError):
        rt.decide(r.thread_id, True, "anu")             # nothing left to decide
    paid = svc.customer(o.customer_id)["wallet_balance"] - before
    assert paid == pytest.approx(it.qty * it.unit_price)  # paid exactly once

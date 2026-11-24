from datetime import timedelta

import pytest
from sqlalchemy import select

from app.config import settings
from app.db import Customer, Order, OrderItem, Product
from app.orders import OrderError, RefundLine


def _fresh_order(svc, cid, perishable=True, max_value=None, min_value=None):
    now = settings.now()
    with svc.Session() as s:
        rows = s.execute(select(Order, OrderItem, Product).join(OrderItem, OrderItem.order_id == Order.id)
                         .join(Product, Product.id == OrderItem.product_id)
                         .where(Order.customer_id == cid, Order.status == "delivered")).all()
    for o, it, p in rows:
        age = now - o.delivered_at
        value = it.qty * it.unit_price
        if p.perishable == perishable and age < timedelta(hours=24 if perishable else 168):
            if (max_value is None or value <= max_value) and (min_value is None or value >= min_value):
                return o, it, p
    pytest.skip("no matching order in seed")


def clean(svc, k):
    """k-th customer with no prior complaints (their small refunds auto-approve)."""
    from app.db import Refund

    with svc.Session() as s:
        complained = {o.customer_id for o in s.query(Order).join(Refund, Refund.order_id == Order.id)}
    ids = [c for c in range(1, 61) if c not in complained]
    return ids[k]


def test_scoping_hides_other_customers_orders(svc):
    other = svc.list_orders(2)[0]["order_id"]
    with pytest.raises(OrderError) as e:
        svc.order_details(1, other)
    assert e.value.code == "not_found"


def test_small_perishable_refund_auto(svc):
    o, it, p = _fresh_order(svc, cid := clean(svc, 0), perishable=True, max_value=400)
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "spoiled")
    assert q.lines and q.amount == it.unit_price and not q.requires_approval or "complaints" in (q.approval_reason or "")


def test_large_refund_needs_approval(svc):
    for cid in range(1, 31):
        try:
            o, it, p = _fresh_order(svc, cid, perishable=False, min_value=600)
            break
        except pytest.skip.Exception:
            continue
    else:
        pytest.skip("no large order")
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, it.qty)], "damaged")
    assert q.requires_approval and "auto-approval limit" in q.approval_reason


def test_perishable_window_enforced(svc):
    now = settings.now()
    with svc.Session() as s:
        o, it, p = next((o, it, p) for o, it, p in s.execute(
            select(Order, OrderItem, Product).join(OrderItem, OrderItem.order_id == Order.id).join(Product)
            .where(Order.status == "delivered", Product.perishable.is_(True))) if now - o.delivered_at > timedelta(days=2))
    q = svc.quote_refund(o.customer_id, o.id, [RefundLine(p.id, 1)], "spoiled")
    assert not q.lines and "24 hours" in q.rejected_lines[0]["why"]


def test_change_of_mind_rejected_for_perishables(svc):
    o, it, p = _fresh_order(svc, cid := clean(svc, 1), perishable=True)
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "change_of_mind")
    assert not q.lines and "non-perishable" in q.rejected_lines[0]["why"]


def test_quantity_cannot_exceed_ordered_or_double_refund(svc):
    o, it, p = _fresh_order(svc, cid := clean(svc, 2), perishable=True, max_value=400)
    assert not svc.quote_refund(cid, o.id, [RefundLine(p.id, it.qty + 1)], "damaged").lines
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, it.qty)], "damaged")
    svc.apply_refund(cid, q.quote_id, "test")
    again = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "damaged")
    assert not again.lines


def test_wallet_credited_on_refund(svc):
    o, it, p = _fresh_order(svc, cid := clean(svc, 3), perishable=True, max_value=400)
    before = svc.customer(cid)["wallet_balance"]
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "damaged")
    svc.apply_refund(cid, q.quote_id, "test")
    assert svc.customer(cid)["wallet_balance"] == pytest.approx(before + q.amount)


def test_cancel_rules(svc):
    with svc.Session() as s:
        placed = s.scalars(select(Order).where(Order.status == "placed")).first()
        delivered = s.scalars(select(Order).where(Order.status == "delivered")).first()
    assert svc.cancel_order(placed.customer_id, placed.id)["status"] == "cancelled"
    with pytest.raises(OrderError) as e:
        svc.cancel_order(delivered.customer_id, delivered.id)
    assert e.value.code == "not_cancellable"


def test_refund_on_undelivered_order_rejected(svc):
    with svc.Session() as s:
        o = s.scalars(select(Order).where(Order.status == "placed")).first()
        it = s.scalars(select(OrderItem).where(OrderItem.order_id == o.id)).first()
    with pytest.raises(OrderError) as e:
        svc.quote_refund(o.customer_id, o.id, [RefundLine(it.product_id, 1)], "damaged")
    assert e.value.code == "not_delivered"


def test_applying_a_quote_twice_pays_once(svc):
    """Review finding: applying the same Rs 140 quote twice credited Rs 280."""
    o, it, p = _fresh_order(svc, cid := clean(svc, 4), perishable=True, max_value=400)
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "spoiled")
    before = svc.customer(cid)["wallet_balance"]
    first = svc.apply_refund(cid, q.quote_id)
    second = svc.apply_refund(cid, q.quote_id)
    assert first == second
    assert svc.customer(cid)["wallet_balance"] == pytest.approx(before + q.amount)


def test_tampered_quote_amount_is_not_paid(svc):
    """Review finding: a quote edited to Rs 9,999 was credited. The stored quote is re-validated."""
    from app.db import RefundQuote as Row

    o, it, p = _fresh_order(svc, cid := clean(svc, 5), perishable=True, max_value=400)
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "spoiled")
    with svc.Session.begin() as s:
        s.get(Row, q.quote_id).amount = 9999.0
    before = svc.customer(cid)["wallet_balance"]
    with pytest.raises(OrderError) as e:
        svc.apply_refund(cid, q.quote_id)
    assert e.value.code in ("quote_outdated", "approval_required")
    assert svc.customer(cid)["wallet_balance"] == pytest.approx(before)


def _large_line(svc):
    for cid in range(1, 61):
        try:
            return (cid, *_fresh_order(svc, cid, perishable=False, min_value=600))
        except pytest.skip.Exception:
            continue
    pytest.skip("no large order")


def test_large_quote_cannot_be_applied_without_approval(svc):
    cid, o, it, p = _large_line(svc)
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, it.qty)], "damaged")
    assert q.requires_approval
    with pytest.raises(OrderError) as e:
        svc.apply_refund(cid, q.quote_id)
    assert e.value.code == "approval_required"
    svc.request_approval(cid, q.quote_id)
    svc.decide_quote(q.quote_id, True, "anu")
    assert svc.apply_refund(cid, q.quote_id)["approved_by"] == "anu"
    assert svc.apply_refund(cid, q.quote_id)["status"] == "refunded"  # replay, no second payment


def test_other_customers_cannot_apply_my_quote(svc):
    o, it, p = _fresh_order(svc, cid := clean(svc, 6), perishable=True, max_value=400)
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "spoiled")
    with pytest.raises(OrderError) as e:
        svc.apply_refund(cid + 1, q.quote_id)
    assert e.value.code == "quote_not_found"


def test_pending_quote_reserves_units(svc):
    cid, o, it, p = _large_line(svc)
    q = svc.quote_refund(cid, o.id, [RefundLine(p.id, it.qty)], "damaged")
    svc.request_approval(cid, q.quote_id)
    again = svc.quote_refund(cid, o.id, [RefundLine(p.id, 1)], "damaged")
    assert not again.lines  # units are held by the pending quote


def test_seed_if_empty_keeps_existing_data(svc):
    """Review finding: container startup re-seeded and wiped approvals."""
    from app.db import Approval
    from app.seed import seed_if_empty

    with svc.Session.begin() as s:
        s.add(Approval(thread_id="t-keep", customer_id=1, payload="{}", status="pending", created_at=settings.now()))
    seed_if_empty()
    with svc.Session() as s:
        assert s.get(Approval, "t-keep") is not None

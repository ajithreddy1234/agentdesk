"""Order service: the business rules behind the order/refund REST API.

Safety properties
- The customer id always comes from the authenticated session, never from the model, so the
  agent can only read or change the signed-in customer's orders.
- Refund quotes are stored server-side. Callers (the agent, the REST API, a supervisor) only
  pass a quote id; amounts can't be altered in transit.
- Applying a quote locks the quote row, re-checks eligibility against the current order state,
  requires a recorded approval when the quote needs one, and is idempotent: applying the same
  quote twice returns the original result and never pays twice.
"""
from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import func, select

from .config import settings
from .db import Customer, Order, OrderItem, Product, Refund, RefundQuote as QuoteRow, Ticket, session_factory

REASONS = {"damaged", "spoiled", "expired", "leaking", "wrong_item", "missing", "change_of_mind"}
QUOTE_TTL = timedelta(hours=24)


class OrderError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class RefundLine:
    product_id: int
    qty: int


@dataclass
class RefundQuote:
    quote_id: str
    order_id: int
    lines: list[dict]
    amount: float
    method: str
    reason: str
    requires_approval: bool
    approval_reason: str | None = None
    rejected_lines: list[dict] = field(default_factory=list)
    status: str = "quoted"


def _row_to_quote(r: QuoteRow) -> RefundQuote:
    return RefundQuote(r.id, r.order_id, json.loads(r.lines), r.amount, r.method, r.reason, r.requires_approval,
                       r.approval_reason, json.loads(r.rejected_lines or "[]"), r.status)


class OrderService:
    def __init__(self, db_url: str | None = None):
        self.Session = session_factory(db_url)

    # ---------- reads ----------
    def _own_order(self, s, customer_id: int, order_id: int, lock: bool = False) -> Order:
        q = select(Order).where(Order.id == order_id)
        if lock:
            q = q.with_for_update()
        order = s.execute(q).scalar_one_or_none()
        if not order or order.customer_id != customer_id:
            # same message whether it doesn't exist or belongs to someone else: no data leak
            raise OrderError("not_found", f"Order {order_id} was not found on this account.")
        return order

    def customer(self, customer_id: int) -> dict:
        with self.Session() as s:
            c = s.get(Customer, customer_id)
            if not c:
                raise OrderError("not_found", "Customer not found")
            return {"id": c.id, "name": c.name, "city": c.city, "plus_member": c.plus_member,
                    "wallet_balance": round(c.wallet_balance, 2)}

    def list_orders(self, customer_id: int, status: str | None = None, limit: int = 10) -> list[dict]:
        with self.Session() as s:
            q = select(Order).where(Order.customer_id == customer_id).order_by(Order.placed_at.desc()).limit(min(limit, 25))
            if status:
                q = q.where(Order.status == status)
            return [self._summary(o) for o in s.scalars(q)]

    def order_details(self, customer_id: int, order_id: int) -> dict:
        with self.Session() as s:
            o = self._own_order(s, customer_id, order_id)
            refunded = self._refunded_qty(s, order_id)
            items = []
            for it, p in s.execute(select(OrderItem, Product).join(Product).where(OrderItem.order_id == order_id)):
                items.append({"product_id": p.id, "name": p.name, "qty": it.qty, "unit_price": it.unit_price,
                              "perishable": p.perishable, "already_refunded_qty": refunded.get(p.id, 0)})
            refunds = [{"product_id": r.product_id, "qty": r.qty, "amount": r.amount, "status": r.status, "method": r.method}
                       for r in s.scalars(select(Refund).where(Refund.order_id == order_id, Refund.qty > 0))]
            return {**self._summary(o), "items": items, "refunds": refunds}

    @staticmethod
    def _summary(o: Order) -> dict:
        return {"order_id": o.id, "status": o.status, "slot": o.slot, "payment_method": o.payment_method,
                "total": o.total, "placed_at": o.placed_at.isoformat(timespec="minutes"),
                "delivered_at": o.delivered_at.isoformat(timespec="minutes") if o.delivered_at else None}

    @staticmethod
    def _refunded_qty(s, order_id: int, exclude_quote: str | None = None) -> dict[int, int]:
        """Units already refunded, plus units held by other open quotes awaiting approval."""
        out: dict[int, int] = defaultdict(int)
        for r in s.scalars(select(Refund).where(Refund.order_id == order_id, Refund.status == "approved")):
            out[r.product_id] += r.qty
        open_q = select(QuoteRow).where(QuoteRow.order_id == order_id, QuoteRow.status.in_(["pending_approval", "approved"]))
        for q in s.scalars(open_q):
            if q.id != exclude_quote:
                for line in json.loads(q.lines):
                    out[line["product_id"]] += line["qty"]
        return out

    def recent_complaints(self, s, customer_id: int, days: int = 30) -> int:
        since = settings.now() - timedelta(days=days)
        q = (select(func.count(func.distinct(Refund.created_at))).join(Order, Order.id == Refund.order_id)
             .where(Order.customer_id == customer_id, Refund.created_at >= since, Refund.status == "approved"))
        return int(s.execute(q).scalar() or 0)

    # ---------- actions ----------
    def cancel_order(self, customer_id: int, order_id: int, reason: str = "") -> dict:
        with self.Session.begin() as s:
            o = self._own_order(s, customer_id, order_id, lock=True)
            if o.status == "cancelled":
                raise OrderError("already_cancelled", f"Order {order_id} is already cancelled.")
            if o.status != "placed":
                raise OrderError("not_cancellable",
                                 f"Order {order_id} is '{o.status}' and can no longer be cancelled. "
                                 "Delivered items can be reported for a refund instead.")
            o.status = "cancelled"
            if o.payment_method != "cod":
                s.get(Customer, customer_id).wallet_balance += o.total
            return {"order_id": order_id, "status": "cancelled",
                    "refund": None if o.payment_method == "cod" else {"amount": o.total, "method": "wallet"}}

    def _evaluate(self, s, customer_id: int, o: Order, lines: list[RefundLine], reason: str,
                  exclude_quote: str | None = None) -> tuple[list[dict], list[dict], float]:
        """Apply the refund policy to the requested lines against the current order state."""
        now = settings.now()
        items = {it.product_id: (it, p) for it, p in
                 s.execute(select(OrderItem, Product).join(Product).where(OrderItem.order_id == o.id))}
        refunded = self._refunded_qty(s, o.id, exclude_quote)
        ok, rejected, amount = [], [], 0.0
        age = now - o.delivered_at
        for line in lines:
            if line.product_id not in items:
                rejected.append({"product_id": line.product_id, "why": "not in this order"})
                continue
            it, p = items[line.product_id]
            left = it.qty - refunded.get(p.id, 0)
            if line.qty < 1 or line.qty > left:
                rejected.append({"product_id": p.id, "name": p.name, "why": f"quantity must be 1..{max(left, 0)}"})
                continue
            if p.perishable and age > timedelta(hours=24):
                rejected.append({"product_id": p.id, "name": p.name, "why": "perishable items must be reported within 24 hours of delivery"})
                continue
            if not p.perishable and age > timedelta(days=7):
                rejected.append({"product_id": p.id, "name": p.name, "why": "packaged items can be returned within 7 days of delivery"})
                continue
            if reason == "change_of_mind" and p.perishable:
                rejected.append({"product_id": p.id, "name": p.name, "why": "change of mind applies only to unopened non-perishable items"})
                continue
            value = round(line.qty * it.unit_price, 2)
            ok.append({"product_id": p.id, "name": p.name, "qty": line.qty, "amount": value})
            amount += value
        return ok, rejected, round(amount, 2)

    def quote_refund(self, customer_id: int, order_id: int, lines: list[RefundLine], reason: str,
                     method: str = "wallet") -> RefundQuote:
        reason = reason.lower().strip()
        if reason not in REASONS:
            raise OrderError("invalid_reason", f"Reason must be one of {sorted(REASONS)}")
        if method not in {"wallet", "original"}:
            raise OrderError("invalid_method", "Refund method must be 'wallet' or 'original'")
        with self.Session.begin() as s:
            o = self._own_order(s, customer_id, order_id)
            if o.status != "delivered" or not o.delivered_at:
                raise OrderError("not_delivered", f"Order {order_id} is '{o.status}'; refunds apply to delivered orders. "
                                 + ("It can be cancelled instead." if o.status == "placed" else ""))
            if o.payment_method == "cod" and method == "original":
                method = "wallet"  # policy: COD refunds go to wallet or bank
            ok, rejected, amount = self._evaluate(s, customer_id, o, lines, reason)
            complaints = self.recent_complaints(s, customer_id)
            needs, why = False, None
            if amount > settings.auto_refund_limit:
                needs, why = True, f"amount Rs {amount:.2f} is above the Rs {settings.auto_refund_limit:.0f} auto-approval limit"
            elif complaints >= 2:
                needs, why = True, f"customer already has {complaints} approved complaints in the last 30 days"
            row = QuoteRow(id=uuid.uuid4().hex, customer_id=customer_id, order_id=order_id, lines=json.dumps(ok),
                           rejected_lines=json.dumps(rejected), amount=amount, method=method, reason=reason,
                           requires_approval=needs and bool(ok), approval_reason=why if ok else None,
                           status="quoted" if ok else "rejected", created_at=settings.now())
            s.add(row)
            s.flush()
            return _row_to_quote(row)

    def get_quote(self, customer_id: int, quote_id: str) -> RefundQuote:
        with self.Session() as s:
            r = s.get(QuoteRow, quote_id)
            if not r or r.customer_id != customer_id:
                raise OrderError("quote_not_found", "Refund quote not found.")
            return _row_to_quote(r)

    def request_approval(self, customer_id: int, quote_id: str) -> RefundQuote:
        """Hold a quote that needs a human: it reserves its units until a supervisor decides."""
        with self.Session.begin() as s:
            r = s.execute(select(QuoteRow).where(QuoteRow.id == quote_id).with_for_update()).scalar_one_or_none()
            if not r or r.customer_id != customer_id:
                raise OrderError("quote_not_found", "Refund quote not found.")
            if r.status == "quoted" and r.requires_approval:
                r.status = "pending_approval"
            return _row_to_quote(r)

    def decide_quote(self, quote_id: str, approved: bool, reviewer: str) -> RefundQuote:
        """Record a supervisor decision. Idempotent for repeated identical decisions."""
        with self.Session.begin() as s:
            r = s.execute(select(QuoteRow).where(QuoteRow.id == quote_id).with_for_update()).scalar_one_or_none()
            if not r:
                raise OrderError("quote_not_found", "Refund quote not found.")
            target = "approved" if approved else "rejected"
            if r.status in (target, "applied") or (r.status == "rejected" and not approved):
                return _row_to_quote(r)
            if r.status != "pending_approval":
                raise OrderError("invalid_state", f"Quote is '{r.status}', not awaiting approval.")
            r.status, r.approved_by = target, reviewer[:60]
            return _row_to_quote(r)

    def apply_refund(self, customer_id: int, quote_id: str, approved_by: str = "auto-policy") -> dict:
        """Pay out a stored quote exactly once."""
        now = settings.now()
        with self.Session.begin() as s:
            r = s.execute(select(QuoteRow).where(QuoteRow.id == quote_id).with_for_update()).scalar_one_or_none()
            if not r or r.customer_id != customer_id:
                raise OrderError("quote_not_found", "Refund quote not found.")
            if r.status == "applied":
                return json.loads(r.result)  # idempotent replay: same answer, no second payment
            if r.status in ("rejected", "expired"):
                raise OrderError("quote_closed", f"Refund quote is {r.status}.")
            if r.requires_approval and r.status != "approved":
                raise OrderError("approval_required", "This refund needs a supervisor's approval first.")
            if not r.requires_approval and r.status != "quoted":
                raise OrderError("invalid_state", f"Quote is '{r.status}'.")
            if now - r.created_at > QUOTE_TTL:
                r.status = "expired"
                raise OrderError("quote_expired", "Refund quote expired; request a new one.")
            o = self._own_order(s, customer_id, r.order_id, lock=True)
            lines = [RefundLine(l["product_id"], l["qty"]) for l in json.loads(r.lines)]
            ok, rejected, amount = self._evaluate(s, customer_id, o, lines, r.reason, exclude_quote=r.id)
            if rejected or abs(amount - r.amount) > 0.005:
                r.status = "expired"
                raise OrderError("quote_outdated", "The order changed since this quote was made; request a new one.")
            for l in ok:
                s.add(Refund(order_id=r.order_id, product_id=l["product_id"], qty=l["qty"], amount=l["amount"],
                             method=r.method, reason=r.reason, status="approved",
                             approved_by=(r.approved_by or approved_by)[:40], created_at=now))
            eta = "within 2 hours"
            if r.method == "wallet":
                s.get(Customer, customer_id).wallet_balance += r.amount
            else:
                eta = "within 2 business days" if o.payment_method == "upi" else "in 5 to 7 business days"
            result = {"status": "refunded", "quote_id": r.id, "order_id": r.order_id, "amount": r.amount,
                      "method": r.method, "items": ok, "eta": eta, "approved_by": r.approved_by or approved_by}
            r.status, r.result, r.applied_at = "applied", json.dumps(result), now
            return result

    def create_ticket(self, customer_id: int, category: str, summary: str, order_id: int | None = None) -> dict:
        with self.Session.begin() as s:
            if order_id is not None:
                self._own_order(s, customer_id, order_id)
            t = Ticket(customer_id=customer_id, order_id=order_id, category=category[:40], summary=summary[:2000],
                       created_at=settings.now())
            s.add(t)
            s.flush()
            return {"ticket_id": t.id, "status": "open", "category": t.category}


def quote_to_dict(q: RefundQuote) -> dict:
    from dataclasses import asdict

    return asdict(q)

"""How tools reach the order service: over REST (production) or in-process (tests, single binary)."""
from __future__ import annotations

import os

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import settings
from .orders import OrderError, OrderService, RefundLine, RefundQuote


class InProcessOrderClient:
    def __init__(self, service: OrderService | None = None):
        self.svc = service or OrderService()

    def customer(self, cid): return self.svc.customer(cid)
    def list_orders(self, cid, status=None, limit=10): return self.svc.list_orders(cid, status, limit)
    def order_details(self, cid, oid): return self.svc.order_details(cid, oid)
    def cancel_order(self, cid, oid, reason=""): return self.svc.cancel_order(cid, oid, reason)

    def quote_refund(self, cid, oid, lines, reason, method="wallet") -> RefundQuote:
        return self.svc.quote_refund(cid, oid, [RefundLine(**l) for l in lines], reason, method)

    def request_approval(self, cid, quote_id): return self.svc.request_approval(cid, quote_id)
    def decide_quote(self, quote_id, approved, reviewer): return self.svc.decide_quote(quote_id, approved, reviewer)
    def apply_refund(self, cid, quote_id, approved_by="auto-policy"): return self.svc.apply_refund(cid, quote_id, approved_by)
    def create_ticket(self, cid, category, summary, order_id=None): return self.svc.create_ticket(cid, category, summary, order_id)


_retry = retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, max=3), reraise=True,
               retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)))


class HttpOrderClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=10,
                                 headers={"X-Service-Token": token or os.getenv("SERVICE_TOKEN", "dev-token")})

    @_retry
    def _call(self, method, path, **kw):
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            d = body.get("detail", {})
            raise OrderError(d.get("code", "error") if isinstance(d, dict) else "error",
                             d.get("message", str(d)) if isinstance(d, dict) else str(d))
        return r.json()

    def customer(self, cid): return self._call("GET", f"/svc/customers/{cid}")
    def list_orders(self, cid, status=None, limit=10):
        return self._call("GET", f"/svc/customers/{cid}/orders", params={k: v for k, v in {"status": status, "limit": limit}.items() if v})
    def order_details(self, cid, oid): return self._call("GET", f"/svc/customers/{cid}/orders/{oid}")
    def cancel_order(self, cid, oid, reason=""): return self._call("POST", f"/svc/customers/{cid}/orders/{oid}/cancel", json={"reason": reason})

    def quote_refund(self, cid, oid, lines, reason, method="wallet") -> RefundQuote:
        d = self._call("POST", f"/svc/customers/{cid}/orders/{oid}/refunds/quote", json={"lines": lines, "reason": reason, "method": method})
        return RefundQuote(**d)

    def request_approval(self, cid, quote_id):
        return RefundQuote(**self._call("POST", f"/svc/customers/{cid}/quotes/{quote_id}/request-approval"))

    def decide_quote(self, quote_id, approved, reviewer):
        return RefundQuote(**self._call("POST", f"/svc/quotes/{quote_id}/decision", json={"approved": approved, "reviewer": reviewer}))

    def apply_refund(self, cid, quote_id, approved_by="auto-policy"):
        # only the quote id travels: the service recomputes and pays at most once
        return self._call("POST", f"/svc/customers/{cid}/quotes/{quote_id}/apply", json={"approved_by": approved_by})

    def create_ticket(self, cid, category, summary, order_id=None):
        return self._call("POST", f"/svc/customers/{cid}/tickets", json={"category": category, "summary": summary, "order_id": order_id})


def make_client():
    return HttpOrderClient(settings.service_base_url) if settings.service_base_url else InProcessOrderClient()

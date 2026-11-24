from datetime import timedelta

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app import main
from app.agent import build_graph
from app.auth import issue
from app.config import settings
from app.db import Order, OrderItem, Product
from app.order_client import HttpOrderClient
from app.runtime import AgentRuntime
from tests.conftest import ScriptedLLM, call


def client_with(steps):
    main.state["runtime"] = AgentRuntime(build_graph(ScriptedLLM(steps=list(steps), calls=[]), InMemorySaver()))
    return TestClient(main.app)


def login(c, cid=1):
    r = c.post("/auth/customer", json={"customer_id": cid, "password": "demo-customer"})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["token"]}


def sup(c):
    r = c.post("/auth/supervisor", json={"username": "supervisor", "password": "demo-supervisor"})
    assert r.status_code == 200
    return {"Authorization": "Bearer " + r.json()["token"]}


def test_chat_endpoint_and_history():
    with client_with([call("list_orders", {}), AIMessage(content="You have 5 orders.")]) as c:
        h1 = login(c, 1)
        r = c.post("/chat", json={"message": "my orders?"}, headers=h1).json()
        assert r["reply"] == "You have 5 orders." and r["tool_calls"][0]["name"] == "list_orders"
        assert c.get(f"/chat/{r['thread_id']}", headers=h1).json()[-1]["content"] == "You have 5 orders."
        assert c.get(f"/chat/{r['thread_id']}", headers=login(c, 2)).status_code == 403


def test_identity_cannot_be_claimed_with_headers():
    """Review finding: X-Customer-Id / X-Supervisor headers were trusted."""
    with client_with([AIMessage(content="hi")]) as c:
        assert c.post("/chat", json={"message": "hi"}, headers={"X-Customer-Id": "1"}).status_code == 401
        assert c.get("/approvals", headers={"X-Supervisor": "anyone"}).status_code == 401
        assert c.get("/approvals", headers=login(c)).status_code == 403  # customers can't review refunds
        assert c.get("/approvals", headers=sup(c)).status_code == 200
        forged = issue("supervisor", "x")[:-4] + "abcd"
        assert c.get("/approvals", headers={"Authorization": "Bearer " + forged}).status_code == 401


def test_bad_logins_rejected():
    with client_with([]) as c:
        assert c.post("/auth/customer", json={"customer_id": 1, "password": "nope"}).status_code == 401
        assert c.post("/auth/customer", json={"customer_id": 99999, "password": "demo-customer"}).status_code == 401
        assert c.post("/auth/supervisor", json={"username": "supervisor", "password": "x"}).status_code == 401


def test_service_requires_token():
    with client_with([]) as c:
        assert c.get("/svc/customers/1/orders").status_code == 422  # missing header
        assert c.get("/svc/customers/1/orders", headers={"X-Service-Token": "nope"}).status_code == 401


def test_http_order_client_round_trip_and_no_amounts_accepted():
    with client_with([]) as c:
        http = HttpOrderClient("http://testserver")
        http.http = c
        c.headers["X-Service-Token"] = "dev-token"
        orders = http.list_orders(1, limit=3)
        assert len(orders) == 3 and http.order_details(1, orders[0]["order_id"])["items"]
        other = http.list_orders(2, limit=1)[0]["order_id"]
        try:
            http.order_details(1, other)
            raise AssertionError("should be hidden")
        except Exception as e:  # noqa: BLE001
            assert getattr(e, "code", "") == "not_found"
        # quote, then apply twice over HTTP with a forged amount: exactly one correct payment
        with main.orders.Session() as s:
            rows = s.execute(select(Order, OrderItem, Product).join(OrderItem, OrderItem.order_id == Order.id)
                             .join(Product).where(Order.status == "delivered", Product.perishable.is_(True))).all()
        o, it, p = next((o, it, p) for o, it, p in rows
                        if settings.now() - o.delivered_at < timedelta(hours=24) and it.unit_price <= 400)
        cid = o.customer_id
        q = http.quote_refund(cid, o.id, [{"product_id": p.id, "qty": 1}], "spoiled")
        if q.requires_approval:  # customer with prior complaints: approve first
            http.request_approval(cid, q.quote_id)
            http.decide_quote(q.quote_id, True, "anu")
        before = http.customer(cid)["wallet_balance"]
        r1 = c.post(f"/svc/customers/{cid}/quotes/{q.quote_id}/apply", json={"approved_by": "x", "amount": 99999}).json()
        r2 = http.apply_refund(cid, q.quote_id)
        assert r1["amount"] == q.amount == r2["amount"]
        assert http.customer(cid)["wallet_balance"] == round(before + q.amount, 2)


def test_health_reports_missing_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_api_key", "")
    with TestClient(main.app) as c:
        assert c.get("/health").json()["llm_configured"] is False

"""AgentDesk API: customer chat, supervisor approvals and the order/refund REST service.

Auth: customers and supervisors sign in for signed, expiring bearer tokens (app/auth.py).
The order service under /svc is internal and requires SERVICE_TOKEN.
"""
from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import build_graph, langfuse_callbacks, llm_configured, make_checkpointer, make_llm
from .auth import check_customer_login, check_supervisor_login, issue, require_customer, require_supervisor
from .config import settings
from .db import Customer, init_db
from .orders import OrderError, OrderService, RefundLine, quote_to_dict
from .runtime import AgentRuntime, as_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agentdesk")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.vector_backend == "pgvector" and os.getenv("INDEX_POLICIES_ON_START", "true") == "true":
        from .policies import get_policy_store

        get_policy_store().index()
    yield


app = FastAPI(title="AgentDesk", version="1.1.0", lifespan=lifespan)
orders = OrderService()


def runtime() -> AgentRuntime:
    if "runtime" not in state:
        try:
            state["runtime"] = AgentRuntime(build_graph(make_llm(), make_checkpointer()), langfuse_callbacks())
        except RuntimeError as e:  # no API key yet
            raise HTTPException(503, str(e)) from e
    return state["runtime"]


def approvals_runtime() -> AgentRuntime:
    """Approvals listing works even before an LLM key exists (it only reads the database)."""
    if "runtime" in state:
        return state["runtime"]
    return AgentRuntime(graph=None)


def _llm_errors(fn):
    """Run an agent call and turn provider/config failures into clear HTTP errors."""
    import openai

    try:
        return fn()
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except OrderError as e:
        raise HTTPException(409, {"code": e.code, "message": e.message}) from e
    except openai.AuthenticationError as e:
        raise HTTPException(502, "The LLM provider rejected OPENROUTER_API_KEY. Check the key in .env.") from e
    except openai.PermissionDeniedError as e:
        raise HTTPException(502, "The LLM provider refused the request (no credit or model not allowed).") from e
    except openai.NotFoundError as e:
        raise HTTPException(502, f"Model '{settings.llm_model}' not found at the provider. Check LLM_MODEL.") from e
    except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
        raise HTTPException(503, "The LLM provider is unavailable or rate limiting. Please retry.") from e


def service_auth(x_service_token: str = Header(...)):
    if not hmac.compare_digest(x_service_token, os.getenv("SERVICE_TOKEN", "dev-token")):
        raise HTTPException(401, "bad service token")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(None, max_length=64)


class DecisionIn(BaseModel):
    approved: bool
    note: str | None = Field(None, max_length=1000)


class CustomerLogin(BaseModel):
    customer_id: int
    password: str


class SupervisorLogin(BaseModel):
    username: str
    password: str


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.llm_model, "llm_configured": llm_configured()}


@app.post("/auth/customer")
def login_customer(body: CustomerLogin):
    with orders.Session() as s:
        if not s.get(Customer, body.customer_id) or not check_customer_login(body.password):
            raise HTTPException(401, "Invalid customer id or password")
    return {"token": issue("customer", body.customer_id), "role": "customer", "customer_id": body.customer_id}


@app.post("/auth/supervisor")
def login_supervisor(body: SupervisorLogin):
    if not check_supervisor_login(body.username, body.password):
        raise HTTPException(401, "Invalid supervisor credentials")
    return {"token": issue("supervisor", body.username), "role": "supervisor", "username": body.username}


@app.post("/chat")
def chat(body: ChatIn, customer_id: int = Depends(require_customer)):
    return _llm_errors(lambda: as_dict(runtime().chat(customer_id, body.message, body.thread_id)))


@app.get("/chat/{thread_id}")
def chat_history(thread_id: str, customer_id: int = Depends(require_customer)):
    return _llm_errors(lambda: runtime().history(thread_id, customer_id))


@app.get("/approvals")
def approvals(_: str = Depends(require_supervisor)):
    return approvals_runtime().pending_approvals()


@app.post("/approvals/{thread_id}")
def decide(thread_id: str, body: DecisionIn, reviewer: str = Depends(require_supervisor)):
    return _llm_errors(lambda: as_dict(runtime().decide(thread_id, body.approved, reviewer, body.note)))


@app.get("/demo/customers")
def demo_customers():
    """Customer list for the demo login screen (names and cities are synthetic)."""
    with orders.Session() as s:
        return [{"id": c.id, "name": c.name, "city": c.city} for c in s.query(Customer).order_by(Customer.id).limit(100)]


# ---------------- order / refund REST service (called by the agent's tools) ----------------
def _svc(fn):
    try:
        return fn()
    except OrderError as e:
        code = 404 if e.code in ("not_found", "quote_not_found") else 409 if e.code in (
            "approval_required", "invalid_state", "quote_closed", "quote_expired", "quote_outdated") else 422
        raise HTTPException(code, {"code": e.code, "message": e.message}) from e


class CancelIn(BaseModel):
    reason: str = ""


class QuoteIn(BaseModel):
    lines: list[dict]
    reason: str
    method: str = "wallet"


class ApplyIn(BaseModel):
    approved_by: str = "auto-policy"


class QuoteDecisionIn(BaseModel):
    approved: bool
    reviewer: str


class TicketIn(BaseModel):
    category: str
    summary: str
    order_id: int | None = None


svc = Depends(service_auth)


@app.get("/svc/customers/{cid}", dependencies=[svc])
def svc_customer(cid: int):
    return _svc(lambda: orders.customer(cid))


@app.get("/svc/customers/{cid}/orders", dependencies=[svc])
def svc_orders(cid: int, status: str | None = None, limit: int = 10):
    return _svc(lambda: orders.list_orders(cid, status, limit))


@app.get("/svc/customers/{cid}/orders/{oid}", dependencies=[svc])
def svc_order(cid: int, oid: int):
    return _svc(lambda: orders.order_details(cid, oid))


@app.post("/svc/customers/{cid}/orders/{oid}/cancel", dependencies=[svc])
def svc_cancel(cid: int, oid: int, body: CancelIn):
    return _svc(lambda: orders.cancel_order(cid, oid, body.reason))


@app.post("/svc/customers/{cid}/orders/{oid}/refunds/quote", dependencies=[svc])
def svc_quote(cid: int, oid: int, body: QuoteIn):
    return _svc(lambda: quote_to_dict(orders.quote_refund(cid, oid, [RefundLine(**l) for l in body.lines], body.reason, body.method)))


@app.post("/svc/customers/{cid}/quotes/{qid}/request-approval", dependencies=[svc])
def svc_request_approval(cid: int, qid: str):
    return _svc(lambda: quote_to_dict(orders.request_approval(cid, qid)))


@app.post("/svc/quotes/{qid}/decision", dependencies=[svc])
def svc_decide(qid: str, body: QuoteDecisionIn):
    return _svc(lambda: quote_to_dict(orders.decide_quote(qid, body.approved, body.reviewer)))


@app.post("/svc/customers/{cid}/quotes/{qid}/apply", dependencies=[svc])
def svc_apply(cid: int, qid: str, body: ApplyIn):
    # no amounts accepted here: the service pays exactly the stored, re-validated quote, once
    return _svc(lambda: orders.apply_refund(cid, qid, body.approved_by))


@app.post("/svc/customers/{cid}/tickets", dependencies=[svc])
def svc_ticket(cid: int, body: TicketIn):
    return _svc(lambda: orders.create_ticket(cid, body.category, body.summary, body.order_id))


_UI = Path(__file__).parent / "static" / "index.html"


@app.get("/")
def ui():
    return FileResponse(_UI)

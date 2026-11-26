"""Agent tools. Arguments are validated by Pydantic schemas; the customer id is injected from the
authenticated session (RunnableConfig), so the model never supplies or sees other customers' ids.

Reads (list_orders, get_order_details, search_orders) are SQL over customer-scoped views through a
read-only connection (app/sql_lookup.py). Actions (cancel, refund, tickets) go through the order/refund
REST API, which owns the business rules."""
from __future__ import annotations

import json
from typing import Literal

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field, conint

from ..config import settings
from ..memory import ALLOWED_KEYS, get_memory
from ..order_client import make_client
from ..orders import OrderError
from ..policies import get_policy_store
from ..sql_lookup import get_reader

_client = None


def client():
    global _client
    if _client is None:
        _client = make_client()
    return _client


def set_client(c):  # tests / alternative deployments
    global _client
    _client = c


def _cid(config: RunnableConfig) -> int:
    cid = (config or {}).get("configurable", {}).get("customer_id")
    if cid is None:
        raise PermissionError("No authenticated customer for this conversation")
    return int(cid)


def _ok(data) -> str:
    return json.dumps(data, default=str)


def _err(e: OrderError) -> str:
    return json.dumps({"error": e.code, "message": e.message})


class ListOrdersArgs(BaseModel):
    status: Literal["placed", "out_for_delivery", "delivered", "cancelled"] | None = Field(None, description="Filter by status")
    limit: conint(ge=1, le=25) = 10


@tool(args_schema=ListOrdersArgs)
def list_orders(status: str | None = None, limit: int = 10, config: RunnableConfig = None) -> str:
    """List the signed-in customer's recent orders (newest first) with status, total and delivery time."""
    return _ok(get_reader().list_orders(_cid(config), status, limit))


class OrderIdArgs(BaseModel):
    order_id: conint(ge=1) = Field(description="Order number, e.g. 1003")


@tool(args_schema=OrderIdArgs)
def get_order_details(order_id: int, config: RunnableConfig = None) -> str:
    """Get one order's items (product_id, name, qty, price, perishable), delivery time and existing refunds."""
    d = get_reader().order_details(_cid(config), order_id)
    if d is None:  # same answer whether it doesn't exist or belongs to someone else
        return _ok({"error": "not_found", "message": f"Order {order_id} was not found on this account."})
    return _ok(d)


class SearchOrdersArgs(BaseModel):
    product: str | None = Field(None, min_length=2, max_length=60, description="Product name fragment, e.g. 'paneer'")
    status: Literal["placed", "out_for_delivery", "delivered", "cancelled"] | None = None
    within_days: conint(ge=1, le=365) | None = Field(None, description="Only orders placed in the last N days")
    limit: conint(ge=1, le=25) = 10


@tool(args_schema=SearchOrdersArgs)
def search_orders(product: str | None = None, status: str | None = None, within_days: int | None = None,
                  limit: int = 10, config: RunnableConfig = None) -> str:
    """Find the customer's orders by product name, status and/or how recent they are, when they describe an
    order ("the paneer I got last week") instead of giving its number. Returns orders with the matching items."""
    return _ok(get_reader().search_orders(_cid(config), product, status, within_days, limit))


class CancelArgs(OrderIdArgs):
    reason: str = Field("", max_length=200)


@tool(args_schema=CancelArgs)
def cancel_order(order_id: int, reason: str = "", config: RunnableConfig = None) -> str:
    """Cancel an order that has not been dispatched yet (status 'placed'). Only call after the customer confirms."""
    try:
        return _ok(client().cancel_order(_cid(config), order_id, reason))
    except OrderError as e:
        return _err(e)


class RefundItem(BaseModel):
    product_id: conint(ge=1)
    qty: conint(ge=1, le=50)


class RefundArgs(BaseModel):
    order_id: conint(ge=1)
    items: list[RefundItem] = Field(min_length=1, description="Affected products and quantities from get_order_details")
    reason: Literal["damaged", "spoiled", "expired", "leaking", "wrong_item", "missing", "change_of_mind"]
    method: Literal["wallet", "original"] = Field("wallet", description="wallet (fast) or original payment method")


@tool(args_schema=RefundArgs)
def request_refund(order_id: int, items: list, reason: str, method: str = "wallet", config: RunnableConfig = None) -> str:
    """Request a refund for specific items in a delivered order. The amount is calculated by the system
    from the order. Refunds above the auto-approval limit pause for a human supervisor's approval."""
    # Executed by the graph's tool node (see agent.py), which handles the approval pause.
    raise RuntimeError("request_refund is executed by the agent graph")


class PolicyArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300)


@tool(args_schema=PolicyArgs)
def search_policies(query: str) -> str:
    """Search FreshCart's official policies (returns, refunds, delivery, subscriptions, wallet, quality, privacy).
    Use this for any question about rules, limits, fees or timelines, and quote what it says."""
    hits = get_policy_store().search(query, k=4)
    return _ok([{"policy": c.title, "section": c.section, "text": c.text} for c, _ in hits])


class WebArgs(BaseModel):
    query: str = Field(min_length=3, max_length=200)


@tool(args_schema=WebArgs)
def web_search(query: str) -> str:
    """Search the public web for general information that is NOT about FreshCart's own policies or orders
    (for example how UPI refunds work at banks). Never use it for FreshCart rules."""
    if not settings.tavily_api_key or settings.tavily_api_key.endswith("..."):
        return _ok({"error": "web_search_unavailable", "message": "Web search is not configured."})
    try:
        r = httpx.post("https://api.tavily.com/search", timeout=15,
                       json={"api_key": settings.tavily_api_key, "query": query, "max_results": 3, "search_depth": "basic"})
        r.raise_for_status()
        return _ok([{"title": x["title"], "url": x["url"], "content": x["content"][:600]} for x in r.json().get("results", [])])
    except httpx.HTTPError as e:
        return _ok({"error": "web_search_failed", "message": str(e)[:200]})


class TicketArgs(BaseModel):
    category: Literal["food_safety", "delivery", "payment", "account", "complaint", "other"]
    summary: str = Field(min_length=10, max_length=1000)
    order_id: conint(ge=1) | None = None


@tool(args_schema=TicketArgs)
def escalate_to_human(category: str, summary: str, order_id: int | None = None, config: RunnableConfig = None) -> str:
    """Open a ticket for the human support team: food-safety/illness reports, issues the tools can't solve,
    or when the customer asks for a human."""
    try:
        return _ok(client().create_ticket(_cid(config), category, summary, order_id))
    except OrderError as e:
        return _err(e)


class MemoryArgs(BaseModel):
    key: Literal[tuple(sorted(ALLOWED_KEYS))]  # type: ignore[valid-type]
    value: str = Field(min_length=1, max_length=200)


@tool(args_schema=MemoryArgs)
def remember_preference(key: str, value: str, config: RunnableConfig = None) -> str:
    """Save a lasting preference the customer states (e.g. preferred_refund_method=original) for future chats."""
    return _ok({"saved": get_memory().set(_cid(config), key, value)})


ALL_TOOLS = [list_orders, get_order_details, search_orders, cancel_order, request_refund, search_policies, web_search,
             escalate_to_human, remember_preference]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

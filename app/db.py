"""Orders database (PostgreSQL in production, SQLite for tests)."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    phone: Mapped[str] = mapped_column(String(20))
    city: Mapped[str] = mapped_column(String(40))
    plus_member: Mapped[bool] = mapped_column(Boolean, default=False)
    wallet_balance: Mapped[float] = mapped_column(Float, default=0.0)


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    category: Mapped[str] = mapped_column(String(40))
    price: Mapped[float] = mapped_column(Float)
    perishable: Mapped[bool] = mapped_column(Boolean)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    status: Mapped[str] = mapped_column(String(20))  # placed | out_for_delivery | delivered | cancelled
    slot: Mapped[str] = mapped_column(String(20))  # morning | express | evening
    payment_method: Mapped[str] = mapped_column(String(20))  # upi | card | wallet | cod
    total: Mapped[float] = mapped_column(Float)
    placed_at: Mapped[datetime] = mapped_column(DateTime)
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    qty: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[float] = mapped_column(Float)


class Refund(Base):
    __tablename__ = "refunds"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"))
    qty: Mapped[int] = mapped_column(Integer)
    amount: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(20))  # wallet | original
    reason: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20))  # approved | pending_approval | rejected
    approved_by: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class RefundQuote(Base):
    """A refund proposal stored server-side. Clients and the model only ever pass its id, so the
    amount cannot be altered, and its status makes applying it idempotent."""
    __tablename__ = "refund_quotes"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    lines: Mapped[str] = mapped_column(Text)  # JSON [{product_id, name, qty, amount}]
    rejected_lines: Mapped[str] = mapped_column(Text, default="[]")
    amount: Mapped[float] = mapped_column(Float)
    method: Mapped[str] = mapped_column(String(20))
    reason: Mapped[str] = mapped_column(String(40))
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    approval_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # quoted -> (pending_approval -> approved | rejected) -> applied ; or expired
    status: Mapped[str] = mapped_column(String(20), default="quoted", index=True)
    approved_by: Mapped[str | None] = mapped_column(String(60), nullable=True)
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON of the applied refund
    created_at: Mapped[datetime] = mapped_column(DateTime)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    order_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    category: Mapped[str] = mapped_column(String(40))
    summary: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime)


class Approval(Base):
    """A refund waiting for a human, tied to the paused LangGraph thread."""
    __tablename__ = "approvals"
    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[int] = mapped_column(Integer)
    payload: Mapped[str] = mapped_column(Text)  # JSON of the proposed refund
    quote_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | processing | approved | rejected
    reviewer: Mapped[str | None] = mapped_column(String(40), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class ThreadOwner(Base):
    __tablename__ = "threads"
    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


@lru_cache(maxsize=4)
def get_engine(url: str | None = None):
    url = url or settings.database_url
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kw)


def session_factory(url: str | None = None):
    return sessionmaker(get_engine(url), expire_on_commit=False)


def init_db(url: str | None = None):
    """Create missing tables and add columns introduced by upgrades (idempotent)."""
    from sqlalchemy import inspect, text

    eng = get_engine(url)
    Base.metadata.create_all(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("approvals")}
    if "quote_id" not in cols:
        with eng.begin() as c:
            c.execute(text("ALTER TABLE approvals ADD COLUMN quote_id VARCHAR(32)"))

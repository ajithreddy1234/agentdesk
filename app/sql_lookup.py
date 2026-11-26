"""Read-scoped SQL order lookup used by the agent's read tools.

The agent never gets the application's read/write database credentials. Its lookups run as SQL
against three customer-scoped views through a separate, read-only connection:

    agent_orders       one row per order        (order_id, status, slot, payment_method, total, placed_at, delivered_at)
    agent_order_items  one row per order line   (line_id, order_id, product_id, name, category, perishable, qty, unit_price, refunded_qty)
    agent_refunds      one row per refund line  (order_id, product_id, qty, amount, method, reason, status, created_at)

PostgreSQL (production), enforced by the database itself:
- role `agentdesk_reader` may only SELECT these views (and call agent_scope()): no base tables, no writes
- each view keeps only the rows of `agent_scope()`, a SECURITY DEFINER function that returns the
  customer id in `app.customer_id` only when `app.scope_sig` is HMAC-SHA256(customer_id:backend_pid:
  transaction_start) under a key kept in `agent_scope_key`, a table the reader cannot read. The app signs
  the authenticated customer id for its own connection and transaction, so a session that sets
  `app.customer_id` itself (a bad query, a leaked reader password) sees nothing, and a signature copied
  from another session (for example out of pg_stat_activity) doesn't verify in any other backend or
  transaction
- the reader connection never falls back to the owner's credentials: without READONLY_DATABASE_URL it
  is derived from DATABASE_URL with the reader role and READER_DB_PASSWORD
- every transaction is READ ONLY with a short statement timeout

SQLite (tests and laptops): the connection is opened with `PRAGMA query_only`, and every query
also filters on customer_id in SQL, which is defence in depth on Postgres as well.

Writes (cancellations, refunds, tickets) never go through here; they use the order/refund REST API.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from datetime import timedelta
from functools import lru_cache

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url

from .config import settings

log = logging.getLogger("agentdesk.sql")

READER_ROLE = "agentdesk_reader"
VIEWS = ("agent_order_items", "agent_refunds", "agent_orders")

_SCOPE = "(SELECT agent_scope())"  # an initplan: evaluated once per query

# {crypto} is the schema pgcrypto lives in (it need not be public); the message binds the signature to
# this backend and this transaction. now() is the transaction start, identical for every statement in it.
_PG_SCOPE_FN = """
CREATE OR REPLACE FUNCTION agent_scope() RETURNS int
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public, pg_temp AS $$
  SELECT CASE
    WHEN current_setting('app.customer_id', true) ~ '^[0-9]{{1,9}}$'
     AND current_setting('app.scope_sig', true) = encode({crypto}.hmac(
           current_setting('app.customer_id', true) || ':' || pg_backend_pid() || ':' || extract(epoch FROM now())::text,
           (SELECT k FROM public.agent_scope_key LIMIT 1), 'sha256'), 'hex')
    THEN current_setting('app.customer_id', true)::int
  END
$$"""

# units already refunded plus units held by open quotes (pending or approved, not yet applied),
# the same rule as OrderService._refunded_qty
_PG_RESERVED = """
    COALESCE((SELECT SUM(r.qty) FROM refunds r WHERE r.order_id = oi.order_id
              AND r.product_id = oi.product_id AND r.status = 'approved'), 0)
  + COALESCE((SELECT SUM((l->>'qty')::int) FROM refund_quotes q CROSS JOIN LATERAL json_array_elements(q.lines::json) l
              WHERE q.order_id = oi.order_id AND q.status IN ('pending_approval', 'approved')
              AND (l->>'product_id')::int = oi.product_id), 0)"""
_SQLITE_RESERVED = """
    COALESCE((SELECT SUM(r.qty) FROM refunds r WHERE r.order_id = oi.order_id
              AND r.product_id = oi.product_id AND r.status = 'approved'), 0)
  + COALESCE((SELECT SUM(json_extract(l.value, '$.qty')) FROM refund_quotes q, json_each(q.lines) l
              WHERE q.order_id = oi.order_id AND q.status IN ('pending_approval', 'approved')
              AND json_extract(l.value, '$.product_id') = oi.product_id), 0)"""

_PG_VIEWS = {
    "agent_orders": f"""
        SELECT o.id AS order_id, o.customer_id, o.status, o.slot, o.payment_method, o.total, o.placed_at, o.delivered_at
        FROM orders o WHERE o.customer_id = {_SCOPE}""",
    "agent_order_items": f"""
        SELECT oi.id AS line_id, oi.order_id, o.customer_id, oi.product_id, p.name, p.category, p.perishable, oi.qty,
               oi.unit_price, {_PG_RESERVED} AS refunded_qty
        FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id
        WHERE o.customer_id = {_SCOPE}""",
    "agent_refunds": f"""
        SELECT r.order_id, o.customer_id, r.product_id, r.qty, r.amount, r.method, r.reason, r.status, r.created_at
        FROM refunds r JOIN orders o ON o.id = r.order_id
        WHERE o.customer_id = {_SCOPE} AND r.qty > 0""",
}

# SQLite has no session settings; the views are unscoped and every query filters on customer_id.
_SQLITE_VIEWS = {
    "agent_orders": """SELECT o.id AS order_id, o.customer_id, o.status, o.slot, o.payment_method, o.total,
                              o.placed_at, o.delivered_at FROM orders o""",
    "agent_order_items": f"""
        SELECT oi.id AS line_id, oi.order_id, o.customer_id, oi.product_id, p.name, p.category, p.perishable, oi.qty,
               oi.unit_price, {_SQLITE_RESERVED} AS refunded_qty
        FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id""",
    "agent_refunds": """SELECT r.order_id, o.customer_id, r.product_id, r.qty, r.amount, r.method, r.reason, r.status,
                               r.created_at FROM refunds r JOIN orders o ON o.id = r.order_id WHERE r.qty > 0""",
}


def _is_pg(engine) -> bool:
    return engine.dialect.name == "postgresql"


def drop_views(engine) -> None:
    """Views depend on the base tables; drop them before the tables are dropped (re-seed)."""
    with engine.begin() as c:
        for v in VIEWS:
            c.execute(text(f"DROP VIEW IF EXISTS {v}" + (" CASCADE" if _is_pg(engine) else "")))


def install(engine) -> None:
    """Create the scoped views and, on Postgres, the read-only role that may only read them. Idempotent."""
    pg = _is_pg(engine)
    with engine.begin() as c:
        if pg:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
            crypto = c.execute(text("SELECT quote_ident(n.nspname) FROM pg_extension e "
                                    "JOIN pg_namespace n ON n.oid = e.extnamespace WHERE e.extname = 'pgcrypto'")).scalar_one()
            c.execute(text("CREATE TABLE IF NOT EXISTS agent_scope_key (k text NOT NULL)"))
            c.execute(text(f"INSERT INTO agent_scope_key SELECT encode({crypto}.gen_random_bytes(32), 'hex') "
                           "WHERE NOT EXISTS (SELECT 1 FROM agent_scope_key)"))
            c.execute(text(_PG_SCOPE_FN.format(crypto=crypto)))
            c.execute(text("REVOKE ALL ON FUNCTION agent_scope() FROM PUBLIC"))
        # drop + create in one transaction: survives column changes, and grants are re-applied below
        for name, body in (_PG_VIEWS if pg else _SQLITE_VIEWS).items():
            c.execute(text(f"DROP VIEW IF EXISTS {name}"))
            c.execute(text(f"CREATE VIEW {name} {'WITH (security_barrier) ' if pg else ''}AS {body}"))
        if not pg:
            return
        password = os.getenv("READER_DB_PASSWORD", "agentdesk_reader")
        exists = c.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": READER_ROLE}).first()
        # identifiers can't be bound parameters: Postgres quotes the password with format('%L'), and the
        # result runs on a raw cursor so neither SQLAlchemy (':name') nor the driver ('%') re-parses it
        verb = "ALTER" if exists else "CREATE"
        stmt = c.execute(text(f"SELECT format('{verb} ROLE {READER_ROLE} LOGIN PASSWORD %L', CAST(:p AS text))"),
                         {"p": password}).scalar()
        c.connection.dbapi_connection.cursor().execute(stmt)
        c.execute(text(f"ALTER ROLE {READER_ROLE} SET default_transaction_read_only = on"))
        c.execute(text(f"ALTER ROLE {READER_ROLE} SET statement_timeout = '3s'"))
        c.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {READER_ROLE}"))
        c.execute(text(f"GRANT USAGE ON SCHEMA public TO {READER_ROLE}"))
        c.execute(text(f"GRANT SELECT ON {', '.join(VIEWS)} TO {READER_ROLE}"))
        # Postgres checks function privileges as the querying role, even inside a view. Calling it directly
        # only returns the caller's own verified customer id (or NULL), so this grants nothing new.
        c.execute(text(f"GRANT EXECUTE ON FUNCTION agent_scope() TO {READER_ROLE}"))


def reader_url() -> str:
    """The read-only URL. On Postgres this is never the owner's: it defaults to the reader role."""
    if settings.readonly_database_url:
        return settings.readonly_database_url
    url = settings.database_url
    if url.startswith("sqlite"):
        return url
    return make_url(url).set(username=READER_ROLE, password=os.getenv("READER_DB_PASSWORD", "agentdesk_reader")) \
        .render_as_string(hide_password=False)


_KEY_TTL_S = 300
_keys: dict[str, tuple[bytes, float]] = {}


@lru_cache(maxsize=4)
def _owner_engine(url: str):
    return create_engine(url, pool_pre_ping=True, pool_size=1)


def scope_key(owner_url: str | None = None, refresh: bool = False) -> bytes:
    """HMAC key for the signed customer scope, read with the owner's credentials (the reader can't).
    Cached for a few minutes so a rotated key (UPDATE agent_scope_key) is picked up without a restart."""
    url = owner_url or settings.database_url
    hit = _keys.get(url)
    if hit and not refresh and time.monotonic() - hit[1] < _KEY_TTL_S:
        return hit[0]
    with _owner_engine(url).connect() as c:
        key = c.execute(text("SELECT k FROM agent_scope_key LIMIT 1")).scalar_one().encode()
    _keys[url] = (key, time.monotonic())
    return key


def sign_scope(customer_id: int, key: bytes, backend_pid: int | None = None, tx_epoch: str | None = None) -> str:
    msg = str(int(customer_id)) if backend_pid is None else f"{int(customer_id)}:{backend_pid}:{tx_epoch}"
    return hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()


@lru_cache(maxsize=4)
def reader_engine(url: str | None = None):
    url = url or reader_url()
    if url.startswith("sqlite"):
        eng = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(eng, "connect")
        def _ro(dbapi_conn, _):  # noqa: ANN001
            dbapi_conn.execute("PRAGMA query_only = ON")

        return eng
    return create_engine(url, pool_pre_ping=True, pool_size=5)


class OrderReader:
    """Customer-scoped, read-only SQL queries. `customer_id` always comes from the authenticated session."""

    def __init__(self, url: str | None = None, owner_url: str | None = None):
        self.engine = reader_engine(url)
        self.pg = _is_pg(self.engine)
        self.owner_url = owner_url

    def _query(self, customer_id: int, sql: str, **params) -> list[dict]:
        with self.engine.connect() as c:
            if self.pg:
                c.execute(text("SET TRANSACTION READ ONLY"))
                c.execute(text("SET LOCAL statement_timeout = '3s'"))
                self._scope(c, customer_id)
            rows = [dict(r) for r in c.execute(text(sql), {"cid": int(customer_id), **params}).mappings()]
            c.rollback()
        return rows

    def _scope(self, c, customer_id: int) -> None:
        """Sign the customer into this transaction and confirm the database accepted it (fail closed)."""
        pid, epoch = c.execute(text("SELECT pg_backend_pid(), extract(epoch FROM now())::text")).one()
        for refresh in (False, True):  # a second try picks up a key rotated since it was cached
            sig = sign_scope(customer_id, scope_key(self.owner_url, refresh=refresh), pid, epoch)
            ok = c.execute(text("SELECT set_config('app.customer_id', :cid, true), set_config('app.scope_sig', :sig, true), "
                                "agent_scope()"), {"cid": str(int(customer_id)), "sig": sig}).one()[2]
            if ok == int(customer_id):
                return
        raise PermissionError("database rejected the customer scope signature")

    @staticmethod
    def _ts(v):
        if v is None:
            return None
        return v.isoformat(timespec="minutes") if hasattr(v, "isoformat") else str(v)[:16].replace(" ", "T")

    def _summary(self, r: dict) -> dict:
        return {"order_id": r["order_id"], "status": r["status"], "slot": r["slot"], "payment_method": r["payment_method"],
                "total": r["total"], "placed_at": self._ts(r["placed_at"]), "delivered_at": self._ts(r["delivered_at"])}

    def list_orders(self, customer_id: int, status: str | None = None, limit: int = 10) -> list[dict]:
        sql = ("SELECT * FROM agent_orders WHERE customer_id = :cid"
               + (" AND status = :status" if status else "")
               + " ORDER BY placed_at DESC LIMIT :limit")
        return [self._summary(r) for r in self._query(customer_id, sql, status=status, limit=min(int(limit), 25))]

    def order_details(self, customer_id: int, order_id: int) -> dict | None:
        rows = self._query(customer_id, "SELECT * FROM agent_orders WHERE customer_id = :cid AND order_id = :oid",
                           oid=int(order_id))
        if not rows:
            return None  # unknown or another customer's: the caller reports both the same way
        items = self._query(customer_id, "SELECT * FROM agent_order_items WHERE customer_id = :cid AND order_id = :oid "
                                         "ORDER BY line_id", oid=int(order_id))
        refunds = self._query(customer_id, "SELECT * FROM agent_refunds WHERE customer_id = :cid AND order_id = :oid",
                              oid=int(order_id))
        return {**self._summary(rows[0]),
                "items": [{"product_id": i["product_id"], "name": i["name"], "qty": i["qty"], "unit_price": i["unit_price"],
                           "perishable": bool(i["perishable"]), "already_refunded_qty": int(i["refunded_qty"])} for i in items],
                "refunds": [{"product_id": r["product_id"], "qty": r["qty"], "amount": r["amount"], "status": r["status"],
                             "method": r["method"]} for r in refunds]}

    def search_orders(self, customer_id: int, product: str | None = None, status: str | None = None,
                      within_days: int | None = None, limit: int = 10) -> list[dict]:
        """Orders matching a product name fragment, status and/or recency, with the matching items."""
        where = ["o.customer_id = :cid"]
        params: dict = {"limit": min(int(limit), 25)}
        if status:
            where.append("o.status = :status")
            params["status"] = status
        if within_days:
            where.append("o.placed_at >= :since")
            since = settings.now() - timedelta(days=int(within_days))
            params["since"] = since if self.pg else since.isoformat(sep=" ")
        if product:
            where.append("EXISTS (SELECT 1 FROM agent_order_items i WHERE i.order_id = o.order_id "
                         "AND i.customer_id = :cid AND LOWER(i.name) LIKE :product)")
            params["product"] = f"%{product.lower().strip()}%"
        orders = self._query(customer_id, f"SELECT * FROM agent_orders o WHERE {' AND '.join(where)} "
                                          "ORDER BY o.placed_at DESC LIMIT :limit", **params)
        out = []
        for o in orders:
            sql = "SELECT name, qty, unit_price FROM agent_order_items WHERE customer_id = :cid AND order_id = :oid"
            extra = {}
            if product:
                sql += " AND LOWER(name) LIKE :product"
                extra["product"] = params["product"]
            items = self._query(customer_id, sql, oid=o["order_id"], **extra)
            out.append({**self._summary(o), "matching_items" if product else "items": items})
        return out


_reader: OrderReader | None = None


def get_reader() -> OrderReader:
    global _reader
    if _reader is None:
        _reader = OrderReader()
    return _reader


def reset_reader() -> None:  # tests
    global _reader
    _reader = None
    reader_engine.cache_clear()
    _keys.clear()

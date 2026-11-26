"""Read-scoped SQL access for the agent.

SQLite tests always run. The Postgres tests check the database-enforced guarantees (read-only role,
per-customer row scoping); they run when TEST_POSTGRES_URL points at a disposable Postgres, e.g.
    docker run -d -p 55432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16
    TEST_POSTGRES_URL=postgresql+psycopg2://postgres:pg@localhost:55432/postgres pytest tests/test_sql_lookup.py
"""
import json
import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError

from app.sql_lookup import OrderReader, reset_reader
from app.tools.definitions import get_order_details, list_orders, search_orders


@pytest.fixture(autouse=True)
def _fresh_reader():
    reset_reader()
    yield
    reset_reader()


def cfg(cid):
    return {"configurable": {"customer_id": cid}}


def test_reader_matches_order_service(svc):
    r = OrderReader()
    assert r.list_orders(1, limit=5) == svc.list_orders(1, limit=5)
    oid = svc.list_orders(1, status="delivered")[0]["order_id"]
    assert r.order_details(1, oid) == svc.order_details(1, oid)


def test_reader_counts_units_reserved_by_open_quotes(svc):
    from app.orders import RefundLine

    big = next(o for o in svc.list_orders(5, status="delivered") if True)
    d = svc.order_details(5, big["order_id"])
    item = d["items"][0]
    q = svc.quote_refund(5, big["order_id"], [RefundLine(item["product_id"], 1)], "damaged")
    if q.lines and q.requires_approval:
        svc.request_approval(5, q.quote_id)
    assert OrderReader().order_details(5, big["order_id"]) == svc.order_details(5, big["order_id"])


def test_reader_url_never_falls_back_to_owner_credentials(monkeypatch):
    from app import sql_lookup

    monkeypatch.setattr(sql_lookup.settings, "readonly_database_url", None)
    monkeypatch.setattr(sql_lookup.settings, "database_url", "postgresql+psycopg2://agentdesk:ownerpw@db:5432/agentdesk")
    monkeypatch.setenv("READER_DB_PASSWORD", "r:p%w")
    url = sql_lookup.reader_url()
    assert url.startswith("postgresql+psycopg2://agentdesk_reader:") and "ownerpw" not in url


def test_reader_hides_other_customers_orders(svc):
    other = svc.list_orders(2)[0]["order_id"]
    assert OrderReader().order_details(1, other) is None
    out = json.loads(get_order_details.invoke({"order_id": other}, cfg(1)))
    assert out["error"] == "not_found"
    assert all(o["order_id"] != other for o in json.loads(list_orders.invoke({"limit": 25}, cfg(1))))


def test_reader_connection_is_read_only():
    with pytest.raises(OperationalError):
        with OrderReader().engine.connect() as c:
            c.execute(text("UPDATE orders SET status = 'cancelled'"))


def test_search_orders_by_product_and_recency(svc):
    oid = svc.list_orders(3, status="delivered")[0]["order_id"]
    name = svc.order_details(3, oid)["items"][0]["name"]
    hits = json.loads(search_orders.invoke({"product": name.split()[0].lower()}, cfg(3)))
    assert oid in [h["order_id"] for h in hits]
    assert all(any(name.split()[0].lower() in i["name"].lower() for i in h["matching_items"]) for h in hits)
    recent = json.loads(search_orders.invoke({"within_days": 1}, cfg(3)))
    old = json.loads(search_orders.invoke({"within_days": 365}, cfg(3)))
    assert 0 < len(recent) < len(old)


def test_search_orders_validates_arguments():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        search_orders.invoke({"within_days": 0}, cfg(1))
    with pytest.raises(ValidationError):
        search_orders.invoke({"status": "shipped"}, cfg(1))


# ---------- Postgres: guarantees enforced by the database ----------
PG = os.getenv("TEST_POSTGRES_URL")
pg = pytest.mark.skipif(not PG, reason="set TEST_POSTGRES_URL to run Postgres read-scope tests")


@pytest.fixture
def pg_reader():
    from sqlalchemy.engine import make_url

    from app.seed import seed

    os.environ["READER_DB_PASSWORD"] = "pw:with%odd'chars"  # must survive SQLAlchemy and driver parsing
    try:
        seed(PG, n_customers=5)
    finally:
        url = make_url(PG).set(username="agentdesk_reader", password=os.environ.pop("READER_DB_PASSWORD"))
    return OrderReader(url.render_as_string(hide_password=False), owner_url=PG)


def _sig_for(c, cid):
    """A valid signature for `cid` in connection c's current transaction, as the app would make it."""
    from app.sql_lookup import scope_key, sign_scope

    pid, epoch = c.execute(text("SELECT pg_backend_pid(), extract(epoch FROM now())::text")).one()
    return sign_scope(cid, scope_key(PG), pid, epoch)


@pg
def test_pg_signature_copied_from_another_session_does_not_verify(pg_reader):
    """Signatures are visible to same-role sessions (pg_stat_activity); they only work where they were made."""
    with pg_reader.engine.connect() as victim, pg_reader.engine.connect() as attacker:
        stolen = _sig_for(victim, 2)
        attacker.execute(text("SELECT set_config('app.customer_id', '2', true), set_config('app.scope_sig', :s, true)"),
                         {"s": stolen})
        assert attacker.execute(text("SELECT count(*) FROM agent_orders")).scalar() == 0
        victim.execute(text("SELECT set_config('app.customer_id', '2', true), set_config('app.scope_sig', :s, true)"),
                       {"s": stolen})
        assert victim.execute(text("SELECT count(*) FROM agent_orders")).scalar() > 0
        victim.rollback()  # new transaction: the same signature no longer verifies
        victim.execute(text("SELECT set_config('app.customer_id', '2', true), set_config('app.scope_sig', :s, true)"),
                       {"s": stolen})
        assert victim.execute(text("SELECT count(*) FROM agent_orders")).scalar() == 0


@pg
def test_pg_rotated_key_is_picked_up_without_restart(pg_reader):
    from sqlalchemy import create_engine

    assert pg_reader.list_orders(1)
    with create_engine(PG).begin() as c:
        c.execute(text("UPDATE agent_scope_key SET k = encode(gen_random_bytes(32), 'hex')"))
    assert pg_reader.list_orders(1)  # stale cached key -> refresh -> works


@pg
def test_pg_works_with_pgcrypto_in_another_schema():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url

    from app.seed import seed

    admin = create_engine(PG, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text("DROP DATABASE IF EXISTS agentdesk_ext_schema"))
        c.execute(text("CREATE DATABASE agentdesk_ext_schema"))
    url = make_url(PG).set(database="agentdesk_ext_schema").render_as_string(hide_password=False)
    with create_engine(url).begin() as c:
        c.execute(text("CREATE SCHEMA extensions"))
        c.execute(text("CREATE EXTENSION pgcrypto SCHEMA extensions"))
    seed(url, n_customers=5)
    reader = make_url(url).set(username="agentdesk_reader", password="agentdesk_reader").render_as_string(hide_password=False)
    assert OrderReader(reader, owner_url=url).list_orders(1)


@pg
def test_pg_reader_role_cannot_touch_base_tables(pg_reader):
    with pg_reader.engine.connect() as c:
        for sql in ("SELECT * FROM orders", "SELECT * FROM customers", "SELECT * FROM refund_quotes",
                    "SELECT * FROM agent_scope_key"):
            with pytest.raises(DBAPIError, match="permission denied"):
                c.execute(text(sql))
            c.rollback()


@pg
def test_pg_reader_role_cannot_write(pg_reader):
    with pg_reader.engine.connect() as c:
        with pytest.raises(DBAPIError):
            c.execute(text("DELETE FROM agent_orders"))


@pg
def test_pg_views_scope_rows_to_the_session_customer(pg_reader):
    with pg_reader.engine.connect() as c:
        c.execute(text("SELECT set_config('app.customer_id', '2', true), set_config('app.scope_sig', :s, true)"),
                  {"s": _sig_for(c, 2)})
        owners = {r[0] for r in c.execute(text("SELECT customer_id FROM agent_orders"))}  # no WHERE clause at all
        assert owners == {2}
        c.rollback()
        assert c.execute(text("SELECT count(*) FROM agent_orders")).scalar() == 0  # nothing without a customer
        c.rollback()


@pg
def test_pg_reader_cannot_forge_another_customers_scope(pg_reader):
    """A session that can run arbitrary SQL as the reader still can't choose whose rows it sees."""
    from app.sql_lookup import scope_key, sign_scope

    with pg_reader.engine.connect() as c:
        for sig in ("", "deadbeef", sign_scope(2, scope_key(PG)), _sig_for(c, 1)):  # none, garbage, unbound, customer 1's
            c.execute(text("SELECT set_config('app.customer_id', '2', true), set_config('app.scope_sig', :s, true)"), {"s": sig})
            for v in ("agent_orders", "agent_order_items", "agent_refunds"):
                assert c.execute(text(f"SELECT count(*) FROM {v}")).scalar() == 0
            c.rollback()
    assert pg_reader.list_orders(1) and all(o["order_id"] for o in pg_reader.list_orders(1))
    other = pg_reader.list_orders(2)[0]["order_id"]
    assert pg_reader.order_details(1, other) is None

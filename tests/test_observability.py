"""Tracing: per-turn trace ids, cost, failure scores; and a no-op when Langfuse isn't configured."""
from langchain_core.messages import AIMessage

from app import observability as obs
from tests.conftest import call


class FakeLangfuse:
    def __init__(self):
        self.scores = []

    def create_score(self, *, trace_id, name, value, comment=None):
        self.scores.append((trace_id, name, value, comment))

    def get_trace_url(self, *, trace_id):
        return f"https://lf.example/trace/{trace_id}"

    def flush(self):
        pass


def test_tracing_is_a_noop_without_keys(make_runtime, monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    rt, _ = make_runtime([AIMessage(content="Hi!")])
    r = rt.chat(1, "hello")
    assert r.trace_id is None and r.trace_url is None and r.reply == "Hi!"


def test_each_turn_is_traced_and_scored(make_runtime, monkeypatch):
    fake = FakeLangfuse()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")  # real handler, but no network export
    monkeypatch.setattr(obs, "client", lambda: fake)
    msg = AIMessage(content="Your order is on its way.", usage_metadata={"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100})
    rt, _ = make_runtime([call("get_order_details", {"order_id": 999999}), msg])
    r = rt.chat(1, "where is order 999999?")
    assert r.trace_id and r.trace_url.endswith(r.trace_id)
    assert r.tool_errors == 1  # not_found
    assert abs(r.cost_usd - (1000 * obs.PRICE_IN + 100 * obs.PRICE_OUT) / 1e6) < 1e-9
    names = {n: v for t, n, v, _ in fake.scores if t == r.trace_id}
    assert names["tool_errors"] == 1 and names["turn_failed"] == 0 and "latency_s" in names and "cost_usd" in names


def test_failed_turn_is_scored(make_runtime, monkeypatch):
    import pytest

    fake = FakeLangfuse()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")  # real handler, but no network export
    monkeypatch.setattr(obs, "client", lambda: fake)

    def boom(_):
        raise RuntimeError("LLM provider down")

    rt, _ = make_runtime([boom])
    with pytest.raises(RuntimeError):
        rt.chat(1, "hello")
    assert any(n == "turn_failed" and v == 1 and "provider down" in c for _, n, v, c in fake.scores)


def test_provider_reported_cost_wins():
    m = AIMessage(content="x", response_metadata={"token_usage": {"cost": 0.0123}},
                  usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})
    assert obs.message_cost(m) == 0.0123


def test_turn_cost_includes_summarisation(make_runtime, monkeypatch):
    """Summarisation removes old messages and its own reply is never stored; cost must still count it."""
    from app.config import settings

    monkeypatch.setattr(settings, "summarize_after_messages", 4)  # turn 3 has 5 messages
    monkeypatch.setattr(settings, "keep_recent_messages", 1)
    u = {"input_tokens": 1_000_000, "output_tokens": 0, "total_tokens": 1_000_000}
    rt, _ = make_runtime([AIMessage(content="a1", usage_metadata=u), AIMessage(content="a2", usage_metadata=u),
                          AIMessage(content="summary", usage_metadata=u), AIMessage(content="a3", usage_metadata=u)])
    r1 = rt.chat(1, "one")
    rt.chat(1, "two", r1.thread_id)
    r3 = rt.chat(1, "three", r1.thread_id)  # this turn = summary call + answer
    assert r3.tokens["input"] == 2_000_000 and abs(r3.cost_usd - 2 * obs.PRICE_IN) < 1e-9


def test_crashed_turn_keeps_its_trace_id(make_runtime, monkeypatch):
    import pytest

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setattr(obs, "client", lambda: FakeLangfuse())

    def boom(_):
        raise RuntimeError("down")

    rt, _ = make_runtime([boom])
    with pytest.raises(RuntimeError) as ei:
        rt.chat(1, "hello")
    assert getattr(ei.value, "agentdesk_trace_id", None)


def test_failed_approval_service_call_is_traced(make_runtime, svc, monkeypatch):
    """A service outage while recording a supervisor decision is a traced, scored failure."""
    import pytest

    from app.orders import OrderError
    from app.tools import definitions

    fake = FakeLangfuse()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    monkeypatch.setattr(obs, "client", lambda: fake)
    from app.db import Approval, session_factory
    from app.config import settings
    with session_factory().begin() as s:
        s.add(Approval(thread_id="t-out", customer_id=1, payload="{}", quote_id="q-missing", status="pending",
                       created_at=settings.now()))

    def down(*a, **k):
        raise OrderError("unavailable", "order service timed out")

    monkeypatch.setattr(definitions.client(), "decide_quote", down)
    rt, _ = make_runtime([])
    with pytest.raises(OrderError) as ei:
        rt.decide("t-out", True, "sup")
    assert getattr(ei.value, "agentdesk_trace_id", None)
    assert any(n == "turn_failed" and v == 1 for _, n, v, _ in fake.scores)

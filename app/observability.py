"""Langfuse tracing, cost accounting and failure signals for every agent turn.

Each chat turn or supervisor decision gets its own Langfuse trace (session = thread, user = customer).
LangChain/LangGraph callbacks record every LLM call, tool call, latency and token count under it. On
top of that each trace gets scores you can chart and filter on in Langfuse:

    latency_s    wall time of the turn
    cost_usd     LLM cost (OpenRouter-reported when available, else the configured price table)
    tool_errors  tool calls that returned an error (validation, not_found, policy refusals ...)
    turn_failed  1 when the turn raised (LLM outage, crash), with the exception as the comment

The eval adds task_success, tool_call_accuracy and faithfulness to the same traces, so a failing
scenario links straight to the trace that explains it (see eval/triage.py).

Tracing is off when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set; the agent works the same.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.callbacks import BaseCallbackHandler

log = logging.getLogger("agentdesk.obs")

# USD per million tokens, used only when the provider doesn't report cost. Override for other models.
PRICE_IN = float(os.getenv("LLM_PRICE_INPUT_PER_MTOK", "0.15"))
PRICE_OUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_MTOK", "0.60"))


def enabled() -> bool:
    pk, sk = os.getenv("LANGFUSE_PUBLIC_KEY", ""), os.getenv("LANGFUSE_SECRET_KEY", "")
    return bool(pk and sk) and not pk.endswith("...") and not sk.endswith("...")


@lru_cache(maxsize=1)
def client():
    from langfuse import get_client

    return get_client()


@dataclass
class Trace:
    trace_id: str | None = None
    callbacks: list = field(default_factory=list)

    @property
    def url(self) -> str | None:
        if not self.trace_id:
            return None
        try:
            return client().get_trace_url(trace_id=self.trace_id)
        except Exception:  # noqa: BLE001
            return None

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        if not self.trace_id:
            return
        try:
            client().create_score(trace_id=self.trace_id, name=name, value=float(value), comment=comment)
        except Exception:  # noqa: BLE001 - observability must never break a turn
            log.warning("langfuse score %s failed", name, exc_info=True)


@contextmanager
def trace_turn(name: str = "chat-turn"):
    """Yields a Trace whose callbacks go into the graph config. Records turn_failed if the body raises."""
    if not enabled():
        yield Trace()
        return
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        tid = Langfuse.create_trace_id()
        t = Trace(tid, [CallbackHandler(trace_context={"trace_id": tid})])
    except Exception:  # noqa: BLE001
        log.warning("langfuse unavailable; tracing disabled for this turn", exc_info=True)
        yield Trace()
        return
    try:
        yield t
    except Exception as e:
        t.score("turn_failed", 1, comment=f"{type(e).__name__}: {e}"[:500])
        try:
            e.agentdesk_trace_id = t.trace_id  # lets the eval link and score a crashed turn
        except AttributeError:
            pass
        raise


class UsageTracker(BaseCallbackHandler):
    """Callback that sums tokens and cost over every LLM call in a turn: agent steps, the forced final
    answer and thread summarisation, none of which can be recovered from the stored messages alone."""

    def __init__(self):
        self.tokens = {"input": 0, "output": 0}
        self.cost = 0.0
        self.calls = 0

    def on_llm_end(self, response, **kwargs):
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                if msg is None:
                    continue
                u = getattr(msg, "usage_metadata", None) or {}
                self.tokens["input"] += u.get("input_tokens", 0)
                self.tokens["output"] += u.get("output_tokens", 0)
                self.cost += message_cost(msg)
                self.calls += 1


def message_cost(msg) -> float:
    """Cost of one AIMessage in USD."""
    meta = getattr(msg, "response_metadata", None) or {}
    usage = meta.get("token_usage") or {}
    if isinstance(usage.get("cost"), (int, float)):  # OpenRouter usage accounting
        return float(usage["cost"])
    u = getattr(msg, "usage_metadata", None) or {}
    return (u.get("input_tokens", 0) * PRICE_IN + u.get("output_tokens", 0) * PRICE_OUT) / 1_000_000


def flush() -> None:
    if enabled():
        try:
            client().flush()
        except Exception:  # noqa: BLE001
            pass

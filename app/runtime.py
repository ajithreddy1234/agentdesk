"""Runs conversation turns and approval decisions against the compiled graph."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from . import observability as obs
from .config import settings
from .db import Approval, ThreadOwner, session_factory


@dataclass
class TurnResult:
    thread_id: str
    reply: str
    status: str  # done | awaiting_approval
    pending_approval: dict | None = None
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    tool_errors: int = 0
    trace_id: str | None = None
    trace_url: str | None = None


class AgentRuntime:
    def __init__(self, graph, callbacks: list | None = None, db_url: str | None = None):
        self.graph = graph
        self.callbacks = callbacks or []
        self.Session = session_factory(db_url)

    def _config(self, thread_id: str, customer_id: int, trace: obs.Trace | None = None,
                tags: list[str] | None = None, metadata: dict | None = None, usage=None) -> dict:
        return {"configurable": {"thread_id": thread_id, "customer_id": customer_id},
                "callbacks": [*self.callbacks, *(trace.callbacks if trace else []), *([usage] if usage else [])],
                "recursion_limit": 40,
                "metadata": {"langfuse_session_id": thread_id, "langfuse_user_id": str(customer_id),
                             "langfuse_tags": tags or [], **(metadata or {})}}

    def _claim(self, thread_id: str, customer_id: int) -> None:
        """Bind a thread to its customer on first use; refuse access from anyone else."""
        with self.Session.begin() as s:
            row = s.get(ThreadOwner, thread_id)
            if row is None:
                s.add(ThreadOwner(thread_id=thread_id, customer_id=customer_id, created_at=settings.now()))
            elif row.customer_id != customer_id:
                raise PermissionError("This conversation belongs to another customer")

    def chat(self, customer_id: int, message: str, thread_id: str | None = None,
             tags: list[str] | None = None, metadata: dict | None = None) -> TurnResult:
        thread_id = thread_id or uuid.uuid4().hex
        self._claim(thread_id, customer_id)  # an access refusal, not a failed turn: raised before tracing
        cfg = self._config(thread_id, customer_id)
        usage = obs.UsageTracker()
        with obs.trace_turn("chat-turn") as trace:  # everything that can fail from here on is traced
            t0 = time.perf_counter()
            existing = self.graph.get_state(cfg)
            before = len((existing.values or {}).get("messages", [])) if existing else 0
            if existing and existing.next and "approval" in existing.next:
                return self._result(thread_id, cfg, 0.0, before,
                                    override="Your refund request is still waiting for a supervisor. I'll update you once it's reviewed.")
            self.graph.invoke({"messages": [HumanMessage(message)]},
                              self._config(thread_id, customer_id, trace, ["chat", *(tags or [])], metadata, usage))
            res = self._result(thread_id, cfg, (time.perf_counter() - t0) * 1000, before)
        return self._finish(res, trace, usage)

    def decide(self, thread_id: str, approved: bool, reviewer: str, note: str | None = None,
               tags: list[str] | None = None, metadata: dict | None = None) -> TurnResult:
        """Apply a supervisor decision and resume the paused thread.

        Recoverable: the approval row moves pending -> processing before the graph resumes and
        only reaches approved/rejected after the resumed run finishes. If the process dies in
        between, the row stays 'processing' and the same decision can simply be submitted
        again; the refund itself is idempotent on its stored quote, so it is never paid twice.
        """
        with self.Session.begin() as s:
            a = s.get(Approval, thread_id)
            if not a or a.status not in ("pending", "processing"):
                raise LookupError("No pending approval for this conversation")
            a.status, a.reviewer = "processing", reviewer
            a.note = f"{'approved' if approved else 'rejected'}: {note or ''}"[:2000]
            customer_id, quote_id = a.customer_id, a.quote_id
        cfg = self._config(thread_id, customer_id)
        usage = obs.UsageTracker()
        with obs.trace_turn("approval-decision") as trace:  # covers the service call and the resumed run
            t0 = time.perf_counter()
            if quote_id:
                from .tools.definitions import client

                client().decide_quote(quote_id, approved, reviewer)  # durable before the graph resumes
            state = self.graph.get_state(cfg)
            before = len(state.values.get("messages", []))
            if state.next and "approval" in state.next:
                self.graph.invoke(Command(resume={"approved": approved, "reviewer": reviewer, "note": note}),
                                  self._config(thread_id, customer_id, trace, ["approval", *(tags or [])],
                                               {"reviewer": reviewer, "approved": approved, **(metadata or {})}, usage))
            with self.Session.begin() as s:
                a = s.get(Approval, thread_id)
                a.status = "approved" if approved else "rejected"
            res = self._result(thread_id, cfg, (time.perf_counter() - t0) * 1000, before)
        return self._finish(res, trace, usage)

    @staticmethod
    def _finish(res: TurnResult, trace: obs.Trace, usage=None) -> TurnResult:
        if usage is not None and usage.calls:  # callback totals include summarisation and forced answers
            res.tokens, res.cost_usd = dict(usage.tokens), round(usage.cost, 6)
        res.trace_id, res.trace_url = trace.trace_id, trace.url
        trace.score("latency_s", res.latency_ms / 1000)
        trace.score("cost_usd", res.cost_usd)
        trace.score("tool_errors", res.tool_errors)
        trace.score("turn_failed", 0)
        return res

    def pending_approvals(self) -> list[dict]:
        with self.Session() as s:
            rows = s.query(Approval).filter_by(status="pending").order_by(Approval.created_at).all()
            return [{"thread_id": r.thread_id, "customer_id": r.customer_id, "refund": json.loads(r.payload),
                     "created_at": r.created_at.isoformat()} for r in rows]

    def history(self, thread_id: str, customer_id: int) -> list[dict]:
        self._claim(thread_id, customer_id)
        st = self.graph.get_state(self._config(thread_id, customer_id))
        out = []
        for m in (st.values or {}).get("messages", []):
            if isinstance(m, HumanMessage):
                out.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage) and m.content:
                out.append({"role": "assistant", "content": m.content})
        return out

    def _result(self, thread_id, cfg, latency_ms, before, override=None) -> TurnResult:
        st = self.graph.get_state(cfg)
        msgs = st.values.get("messages", [])
        new = msgs[before:] if before <= len(msgs) else msgs
        calls, tokens, cost = [], {"input": 0, "output": 0}, 0.0
        results = {m.tool_call_id: m.content for m in new if isinstance(m, ToolMessage)}
        for m in new:
            if isinstance(m, AIMessage):
                for c in m.tool_calls:
                    calls.append({"name": c["name"], "args": c["args"], "result": _safe_json(results.get(c["id"]))})
                u = getattr(m, "usage_metadata", None) or {}
                tokens["input"] += u.get("input_tokens", 0)
                tokens["output"] += u.get("output_tokens", 0)
                cost += obs.message_cost(m)
        errors = sum(1 for c in calls if isinstance(c["result"], dict) and "error" in c["result"])
        extra = {"cost_usd": round(cost, 6), "tool_errors": errors}
        pending = None
        if st.interrupts:
            pending = st.interrupts[0].value
            with self.Session.begin() as s:
                if not s.get(Approval, thread_id) or s.get(Approval, thread_id).status != "pending":
                    s.merge(Approval(thread_id=thread_id, customer_id=pending["customer_id"], payload=json.dumps(pending),
                                     quote_id=pending.get("quote_id"), status="pending", created_at=settings.now()))
            reply = override or (f"Your refund of Rs {pending['amount']:.2f} for order {pending['order_id']} needs a quick "
                                 f"supervisor review ({pending.get('approval_reason')}). You'll be updated here once it's approved.")
            return TurnResult(thread_id, reply, "awaiting_approval", pending, calls, round(latency_ms, 1), tokens, **extra)
        last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and m.content), None)
        reply = override or (last_ai.content if last_ai else "")
        return TurnResult(thread_id, reply, "done", None, calls, round(latency_ms, 1), tokens, **extra)


def _safe_json(s):
    try:
        return json.loads(s) if s else None
    except (TypeError, json.JSONDecodeError):
        return s


def as_dict(r: TurnResult) -> dict:
    return asdict(r)

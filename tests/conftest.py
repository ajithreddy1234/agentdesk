import os
import sys
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ.setdefault("AGENTDESK_NOW", "2026-11-20T09:00:00")  # frozen clock for deterministic tests
os.environ.pop("SERVICE_BASE_URL", None)
os.environ.pop("REDIS_URL", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class ScriptedLLM(BaseChatModel):
    """Fake chat model: each step is an AIMessage or a function(messages) -> AIMessage."""

    steps: list = []
    calls: list = []

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls.append(messages)
        step = self.steps.pop(0) if self.steps else AIMessage(content="Done.")
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id, "type": "tool_call"}])


@pytest.fixture(autouse=True)
def fresh_db():
    from app.seed import seed

    seed()
    yield


@pytest.fixture
def make_runtime():
    from langgraph.checkpoint.memory import InMemorySaver

    from app.agent import build_graph
    from app.runtime import AgentRuntime

    def _make(steps):
        llm = ScriptedLLM(steps=list(steps), calls=[])
        return AgentRuntime(build_graph(llm, InMemorySaver())), llm

    return _make


@pytest.fixture
def svc():
    from app.orders import OrderService

    return OrderService()

# AgentDesk: Multi-Tool AI Support Agent for E-commerce Operations

A LangGraph agent that resolves customer-support requests for a grocery delivery business from start to
finish. It looks up orders in SQL, calls the order/refund REST API, searches policy documents (RAG on
pgvector) and the web, and remembers customer preferences across chats. Refunds above Rs 500 pause for a
human supervisor before any money moves.

```
                 ┌──────────── prepare: roll old turns into a summary (context management)
 customer ──►    │
  /chat     ──►  agent (LLM + tools) ──► tools ──► small refund: auto-approved by policy
                     ▲                    │
                     │                    └──► refund > Rs 500 ──► approval (LangGraph interrupt)
                     └──────────────────────────────────────────────── supervisor /approvals ──┘

tools: list_orders · get_order_details · cancel_order · request_refund · search_policies (pgvector + full-text)
       web_search (Tavily) · escalate_to_human · remember_preference (Redis)
state: Postgres checkpoints (paused threads survive restarts) · Langfuse traces (latency, tokens, cost)
```

## What makes it production-shaped

| Concern | How it's handled |
| --- | --- |
| Multi-step tool use | The agent chains list → details → refund on its own, for example "the paneer in my last order was spoiled". |
| Context management | Thread state lives in LangGraph checkpoints. Long threads are summarised and old messages removed, always cutting at a user turn so tool calls stay paired. Profile, saved preferences and summary go into the system prompt. |
| Long-term memory | Redis hash per customer (allow-listed keys, size-capped, 180-day TTL) for preferences such as the refund method. |
| Auth | Customers and supervisors sign in for signed, expiring bearer tokens with roles. Customers can't see the approval queue, a conversation belongs to the customer who started it, and the internal order service requires a service token. (The demo uses shared demo passwords; swap `app/auth.py` logins for your identity provider.) |
| Guardrails | Pydantic schemas validate every tool call; validation errors go back to the model to self-correct. The customer id comes from the token, never from the model. The order service hides other customers' orders and enforces policy windows (24 h perishable, 7 days packaged, change of mind only for non-perishables). Tool-round limit; prompt-injection scenarios in the eval. |
| Refund safety | Refunds are **server-side quotes**: only a quote id ever travels, so amounts can't be altered. Applying a quote locks it, re-checks eligibility against the current order, requires a recorded approval when needed, and is idempotent, so a replayed or retried apply returns the original result without paying twice. Pending quotes reserve their units. |
| Human in the loop | Refunds above `AUTO_REFUND_LIMIT` (Rs 500), or from customers with ≥2 complaints in 30 days, pause the graph with `interrupt()`; smaller refunds are auto-approved by policy. The supervisor's decision is recorded durably before the thread resumes from its Postgres checkpoint. If the process dies mid-way the approval stays `processing` and the same decision can be resubmitted safely. |
| Policy RAG | Section-level chunks of six policy documents; `bge-small` embeddings in pgvector plus Postgres full-text search, fused with RRF. |
| Observability | Langfuse callback traces every LLM and tool call (session = thread, user = customer). Each API response returns latency, tokens and the tools used. |
| Reliability | HTTP order client with retries and backoff, a service token on internal endpoints, LLM retries, and tool errors returned to the model instead of crashing the turn. |

## Evaluation: 120 scenarios

`eval/scenarios.py` builds 120 labelled scenarios from the seeded database, so the expected facts are real:

| Category | n | Pass criteria |
| --- | --- | --- |
| Order status / details | 15 / 10 | right tool + reply names the order/status/items |
| Auto refund | 15 | `request_refund` with right order & reason; refund row approved; reply states amount |
| Refund needing approval | 10 | thread pauses; after supervisor approval the refund row exists |
| Ineligible refund | 10 | no refund row; reply explains the 24-hour rule |
| Cancel allowed / blocked | 8 / 6 | order cancelled / unchanged with an explanation |
| Policy questions | 25 | `search_policies` + reply contains the exact figure |
| Cross-customer access | 6 | no data from the other account, no refund |
| Prompt injection | 5 | no large refund, no cancellations |
| Food safety | 5 | `escalate_to_human(food_safety)` + helpline in reply |
| Long-term memory | 5 | preference saved in chat 1 and used for the refund method in chat 2 |

```bash
python -m eval.run_eval --tag baseline
# change a prompt / tool description / model, then:
python -m eval.run_eval --tag v2 --compare baseline
```

The report gives **task success**, **tool-call accuracy** (the share of scenarios where every required tool was called
with the right key arguments and no forbidden tool was used), **LLM-as-judge faithfulness** (1–5, and the share
of replies ≥4), **p50/p95 turn latency**, tokens per scenario, and a per-category breakdown. CI runs the
suite on every pull request when `OPENROUTER_API_KEY` is set and fails below 80% task success.

| Run | Task success | Tool-call accuracy | Faithfulness | p95 latency |
| --- | --- | --- | --- | --- |
| baseline | run locally | | | |
| after fixes | | | | |

## Run it

```bash
cp .env.example .env          # add OPENROUTER_API_KEY (+ optional TAVILY / LANGFUSE keys)
docker compose up --build     # http://localhost:8000: customer chat + supervisor approval panel
```

Sign in on the page as any customer with the demo customer password, and as `supervisor` in the approvals panel
(`DEMO_CUSTOMER_PASSWORD` and `SUPERVISORS` in `.env`). Data is seeded only into an empty database
(`python -m app.seed` resets it on purpose). Postgres and Redis use named volumes, so chats, approvals and
preferences survive restarts.

Without Docker: `pip install -r requirements.txt && python -m app.seed && uvicorn app.main:app` (SQLite,
in-memory vectors and memory). Tests use a scripted chat model and need no key: `pytest -q`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/auth/customer`, `/auth/supervisor` | Sign in → bearer token |
| POST | `/chat` (customer token) | `{message, thread_id?}` → reply, status (`done`/`awaiting_approval`), tool calls, latency, tokens |
| GET | `/chat/{thread_id}` (customer token) | Conversation history (owner only) |
| GET | `/approvals` (supervisor token) | Refunds waiting for review |
| POST | `/approvals/{thread_id}` (supervisor token) | `{approved, note}` → records the decision and resumes the paused agent |
| `*` | `/svc/...` (`X-Service-Token`) | Internal order/refund API the tools call. Quotes, approval and apply all take a quote id, never an amount. |

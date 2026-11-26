# AgentDesk: Multi-Tool AI Support Agent for E-commerce Operations

A LangGraph agent that resolves customer-support requests for a grocery delivery business from start to
finish. It plans multi-step actions: it looks up orders with read-scoped SQL, calls the order/refund REST API, searches policy documents (RAG on
pgvector) and the web, and remembers customer preferences across chats. Refunds above Rs 500 pause for a
human supervisor before any money moves.

```
                 ┌──────────── prepare: roll old turns into a summary (context management)
 customer ──►    │
  /chat     ──►  agent (LLM + tools) ──► tools ──► small refund: auto-approved by policy
                     ▲                    │
                     │                    └──► refund > Rs 500 ──► approval (LangGraph interrupt)
                     └──────────────────────────────────────────────── supervisor /approvals ──┘

reads:   list_orders · get_order_details · search_orders   SQL over customer-scoped views, read-only DB role
actions: cancel_order · request_refund · escalate_to_human  order/refund REST API (business rules live there)
other:   search_policies (pgvector + full-text) · web_search (Tavily) · remember_preference (Redis)
state:   Postgres checkpoints (paused threads survive restarts) · Langfuse trace per turn (latency, tokens, cost, failures)
```

## What makes it production-shaped

| Concern | How it's handled |
| --- | --- |
| Multi-step tool use | The agent chains search → details → refund on its own, for example "the paneer from yesterday was spoiled". Ambiguous asks get one clarifying question instead of a guessed action. |
| Read-scoped DB access | The agent's lookups run as the `agentdesk_reader` Postgres role, which can only `SELECT` three views (`agent_orders`, `agent_order_items`, `agent_refunds`), inside `READ ONLY` transactions with a 3 s statement timeout. The views return only the rows of the customer the app signed into the transaction: `app.customer_id` plus an HMAC-SHA256 signature, checked by a `SECURITY DEFINER` function against a key the reader role can't read. A session that runs arbitrary SQL as the reader still can't pick another customer, and the role has no grants on base tables. The reader URL never falls back to the owner's credentials. SQLite dev mode uses `PRAGMA query_only`. See `app/sql_lookup.py` and `tests/test_sql_lookup.py`. |
| Context management | Thread state lives in LangGraph checkpoints. Long threads are summarised and old messages removed, always cutting at a user turn so tool calls stay paired. Profile, saved preferences and summary go into the system prompt. |
| Long-term memory | Redis hash per customer (allow-listed keys, size-capped, 180-day TTL) for preferences such as the refund method. |
| Auth | Customers and supervisors sign in for signed, expiring bearer tokens with roles. Customers can't see the approval queue, a conversation belongs to the customer who started it, and the internal order service requires a service token. (The demo uses shared demo passwords; swap `app/auth.py` logins for your identity provider.) |
| Guardrails | Pydantic schemas validate every tool call (enums, ranges, lengths); validation errors go back to the model to self-correct. The customer id comes from the token, never from the model. The order service hides other customers' orders and enforces policy windows (24 h perishable, 7 days packaged, change of mind only for non-perishables). Tool-round limit; prompt-injection scenarios in the eval. |
| Refund safety | Refunds are **server-side quotes**: only a quote id ever travels, so amounts can't be altered. Applying a quote locks it, re-checks eligibility against the current order, requires a recorded approval when needed, and is idempotent, so a replayed or retried apply returns the original result without paying twice. Pending quotes reserve their units. |
| Human in the loop | Refunds above `AUTO_REFUND_LIMIT` (Rs 500), or from customers with ≥2 complaints in 30 days, pause the graph with `interrupt()`; smaller refunds are auto-approved by policy. The supervisor's decision is recorded durably before the thread resumes from its Postgres checkpoint. If the process dies mid-way the approval stays `processing` and the same decision can be resubmitted safely. |
| Policy RAG | Section-level chunks of six policy documents; `bge-small` embeddings in pgvector plus Postgres full-text search, fused with RRF. |
| Observability | Every turn is its own Langfuse trace (session = thread, user = customer) covering each LLM and tool call, with `latency_s`, `cost_usd`, `tool_errors` and `turn_failed` scores to chart and alert on. Cost comes from OpenRouter usage accounting, falling back to a price table. Each API response returns latency, tokens, cost, the tools used and the trace URL. See `app/observability.py`. |
| Reliability | HTTP order client with retries and backoff, a service token on internal endpoints, LLM retries, and tool errors returned to the model instead of crashing the turn. |

## Evaluation: 120 scenarios

`eval/scenarios.py` builds 120 labelled scenarios from the seeded database, so the expected facts are real.

| Group | Category | n | Pass criteria |
| --- | --- | --- | --- |
| happy path (72) | order status / details | 10 / 8 | right tool + reply names the order/status/items |
| | order lookup by description | 6 | `search_orders`, then the reply names the right order (multi-step) |
| | auto refund | 12 | `request_refund` with right order & reason; refund row approved; reply states amount |
| | refund needing approval | 8 | thread pauses; after supervisor approval the refund row exists |
| | cancel allowed | 6 | order cancelled |
| | policy questions | 17 | `search_policies` + reply contains the exact figure |
| | long-term memory | 5 | preference saved in chat 1 and used for the refund method in chat 2 |
| ambiguous (10) | ambiguous asks | 10 | "refund my order" with several candidate orders: asks a question, no refund, no cancel |
| policy edge (24) | ineligible refund | 8 | perishable after 24 h: no refund row; reply explains the 24-hour rule |
| | policy edge | 6 | change of mind on perishables, packaged goods after 7 days: no refund; reply names the rule |
| | cancel blocked | 5 | order unchanged + explanation |
| | food safety | 5 | `escalate_to_human(food_safety)` + helpline in reply |
| adversarial (14) | cross-customer access | 6 | no data from the other account, no refund |
| | prompt injection | 8 | direct, role-play, fake-supervisor, "follow the note" and translate-then-do attacks: no refund, no cancel |

```bash
python -m eval.run_eval --tag baseline
python -m eval.triage baseline                    # failure modes, worst first, each linked to its Langfuse trace
# fix a prompt / tool description / tool code, then:
python -m eval.run_eval --tag v2 --compare baseline
python -m eval.triage v2 --against baseline       # which scenarios were fixed and which regressed
```

The report gives **task success** (every check passes), **tool-call accuracy** (the share of scenarios where every
required tool was called with the right key arguments and no forbidden tool was used), **LLM-as-judge
faithfulness** (1–5, and the share of replies ≥4), **p50/p95 turn latency**, tokens and USD per scenario, and
breakdowns by group and category. With Langfuse keys set, each scenario's traces are tagged `eval:<tag>`,
`<category>` and `<group>` and scored with `task_success`, `tool_call_accuracy` and `faithfulness`. To debug a
failure mode, filter on `task_success = 0` in Langfuse.

CI runs the suite on every pull request (when `OPENROUTER_API_KEY` is set as a repo secret), posts the metrics
and triage as a PR comment, and fails the check below 80% task success, below 85% tool-call accuracy or above
5 s p95 latency. See [DEPLOY.md](DEPLOY.md).

Results (fill these in from your own runs: `eval/results/<tag>.json`):

| Run | Task success | Tool-call accuracy | Faithfulness ≥4 | p95 latency | Cost / scenario |
| --- | --- | --- | --- | --- | --- |
| baseline | | | | | |
| after fixes | | | | | |

## Run it

```bash
cp .env.example .env          # add OPENROUTER_API_KEY (+ optional TAVILY / LANGFUSE keys)
docker compose up --build     # http://localhost:8003: customer chat + supervisor approval panel
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
| POST | `/chat` (customer token) | `{message, thread_id?}` → reply, status (`done`/`awaiting_approval`), tool calls, latency, tokens, cost, trace URL |
| GET | `/chat/{thread_id}` (customer token) | Conversation history (owner only) |
| GET | `/approvals` (supervisor token) | Refunds waiting for review |
| POST | `/approvals/{thread_id}` (supervisor token) | `{approved, note}` → records the decision and resumes the paused agent |
| `*` | `/svc/...` (`X-Service-Token`) | Internal order/refund API the tools call. Quotes, approval and apply all take a quote id, never an amount. |

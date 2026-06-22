# shadow-chat

**A token-efficient AI chat engine** — drop-in OpenAI-compatible server that dramatically reduces LLM input costs by only sending what the model actually needs, instead of replaying the full conversation on every turn.

---

## The problem with traditional AI chat

Every mainstream AI chat client works the same way: on each message, it sends the **entire conversation history** to the LLM from scratch.

```
Turn 1:  [user msg]                              → ~50 tokens in
Turn 5:  [msg1 + msg2 + msg3 + msg4 + user msg] → ~900 tokens in
Turn 10: [all 9 prior turns + user msg]          → ~2,000 tokens in
Turn 20: [all 19 prior turns + user msg]         → ~5,000 tokens in
```

Input token cost grows linearly with conversation length. In agentic workflows (MCP tool calling), it gets worse — every tool result (often 10,000–15,000 tokens of API data) gets re-sent on every subsequent turn.

---

## How shadow-chat is different

shadow-chat flips the model: **the LLM starts with only the current message** and pulls history on-demand via internal tools, only when it actually needs context.

```
Turn 1:  [user msg]          → ~50 tokens in   (same)
Turn 5:  [user msg]          → ~50 tokens in   (history not sent)
Turn 10: [user msg]          → ~50 tokens in   (history not sent)
Turn 20: [user msg]          → ~50 tokens in   (history not sent)
```

If the LLM needs past context to answer, it calls `get_chat_history` or `semantic_search` — fetching only what's relevant. If the question is self-contained, no history is sent at all.

### Measured savings

Based on real session logs:

| Scenario                                        | Traditional                           | shadow-chat      | Saved               |
|-------------------------------------------------|---------------------------------------|------------------|---------------------|
| Simple question in a 10-turn session            | ~2,000 tokens                         | ~50 tokens       | **~97%**            |
| Follow-up question needing 2 prior turns        | ~2,000 tokens                         | ~400 tokens      | **~80%**            |
| MCP tool call on turn 8 (e.g. fetch portfolio)  | ~5,300 tokens of prior history re-sent | 0 tokens re-sent | **100% of history** |
| MCP tool call on turn 12 (after prior fetches)  | ~10,000 tokens of history re-sent     | 0 tokens re-sent | **100% of history** |

In a typical 20-turn conversation with MCP tools, shadow-chat avoids sending **5,000–15,000 tokens of prior history** per turn to the LLM — history that sits in the client but never needs to reach the model.

---

## Features

- **Lazy history loading** — only the latest user message is sent to the LLM; full conversation history is loaded on-demand via internal tools
- **MCP tool calling** — client-side MCP tools (Continue.dev executes them, server orchestrates); cached results reused across turns via `get_tool_result`
- **Turn-based data model** — every user interaction stored with full execution trace: all LLM calls, tool calls, token counts, and timestamps
- **Per-session and cross-session vector search** — FAISS + sentence-transformers for semantic search within and across all past sessions
- **Multi-provider** — Gemini, Anthropic, and any OpenAI-compatible endpoint

---

## Installation

**Requirements:** Python 3.11 or 3.12

```bash
git clone https://github.com/deepak-s-2000/shadow-chat.git
cd shadow-chat
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
API_KEY=your_bearer_token_here      # used to authenticate requests to this server
```

For the LLM providers, the API key is passed per-request in the `Authorization: Bearer` header by the client — no server-side provider keys needed.

### Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server auto-creates the SQLite database (`chatbot.db`) and all tables on first run.

---

## API

| Method     | Path                     | Description                       |
|------------|--------------------------|-----------------------------------|
| `GET`      | `/health`                | Health check                      |
| `GET`      | `/v1/models`             | List available models             |
| `POST`     | `/v1/chat/completions`   | OpenAI-compatible chat endpoint   |
| `GET`      | `/sessions`              | List all sessions                 |
| `POST`     | `/sessions`              | Create a session                  |
| `GET`      | `/sessions/{id}/history` | Message history                   |
| `GET`      | `/sessions/{id}/stats`   | Token usage and turn details      |
| `DELETE`   | `/sessions/{id}`         | Delete session and vector index   |

### Model name format

The model string in the request body controls which provider is used:

```
gemini/gemma-4-31b-it         → Google Gemini API
anthropic/claude-haiku-4-5-20251001 → Anthropic API
openai/gpt-4o-mini            → OpenAI API
```

---

## Internal tools

The server gives the LLM a set of built-in tools to retrieve context on demand instead of loading full history upfront.

### Current-session tools

| Tool                                        | Description                                                                                                  |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `get_chat_history(limit)`                   | Fetches the N most recent messages from the current session's DB                                             |
| `search_messages(query, limit)`             | Keyword search through the current session's message history                                                 |
| `get_conversation_stats()`                  | Returns user/assistant message counts for the current session                                                |
| `semantic_search(query, limit)`             | FAISS cosine-similarity search over the current session's vector index                                       |
| `get_tool_result(tool_name, max_age_turns)` | Retrieves a cached external MCP tool result from a recent turn — use this **before** calling any MCP tool to avoid redundant fetches |

### Cross-session tools

| Tool                                        | Description                                                                                    |
|---------------------------------------------|------------------------------------------------------------------------------------------------|
| `search_all_sessions(query, limit)`         | Semantic search across **all** sessions — useful when the user asks about a past conversation  |
| `search_history_all_sessions(query, limit)` | Keyword search across **all** sessions                                                         |

---

## Integration with Continue.dev

[Continue.dev](https://docs.continue.dev) is a VS Code / JetBrains AI coding extension that supports custom OpenAI-compatible models and MCP servers.

### 1. Add shadow-chat as a model

Edit `~/.continue/config.yaml`:

```yaml
models:
  - name: shadow-chat
    provider: openai
    model: gemini/gemma-4-31b-it    # or any supported model string
    apiBase: http://localhost:8000/v1
    apiKey: your_bearer_token_here   # must match API_KEY in your .env
    capabilities:
      - tool_use                     # required — enables MCP tool calling
    roles:
      - chat
      - edit
      - apply
```

> **Important:** The `capabilities: [tool_use]` line is required. Without it, Continue.dev won't send tools to the server for unrecognised model names and MCP tool calling will silently not work.

### 2. Add an MCP server (optional)

To use external MCP tools (e.g. Kite trading API), add them under `mcpServers`:

```yaml
mcpServers:
  - name: Kite MCP
    url: https://mcp.kite.trade/sse
```

Continue.dev connects to the MCP server, discovers its tools, and forwards them to shadow-chat with each request. The server routes them back to Continue.dev for execution.

### 3. How the MCP flow works

```
User message
    │
    ▼
shadow-chat receives request + tools list from Continue.dev
    │
    ▼
LLM decides to call an external tool (e.g. kite_mcp_get_holdings)
    │
    ▼
shadow-chat returns finish_reason: tool_calls to Continue.dev
    │
    ▼
Continue.dev executes the MCP tool
    │
    ▼
Continue.dev sends tool result back to shadow-chat
    │
    ▼
LLM produces final answer
```

The server stores the complete turn — both LLM calls and all tool results — in the `turns` table.

### 4. Avoiding redundant MCP fetches

If the LLM fetches a large external dataset (e.g. portfolio holdings), it will re-fetch the same data on follow-up questions unless you tell it not to. The `get_tool_result` internal tool handles this:

The LLM is instructed to call `get_tool_result("kite_mcp_get_holdings")` first — if a cached result from a recent turn exists, it uses that instead of triggering another MCP round-trip.

---

## Token saving

The server logs token usage after every completed turn:

**Standard mode (no MCP tools):**
```
actual input    :   1200 tokens
actual output   :    180 tokens
actual total    :   1380 tokens
est. baseline   :   8400 tokens  (naive full-history approach)
est. saved      :   7200 tokens
```
`est. saved` = tokens that would have been sent if the client passed the full conversation history to the LLM on every turn.

**MCP mode:**
```
actual input    :  21595 tokens
actual output   :   2240 tokens
actual total    :  23835 tokens
history not sent:   5336 tokens  (prior turns kept client-side)
```
`history not sent` = chars of prior conversation context that Continue.dev sent to the server but the server chose not to forward to the LLM.

---

## Project structure

```
app/
├── main.py                  FastAPI app setup
├── api/routes/
│   ├── openai_compat.py     /v1/chat/completions — main endpoint
│   ├── sessions.py          session CRUD + stats
│   └── chat.py              /chat/stream — native streaming endpoint
├── core/
│   ├── graph.py             LangGraph state machine (standard mode)
│   ├── tools.py             Internal tool definitions
│   ├── providers.py         LLM provider factory (Gemini / Anthropic / OpenAI)
│   └── vectorstore.py       FAISS per-session and cross-session search
├── db/
│   ├── database.py          SQLAlchemy engine + session
│   └── models.py            Session, Message, Turn, TokenUsage tables
└── schemas/
    └── chat.py              Pydantic request/response models
```

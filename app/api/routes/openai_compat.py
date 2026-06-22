"""
OpenAI-compatible /v1 endpoint.

Standard mode (no body.tools, last message is user):
  Only the latest user message is sent to the LLM; history loaded on-demand
  via internal tools. Uses LangGraph for the internal tool loop.

MCP mode (body.tools provided OR last message is a tool result):
  External MCP tools are bound alongside internal tools. When the LLM calls
  an external tool we return finish_reason: tool_calls so Chatbox can execute
  it via MCP, then resume on the follow-up request with tool results.
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from pydantic import BaseModel

from app.core.providers import create_model
from app.core.tools import create_tools
from app.core.graph import build_graph
from app.core.vectorstore import vector_store
from app.db.database import SessionLocal
from app.db.models import Session as DBSession, Message as DBMessage, Turn as DBTurn
from app.schemas.chat import ProviderConfig

router = APIRouter(prefix="/v1", tags=["openai-compatible"])
log = logging.getLogger("chat.compat")

AVAILABLE_MODELS = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-pro",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
]

_NAMING_PHRASES = (
    "give this conversation a name",
    "name this conversation",
    "title this conversation",
    "summarize this conversation",
    "please reply with a title",   # Continue.dev auto-naming prompt
    "title for the chat",          # Continue.dev auto-naming prompt
)


# ── schemas ────────────────────────────────────────────────────────────────────

class OAIToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class OAITool(BaseModel):
    type: str = "function"
    function: OAIToolFunction


class OAIMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None   # present on assistant tool-call turns
    tool_call_id: Optional[str] = None        # present on tool-result turns
    name: Optional[str] = None


class OAIRequest(BaseModel):
    model: str
    messages: list[OAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[list[OAITool]] = None     # MCP tools forwarded by Chatbox


# ── helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _time_system_message() -> SystemMessage:
    now = datetime.now(timezone.utc)
    return SystemMessage(content=f"Current date and time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}")


def _parse_model(model_str: str) -> tuple[str, str, Optional[str]]:
    base_url = None
    if "|" in model_str:
        model_str, base_url = model_str.split("|", 1)
    if model_str.startswith("gemini/"):
        return "gemini", model_str[len("gemini/"):], None
    if model_str.startswith("anthropic/"):
        return "anthropic", model_str[len("anthropic/"):], None
    if model_str.startswith("openai/"):
        return "openai_compatible", model_str[len("openai/"):], base_url
    return "openai_compatible", model_str, base_url


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return content if isinstance(content, str) else ""


def _derive_session_id(first_msg: str) -> str:
    return "auto-" + hashlib.md5(first_msg.encode()).hexdigest()[:16]


def _oai_chunk(text: str, model: str, cid: str) -> str:
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }) + "\n\n"


def _oai_tool_calls_chunks(ai_message, model: str, cid: str) -> list[str]:
    """
    Return the proper SSE sequence for tool_calls as a LIST so each item
    can be yielded separately (one HTTP chunk per SSE event, matching OpenAI).

    Sequence per OpenAI spec:
      1. role announcement          (finish_reason: null)
      2. tool name + id per call    (finish_reason: null)
      3. full arguments per call    (finish_reason: null)
      4. finish chunk               (finish_reason: "tool_calls")
    """
    created = int(time.time())
    tcs = ai_message.tool_calls or []

    def _evt(delta, finish=None):
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
        }) + "\n\n"

    chunks = [_evt({"role": "assistant", "content": None})]

    for i, tc in enumerate(tcs):
        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
        chunks.append(_evt({"tool_calls": [{
            "index": i, "id": tc_id, "type": "function",
            "function": {"name": tc.get("name", ""), "arguments": ""},
        }]}))
        chunks.append(_evt({"tool_calls": [{
            "index": i,
            "function": {"arguments": json.dumps(tc.get("args", {}))},
        }]}))

    chunks.append(_evt({}, finish="tool_calls"))
    return chunks


def _oai_done(model: str, cid: str) -> str:
    return (
        "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }) + "\n\ndata: [DONE]\n\n"
    )


def _extract_external_tool_results(
    messages: list, last_user_idx: int, turn_builder: dict
) -> None:
    """
    Scan the current turn's message slice for external tool call/result pairs
    and append them to turn_builder["tool_executions"].
    Called once per tool-continuation request before the LLM loop starts.
    """
    tool_call_args: dict[str, dict] = {}
    for msg in messages[last_user_idx:]:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                tool_call_args[tc.get("id", "")] = {
                    "name": fn.get("name", ""),
                    "args": args,
                }
    for msg in messages[last_user_idx:]:
        if msg.role == "tool":
            tc_info = tool_call_args.get(msg.tool_call_id or "", {})
            turn_builder["tool_executions"].append({
                "type": "external",
                "name": msg.name or tc_info.get("name", "unknown"),
                "args": tc_info.get("args", {}),
                "result": (msg.content or "")[:10000],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })


def _to_lc_message(m: OAIMessage):
    """Convert an OAIMessage to the appropriate LangChain message type."""
    if m.role == "system":
        return SystemMessage(content=m.content or "")
    if m.role == "user":
        return HumanMessage(content=m.content or "")
    if m.role == "assistant":
        if m.tool_calls:
            lc_tcs = []
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                lc_tcs.append({
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "name": fn.get("name", ""),
                    "args": args,
                    "type": "tool_call",
                })
            return AIMessage(content=m.content or "", tool_calls=lc_tcs)
        return AIMessage(content=m.content or "")
    if m.role == "tool":
        return ToolMessage(content=m.content or "", tool_call_id=m.tool_call_id or "", name=m.name or "")
    return HumanMessage(content=m.content or "")


# ── routes ─────────────────────────────────────────────────────────────────────

@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": 0, "owned_by": "user"} for m in AVAILABLE_MODELS],
    }


@router.post("/chat/completions")
async def chat_completions(
    body: OAIRequest,
    authorization: Optional[str] = Header(default=None),
    x_base_url: Optional[str] = Header(default=None, alias="x-base-url"),
    x_session_id: Optional[str] = Header(default=None, alias="x-session-id"),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    api_key = authorization[len("Bearer "):]

    if not body.messages:
        raise HTTPException(status_code=422, detail="messages must not be empty")

    # Deflect Chatbox auto-naming requests cheaply
    last_content = (body.messages[-1].content or "").lower()
    if any(phrase in last_content for phrase in _NAMING_PHRASES):
        log.info("[COMPAT] deflecting UI naming request")
        cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        if body.stream:
            async def _naming():
                yield _oai_chunk("Chat", body.model, cid)
                yield _oai_done(body.model, cid)
            return StreamingResponse(_naming(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache"})
        return {
            "id": cid, "object": "chat.completion", "created": int(time.time()),
            "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Chat"}, "finish_reason": "stop"}],
        }

    provider_type, model_name, base_url = _parse_model(body.model)
    base_url = x_base_url or base_url

    first_content = body.messages[0].content or ""
    session_id = x_session_id or _derive_session_id(first_content)

    is_tool_continuation = (body.messages[-1].role == "tool")
    has_external_tools = bool(body.tools)
    use_mcp_loop = has_external_tools or is_tool_continuation

    async def generate():
        db = SessionLocal()
        try:
            # ── 1. session ─────────────────────────────────────────────────────
            session = db.query(DBSession).filter(DBSession.id == session_id).first()
            if not session:
                title = first_content[:100]
                session = DBSession(id=session_id, title=title)
                db.add(session)
                db.commit()
                log.info("[COMPAT] new session  id=%s  title=%r", session_id, title)
            else:
                log.info("[COMPAT] existing session  id=%s", session_id)

            # ── 2. sync prior real messages to DB ──────────────────────────────
            # Skip tool-intermediate messages (tool_calls with null content, role=tool).
            # Those are MCP implementation details; we only store conversation turns.
            if not is_tool_continuation:
                real_prior = [
                    m for m in body.messages[:-1]
                    if m.role in ("user", "assistant") and (m.content or "").strip()
                ]
                db_count = db.query(DBMessage).filter(DBMessage.session_id == session_id).count()
                if db_count < len(real_prior):
                    new_msgs = real_prior[db_count:]
                    for m in new_msgs:
                        db.add(DBMessage(session_id=session_id, role=m.role, content=m.content))
                    db.commit()
                    db_count += len(new_msgs)
                    log.info("[COMPAT] synced %d prior messages to DB", len(new_msgs))
                else:
                    db_count = db.query(DBMessage).filter(DBMessage.session_id == session_id).count()
                    log.info("[COMPAT] DB up to date  db_count=%d", db_count)
            else:
                db_count = db.query(DBMessage).filter(DBMessage.session_id == session_id).count()
                log.info("[COMPAT] tool continuation  db_count=%d", db_count)

            # ── 3. locate the user message for this turn ───────────────────────
            last_user_msg = next((m for m in reversed(body.messages) if m.role == "user"), None)
            if not last_user_msg:
                yield _oai_done(body.model, f"chatcmpl-{uuid.uuid4().hex[:8]}")
                return
            latest_content = last_user_msg.content or ""

            log.info(
                "[COMPAT] request  session=%s  provider=%s/%s  mode=%s  history_in_db=%d  message=%r",
                session_id, provider_type, model_name,
                "mcp" if use_mcp_loop else "standard",
                db_count, latest_content[:80],
            )

            # ── 4. build LLM + internal tools ──────────────────────────────────
            try:
                llm = create_model(ProviderConfig(
                    type=provider_type, api_key=api_key, model=model_name, base_url=base_url,
                ))
            except Exception as e:
                log.error("[COMPAT] provider init failed: %s", e)
                yield _oai_done(body.model, f"chatcmpl-{uuid.uuid4().hex[:8]}")
                return

            def get_history(limit: int = 100):
                rows = (
                    db.query(DBMessage)
                    .filter(DBMessage.session_id == session_id)
                    .order_by(DBMessage.created_at.asc())
                    .limit(limit)
                    .all()
                )
                return [{"role": r.role, "content": r.content} for r in rows]

            def search_vectors_fn(query: str, k: int = 3) -> list:
                vector_store.backfill_if_needed(session_id, get_history(limit=10000))
                return vector_store.search(session_id, query, k)

            def search_all_vectors_fn(query: str, k: int = 5) -> list:
                return vector_store.search_all(query, k)

            def search_all_history_fn(query: str, limit: int = 5) -> list:
                rows = (
                    db.query(DBMessage)
                    .filter(DBMessage.content.ilike(f"%{query}%"))
                    .order_by(DBMessage.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "session_id": r.session_id,
                        "role": r.role,
                        "content": r.content,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in rows
                ]

            def get_recent_turns_fn(limit: int = 10) -> list:
                rows = (
                    db.query(DBTurn)
                    .filter(DBTurn.session_id == session_id)
                    .order_by(DBTurn.created_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    {
                        "id": r.id,
                        "user_message": r.user_message,
                        "tool_executions": r.tool_executions,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in rows
                ]

            internal_tools = create_tools(
                get_history,
                search_vectors=search_vectors_fn,
                get_recent_turns=get_recent_turns_fn,
                search_all_vectors=search_all_vectors_fn,
                search_all_history=search_all_history_fn,
            )
            tools_map = {t.name: t for t in internal_tools}
            internal_tool_names = set(tools_map.keys())

            # Turn builder accumulates all data for the Turn row saved at the end.
            # last_user_idx is resolved now so both turn_builder init and lc_messages
            # construction below can use the same value.
            last_user_idx = next(
                (i for i, m in reversed(list(enumerate(body.messages))) if m.role == "user"),
                0,
            )
            turn_builder: dict = {
                "user_message": latest_content,
                "tool_executions": [],
                "llm_call_log": [],
                "mode": "mcp" if use_mcp_loop else "standard",
            }
            if is_tool_continuation:
                _extract_external_tool_results(body.messages, last_user_idx, turn_builder)

            prior_history = get_history(limit=10000)
            if use_mcp_loop:
                # Saving = the conversation history from body.messages that we chose NOT to
                # forward to the LLM (everything before the current turn's user message).
                # Using direct char-count avoids the tool-definition inflation problem:
                # tool defs are in actual_input_tokens but not in this baseline, so comparing
                # baseline vs actual would always give 0. Instead, saving = chars we sliced away.
                prior_in_body_chars = sum(
                    len(m.content or "") for m in body.messages[:last_user_idx]
                )
                estimated_baseline_tokens = prior_in_body_chars // 4
            else:
                # Standard mode: baseline = cost of naively sending full DB history every turn.
                baseline_chars = sum(len(m["content"]) for m in prior_history) + len(latest_content)
                estimated_baseline_tokens = baseline_chars // 4

            cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            full_response = ""
            llm_calls = 0
            # Pre-seed with any external tools already extracted from this continuation's messages
            tools_called: list[str] = [
                f"[MCP]{ex['name']}" for ex in turn_builder["tool_executions"]
                if ex.get("type") == "external"
            ]
            actual_input_tokens = 0
            actual_output_tokens = 0

            # For MCP multi-step turns (fresh → tool-handoff → continuation), carry forward
            # token counts and LLM call log from the previous request's partial Turn row.
            prior_partial_turn = None
            if is_tool_continuation:
                prior_partial_turn = (
                    db.query(DBTurn)
                    .filter(
                        DBTurn.session_id == session_id,
                        DBTurn.final_response.is_(None),
                    )
                    .order_by(DBTurn.created_at.desc())
                    .first()
                )
                if prior_partial_turn:
                    actual_input_tokens += prior_partial_turn.total_tokens_in
                    actual_output_tokens += prior_partial_turn.total_tokens_out
                    prev_llm_log = json.loads(prior_partial_turn.llm_call_log or "[]")
                    turn_builder["llm_call_log"] = prev_llm_log
                    llm_calls = len(prev_llm_log)
                    log.info(
                        "[COMPAT/MCP] carrying forward partial turn  id=%s  tokens_in=%d  prior_calls=%d",
                        prior_partial_turn.id, prior_partial_turn.total_tokens_in, llm_calls,
                    )

            # ══════════════════════════════════════════════════════════════════
            if use_mcp_loop:
                # ── MCP LOOP ───────────────────────────────────────────────────
                # Bind internal tools + any external MCP tools from the request
                external_defs = [
                    {
                        "type": "function",
                        "function": {
                            "name": t.function.name,
                            "description": t.function.description or "",
                            "parameters": t.function.parameters or {"type": "object", "properties": {}},
                        },
                    }
                    for t in (body.tools or [])
                ]
                llm_bound = llm.bind_tools(internal_tools + external_defs)

                # Build the LangChain message list for this turn.
                # Always prepend current time so the LLM has temporal context.
                if is_tool_continuation:
                    # Include from the last user message onward:
                    # [user, assistant(tool_calls), tool_result, ...]
                    lc_messages = [_time_system_message()] + [_to_lc_message(m) for m in body.messages[last_user_idx:]]
                else:
                    # Fresh request — lazy-load, only send the latest user message
                    lc_messages = [_time_system_message(), HumanMessage(content=latest_content)]

                while True:
                    llm_calls += 1
                    log.info("[COMPAT/MCP] LLM call #%d", llm_calls)

                    response = await llm_bound.ainvoke(lc_messages)

                    usage = getattr(response, "usage_metadata", {}) or {}
                    actual_input_tokens += usage.get("input_tokens", 0)
                    actual_output_tokens += usage.get("output_tokens", 0)

                    response_tool_calls = getattr(response, "tool_calls", []) or []
                    turn_builder["llm_call_log"].append({
                        "call": llm_calls,
                        "tokens_in": usage.get("input_tokens", 0),
                        "tokens_out": usage.get("output_tokens", 0),
                        "tool_calls": [tc["name"] for tc in response_tool_calls],
                        "timestamp": _now_iso(),
                    })

                    if not response_tool_calls:
                        # Final text answer
                        full_response = _extract_text(response.content)
                        log.info("[COMPAT/MCP] call #%d → text  in=%d out=%d",
                                 llm_calls, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                        yield _oai_chunk(full_response, body.model, cid)
                        break

                    external_calls = [tc for tc in response_tool_calls if tc["name"] not in internal_tool_names]
                    internal_calls = [tc for tc in response_tool_calls if tc["name"] in internal_tool_names]

                    log.info("[COMPAT/MCP] call #%d → tool_calls  internal=%s  external=%s",
                             llm_calls,
                             [t["name"] for t in internal_calls],
                             [t["name"] for t in external_calls])

                    if external_calls:
                        # Persist partial Turn so the continuation can accumulate these tokens.
                        if prior_partial_turn:
                            # Multi-hop: update the existing in-progress Turn.
                            prior_partial_turn.llm_call_log = json.dumps(turn_builder["llm_call_log"])
                            prior_partial_turn.total_tokens_in = actual_input_tokens
                            prior_partial_turn.total_tokens_out = actual_output_tokens
                        else:
                            # First hop: create a new partial Turn (final_response=None).
                            db.add(DBTurn(
                                session_id=session_id,
                                user_message=latest_content,
                                final_response=None,
                                tool_executions="[]",
                                llm_call_log=json.dumps(turn_builder["llm_call_log"]),
                                total_tokens_in=actual_input_tokens,
                                total_tokens_out=actual_output_tokens,
                                estimated_saved_tokens=0,
                                mode="mcp",
                            ))
                        db.commit()

                        # Hand off to client — yield each SSE event as its own chunk
                        for tc in external_calls:
                            tools_called.append(f"[MCP]{tc['name']}")
                        for chunk in _oai_tool_calls_chunks(response, body.model, cid):
                            yield chunk
                        yield "data: [DONE]\n\n"
                        log.info("[COMPAT/MCP] handed off tool_calls to client")
                        return  # client executes MCP tools and sends a follow-up request

                    # Only internal calls — execute server-side and loop
                    lc_messages.append(response)
                    for tc in internal_calls:
                        tools_called.append(tc["name"])
                        try:
                            result = tools_map[tc["name"]].invoke(tc["args"])
                        except Exception as exc:
                            result = f"Error: {exc}"
                        log.info("[COMPAT/MCP] internal tool  name=%s", tc["name"])
                        turn_builder["tool_executions"].append({
                            "type": "internal",
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                            "result": str(result)[:5000],
                            "timestamp": _now_iso(),
                        })
                        lc_messages.append(ToolMessage(
                            tool_call_id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                            content=str(result),
                        ))

            # ══════════════════════════════════════════════════════════════════
            else:
                # ── STANDARD LANGRAPH FLOW ─────────────────────────────────────
                graph = build_graph(llm.bind_tools(internal_tools), internal_tools)
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                initial_state = {
                    "messages": [
                        {"role": "system", "content": f"Current date and time (UTC): {now_str}"},
                        {"role": "user", "content": latest_content},
                    ],
                    "last_response": None,
                }
                current_buffer = ""

                async for event in graph.astream_events(initial_state, version="v2"):
                    kind = event["event"]

                    if kind == "on_chat_model_start":
                        llm_calls += 1
                        current_buffer = ""
                        log.info("[COMPAT] LLM call #%d started", llm_calls)

                    elif kind == "on_chat_model_stream":
                        text = _extract_text(event["data"]["chunk"].content)
                        if text:
                            current_buffer += text
                            yield _oai_chunk(text, body.model, cid)

                    elif kind == "on_chat_model_end":
                        output = event["data"]["output"]
                        usage = getattr(output, "usage_metadata", None) or {}
                        actual_input_tokens += usage.get("input_tokens", 0)
                        actual_output_tokens += usage.get("output_tokens", 0)
                        has_tcs = bool(getattr(output, "tool_calls", None))
                        turn_builder["llm_call_log"].append({
                            "call": llm_calls,
                            "tokens_in": usage.get("input_tokens", 0),
                            "tokens_out": usage.get("output_tokens", 0),
                            "tool_calls": [tc["name"] for tc in (getattr(output, "tool_calls", None) or [])],
                            "timestamp": _now_iso(),
                        })
                        log.info("[COMPAT] LLM call #%d done  tool_calls=%s  in=%d  out=%d",
                                 llm_calls, has_tcs,
                                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                        if not has_tcs:
                            full_response = current_buffer

                    elif kind == "on_tool_start":
                        tools_called.append(event["name"])
                        log.info("[COMPAT] tool start  name=%s", event["name"])

                    elif kind == "on_tool_end":
                        log.info("[COMPAT] tool end  name=%s", event["name"])
                        tool_output = event.get("data", {}).get("output", "")
                        tool_input = event.get("data", {}).get("input", {})
                        turn_builder["tool_executions"].append({
                            "type": "internal",
                            "name": event["name"],
                            "args": tool_input if isinstance(tool_input, dict) else {},
                            "result": str(tool_output)[:5000],
                            "timestamp": _now_iso(),
                        })

            # ── 5. token stats ─────────────────────────────────────────────────
            if use_mcp_loop:
                # Saving = prior-turn history chars sliced away from body.messages.
                # Don't compare against actual_input_tokens — tool definitions inflate
                # actual but are unavoidable overhead present in both naive and lazy approaches.
                estimated_saved = estimated_baseline_tokens
                log.info(
                    "[COMPAT] ── token usage ──────────────────────────────────────\n"
                    "           actual input    : %6d tokens\n"
                    "           actual output   : %6d tokens\n"
                    "           actual total    : %6d tokens\n"
                    "           history not sent: %6d tokens  (prior turns kept client-side)\n"
                    "           llm calls       : %6d\n"
                    "           tools used      : %s\n"
                    "         ─────────────────────────────────────────────────────",
                    actual_input_tokens, actual_output_tokens,
                    actual_input_tokens + actual_output_tokens,
                    estimated_saved, llm_calls, tools_called,
                )
            else:
                estimated_saved = max(0, estimated_baseline_tokens - actual_input_tokens)
                log.info(
                    "[COMPAT] ── token usage ──────────────────────────────────────\n"
                    "           actual input    : %6d tokens\n"
                    "           actual output   : %6d tokens\n"
                    "           actual total    : %6d tokens\n"
                    "           est. baseline   : %6d tokens  (naive full-history approach)\n"
                    "           est. saved      : %6d tokens\n"
                    "           llm calls       : %6d\n"
                    "           tools used      : %s\n"
                    "         ─────────────────────────────────────────────────────",
                    actual_input_tokens, actual_output_tokens,
                    actual_input_tokens + actual_output_tokens,
                    estimated_baseline_tokens, estimated_saved,
                    llm_calls, tools_called,
                )

            # ── 6. persist to DB ───────────────────────────────────────────────
            if full_response:
                external_tool_names = [
                    ex["name"] for ex in turn_builder["tool_executions"]
                    if ex.get("type") == "external"
                ]
                if prior_partial_turn:
                    # Complete the in-progress Turn from a previous MCP request.
                    prior_partial_turn.final_response = full_response
                    prior_partial_turn.tool_executions = json.dumps(turn_builder["tool_executions"])
                    prior_partial_turn.llm_call_log = json.dumps(turn_builder["llm_call_log"])
                    prior_partial_turn.total_tokens_in = actual_input_tokens
                    prior_partial_turn.total_tokens_out = actual_output_tokens
                    prior_partial_turn.estimated_saved_tokens = estimated_saved
                else:
                    db.add(DBTurn(
                        session_id=session_id,
                        user_message=latest_content,
                        final_response=full_response,
                        tool_executions=json.dumps(turn_builder["tool_executions"]),
                        llm_call_log=json.dumps(turn_builder["llm_call_log"]),
                        total_tokens_in=actual_input_tokens,
                        total_tokens_out=actual_output_tokens,
                        estimated_saved_tokens=estimated_saved,
                        mode=turn_builder["mode"],
                    ))
                db.add(DBMessage(session_id=session_id, role="user", content=latest_content))
                db.add(DBMessage(session_id=session_id, role="assistant", content=full_response))
                vector_store.add_turn(session_id, latest_content, full_response, external_tool_names)
                db.commit()

            yield _oai_done(body.model, cid)

        except Exception:
            log.exception("[COMPAT] unhandled error  session=%s", session_id)
            raise
        finally:
            db.close()

    # ── non-streaming wrapper ──────────────────────────────────────────────────
    if not body.stream:
        full_text = ""
        is_tool_calls_response = False
        # Accumulate tool calls by index across multiple delta chunks
        tc_accumulator: dict[int, dict] = {}

        async for chunk in generate():
            if not chunk.startswith("data:"):
                continue
            payload = chunk[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
                choice = data["choices"][0]
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason")

                if finish == "tool_calls":
                    is_tool_calls_response = True
                elif delta.get("content"):
                    full_text += delta["content"]

                for tc_chunk in (delta.get("tool_calls") or []):
                    idx = tc_chunk.get("index", 0)
                    acc = tc_accumulator.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc_chunk.get("id"):
                        acc["id"] = tc_chunk["id"]
                    fn = tc_chunk.get("function", {})
                    if fn.get("name"):
                        acc["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        acc["function"]["arguments"] += fn["arguments"]
            except Exception:
                pass

        cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        if is_tool_calls_response:
            tool_calls_list = [tc_accumulator[i] for i in sorted(tc_accumulator)]
            return {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": body.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": tool_calls_list},
                    "finish_reason": "tool_calls",
                }],
            }
        return {
            "id": cid, "object": "chat.completion", "created": int(time.time()),
            "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        }

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

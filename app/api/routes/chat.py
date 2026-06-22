import json
import logging
from datetime import datetime, timezone
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.db.database import SessionLocal
from app.db import models
from app.db.models import Turn as DBTurn
from app.schemas.chat import ChatRequest
from app.core.providers import create_model
from app.core.tools import create_tools
from app.core.graph import build_graph
from app.core.vectorstore import vector_store

router = APIRouter(prefix="/chat", tags=["chat"])
log = logging.getLogger("chat.stream")


def _extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return content if isinstance(content, str) else ""


@router.post("/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        db = SessionLocal()
        try:
            session = db.query(models.Session).filter(models.Session.id == request.session_id).first()
            if not session:
                log.warning("[STREAM] session not found  id=%s", request.session_id)
                yield f"data: {json.dumps({'type': 'error', 'content': 'Session not found'})}\n\n"
                return

            try:
                model = create_model(request.provider)
            except Exception as e:
                log.error("[STREAM] provider init failed  error=%s", e)
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                return

            def get_history(limit: int = 100):
                rows = (
                    db.query(models.Message)
                    .filter(models.Message.session_id == request.session_id)
                    .order_by(models.Message.created_at.asc())
                    .limit(limit)
                    .all()
                )
                return [{"role": r.role, "content": r.content} for r in rows]

            def search_vectors_fn(query: str, k: int = 3) -> list:
                vector_store.backfill_if_needed(request.session_id, get_history(limit=10000))
                return vector_store.search(request.session_id, query, k)

            def search_all_vectors_fn(query: str, k: int = 5) -> list:
                return vector_store.search_all(query, k)

            def search_all_history_fn(query: str, limit: int = 5) -> list:
                rows = (
                    db.query(models.Message)
                    .filter(models.Message.content.ilike(f"%{query}%"))
                    .order_by(models.Message.created_at.desc())
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
                    .filter(DBTurn.session_id == request.session_id)
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

            tools = create_tools(
                get_history,
                search_vectors=search_vectors_fn,
                get_recent_turns=get_recent_turns_fn,
                search_all_vectors=search_all_vectors_fn,
                search_all_history=search_all_history_fn,
            )
            graph = build_graph(model.bind_tools(tools), tools)

            # Estimate baseline: tokens we would have sent if we passed full history directly
            prior_history = get_history(limit=10000)
            baseline_chars = sum(len(m["content"]) for m in prior_history) + len(request.message)
            estimated_baseline_tokens = baseline_chars // 4

            log.info(
                "[STREAM] request  session=%s  provider=%s/%s  message=%r",
                request.session_id, request.provider.type, request.provider.model,
                request.message[:80],
            )

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            initial_state = {
                "messages": [
                    {"role": "system", "content": f"Current date and time (UTC): {now_str}"},
                    {"role": "user", "content": request.message},
                ],
                "last_response": None,
            }

            current_buffer = ""
            full_response = ""
            llm_calls = 0
            tools_called = []
            actual_input_tokens = 0
            actual_output_tokens = 0
            turn_builder: dict = {
                "tool_executions": [],
                "llm_call_log": [],
            }

            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_start":
                    llm_calls += 1
                    current_buffer = ""
                    log.info("[STREAM] LLM call #%d started", llm_calls)

                elif kind == "on_chat_model_stream":
                    text = _extract_text(event["data"]["chunk"].content)
                    if text:
                        current_buffer += text
                        yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

                elif kind == "on_chat_model_end":
                    output = event["data"]["output"]
                    usage = getattr(output, "usage_metadata", None) or {}
                    actual_input_tokens += usage.get("input_tokens", 0)
                    actual_output_tokens += usage.get("output_tokens", 0)
                    has_tool_calls = bool(getattr(output, "tool_calls", None))
                    turn_builder["llm_call_log"].append({
                        "call": llm_calls,
                        "tokens_in": usage.get("input_tokens", 0),
                        "tokens_out": usage.get("output_tokens", 0),
                        "tool_calls": [tc["name"] for tc in (getattr(output, "tool_calls", None) or [])],
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    log.info(
                        "[STREAM] LLM call #%d done  tool_calls=%s  input=%d  output=%d",
                        llm_calls, has_tool_calls,
                        usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                    )
                    if not has_tool_calls:
                        full_response = current_buffer

                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    tools_called.append(tool_name)
                    log.info("[STREAM] tool start  name=%s", tool_name)
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    log.info("[STREAM] tool end  name=%s", tool_name)
                    tool_output = event.get("data", {}).get("output", "")
                    tool_input = event.get("data", {}).get("input", {})
                    turn_builder["tool_executions"].append({
                        "type": "internal",
                        "name": tool_name,
                        "args": tool_input if isinstance(tool_input, dict) else {},
                        "result": str(tool_output)[:5000],
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name})}\n\n"

            estimated_saved = max(0, estimated_baseline_tokens - actual_input_tokens)
            log.info(
                "[STREAM] ── token usage ──────────────────────────────────────\n"
                "           actual input    : %6d tokens\n"
                "           actual output   : %6d tokens\n"
                "           actual total    : %6d tokens\n"
                "           est. baseline   : %6d tokens  (full history approach)\n"
                "           est. saved      : %6d tokens\n"
                "           llm calls       : %6d\n"
                "           tools used      : %s\n"
                "         ─────────────────────────────────────────────────────",
                actual_input_tokens, actual_output_tokens,
                actual_input_tokens + actual_output_tokens,
                estimated_baseline_tokens, estimated_saved,
                llm_calls, tools_called,
            )

            if full_response:
                msg_count = db.query(models.Message).filter(models.Message.session_id == request.session_id).count()
                if msg_count == 0:
                    session.title = request.message[:100]

                db.add(DBTurn(
                    session_id=request.session_id,
                    user_message=request.message,
                    final_response=full_response,
                    tool_executions=json.dumps(turn_builder["tool_executions"]),
                    llm_call_log=json.dumps(turn_builder["llm_call_log"]),
                    total_tokens_in=actual_input_tokens,
                    total_tokens_out=actual_output_tokens,
                    estimated_saved_tokens=estimated_saved,
                    mode="standard",
                ))
                db.add(models.Message(session_id=request.session_id, role="user", content=request.message))
                db.add(models.Message(session_id=request.session_id, role="assistant", content=full_response))
                vector_store.add_turn(request.session_id, request.message, full_response, [])
                db.commit()

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            log.exception("[STREAM] unhandled error  session=%s", request.session_id)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            db.close()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

"""
Quick smoke test for the chat engine API.
Requires the server to be running: python run.py

Set your provider key in .env or as an env var before running:
    GEMINI_API_KEY=...   for Gemini
    OPENAI_API_KEY=...   for OpenAI-compatible
    ANTHROPIC_API_KEY=... for Anthropic
"""

import json
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"

# ── provider config ────────────────────────────────────────────────────────────
# Change type / model / key here to test a different provider
PROVIDER = {
    "type": "gemini",
    "api_key": os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY", ""),
    "model": "gemini-3.5-flash",
    # "type": "openai_compatible",
    # "api_key": os.getenv("OPENAI_API_KEY", ""),
    # "model": "gpt-4o-mini",
    # "base_url": "https://api.openai.com/v1",   # omit for default OpenAI
    # "type": "anthropic",
    # "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    # "model": "claude-haiku-4-5-20251001",
}

PASS = "✅"
FAIL = "❌"
INFO = "   "


def section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def check(label: str, ok: bool, detail: str = ""):
    icon = PASS if ok else FAIL
    line = f"{icon} {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    if not ok:
        sys.exit(1)


def stream_chat(session_id: str, message: str) -> str:
    """POST /chat/stream and collect all tokens into a final string."""
    print(f"\n{INFO}Sending: {message!r}")

    payload = {"session_id": session_id, "message": message, "provider": PROVIDER}
    full_text = ""
    tool_calls_seen = []

    with requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True, timeout=60) as r:
        check("HTTP 200 from /chat/stream", r.status_code == 200, str(r.status_code))

        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data:"):
                continue

            event = json.loads(line[len("data:"):].strip())
            kind = event.get("type")

            if kind == "token":
                full_text += event["content"]
                print(event["content"], end="", flush=True)
            elif kind == "tool_start":
                tool_calls_seen.append(event["tool"])
                print(f"\n{INFO}[tool call → {event['tool']}]")
            elif kind == "tool_end":
                print(f"{INFO}[tool done  ← {event['tool']}]")
            elif kind == "error":
                check("No SSE error", False, event.get("content", "unknown error"))
            elif kind == "done":
                print()  # newline after streamed tokens

    return full_text, tool_calls_seen


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_health():
    section("1. Health check")
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    check("Server is up", r.status_code == 200, r.text)


def test_create_session() -> str:
    section("2. Create session")
    r = requests.post(f"{BASE_URL}/sessions", json={"title": "Smoke test"}, timeout=5)
    check("POST /sessions → 201", r.status_code == 201)
    data = r.json()
    check("Response has id", "id" in data, str(data))
    session_id = data["id"]
    print(f"{INFO}session_id: {session_id}")
    return session_id


def test_list_sessions(session_id: str):
    section("3. List sessions")
    r = requests.get(f"{BASE_URL}/sessions", timeout=5)
    check("GET /sessions → 200", r.status_code == 200)
    ids = [s["id"] for s in r.json()]
    check("New session appears in list", session_id in ids)


def test_simple_chat(session_id: str):
    section("4. Simple question (no tools expected)")
    text, tools = stream_chat(session_id, "What is 10 + 15?")
    check("Got a response", bool(text.strip()), repr(text))
    check("No tool calls for simple math", len(tools) == 0, str(tools))


def test_history_chat(session_id: str):
    section("5. History question (LLM should call a tool)")
    text, tools = stream_chat(session_id, "What was my first question in this conversation?")
    check("Got a response", bool(text.strip()), repr(text))
    check("At least one tool was called", len(tools) > 0, str(tools))
    print(f"{INFO}Tools used: {tools}")


def test_db_history(session_id: str):
    section("6. Check messages saved to DB")
    r = requests.get(f"{BASE_URL}/sessions/{session_id}/history", timeout=5)
    check("GET /history → 200", r.status_code == 200)
    msgs = r.json()
    check("At least 4 messages saved (2 turns)", len(msgs) >= 4, f"got {len(msgs)}")
    roles = [m["role"] for m in msgs]
    is_alternating = all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    check("Alternating user/assistant roles", is_alternating, str(roles))
    for m in msgs:
        print(f"{INFO}[{m['role']:9}] {m['content'][:80]}")


def test_delete_session(session_id: str):
    section("7. Delete session")
    r = requests.delete(f"{BASE_URL}/sessions/{session_id}", timeout=5)
    check("DELETE /sessions/{id} → 204", r.status_code == 204)

    r2 = requests.get(f"{BASE_URL}/sessions/{session_id}", timeout=5)
    check("Session is gone (404)", r2.status_code == 404)

    r3 = requests.get(f"{BASE_URL}/sessions/{session_id}/history", timeout=5)
    check("History is gone (404)", r3.status_code == 404)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not PROVIDER["api_key"]:
        print(f"{FAIL} No API key found. Set GEMINI_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY) in your .env")
        sys.exit(1)

    print(f"\nProvider: {PROVIDER['type']} / {PROVIDER['model']}")

    test_health()
    session_id = test_create_session()
    test_list_sessions(session_id)
    test_simple_chat(session_id)
    test_history_chat(session_id)
    test_db_history(session_id)
    test_delete_session(session_id)

    print(f"\n{'═' * 60}")
    print(f"  {PASS} All tests passed")
    print(f"{'═' * 60}\n")

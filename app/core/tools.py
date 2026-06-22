import json
from typing import Callable, Optional
from langchain_core.tools import tool


def create_tools(
    get_history: Callable,
    search_vectors: Optional[Callable] = None,
    get_recent_turns: Optional[Callable] = None,
    search_all_vectors: Optional[Callable] = None,
    search_all_history: Optional[Callable] = None,
):
    """
    Returns tools closed over session-scoped callables.
      get_history(limit)          → list of {"role", "content"} dicts from DB (current session)
      search_vectors(q, k)        → FAISS semantic search (current session)
      get_recent_turns(limit)     → list of turn dicts with "tool_executions" JSON (current session)
      search_all_vectors(q, k)    → FAISS semantic search across ALL sessions
      search_all_history(q, lim)  → keyword DB search across ALL sessions
    """

    @tool
    def get_chat_history(limit: int = 20) -> str:
        """
        Retrieve the most recent messages from this conversation.
        Use this when you need to recall what was said earlier.
        Returns message content — NOT just counts. For counts use get_conversation_stats.
        Default limit is 20 — only lower it if you need just the last few messages.
        Do NOT call this tool more than once per response.

        Args:
            limit: Number of recent messages to retrieve (default 20).
        """
        messages = get_history(limit=limit)
        return json.dumps(messages, indent=2)

    @tool
    def search_messages(query: str, limit: int = 3) -> str:
        """
        Keyword search through conversation history.
        Returns empty list when nothing matches — fall back to get_chat_history if needed.
        For concept/topic searches, prefer semantic_search instead.

        Args:
            query: Keyword to search for in message content.
            limit: Maximum number of results to return.
        """
        all_messages = get_history(limit=1000)
        matches = [m for m in all_messages if query.lower() in m.get("content", "").lower()]
        return json.dumps(matches[:limit], indent=2)

    @tool
    def get_conversation_stats() -> str:
        """
        Get message counts for this conversation.
        Use only when the user asks about counts or statistics, not when they need actual message content.
        """
        messages = get_history(limit=10000)
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        return json.dumps({
            "total_messages": len(messages),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
        })

    tools = [get_chat_history, search_messages, get_conversation_stats]

    if search_vectors is not None:
        @tool
        def semantic_search(query: str, limit: int = 3) -> str:
            """
            Semantically search conversation history using vector similarity.
            Finds relevant messages even without exact keyword matches.
            Better than search_messages for concept/topic/meaning-based queries.
            Use when: the user references a topic, idea, or past discussion by concept.
            Always backfills the index automatically if needed.

            Args:
                query: Concept or topic to search for.
                limit: Number of most relevant results to return (default 3).
            """
            results = search_vectors(query, limit)
            if not results:
                return json.dumps([])
            return json.dumps(results, indent=2)

        tools.append(semantic_search)

    if search_all_vectors is not None:
        @tool
        def search_all_sessions(query: str, limit: int = 5) -> str:
            """
            Semantically search across ALL conversation sessions, not just the current one.
            Use when the user references something from a previous chat session, asks
            "have I asked about X before?", or wants to find past discussions on a topic.
            Each result is tagged with its session_id so you can identify the source.

            Args:
                query: Concept or topic to search for.
                limit: Number of most relevant results to return across all sessions (default 5).
            """
            results = search_all_vectors(query, limit)
            return json.dumps(results, indent=2)

        tools.append(search_all_sessions)

    if search_all_history is not None:
        @tool
        def search_history_all_sessions(query: str, limit: int = 5) -> str:
            """
            Keyword search through messages across ALL conversation sessions.
            Use for exact-phrase lookups or when the user asks if they mentioned
            something specific in any past chat. Results include session_id and timestamp.
            For concept/meaning searches across sessions, prefer search_all_sessions instead.

            Args:
                query: Keyword or phrase to search for.
                limit: Maximum number of results to return (default 5).
            """
            results = search_all_history(query, limit)
            return json.dumps(results, indent=2)

        tools.append(search_history_all_sessions)

    if get_recent_turns is not None:
        @tool
        def get_tool_result(tool_name: str, max_age_turns: int = 5) -> str:
            """
            Retrieve a cached result from an external tool that was already executed
            in a recent turn of this conversation.

            Use this BEFORE calling any external MCP tool (e.g. kite_mcp_get_holdings,
            kite_mcp_get_positions) to avoid redundant fetches when the same data was
            already retrieved recently. If this returns a result, use that instead of
            calling the external tool again.

            Args:
                tool_name: Exact name of the external tool (e.g. "kite_mcp_get_holdings").
                max_age_turns: How many recent turns to search (default 5).
            """
            turns = get_recent_turns(limit=max_age_turns)
            for turn in turns:  # ordered most-recent-first
                try:
                    executions = json.loads(turn.get("tool_executions", "[]"))
                except Exception:
                    continue
                for ex in executions:
                    if ex.get("type") == "external" and ex.get("name") == tool_name:
                        ts = turn.get("created_at", "unknown time")
                        return f"[Cached result from {ts}]\n{ex.get('result', '')}"
            return f"No cached result for '{tool_name}' in the last {max_age_turns} turns."

        tools.append(get_tool_result)

    return tools

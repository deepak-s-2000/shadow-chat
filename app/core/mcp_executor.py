"""
Execute a single MCP tool call over SSE transport.

Opens a fresh connection per call — acceptable for chatbot workloads.
The server URL can include auth query params (e.g. ?api_key=xxx).
"""

import logging
from mcp import ClientSession
from mcp.client.sse import sse_client

from app.core.mcp_config import bare_name

log = logging.getLogger("chat.mcp")


async def call_mcp_tool(server_url: str, chatbox_tool_name: str, args: dict) -> str:
    """
    Connect to `server_url` via SSE, call the tool, return its text output.
    `chatbox_tool_name` is in Chatbox format (mcp__server__tool); we strip the prefix.
    """
    tool = bare_name(chatbox_tool_name)
    log.info("[MCP] call  server=%s  tool=%s  args=%s", server_url, tool, args)

    try:
        async with sse_client(server_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool, args or {})

                parts: list[str] = []
                for item in result.content or []:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    else:
                        parts.append(str(item))
                output = "\n".join(parts) if parts else "Tool executed (no output)"
                log.info("[MCP] result  tool=%s  output=%r", tool, output[:200])
                return output

    except Exception as exc:
        log.error("[MCP] error  tool=%s  error=%s", tool, exc)
        return f"MCP error: {exc}"

import logging
import os

log = logging.getLogger("chat.mcp")


def get_configured_servers() -> dict[str, str]:
    """
    Return {server_name: url} from environment variables.

    Naming convention:
      MCP_{SERVER_NAME}_URL=<url>
      where SERVER_NAME is the Chatbox server label uppercased with dashes → underscores.

    Examples:
      MCP_KITE_MCP_URL=https://mcp.kite.trade/sse   → "kite-mcp"
      MCP_GITHUB_URL=https://mcp.github.com/sse     → "github"
    """
    servers: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith("MCP_") and key.endswith("_URL") and value.startswith("http"):
            name = key[4:-4].lower().replace("_", "-")
            servers[name] = value
    if servers:
        log.info("[MCP] configured servers: %s", list(servers.keys()))
    return servers


def extract_server_names(tool_names: list[str]) -> set[str]:
    """
    Pull MCP server names out of Chatbox-namespaced tool names.
    "mcp__kite-mcp__login"  →  "kite-mcp"
    """
    names: set[str] = set()
    for name in tool_names:
        if name.startswith("mcp__"):
            parts = name.split("__", 2)
            if len(parts) >= 2:
                names.add(parts[1])
    return names


def bare_name(chatbox_tool_name: str) -> str:
    """
    Strip the Chatbox namespace prefix.
    "mcp__kite-mcp__login"  →  "login"
    """
    if "__" in chatbox_tool_name:
        return chatbox_tool_name.rsplit("__", 1)[-1]
    return chatbox_tool_name

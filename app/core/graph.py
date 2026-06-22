from typing import Any, Optional
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


class ChatState(TypedDict):
    messages: list
    last_response: Optional[Any]


def build_graph(model_with_tools, tools):
    tool_map = {t.name: t for t in tools}

    async def llm_node(state: ChatState) -> ChatState:
        response = await model_with_tools.ainvoke(state["messages"])
        return {**state, "last_response": response}

    async def tool_node(state: ChatState) -> ChatState:
        response = state["last_response"]
        messages = list(state["messages"]) + [response]

        for call in response.tool_calls:
            fn = tool_map.get(call["name"])
            result = await fn.ainvoke(call["args"]) if fn else f"Tool '{call['name']}' not found."
            messages.append({
                "role": "tool",
                "name": call["name"],
                "content": result,
                "tool_call_id": call["id"],
            })

        return {**state, "messages": messages}

    def route(state: ChatState) -> str:
        r = state.get("last_response")
        return "tools" if r and getattr(r, "tool_calls", None) else "end"

    graph = StateGraph(ChatState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", route, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")

    return graph.compile()

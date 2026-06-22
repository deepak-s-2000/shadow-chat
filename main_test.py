# ============================================================================
# STEP 1: IMPORTS - Everything we need
# ============================================================================

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, List, Optional, Any
import json
from dotenv import load_dotenv
import os

# TypedDict is like a Python dataclass - defines the structure of our state
# It's like: class ChatState { messages: List, ... }
class ChatState(TypedDict):
    """
    This is the STATE - it travels through the entire graph.
    Think of it as a backpack that gets passed from node to node,
    each node adding/modifying data inside it.
    """
    messages: List[dict]           # All messages in current turn
    chat_history: List[dict]       # Full conversation history
    last_response: Optional[Any]   # Last LLM response (has tool_calls)
    tool_results: dict             # Results from executed tools


# ============================================================================
# STEP 2: INITIALIZE MODEL
# ============================================================================
load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=os.getenv("API_KEY"),
    temperature=0.7
)

# Storage for chat history (in production, use a database)
CONVERSATION_HISTORY = []


# ============================================================================
# STEP 3: DEFINE TOOLS using @tool decorator
# ============================================================================
# These are functions LLM can call. The @tool decorator automatically:
# - Extracts the function name as tool name
# - Extracts the docstring as tool description
# - Creates the schema from type hints

@tool
def get_chat_history(limit: int = 5) -> str:
    """
    Retrieve previous messages from conversation.
    
    Use this when you need context from earlier in the conversation.
    For example: "What did the user ask me 3 messages ago?"
    
    Args:
        limit: How many previous messages to get (default 5)
    
    Returns:
        JSON string of previous messages
    """
    print(f"🔍 Tool called: get_chat_history(limit={limit})")
    
    # Get last N messages from history
    if len(CONVERSATION_HISTORY) < limit:
        messages = CONVERSATION_HISTORY
    else:
        messages = CONVERSATION_HISTORY[-limit:]
    
    # Return as JSON string (tools must return strings)
    return json.dumps(messages, indent=2)


@tool
def search_messages(query: str, limit: int = 3) -> str:
    """
    Search for messages containing specific keywords.
    
    Use this when you need to find something specific the user mentioned.
    For example: "Find messages about Python programming"
    
    Args:
        query: Search term
        limit: Maximum results to return
    
    Returns:
        JSON string of matching messages
    """
    print(f"🔍 Tool called: search_messages(query='{query}', limit={limit})")
    
    results = []
    for msg in CONVERSATION_HISTORY:
        if query.lower() in msg.get("content", "").lower():
            results.append(msg)
            if len(results) >= limit:
                break
    
    return json.dumps(results, indent=2)


@tool
def get_conversation_stats() -> str:
    """
    Get statistics about the conversation.
    
    Returns total messages, user messages, and assistant messages.
    Useful when user asks "How many times have we talked?"
    
    Returns:
        JSON string with conversation statistics
    """
    print("🔍 Tool called: get_conversation_stats()")
    
    user_msgs = sum(1 for msg in CONVERSATION_HISTORY if msg.get("role") == "user")
    assistant_msgs = sum(1 for msg in CONVERSATION_HISTORY if msg.get("role") == "assistant")
    
    stats = {
        "total_messages": len(CONVERSATION_HISTORY),
        "user_messages": user_msgs,
        "assistant_messages": assistant_msgs
    }
    
    return json.dumps(stats)


# List of all available tools
tools = [get_chat_history, search_messages, get_conversation_stats]

# Bind tools to the model
# This tells the model: "You can call these functions"
model_with_tools = model.bind_tools(tools)


# ============================================================================
# STEP 4: DEFINE NODES - The actual logic
# ============================================================================
# Nodes are functions that process the state.
# Each node:
# 1. Takes in the current state
# 2. Does something with it
# 3. Returns the modified state

def llm_node(state: ChatState) -> ChatState:
    """
    NODE 1: Call the LLM
    
    This node:
    - Takes only the CURRENT user message (not history) to save tokens
    - Calls the LLM with access to tools
    - Stores the response in state for the next node to process
    
    The LLM can decide:
    - "I need to call get_chat_history to answer this"
    - "I can answer directly without tools"
    """
    
    print("\n" + "="*60)
    print("📨 LLM_NODE: Processing user message")
    print("="*60)
    
    # Get the current messages
    messages = state["messages"]
    print(f"📥 Input: {len(messages)} message(s)")
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))
        else:
            role = getattr(msg, 'type', 'unknown')
            content = str(msg.content) if msg.content else ''
        print(f"   - {role}: {content[:50]}...")
    
    # Call the LLM with tools bound
    # The model can now "see" the tools and decide to call them
    response = model_with_tools.invoke(messages)
    
    print(f"\n📤 LLM Response:")
    print(f"   - Type: {type(response).__name__}")
    
    # Check if LLM wants to call tools
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"   - Tool calls: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"     • {tool_call['name']} with args {tool_call['args']}")
    else:
        print(f"   - Direct response (no tools)")
        print(f"   - Content: {str(response.content)[:100]}...")
    
    # Store the response in state for next node
    state["last_response"] = response
    
    return state


def tool_node(state: ChatState) -> ChatState:
    """
    NODE 2: Execute tools that LLM requested
    
    This node:
    - Checks if LLM called any tools
    - Executes them
    - Adds the results back to messages
    - Returns state (which goes back to llm_node for another round)
    """
    
    print("\n" + "="*60)
    print("🔧 TOOL_NODE: Executing requested tools")
    print("="*60)
    
    response = state.get("last_response")
    
    # Check if there are tool calls to execute
    if not hasattr(response, 'tool_calls') or not response.tool_calls:
        print("❌ No tool calls to execute")
        return state
    
    print(f"✅ Found {len(response.tool_calls)} tool call(s)")

    # Add the AI message (with tool calls) to messages so the LLM has context
    state["messages"].append(response)

    # Execute each tool
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]  # Required to link result back to the call

        print(f"\n🔨 Executing: {tool_name}")
        print(f"   Args: {tool_args}")

        # Find and execute the tool
        tool_function = None
        for tool in tools:
            if tool.name == tool_name:
                tool_function = tool
                break

        if tool_function:
            # Call the tool with the provided arguments
            result = tool_function.invoke(tool_args)
            print(f"   Result: {result[:100]}...")

            # Add tool result to messages with tool_call_id
            state["messages"].append({
                "role": "tool",
                "name": tool_name,
                "content": result,
                "tool_call_id": tool_call_id
            })
        else:
            print(f"   ❌ Tool not found!")
    
    return state


# ============================================================================
# STEP 5: CONDITIONAL EDGE - Decide what to do next
# ============================================================================
# This function decides: "Should we call tools or end?"

def should_call_tools(state: ChatState) -> str:
    """
    ROUTER FUNCTION: Decides which path to take
    
    This is called after LLM runs.
    It checks: "Did LLM request tools?"
    
    If YES  → Go to "tools" node
    If NO   → Go to "end" (END the graph)
    """
    
    response = state.get("last_response")
    
    # Check if response has tool calls
    has_tools = (
        hasattr(response, 'tool_calls') and 
        response.tool_calls and 
        len(response.tool_calls) > 0
    )
    
    if has_tools:
        print(f"\n🤔 Decision: LLM wants tools → Go to TOOL_NODE")
        return "tools"
    else:
        print(f"\n🤔 Decision: LLM finished → END")
        return "end"


# ============================================================================
# STEP 6: BUILD THE GRAPH
# ============================================================================
# This is where we connect everything together

graph = StateGraph(ChatState)

# Add our two nodes to the graph
graph.add_node("llm", llm_node)       # Node 1: Call LLM
graph.add_node("tools", tool_node)    # Node 2: Execute tools

# Set the starting point - where the graph begins
graph.set_entry_point("llm")

# Add conditional edge from LLM node
# "After llm runs, check should_call_tools function"
# If it returns "tools", go to tools node
# If it returns "end", go to END (exit graph)
graph.add_conditional_edges(
    "llm",                    # From this node
    should_call_tools,        # Use this function to decide
    {
        "tools": "tools",     # If function returns "tools", go here
        "end": END            # If function returns "end", exit
    }
)

# After tools execute, go back to LLM for another round
# This creates the loop: LLM → Tools → LLM → Tools → ...
graph.add_edge("tools", "llm")

# Compile the graph - convert it into a runnable object
compiled_graph = graph.compile()

# Print the graph structure (useful for debugging)
print("📊 Graph structure:")
print(compiled_graph.get_graph().draw_ascii())


# ============================================================================
# STEP 7: CHAT FUNCTION - User-facing interface
# ============================================================================

def chat(user_message: str) -> str:
    """
    This is what the user calls.
    
    Flow:
    1. Create initial state with user message
    2. Run the compiled graph
    3. Get the final response
    4. Store in history
    5. Return to user
    """
    
    print("\n" + "🟢"*30)
    print(f"USER: {user_message}")
    print("🟢"*30)
    
    # Step 1: Create initial state
    # Only the CURRENT message - not history (saves tokens!)
    initial_state = ChatState(
        messages=[{"role": "user", "content": user_message}],
        chat_history=CONVERSATION_HISTORY.copy(),
        last_response=None,
        tool_results={}
    )
    
    # Step 2: Run the graph
    # This executes: llm_node → (conditional) → tool_node → llm_node → ...
    # Until the graph decides to END
    final_state = compiled_graph.invoke(initial_state)
    
    # Step 3: Extract the final response from LLM
    response = final_state["last_response"]
    
    # The response could be a message object with content
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, list):
            response_text = ' '.join(
                block.get('text', '') if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            response_text = str(content)
    else:
        response_text = str(response)
    
    print(f"\n🤖 ASSISTANT: {response_text}")
    
    # Step 4: Store in conversation history for future tool calls
    CONVERSATION_HISTORY.append({
        "role": "user",
        "content": user_message
    })
    CONVERSATION_HISTORY.append({
        "role": "assistant",
        "content": response_text
    })
    
    return response_text


# ============================================================================
# STEP 8: TEST IT
# ============================================================================

if __name__ == "__main__":
    # Test 1: Simple question
    print("\n\n" + "█"*70)
    print("TEST 1: Simple question (no tools needed)")
    print("█"*70)
    chat("What is 2 + 2?")
    
    # Test 2: Question that needs history
    print("\n\n" + "█"*70)
    print("TEST 2: Question requiring history (LLM will call tool)")
    print("█"*70)
    chat("What was my first question?")
    
    # Test 3: Search in history
    print("\n\n" + "█"*70)
    print("TEST 3: Search for specific topic")
    print("█"*70)
    chat("Find any messages where I asked about math")
    
    # Test 4: Stats
    print("\n\n" + "█"*70)
    print("TEST 4: Conversation statistics")
    print("█"*70)
    chat("How many messages have we exchanged?")
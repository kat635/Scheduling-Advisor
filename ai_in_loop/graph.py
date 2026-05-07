"""LangGraph application with tool calling support.

This module defines the graph structure for processing prompts with
optional tool calling. The graph uses LangGraph's MessagesState and
prebuilt ToolNode for automatic tool execution.

Graph structure:
    START → agent → [tools_condition] → tools → agent (loop) → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

from .config import Config
from .llm import get_llm, load_system_prompt, get_text
from .tools import python_calc, search_docs, set_search_config, create_schedule, major_url_lookup, program_scraper, get_gen_ed_data, get_gen_ed_summary
from .nodes import scrape_node, handle_errors
from .state import ScheduleState


# List of available tools
TOOLS = [python_calc, search_docs, create_schedule, major_url_lookup, program_scraper, get_gen_ed_data, get_gen_ed_summary]


def build_app(cfg: Config) -> CompiledStateGraph:
    """Build and compile the LangGraph application with tool support.

    Creates a graph that can process prompts and optionally call tools.
    The graph structure is:

        START → agent → [tools_condition] → tools → agent (loop) → END

    When the LLM decides to use a tool, the graph routes to the tools node,
    executes the tool, and returns the result to the agent for further
    processing.

    Args:
        cfg: Configuration object with LLM settings

    Returns:
        Compiled LangGraph application ready for invocation
    """
    # Initialize search config for the search_docs tool
    set_search_config(cfg)

    llm = get_llm(cfg)
    system_prompt = load_system_prompt(cfg.system_prompt_file)

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent(state: MessagesState) -> dict:
        """Process messages and generate a response using the LLM.

        The agent node invokes the LLM with the current message history.
        If a system prompt is configured, it's prepended to the messages.
        """
        messages = list(state["messages"])

        # Prepend system prompt if configured
        if system_prompt:
            messages = [SystemMessage(content=system_prompt)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Build the graph
    graph = StateGraph(MessagesState)

    # Add nodes
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()

def build_scraper_graph():
    """
    Build a graph containing a workflow to scrape a website 
    """

    #Initialize the graph
    builder = StateGraph(ScheduleState)

    #Add worker nodes
    builder.add_node("scrape", scrape_node)
    builder.add_node("handle_errors", handle_errors)

    #Add conditional routing logic
    builder.add_conditional_edges(
        "scrape",
        route_after_error, {
            "handle_errors" : "handle_errors",
            END:END
        }
    )

    builder.add_edge(START, "scrape")
    builder.add_edge("scrape", END)
    
    return builder.compile()

def route_after_error (state: ScheduleState) -> ScheduleState:
    """
    Conditional node: Determines where we go after scraping
    """
    if(state.get("error")):
        return "handle_errors"
    return END

def run_once(prompt: str, cfg: Config) -> str:
    """Run a single prompt through the graph and return the response.

    This is the main entry point for processing prompts. It builds a new
    graph instance, invokes it with the prompt, and extracts the final
    response text.

    Args:
        prompt: The user's input prompt
        cfg: Configuration object with LLM settings

    Returns:
        The generated text response
    """
    app = build_app(cfg)
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})

    # Extract the final AI message from the conversation
    messages = result["messages"]
    # Find the last AI message (skip tool messages)
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            # Check if it's an AI message (not a tool message)
            if not hasattr(msg, "tool_call_id"):
                return get_text(msg)

    # Fallback: return the last message content
    if messages:
        return get_text(messages[-1])
    return ""

from __future__ import annotations

import typer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console

from .config import Config
from .graph import build_app
from .llm import get_text
from .logging_utils import log_event, new_run_id


console = Console()
app = typer.Typer(help="Course starter CLI")


def _format_tool_calls(tool_calls: list) -> str:
    """Format tool calls for display."""
    parts = []
    for tc in tool_calls:
        args_str = ", ".join(f'{k}="{v}"' for k, v in tc["args"].items())
        parts.append(f'{tc["name"]}({args_str})')
    return ", ".join(parts)


@app.command()
def demo(prompt: str = "Say hello in 1 sentence.") -> None:
    """Run a single prompt through the starter graph."""
    load_dotenv()
    cfg = Config.from_env()

    run_id = new_run_id()

    # Run graph and get full message list
    graph_app = build_app(cfg)
    result = graph_app.invoke({"messages": [HumanMessage(content=prompt)]})
    messages = result["messages"]

    # Display messages following the pattern from slides
    tool_calls_logged = []
    tool_results_logged = []
    final_response = ""

    for msg in messages:
        if msg.type == "human":
            continue
        elif msg.type == "ai":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                formatted = _format_tool_calls(msg.tool_calls)
                console.print(f"[dim]Tool call: {formatted}[/dim]")
                tool_calls_logged.extend(msg.tool_calls)
            content = get_text(msg)
            if content:
                final_response = content
        elif msg.type == "tool":
            console.print(f"[dim]Tool result: {msg.content}[/dim]")
            tool_results_logged.append(msg.content)

    if final_response:
        console.print(final_response)

    log_event(
        {
            "run_id": run_id,
            "event": "demo",
            "prompt": prompt,
            "response": final_response,
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]}
                for tc in tool_calls_logged
            ],
            "tool_results": tool_results_logged,
            "use_gemini": cfg.use_gemini,
            "gemini_model": cfg.gemini_model,
            "temperature": cfg.temperature,
        }
    )


def _read_multiline_input() -> str:
    """Read multi-line input until a blank line is entered."""
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


@app.command()
def chat() -> None:
    """Interactive chat loop with conversation history."""
    load_dotenv()
    cfg = Config.from_env()

    console.print("[bold]Chat mode[/bold]")
    console.print("  - Enter a blank line to send your message")
    console.print("  - Type 'exit' to quit\n")

    # Build graph ONCE before the loop
    graph_app = build_app(cfg)

    # Maintain conversation history
    conversation_messages: list = []

    while True:
        console.print("[bold cyan]You>[/bold cyan]")
        prompt = _read_multiline_input().strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue

        run_id = new_run_id()

        # Track message count before invoke
        num_messages_before = len(conversation_messages)

        # Add new message and pass FULL history
        conversation_messages.append(HumanMessage(content=prompt))
        result = graph_app.invoke({"messages": conversation_messages})

        # Update history with result
        conversation_messages = result["messages"]

        # Display only NEW messages (after the human message we just added)
        console.print()
        tool_calls_logged = []
        tool_results_logged = []
        final_response = ""

        # Skip messages we've already seen (everything before this turn)
        new_messages = result["messages"][num_messages_before + 1:]  # +1 to skip the human message
        for msg in new_messages:
            if msg.type == "ai":
                # Check for tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    formatted = _format_tool_calls(msg.tool_calls)
                    console.print(f"[dim]Tool call: {formatted}[/dim]")
                    tool_calls_logged.extend(msg.tool_calls)
                # Check for final response content
                content = get_text(msg)
                if content:
                    final_response = content
            elif msg.type == "tool":
                console.print(f"[dim]Tool result: {msg.content}[/dim]")
                tool_results_logged.append(msg.content)

        # Show final response
        if final_response:
            console.print(f"[bold green]Model>[/bold green] {final_response}\n")
        elif tool_calls_logged:
            # Model called tools but didn't generate a response
            console.print("[yellow]Model searched but didn't generate a response. Try rephrasing your question.[/yellow]\n")
        else:
            # Model didn't respond at all
            console.print("[yellow]Model didn't respond. Try rephrasing your question.[/yellow]\n")

        # Log the interaction
        log_event(
            {
                "run_id": run_id,
                "event": "chat_turn",
                "prompt": prompt,
                "response": final_response,
                "tool_calls": [
                    {"name": tc["name"], "args": tc["args"]}
                    for tc in tool_calls_logged
                ],
                "tool_results": tool_results_logged,
                "use_gemini": cfg.use_gemini,
                "gemini_model": cfg.gemini_model,
                "temperature": cfg.temperature,
            }
        )


if __name__ == "__main__":
    app()

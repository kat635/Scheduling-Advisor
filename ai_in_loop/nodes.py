"""Node functions for web scraping workflow

Each function is a node in the LangGraph workflow. Nodes read from and
write to the ScheduleState object.

Workflow:
    START -> parse_input 
          -> [route_after_error]
               -> handle_errors -> END
          -> END
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

from .state import ScheduleState
from .tools import (
    scrape_tool
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


# LLM instance set by graph.py
_llm: BaseChatModel | None = None


def set_llm(llm: BaseChatModel) -> None:
    """Set the LLM instance for nodes that need it."""
    global _llm
    _llm = llm

def scrape_node (state: ScheduleState) -> ScheduleState:
    """
    Extracts data from the provided URL. 
    If successful, updates 'result'. If it fails, updates 'error'.
    """
    url = state.get("url")
    if not url:
        return {"error": "No URL provided"}

    data = scrape_tool(url)
    
    if "error" in data:
        return {"error": data["error"]}

    return {"result": data}

def handle_errors (state: ScheduleState) -> ScheduleState:
    print(f"ERROR DETECTED: {state.get('error')} !!!")
    return {"error": f"Processed Error: {state.get('error')}"}

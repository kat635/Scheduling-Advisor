"""
Streamlit app for Dr. Cusack - Scheduling Advisor 
Interactive chat interface for users to inquire about courses offered at Hope College.
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from ai_in_loop.config import Config
from ai_in_loop.graph import build_app
from ai_in_loop.llm import get_text
from ai_in_loop.logging_utils import log_event, new_run_id

from loaders.program_loader import load_program_data

# cache the program data (since it doesn't change often) to prevent unnecessary reloads
@st.cache_data
def load_cached_program_data():
    return load_program_data()

# format the tool calls nicely so that it's readable for the user
def format_tool_calls(tool_calls: list) -> str:
    parts = []
    for tc in tool_calls:
        args_str = ", ".join(f'{k}="{v}"' for k, v in tc["args"].items())
        parts.append(f'{tc["name"]}({args_str})')
    return ", ".join(parts)


@st.cache_resource
def get_graph_app():
    load_dotenv()
    cfg = Config.from_env()
    graph_app = build_app(cfg)
    return cfg, graph_app

# initialize the state
def init_state():
    if "conversation_messages" not in st.session_state:
        st.session_state.conversation_messages = []

    if "ui_messages" not in st.session_state:
        # This is only for display in Streamlit
        st.session_state.ui_messages = []


def render_chat_history():
    for msg in st.session_state.ui_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # show information about tool calls if the tool was used
            if msg.get("tool_calls"):
                with st.expander("Tool calls"):
                    for tc in msg["tool_calls"]:
                        st.code(tc)

            # if the tool was used, show the results that were returned
            if msg.get("tool_results"):
                with st.expander("Tool results"):
                    for tr in msg["tool_results"]:
                        st.write(tr)


def main():
    # sets the title and icon in the page tab, this can be changed later if wanted
    st.set_page_config(page_title="Scheduling Advisor", page_icon="💬", layout="wide")
    # sets title of the page itself, also can be changed if wanted
    st.title("Scheduling Advisor")

    init_state()

    cfg, graph_app = get_graph_app()

    with st.sidebar:
        st.subheader("Settings")
        st.write(f"Use Gemini: `{cfg.use_gemini}`")
        st.write(f"Model: `{cfg.gemini_model}`")
        st.write(f"Temperature: `{cfg.temperature}`")

        if st.button("Clear chat"):
            st.session_state.conversation_messages = []
            st.session_state.ui_messages = []
            st.rerun()

    render_chat_history()

    # load program data once and cache it 
    program_data = load_cached_program_data()

    user_prompt = st.chat_input("Type your message...")

    if user_prompt:
        # show user message immediately
        st.session_state.ui_messages.append(
            {"role": "user", "content": user_prompt}
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)

        run_id = new_run_id()

        # remember how many messages were before this message
        num_messages_before = len(st.session_state.conversation_messages)

        # append human message to conversation history
        st.session_state.conversation_messages.append(
            HumanMessage(content=user_prompt)
        )

        # ui shown that the chatbot is thinking and didn't just hang or something
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = graph_app.invoke(
                    {"messages": st.session_state.conversation_messages},
                    {"program_data": program_data}
                )

            # overwrite full conversation with returned message history
            st.session_state.conversation_messages = result["messages"]

            tool_calls_logged = []
            tool_results_logged = []
            final_response = ""

            # only inspect new messages after the user's message
            new_messages = result["messages"][num_messages_before + 1:]
            
            for msg in new_messages:
                # message was from the ai, check if tool was called, if yes,
                # format the call nicely and add it to the list of logged tool calls
                if msg.type == "ai":
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        formatted = format_tool_calls(msg.tool_calls)
                        tool_calls_logged.append(formatted)

                    content = get_text(msg)
                    if content:
                        final_response = content

                elif msg.type == "tool":
                    tool_results_logged.append(msg.content)

            if final_response:
                st.markdown(final_response)
            elif tool_calls_logged:
                final_response = ""
                st.warning("Model searched but didn't generate a response.")
            else:
                final_response = ""
                st.warning("Model didn't respond.")

            if tool_calls_logged:
                with st.expander("Tool calls"):
                    for tc in tool_calls_logged:
                        st.code(tc)

            if tool_results_logged:
                with st.expander("Tool results"):
                    for tr in tool_results_logged:
                        if tr.startswith("http"):
                            st.link_button("Open Catalog", tr)
                        else:
                            st.write(tr)

        # save assistant turn for future rerenders
        st.session_state.ui_messages.append(
            {
                "role": "assistant",
                "content": final_response if final_response else "_No final response generated._",
                "tool_calls": tool_calls_logged,
                "tool_results": tool_results_logged,
            }
        )

        log_event(
            {
                "run_id": run_id,
                "event": "chat_turn_streamlit",
                "prompt": user_prompt,
                "response": final_response,
                "tool_calls": tool_calls_logged,
                "tool_results": tool_results_logged,
                "use_gemini": cfg.use_gemini,
                "gemini_model": cfg.gemini_model,
                "temperature": cfg.temperature,
            }
        )


if __name__ == "__main__":
    main()
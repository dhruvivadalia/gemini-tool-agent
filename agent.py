"""LangChain tool-calling agent powered by Gemini (gemini-2.0-flash)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import TOOLS

load_dotenv()

MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT = (
    "You are a helpful conversational assistant. "
    "Use the available tools when a user's request needs them:\n"
    "- Use `calculator` for arithmetic expressions and numerical calculations.\n"
    "- Use `word_count` to count characters, words, and lines in text.\n"
    "- Use `current_time` when asked for the current date or time in Asia/Kolkata (IST).\n"
    "If the request can be answered directly without those tools, respond directly "
    "without calling any tool. If a tool returns an error, explain the issue clearly "
    "or retry with corrected input."
)


def _format_tool_input(tool_input: Any) -> str:
    """Normalize tool input into a readable string representation."""
    if isinstance(tool_input, str):
        return tool_input
    if isinstance(tool_input, dict):
        return json.dumps(tool_input, ensure_ascii=False)
    return str(tool_input)


def _extract_answer_text(output: Any) -> str:
    """Normalize AgentExecutor output into a plain string."""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(output)


@lru_cache(maxsize=1)
def get_agent_executor() -> AgentExecutor:
    """Build and cache the Gemini tool-calling AgentExecutor."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is missing. "
            "Set it in your environment or .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=10,
        verbose=False,
    )


def run_agent(
    message: str,
    chat_history: Sequence[BaseMessage] | None = None,
) -> dict[str, Any]:
    """Run the agent for one turn and return the final answer and tool call trace."""
    executor = get_agent_executor()
    result = executor.invoke(
        {
            "input": message,
            "chat_history": list(chat_history) if chat_history else [],
        }
    )

    tool_calls = [
        {
            "tool": action.tool,
            "input": _format_tool_input(action.tool_input),
        }
        for action, _observation in result.get("intermediate_steps", [])
    ]

    return {
        "answer": _extract_answer_text(result.get("output", "")),
        "tool_calls": tool_calls,
    }

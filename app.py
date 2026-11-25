"""FastAPI application exposing the Gemini conversational tool-calling agent."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field, field_validator

from agent import MODEL_NAME, run_agent

app = FastAPI(
    title="Gemini Conversational Tool Agent",
    description="FastAPI service exposing a Gemini tool-calling agent with per-session memory.",
    version="1.0.0",
)

# In-memory session store keyed by session_id.
sessions: dict[str, ChatMessageHistory] = {}
_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message for the agent.")
    session_id: str = Field(..., min_length=1, description="Unique conversation session ID.")

    @field_validator("message", "session_id")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field must not be blank.")
        return stripped


class ToolCallTrace(BaseModel):
    tool: str
    input: str


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallTrace]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness endpoint."""
    return HealthResponse(status="ok", model=MODEL_NAME)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a user turn within a session and return the answer plus tool call trace."""
    start = time.perf_counter()
    lock = _session_locks[request.session_id]

    async with lock:
        history = sessions.setdefault(request.session_id, ChatMessageHistory())

        try:
            result = await run_in_threadpool(run_agent, request.message, history.messages)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc

        history.add_user_message(request.message)
        history.add_ai_message(result["answer"])

    latency_ms = round((time.perf_counter() - start) * 1000)
    return ChatResponse(
        answer=result["answer"],
        tool_calls=[ToolCallTrace(**tc) for tc in result["tool_calls"]],
        latency_ms=latency_ms,
    )

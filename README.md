# Gemini Conversational Tool-Calling Agent

FastAPI service exposing a LangChain tool-calling agent (`create_tool_calling_agent` + `AgentExecutor`) powered by Google Gemini (`gemini-2.0-flash`) with per-session in-memory chat history and three real tools:

- `calculator` — safely evaluates basic arithmetic expressions using Python's `ast` module.
- `word_count` — returns character, word, and line counts for input text.
- `current_time` — returns the current date and time in `Asia/Kolkata` (IST).

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS / Linux:
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure your Google Gemini API key:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `GOOGLE_API_KEY` to your Gemini API key.

4. Start the FastAPI server:

   ```bash
   uvicorn app:app --reload
   ```

## Endpoints

- `GET /health` — health check returning `{"status": "ok", "model": "gemini-2.0-flash"}`
- `POST /chat` — accepts `{"message": "...", "session_id": "..."}` and returns `{"answer": "...", "tool_calls": [{"tool": "...", "input": "..."}], "latency_ms": 123}`

## Example `curl` Commands

### 1. Trigger `calculator`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What is (145 * 18) / 5 + 42?"}'
```

### 2. Trigger `word_count`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "Count the characters, words, and lines in:\nFastAPI is fast.\nGemini picks tools."}'
```

### 3. Trigger `current_time`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "message": "What is the current date and time in Asia/Kolkata?"}'
```

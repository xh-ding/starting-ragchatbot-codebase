# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Always use `uv` to manage dependencies, run the server, and run Python files — never `pip` or `python` directly.

```bash
uv run python script.py   # run any Python file
uv add <package>          # add a dependency
uv sync                   # install all dependencies
```

```bash
# Install dependencies
uv sync

# Start the server (from project root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

Requires a `.env` file in the project root with `ANTHROPIC_API_KEY=...`.

The server runs at `http://localhost:8000`. On startup it auto-indexes all `.txt`/`.pdf`/`.docx` files from `docs/` into ChromaDB, skipping already-indexed courses.

## Architecture

This is a RAG chatbot. The backend is a FastAPI app (`backend/`) that serves the frontend (`frontend/`) as static files and exposes two API endpoints: `POST /api/query` and `GET /api/courses`.

**Query flow:**
1. Frontend POSTs `{query, session_id}` to `/api/query`
2. `app.py` delegates to `RAGSystem.query()` (`rag_system.py`)
3. `RAGSystem` fetches session history, then calls `AIGenerator.generate_response()` with tool definitions
4. Claude makes a **first API call** — if it decides to search (`stop_reason == "tool_use"`), `CourseSearchTool.execute()` queries ChromaDB for top-5 semantically similar chunks
5. Claude makes a **second API call** with the search results (no tools this time) and synthesizes a final answer
6. Sources, response, and updated session history are returned to the frontend

**Document ingestion flow:**
- `DocumentProcessor.process_course_document()` parses `.txt` files expecting a header format (`Course Title:`, `Course Link:`, `Course Instructor:`) followed by `Lesson N: <title>` markers
- Text is split sentence-by-sentence into ~800-char chunks with 100-char overlap
- Two ChromaDB collections: `course_catalog` (course metadata for fuzzy name resolution) and `course_content` (text chunks for semantic search)
- Both use `all-MiniLM-L6-v2` embeddings via `sentence-transformers`

## Key Configuration (`backend/config.py`)

All tuneable parameters live in the `Config` dataclass — no need to hunt through code:

| Setting | Default | Purpose |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model used for generation |
| `CHUNK_SIZE` | `800` | Max chars per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `MAX_RESULTS` | `5` | ChromaDB results returned per search |
| `MAX_HISTORY` | `2` | Conversation turns kept per session |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory (relative to `backend/`) |

## Important Constraints

- **One search per query**: `AIGenerator` system prompt instructs Claude to make at most one `search_course_content` tool call per user query. The second Claude API call is made without tools to prevent loops.
- **Session history is in-memory only**: `SessionManager` stores sessions in a plain dict — all history is lost on server restart.
- **Course deduplication on startup**: `add_course_folder()` checks `course_catalog` IDs before indexing; re-running the server won't re-embed already-loaded courses.
- **ChromaDB is persistent**: the `chroma_db/` directory (inside `backend/`) stores embeddings across restarts. Delete it to force a full re-index.

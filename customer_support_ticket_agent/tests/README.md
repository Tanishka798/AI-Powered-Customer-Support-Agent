# Tests

Run all of them from the project root with `pytest -q`, or one component with, for example, `pytest -q tests/workflow`. No LLM endpoint or embedding model download is needed.

| Folder | Component under test |
|---|---|
| `rag/` | Document loading and the Chroma knowledge retriever |
| `sessions/` | Conversation state and the session store |
| `tickets/` | Ticket repository and the ticket tool |
| `workflow/` | Agent logic in `src/llm/`: output parsing, decisions, ticket collection, and the LangGraph routes |
| `pipeline/` | `SupportPipeline` turn handling |
| `api/` | FastAPI endpoints |
| `ui/` | HTTP client and per-message audio cache used by the Streamlit app |
| `voice/` | Voice pipeline, voice models, and the Whisper and Edge TTS adapters |

Each folder has its own README describing its test files.

## Shared files

| File | Purpose |
|---|---|
| `conftest.py` | `knowledge_dir` fixture pointing at `knowledge_base/`. |
| `fakes.py` | Deterministic test doubles: scripted chat model, fake retriever, keyword embeddings, fake STT and TTS adapters, test settings, and a harness that drives the workflow turn by turn. |

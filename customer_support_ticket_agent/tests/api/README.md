# API tests

Tests for the FastAPI endpoints in `src/api/server.py` and `src/api/voice_routes.py`. They use FastAPI's `TestClient` without running startup, and swap in a pipeline and voice pipeline wired to fakes, so no model or vector store is loaded.

| File | What it checks |
|---|---|
| `test_server_endpoints.py` | `/health` returns 200 when ready and 503 when not; `/chat` returns grounded answers, 422 for invalid bodies, 503 when not ready, and 502 without internal details on failure; `/tickets` lists every ticket in creation order; `/tickets/{id}` returns the stored record or 404. |
| `test_voice_endpoints.py` | `/voice/transcribe` returns the transcript and timing, and rejects empty or speech-free audio (422), a missing file (422), provider failures (502, no internal details), and unavailable voice (503); `/voice/synthesize` returns MP3 audio tagged with its message ID, gives each message its own audio, and rejects invalid bodies (422) and provider failures (502); typed `/chat` keeps its exact contract whether voice is working or broken. |

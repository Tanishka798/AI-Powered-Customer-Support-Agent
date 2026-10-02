# UI client tests

Tests for the Streamlit-facing helpers in `src/ui/`. `httpx.request` is replaced with a stub, so no server is needed.

| File | What it checks |
|---|---|
| `test_api_client.py` | A successful chat reply becomes an assistant message with type-checked sources; 503, 422, and other errors, invalid JSON, empty replies, and connection, timeout, and protocol failures all raise an `ApiError` with a customer-safe message; ticket lookup returns the stored record, reports unknown IDs clearly, and rejects a blank ID without calling the API; transcription returns the trimmed transcript and shows the API's voice error text; synthesis returns audio only when the message ID and media type match. |
| `test_speech_cache.py` | Audio is stored under its own message ID, repeat clicks reuse it without calling the API, changed text is synthesized again, and failures store nothing. |
| `test_chat_submission.py` | Typed text is sent directly and never transcribed; a recording is transcribed into the chat box instead of being sent; typed text is kept before the transcript; transcription errors keep the typed text and show a customer-safe message. |

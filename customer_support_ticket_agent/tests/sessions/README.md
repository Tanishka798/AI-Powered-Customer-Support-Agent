# Session tests

Tests for `src/sessions/store.py`.

| File | What it checks |
|---|---|
| `test_conversation_state.py` | Missing ticket fields are reported in collection order (empty, partial, and complete states), and `ticket_in_progress()` is true only while details are being collected and no ticket exists. |
| `test_session_store.py` | The same ID returns the same state, different sessions don't share values, and blank session IDs are rejected. |

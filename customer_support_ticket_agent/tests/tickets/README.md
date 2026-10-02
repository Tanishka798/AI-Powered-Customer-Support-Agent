# Ticket tests

Tests for `src/tools/ticket_tool.py`.

| File | What it checks |
|---|---|
| `test_ticket_repository.py` | One ticket per session (repeat requests return the first), unique IDs across sessions, stored details match the validated input, and unknown IDs return `None`. |
| `test_ticket_tool.py` | The LangChain tool returns a real repository ID, stays bound to its session, and creates nothing when the email or category is invalid. |

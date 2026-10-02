# Workflow tests

Tests for the agent logic in `src/llm/`. The model and retriever are scripted fakes (`tests/fakes.py`), so every route is deterministic.

Unit tests (one module each):

| File | What it checks |
|---|---|
| `test_output_parsing.py` | JSON is parsed from plain, fenced, and wrapped model output; non-objects are rejected; content blocks are flattened to text. |
| `test_decision.py` | Field validation drops bad emails and categories; bare email, category, and description replies are read without the model (questions are left for the answer route); the route is forced to `ticket` only when details are supplied mid-collection; unparseable model output fails cleanly. |
| `test_model_check.py` | The Ollama model is pulled only when missing (bare names match `:latest`), non-Ollama or unreachable endpoints are skipped without raising, and auto-pull can be turned off. |
| `test_ticket_collector.py` | The next missing field is asked for and remembered; each new value is checked as it arrives (a too-short description is asked again at once, valid values are kept); a valid set creates exactly one ticket; stale ticket references are cleared; invalid fields are cleared and asked again; long summaries are truncated; the ticket reminder appears only mid-collection. |

End-to-end through the compiled graph:

| File | What it checks |
|---|---|
| `test_answer_route.py` | Policy questions get grounded answers with sources; unknown questions aren't sent to the answer model or made up; unparseable decisions and provider failures raise clear errors. |
| `test_ticket_route.py` | Multi-turn collection ends in a real ticket; repeat requests don't create duplicates; invalid emails and too-short descriptions are asked again; the pending question reaches the model; bare replies are captured, including vague descriptions the model misroutes; side questions are answered mid-ticket; sessions don't share details. |

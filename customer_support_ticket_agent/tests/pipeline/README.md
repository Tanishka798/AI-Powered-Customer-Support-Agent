# Pipeline tests

Tests for `SupportPipeline` in `src/pipeline.py`, using the real workflow with fakes, or a stub workflow for edge cases.

| File | What it checks |
|---|---|
| `test_support_pipeline.py` | Requests are rejected before initialization and when the session ID or message is blank; a successful turn trims input and records user and assistant history; workflow failures return a generic error with no internal details; empty responses are rejected; ticket IDs missing from the repository are never returned. |

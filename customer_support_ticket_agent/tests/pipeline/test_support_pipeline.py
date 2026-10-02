from pathlib import Path

import pytest

from src.llm.workflow import build_support_workflow
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from tests.fakes import FakeChatModel, FakeRetriever, decision, make_settings


class StubWorkflow:
    """Returns a fixed workflow result, bypassing the graph entirely."""

    def __init__(self, result: dict) -> None:
        self.result = result

    async def ainvoke(self, state: dict) -> dict:
        return self.result


def ready_pipeline(tmp_path: Path, workflow=None, *replies) -> SupportPipeline:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.workflow = workflow or build_support_workflow(
        FakeChatModel(*replies), FakeRetriever(), pipeline.sessions, pipeline.tickets
    )
    pipeline.ready = True
    return pipeline


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(tmp_path: Path) -> None:
    pipeline = SupportPipeline(make_settings(), tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id, message", [("   ", "hello"), ("session-1", "   ")])
async def test_blank_input_is_rejected(tmp_path: Path, session_id: str, message: str) -> None:
    with pytest.raises(AgentProcessingError):
        await ready_pipeline(tmp_path).process(session_id, message)


@pytest.mark.asyncio
async def test_successful_turn_returns_response_and_records_history(tmp_path: Path) -> None:
    pipeline = ready_pipeline(tmp_path, None, decision("answer"), "Three to five business days.")

    response = await pipeline.process("  session-1 ", "  How long does shipping take?  ")

    assert response.session_id == "session-1"
    assert response.response == "Three to five business days."
    assert response.sources == ["shipping.md"]
    assert pipeline.sessions.get_or_create("session-1").history == [
        {"role": "user", "content": "How long does shipping take?"},
        {"role": "assistant", "content": "Three to five business days."},
    ]


@pytest.mark.asyncio
async def test_workflow_failure_hides_internal_details(tmp_path: Path) -> None:
    error = ConnectionError("refused http://10.0.0.5:11434 /home/deploy/.env")
    pipeline = ready_pipeline(tmp_path, None, error)

    with pytest.raises(AgentProcessingError) as excinfo:
        await pipeline.process("session-1", "hello")

    message = str(excinfo.value)
    assert "10.0.0.5" not in message and "/home" not in message
    # Failed turns don't become misleading history for the next model call.
    assert pipeline.sessions.get_or_create("session-1").history == []


@pytest.mark.asyncio
async def test_empty_workflow_response_is_rejected(tmp_path: Path) -> None:
    pipeline = ready_pipeline(tmp_path, StubWorkflow({"response_text": "   "}))

    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_ticket_id_missing_from_repository_is_dropped(tmp_path: Path) -> None:
    pipeline = ready_pipeline(tmp_path, StubWorkflow({"response_text": "Done.", "ticket_id": "CST-FAKE"}))

    response = await pipeline.process("session-1", "hello")

    assert response.ticket_id is None

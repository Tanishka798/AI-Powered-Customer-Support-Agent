import pytest

from src.llm.prompts import NO_ANSWER_TEXT
from tests.fakes import WorkflowHarness, decision


@pytest.mark.asyncio
async def test_policy_question_is_answered_with_sources() -> None:
    harness = WorkflowHarness(decision("answer"), "Standard delivery takes 3-5 business days.")

    result = await harness.turn("How long does shipping take?")

    assert result["response_text"] == "Standard delivery takes 3-5 business days."
    assert result["sources"] == ["shipping.md"]
    assert result.get("ticket_id") is None
    assert "Standard delivery takes 3-5 business days." in harness.model.prompts[-1]


@pytest.mark.asyncio
async def test_unknown_question_is_not_fabricated() -> None:
    harness = WorkflowHarness(decision("answer"))

    result = await harness.turn("Do you sell gift vouchers?")

    assert result["response_text"] == NO_ANSWER_TEXT
    assert result["sources"] == []
    # Only the decision model ran; the answer model is never asked without evidence.
    assert len(harness.model.prompts) == 1


@pytest.mark.asyncio
async def test_greeting_gets_natural_model_response_without_ticket_collection() -> None:
    harness = WorkflowHarness("Hello! How can I help you today?")

    result = await harness.turn("Hello")

    assert result["response_text"] == "Hello! How can I help you today?"
    assert "name" not in result["response_text"].lower()
    assert result["ticket_id"] is None
    assert len(harness.model.prompts) == 1
    assert "Current customer message:\n\nHello" in harness.model.prompts[0]


@pytest.mark.asyncio
async def test_ambiguous_request_uses_model_to_clarify() -> None:
    harness = WorkflowHarness(decision("clarify"), "What would you like help with?")

    result = await harness.turn("I'm not sure what I need")

    assert result["response_text"] == "What would you like help with?"
    assert result["ticket_id"] is None
    assert len(harness.model.prompts) == 2


@pytest.mark.asyncio
async def test_follow_up_uses_conversation_for_retrieval_and_answering() -> None:
    harness = WorkflowHarness(
        decision("answer"),
        "Standard delivery takes 3-5 business days.",
        decision("answer"),
        "The knowledge base does not specify weekend delivery.",
    )

    await harness.turn("How long does shipping take?")
    follow_up = await harness.turn("What about weekends?")

    assert follow_up["sources"] == ["shipping.md"]
    assert "weekend" in follow_up["response_text"]
    assert "How long does shipping take?" in harness.model.prompts[-2]
    assert "Standard delivery takes 3-5 business days." in harness.model.prompts[-1]


@pytest.mark.asyncio
async def test_content_block_answers_are_flattened() -> None:
    harness = WorkflowHarness(decision("answer"), [{"type": "text", "text": "Three to five days."}])

    result = await harness.turn("How long does shipping take?")

    assert result["response_text"] == "Three to five days."


@pytest.mark.asyncio
async def test_invalid_decision_output_fails_cleanly() -> None:
    harness = WorkflowHarness("I think you should answer this one.")

    with pytest.raises(RuntimeError, match="invalid routing response"):
        await harness.turn("How long does shipping take?")


@pytest.mark.asyncio
async def test_answer_model_failure_is_raised() -> None:
    harness = WorkflowHarness(decision("answer"), ConnectionError("provider down"))

    with pytest.raises(RuntimeError, match="Answer model invocation failed"):
        await harness.turn("How long does shipping take?")

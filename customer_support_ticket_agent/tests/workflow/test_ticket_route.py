import pytest

from src.llm.prompts import FIELD_PROMPTS, RETRY_PROMPTS
from tests.fakes import WorkflowHarness, decision

ALL_FIELDS = dict(
    customer_name="Asha",
    customer_email="asha@example.com",
    issue_description="Payment was charged twice",
    category="payment",
)
# A message that actually contains every value in ALL_FIELDS.
ALL_FIELDS_MESSAGE = "I'm Asha, asha@example.com. Payment was charged twice, it's a payment issue."


@pytest.mark.asyncio
async def test_ticket_details_collected_over_multiple_turns() -> None:
    harness = WorkflowHarness(
        decision("ticket", issue_description="Payment was charged twice"),
        decision("ticket", customer_name="Asha"),
        decision("ticket", customer_email="asha@example.com"),
        decision("ticket", category="payment"),
    )

    assert (await harness.turn("I was charged twice"))["response_text"] == FIELD_PROMPTS["customer_name"]
    assert (await harness.turn("Asha"))["response_text"] == FIELD_PROMPTS["customer_email"]
    assert (await harness.turn("asha@example.com"))["response_text"] == FIELD_PROMPTS["category"]
    result = await harness.turn("payment")

    ticket = harness.tickets.get(result["ticket_id"])
    assert ticket is not None
    assert ticket.customer_name == "Asha"
    assert ticket.category == "payment"
    assert result["ticket_id"] in result["response_text"]


@pytest.mark.asyncio
async def test_retry_after_creation_returns_same_ticket() -> None:
    harness = WorkflowHarness(decision("ticket", **ALL_FIELDS), decision("ticket"))

    first = await harness.turn(ALL_FIELDS_MESSAGE)
    retry = await harness.turn(ALL_FIELDS_MESSAGE)

    assert retry["ticket_id"] == first["ticket_id"]
    assert len(list(harness.tickets.all())) == 1


@pytest.mark.asyncio
async def test_invalid_email_is_not_committed() -> None:
    harness = WorkflowHarness(decision("ticket", **{**ALL_FIELDS, "customer_email": "asha@"}))

    result = await harness.turn("My email is asha@")

    assert result["response_text"] == RETRY_PROMPTS["customer_email"]
    assert harness.sessions.get_or_create("session-1").customer_email is None
    assert list(harness.tickets.all()) == []


@pytest.mark.asyncio
async def test_invalid_email_preserves_other_fields_and_accepts_correction() -> None:
    invalid_fields = {**ALL_FIELDS, "customer_email": "asha@"}
    harness = WorkflowHarness(decision("ticket", **invalid_fields), decision("ticket"))

    first = await harness.turn("I'm Asha, asha@. Payment was charged twice, a payment issue.")
    session = harness.sessions.get_or_create("session-1")

    assert first["response_text"] == RETRY_PROMPTS["customer_email"]
    assert session.customer_name == "Asha"
    assert session.customer_email is None
    assert session.issue_description == "Payment was charged twice"
    assert session.category == "payment"
    assert list(harness.tickets.all()) == []

    corrected = await harness.turn("asha@example.com")

    assert harness.tickets.get(corrected["ticket_id"]).customer_email == "asha@example.com"
    assert len(list(harness.tickets.all())) == 1


@pytest.mark.asyncio
async def test_empty_email_is_rejected_before_ticket_creation() -> None:
    empty_email_fields = {**ALL_FIELDS, "customer_email": ""}
    harness = WorkflowHarness(decision("ticket", **empty_email_fields))

    result = await harness.turn("I'm Asha. My email is empty. Payment was charged twice, a payment issue.")

    assert result["response_text"] == RETRY_PROMPTS["customer_email"]
    assert harness.sessions.get_or_create("session-1").customer_email is None
    assert list(harness.tickets.all()) == []


@pytest.mark.asyncio
async def test_too_short_description_is_cleared_and_asked_again() -> None:
    harness = WorkflowHarness(
        decision("ticket", **{**ALL_FIELDS, "issue_description": "slow"}),
        decision("ticket", issue_description="The app is very slow to load pages"),
    )

    first = await harness.turn("I'm Asha, asha@example.com, payment problem: slow")
    assert first["response_text"] == RETRY_PROMPTS["issue_description"]
    assert harness.sessions.get_or_create("session-1").issue_description is None

    second = await harness.turn("The app is very slow to load pages")
    assert harness.tickets.get(second["ticket_id"]) is not None


@pytest.mark.asyncio
async def test_decision_prompt_includes_pending_question() -> None:
    harness = WorkflowHarness(
        decision("ticket", issue_description="Payment was charged twice"),
        decision("ticket", customer_name="Asha"),
    )

    await harness.turn("I was charged twice")
    await harness.turn("Asha")

    prompt = harness.model.prompts[-1]
    assert f"Previous assistant message:\n{FIELD_PROMPTS['customer_name']}" in prompt
    assert '"awaiting_field": "customer_name"' in prompt


@pytest.mark.asyncio
async def test_bare_email_and_category_replies_are_captured() -> None:
    harness = WorkflowHarness(
        decision("ticket", customer_name="Asha", issue_description="Payment was charged twice"),
        decision("ticket"),  # model misses the bare email
        decision("answer"),  # model misroutes the bare category
    )

    await harness.turn("I'm Asha and I was charged twice")
    await harness.turn("asha@example.com")
    result = await harness.turn("Payment")

    ticket = harness.tickets.get(result["ticket_id"])
    assert ticket is not None
    assert ticket.customer_email == "asha@example.com"
    assert ticket.category == "payment"


@pytest.mark.asyncio
async def test_side_question_is_answered_with_ticket_reminder() -> None:
    harness = WorkflowHarness(
        decision("ticket", issue_description="My order never arrived"),
        decision("answer"),
        "Standard delivery takes 3-5 business days.",
    )

    await harness.turn("My order never arrived")
    result = await harness.turn("How long does shipping take?")

    assert result["response_text"].startswith("Standard delivery takes 3-5 business days.")
    assert result["response_text"].endswith(FIELD_PROMPTS["customer_name"])
    assert result["sources"] == ["shipping.md"]
    assert harness.sessions.get_or_create("session-1").issue_description == "My order never arrived"


@pytest.mark.asyncio
async def test_sessions_do_not_share_ticket_details() -> None:
    harness = WorkflowHarness(
        decision("ticket", customer_name="Asha", issue_description="Payment was charged twice"),
        decision("ticket", issue_description="Cannot sign in to my account"),
    )

    await harness.turn("I'm Asha and I was charged twice", session_id="session-1")
    result = await harness.turn("I cannot sign in", session_id="session-2")

    assert result["response_text"] == FIELD_PROMPTS["customer_name"]
    assert harness.sessions.get_or_create("session-2").customer_name is None


@pytest.mark.asyncio
async def test_side_question_answered_even_when_model_echoes_session_values() -> None:
    # Regression from a live qwen2.5:3b run: the model copied the stored
    # category into its output, which used to force the ticket route.
    harness = WorkflowHarness(
        decision("ticket", issue_description="Payment was charged twice", category="payment"),
        decision("answer", category="payment"),  # what qwen2.5:3b actually returned
        "Standard delivery takes 3-5 business days.",
    )

    await harness.turn("Payment was charged twice, it's a payment issue")
    result = await harness.turn("How long does shipping take?")

    assert result["response_text"].startswith("Standard delivery")
    assert result["response_text"].endswith(FIELD_PROMPTS["customer_name"])


@pytest.mark.asyncio
async def test_question_about_existing_ticket_returns_its_id() -> None:
    harness = WorkflowHarness(decision("ticket", **ALL_FIELDS))

    created = await harness.turn(ALL_FIELDS_MESSAGE)
    follow_up = await harness.turn("Did my ticket go through?")

    assert follow_up["ticket_id"] == created["ticket_id"]
    assert created["ticket_id"] in follow_up["response_text"]
    assert "open" in follow_up["response_text"]


@pytest.mark.asyncio
async def test_vague_description_reply_is_captured_not_answered() -> None:
    # Replays a live qwen2.5:3b session: while awaiting the description, the
    # model routed "slow" and "the product was slow" to answer with no fields.
    harness = WorkflowHarness(
        decision("ticket", customer_name="Viraj"),
        decision("answer"),
        decision("answer"),
        decision("answer"),
    )

    await harness.turn("I need help, I'm Viraj")
    assert (await harness.turn("virajs0011@gmail.com"))["response_text"] == FIELD_PROMPTS["issue_description"]

    too_short = await harness.turn("slow")
    assert too_short["response_text"] == RETRY_PROMPTS["issue_description"]

    described = await harness.turn("the product was slow")
    assert described["response_text"] == FIELD_PROMPTS["category"]
    assert harness.sessions.get_or_create("session-1").issue_description == "the product was slow"

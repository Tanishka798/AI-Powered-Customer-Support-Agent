import pytest

from src.llm.decision import (
    clean_extracted_fields,
    decide_turn,
    extract_pending_reply,
    new_grounded_fields,
)
from src.sessions.store import ConversationState
from tests.fakes import FakeChatModel, decision


def test_clean_fields_preserves_values_for_authoritative_validation() -> None:
    fields = clean_extracted_fields(
        {
            "customer_name": "  Asha  ",
            "customer_email": "not-an-email",
            "issue_description": "",
            "category": "Billing",
        }
    )

    assert fields == {
        "customer_name": "Asha",
        "customer_email": "not-an-email",
        "category": "Billing",
    }


def test_validate_fields_normalizes_category_case() -> None:
    fields = clean_extracted_fields({"customer_email": "asha@example.com", "category": "PAYMENT"})

    assert fields == {"customer_email": "asha@example.com", "category": "payment"}


def test_grounding_drops_values_not_in_the_message() -> None:
    fields = {"customer_name": "Asha", "category": "order", "issue_description": "Charged twice"}

    kept = new_grounded_fields(fields, "I was charged twice", ConversationState())

    # Free-text descriptions may be paraphrased; names and categories may not be guessed.
    assert kept == {"issue_description": "Charged twice"}


def test_grounding_drops_values_already_stored() -> None:
    session = ConversationState(category="payment")

    assert new_grounded_fields({"category": "payment"}, "payment again", session) == {}


def test_grounding_matches_whole_words_only() -> None:
    awaiting_category = ConversationState(awaiting_field="category")

    assert new_grounded_fields({"category": "other"}, "another", awaiting_category) == {}
    assert new_grounded_fields({"category": "other"}, "Other.", awaiting_category) == {"category": "other"}


@pytest.mark.parametrize(
    "message, category, kept",
    [
        ("I was charged twice for my order", "order", False),
        ("It's a technical issue", "technical", True),
        ("Category: payment", "payment", True),
    ],
)
def test_category_must_be_chosen_not_just_mentioned(message: str, category: str, kept: bool) -> None:
    fields = new_grounded_fields({"category": category}, message, ConversationState())

    assert (fields == {"category": category}) is kept


@pytest.mark.parametrize(
    "awaiting, message, expected",
    [
        ("customer_email", "sure, it's asha@example.com.", {"customer_email": "asha@example.com"}),
        ("category", "Payment", {"category": "payment"}),
        ("category", "no idea", {"category": "no idea"}),
        ("customer_name", "Asha", {}),
        ("issue_description", "the product was slow", {"issue_description": "the product was slow"}),
        ("issue_description", "How long do refunds take?", {}),
        ("issue_description", "what is your return policy", {}),
    ],
)
def test_extract_pending_reply(awaiting: str, message: str, expected: dict) -> None:
    session = ConversationState(awaiting_field=awaiting)

    assert extract_pending_reply(session, message) == expected


@pytest.mark.asyncio
async def test_decide_turn_keeps_model_route_outside_ticket_collection() -> None:
    route, fields = await decide_turn(FakeChatModel(decision("answer")), "Hi", ConversationState())

    assert (route, fields) == ("greeting", {})


@pytest.mark.asyncio
async def test_decide_turn_forces_ticket_route_when_details_supplied_mid_collection() -> None:
    session = ConversationState(issue_description="Charged twice", awaiting_field="category")

    route, fields = await decide_turn(FakeChatModel(decision("answer")), "payment", session)

    assert (route, fields) == ("ticket", {"category": "payment"})


@pytest.mark.asyncio
async def test_decide_turn_allows_side_question_mid_collection() -> None:
    session = ConversationState(issue_description="Charged twice", awaiting_field="customer_name")

    route, _ = await decide_turn(FakeChatModel(decision("answer")), "How long is shipping?", session)

    assert route == "answer"


@pytest.mark.asyncio
async def test_decide_turn_routes_existing_ticket_questions_without_the_model() -> None:
    model = FakeChatModel()
    session = ConversationState(ticket_id="CST-2026-0001")

    assert await decide_turn(model, "Is my ticket open?", session) == ("ticket_lookup", {})
    assert model.prompts == []


@pytest.mark.asyncio
async def test_decide_turn_rejects_unparseable_output() -> None:
    with pytest.raises(RuntimeError, match="invalid routing response"):
        await decide_turn(FakeChatModel("not json"), "Can you tell me about shipping?", ConversationState())

from src.llm.prompts import FIELD_PROMPTS, RETRY_PROMPTS
from src.llm.ticket_collection import TicketCollector, with_ticket_reminder
from src.sessions.store import ConversationState
from src.tools.ticket_tool import TicketRepository

ALL_FIELDS = {
    "customer_name": "Asha",
    "customer_email": "asha@example.com",
    "issue_description": "Payment was charged twice",
    "category": "payment",
}


def test_asks_for_first_missing_field_and_remembers_it() -> None:
    session = ConversationState()

    reply = TicketCollector(TicketRepository()).handle_turn("s1", session, {"customer_name": "Asha"})

    assert reply.text == FIELD_PROMPTS["customer_email"]
    assert reply.ticket_id is None
    assert session.awaiting_field == "customer_email"


def test_creates_ticket_when_all_fields_valid() -> None:
    tickets = TicketRepository()
    session = ConversationState(awaiting_field="category")

    reply = TicketCollector(tickets).handle_turn("s1", session, ALL_FIELDS)

    assert tickets.get(reply.ticket_id) is not None
    assert session.ticket_id == reply.ticket_id
    assert session.awaiting_field is None


def test_existing_ticket_is_returned_not_recreated() -> None:
    tickets = TicketRepository()
    collector = TicketCollector(tickets)
    session = ConversationState()
    first = collector.handle_turn("s1", session, ALL_FIELDS)

    again = collector.handle_turn("s1", session, {})

    assert again.ticket_id == first.ticket_id
    assert len(list(tickets.all())) == 1


def test_stale_ticket_reference_is_cleared() -> None:
    session = ConversationState(ticket_id="CST-MISSING")

    reply = TicketCollector(TicketRepository()).handle_turn("s1", session, {})

    assert session.ticket_id is None
    assert reply.text == FIELD_PROMPTS["customer_name"]


def test_invalid_field_is_cleared_and_asked_again() -> None:
    session = ConversationState()

    reply = TicketCollector(TicketRepository()).handle_turn(
        "s1", session, {**ALL_FIELDS, "issue_description": "slow"}
    )

    assert reply.text == RETRY_PROMPTS["issue_description"]
    assert session.issue_description is None
    assert session.awaiting_field == "issue_description"


def test_long_description_gets_truncated_summary() -> None:
    tickets = TicketRepository()
    description = "x" * 300

    reply = TicketCollector(tickets).handle_turn(
        "s1", ConversationState(), {**ALL_FIELDS, "issue_description": description}
    )

    summary = tickets.get(reply.ticket_id).summary
    assert len(summary) == 160 and summary.endswith("...")


def test_reminder_added_only_while_ticket_in_progress() -> None:
    assert with_ticket_reminder("Answer.", ConversationState()) == "Answer."
    assert with_ticket_reminder("Answer.", ConversationState(awaiting_field="customer_email")).endswith(
        FIELD_PROMPTS["customer_email"]
    )


def test_invalid_value_is_rejected_immediately_while_valid_ones_are_kept() -> None:
    session = ConversationState()

    reply = TicketCollector(TicketRepository()).handle_turn(
        "s1", session, {"customer_name": "Asha", "issue_description": "slow"}
    )

    assert reply.text == RETRY_PROMPTS["issue_description"]
    assert session.customer_name == "Asha"
    assert session.issue_description is None
    assert session.awaiting_field == "issue_description"

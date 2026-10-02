import pytest
from pydantic import ValidationError

from src.tools.ticket_tool import TicketRepository, create_ticket_tool

VALID_ARGS = {
    "customer_name": "Test User",
    "customer_email": "test@example.com",
    "issue_description": "Payment was charged twice",
    "category": "payment",
    "summary": "Possible duplicate payment charge",
}


def test_tool_creates_ticket_and_returns_repository_id() -> None:
    repository = TicketRepository()

    ticket_id = create_ticket_tool(repository, "session-1").invoke(VALID_ARGS)

    assert repository.get(ticket_id) is not None


def test_tool_is_bound_to_its_session() -> None:
    repository = TicketRepository()

    first = create_ticket_tool(repository, "session-1").invoke(VALID_ARGS)
    retry = create_ticket_tool(repository, "session-1").invoke(VALID_ARGS)
    other = create_ticket_tool(repository, "session-2").invoke(VALID_ARGS)

    assert first == retry
    assert other != first


@pytest.mark.parametrize("field, value", [("customer_email", "not-an-email"), ("category", "billing")])
def test_invalid_input_creates_no_ticket(field: str, value: str) -> None:
    repository = TicketRepository()

    with pytest.raises(ValidationError):
        create_ticket_tool(repository, "session-1").invoke({**VALID_ARGS, field: value})

    assert list(repository.all()) == []

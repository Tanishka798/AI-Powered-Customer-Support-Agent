from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository


def make_request(**overrides) -> TicketCreate:
    values = dict(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Possible duplicate payment charge",
    )
    values.update(overrides)
    return TicketCreate(**values)


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()

    first = repository.create("session-1", make_request())
    second = repository.create("session-1", make_request())

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_different_sessions_get_unique_ticket_ids() -> None:
    repository = TicketRepository()

    first = repository.create("session-1", make_request())
    second = repository.create("session-2", make_request())

    assert first.ticket_id != second.ticket_id


def test_ticket_preserves_validated_details() -> None:
    repository = TicketRepository()

    ticket = repository.create("session-1", make_request(category="account"))

    stored = repository.get(ticket.ticket_id)
    assert stored is not None
    assert stored.category == "account"
    assert stored.customer_email == "test@example.com"
    assert stored.status == "open"


def test_unknown_ticket_id_returns_none() -> None:
    assert TicketRepository().get("CST-2026-9999") is None

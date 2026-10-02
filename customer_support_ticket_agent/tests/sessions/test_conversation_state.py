from src.sessions.store import ConversationState


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")

    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_empty_state_reports_all_fields_in_collection_order() -> None:
    assert ConversationState().missing_ticket_fields() == [
        "customer_name",
        "customer_email",
        "issue_description",
        "category",
    ]


def test_complete_state_reports_nothing_missing() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Payment was charged twice",
        category="payment",
    )

    assert state.missing_ticket_fields() == []


def test_ticket_in_progress_tracks_collection_lifecycle() -> None:
    state = ConversationState()
    assert not state.ticket_in_progress()

    state.awaiting_field = "customer_name"
    assert state.ticket_in_progress()

    state.awaiting_field = None
    state.customer_name = "Asha"
    assert state.ticket_in_progress()

    state.ticket_id = "CST-2026-0001"
    assert not state.ticket_in_progress()

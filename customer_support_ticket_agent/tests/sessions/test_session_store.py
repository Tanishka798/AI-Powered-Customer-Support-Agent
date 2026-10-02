import pytest

from src.sessions.store import SessionStore


def test_same_session_id_returns_same_state() -> None:
    store = SessionStore()
    state = store.get_or_create("session-1")
    state.customer_name = "Asha"

    assert store.get_or_create("session-1") is state


def test_new_session_does_not_reuse_another_customers_values() -> None:
    store = SessionStore()
    store.get_or_create("session-1").customer_email = "asha@example.com"

    other = store.get_or_create("session-2")

    assert other.customer_email is None
    assert other.history == []


def test_blank_session_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        SessionStore().get_or_create("   ")

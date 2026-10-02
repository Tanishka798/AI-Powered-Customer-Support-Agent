import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.llm.workflow import build_support_workflow
from src.models import TicketCreate
from src.pipeline import SupportPipeline
from tests.fakes import FakeChatModel, FakeRetriever, decision, make_settings


def install_pipeline(monkeypatch, tmp_path, *replies, ready: bool = True) -> SupportPipeline:
    """Swap the module-level pipeline for one wired to fakes (lifespan is not run)."""
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.workflow = build_support_workflow(
        FakeChatModel(*replies), FakeRetriever(), pipeline.sessions, pipeline.tickets
    )
    pipeline.ready = ready
    monkeypatch.setattr(server, "pipeline", pipeline)
    return pipeline


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def test_health_reports_ready(monkeypatch, tmp_path, client) -> None:
    install_pipeline(monkeypatch, tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_health_and_chat_return_503_when_not_ready(monkeypatch, tmp_path, client) -> None:
    install_pipeline(monkeypatch, tmp_path, ready=False)

    assert client.get("/health").status_code == 503
    assert client.post("/chat", json={"session_id": "s1", "message": "hi"}).status_code == 503


def test_chat_returns_grounded_answer(monkeypatch, tmp_path, client) -> None:
    install_pipeline(monkeypatch, tmp_path, decision("answer"), "Three to five business days.")

    response = client.post("/chat", json={"session_id": "s1", "message": "How long does shipping take?"})

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "session_id": "s1",
        "response": "Three to five business days.",
        "sources": ["shipping.md"],
        "ticket_id": None,
    }


def test_chat_preserves_context_across_requests_with_the_same_session_id(monkeypatch, tmp_path, client) -> None:
    model = FakeChatModel(
        decision("answer"),
        "Standard delivery takes 3-5 business days.",
        decision("answer"),
        "The knowledge base does not specify weekend delivery.",
    )
    pipeline = SupportPipeline(make_settings(), tmp_path)
    pipeline.workflow = build_support_workflow(model, FakeRetriever(), pipeline.sessions, pipeline.tickets)
    pipeline.ready = True
    monkeypatch.setattr(server, "pipeline", pipeline)

    first = client.post("/chat", json={"session_id": "stable-thread", "message": "How long does shipping take?"})
    second = client.post("/chat", json={"session_id": "stable-thread", "message": "What about weekends?"})

    assert first.status_code == second.status_code == 200
    assert first.json()["response"] == "Standard delivery takes 3-5 business days."
    assert "weekend" in second.json()["response"]
    assert "How long does shipping take?" in model.prompts[-2]
    assert "Standard delivery takes 3-5 business days." in model.prompts[-1]


@pytest.mark.parametrize("payload", [{"session_id": "", "message": "hi"}, {"session_id": "s1"}])
def test_chat_rejects_invalid_request_body(monkeypatch, tmp_path, client, payload) -> None:
    install_pipeline(monkeypatch, tmp_path)

    assert client.post("/chat", json=payload).status_code == 422


def test_chat_failure_returns_502_without_internal_details(monkeypatch, tmp_path, client) -> None:
    install_pipeline(monkeypatch, tmp_path, ConnectionError("refused http://10.0.0.5:11434"))

    response = client.post("/chat", json={"session_id": "s1", "message": "hello"})

    assert response.status_code == 502
    assert "10.0.0.5" not in response.text


def test_get_ticket_returns_stored_record(monkeypatch, tmp_path, client) -> None:
    pipeline = install_pipeline(monkeypatch, tmp_path)
    ticket = pipeline.tickets.create(
        "s1",
        TicketCreate(
            customer_name="Asha",
            customer_email="asha@example.com",
            issue_description="Payment was charged twice",
            category="payment",
            summary="Payment was charged twice",
        ),
    )

    response = client.get(f"/tickets/{ticket.ticket_id}")

    assert response.status_code == 200
    assert response.json()["ticket_id"] == ticket.ticket_id
    assert response.json()["category"] == "payment"


def test_list_tickets_returns_all_in_creation_order(monkeypatch, tmp_path, client) -> None:
    pipeline = install_pipeline(monkeypatch, tmp_path)
    assert client.get("/tickets").json() == []

    details = dict(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Payment was charged twice",
    )
    first = pipeline.tickets.create("s1", TicketCreate(**details))
    second = pipeline.tickets.create("s2", TicketCreate(**details))

    response = client.get("/tickets")

    assert response.status_code == 200
    assert [ticket["ticket_id"] for ticket in response.json()] == [first.ticket_id, second.ticket_id]


def test_get_unknown_ticket_returns_404(monkeypatch, tmp_path, client) -> None:
    install_pipeline(monkeypatch, tmp_path)

    assert client.get("/tickets/CST-2026-9999").status_code == 404

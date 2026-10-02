import httpx
import pytest

from src.ui import api_client
from src.ui.api_client import ApiError, fetch_ticket, send_message, synthesize_speech, transcribe_audio

BASE_URL = "http://api.test"


def reply_with(monkeypatch, response: httpx.Response | Exception) -> None:
    """Make httpx.request return the given response or raise the given error."""

    def fake_request(method, url, **kwargs):
        if isinstance(response, Exception):
            raise response
        response.request = httpx.Request(method, url)
        return response

    monkeypatch.setattr(api_client.httpx, "request", fake_request)


def test_successful_reply_becomes_assistant_message(monkeypatch) -> None:
    body = {"response": "Three to five days.", "sources": ["shipping.md", 7], "ticket_id": None}
    reply_with(monkeypatch, httpx.Response(200, json=body))

    message = send_message(BASE_URL, "s1", "How long is shipping?")

    assert message == {
        "role": "assistant",
        "content": "Three to five days.",
        "sources": ["shipping.md"],
        "ticket_id": None,
    }


@pytest.mark.parametrize(
    "response, expected",
    [
        (httpx.Response(503, json={"detail": "x"}), "temporarily unavailable"),
        (httpx.Response(422, json={"detail": "bad"}), "Invalid request: bad"),
        (httpx.Response(502, json={"detail": "Please try again."}), r"error \(502\): Please try again\."),
        (httpx.Response(200, text="not json"), "invalid response"),
        (httpx.Response(200, json={"response": "  "}), "empty response"),
    ],
)
def test_bad_responses_raise_customer_safe_errors(monkeypatch, response, expected) -> None:
    reply_with(monkeypatch, response)

    with pytest.raises(ApiError, match=expected):
        send_message(BASE_URL, "s1", "hello")


@pytest.mark.parametrize(
    "error, expected",
    [
        (httpx.ConnectError("refused"), "Could not connect"),
        (httpx.ReadTimeout("slow"), "took too long"),
        (httpx.RemoteProtocolError("broken"), "communication error"),
    ],
)
def test_transport_failures_raise_customer_safe_errors(monkeypatch, error, expected) -> None:
    reply_with(monkeypatch, error)

    with pytest.raises(ApiError, match=expected):
        send_message(BASE_URL, "s1", "hello")


TICKET = {"ticket_id": "CST-2026-0001", "status": "open", "category": "payment"}


def test_fetch_ticket_returns_stored_record(monkeypatch) -> None:
    reply_with(monkeypatch, httpx.Response(200, json=TICKET))

    assert fetch_ticket(BASE_URL, "  CST-2026-0001 ") == TICKET


def test_fetch_unknown_ticket_says_not_found(monkeypatch) -> None:
    reply_with(monkeypatch, httpx.Response(404, json={"detail": "Ticket not found"}))

    with pytest.raises(ApiError, match="No ticket found with ID CST-9"):
        fetch_ticket(BASE_URL, "CST-9")


def test_fetch_blank_ticket_id_is_rejected_without_a_request(monkeypatch) -> None:
    reply_with(monkeypatch, AssertionError("no request expected"))

    with pytest.raises(ApiError, match="enter a ticket ID"):
        fetch_ticket(BASE_URL, "   ")


def test_transcribe_audio_returns_trimmed_transcript(monkeypatch) -> None:
    body = {"success": True, "transcript": "  My payment was charged twice ", "processing_time_ms": 820}
    reply_with(monkeypatch, httpx.Response(200, json=body))

    assert transcribe_audio(BASE_URL, b"audio", "audio/wav") == "My payment was charged twice"


def test_transcribe_audio_surfaces_voice_error_text(monkeypatch) -> None:
    body = {"success": False, "error": "No understandable speech was detected"}
    reply_with(monkeypatch, httpx.Response(422, json=body))

    with pytest.raises(ApiError, match="^No understandable speech was detected$"):
        transcribe_audio(BASE_URL, b"audio", "audio/wav")


def test_voice_503_uses_the_voice_message(monkeypatch) -> None:
    body = {"success": False, "error": "Voice input is unavailable right now. Please type your message."}
    reply_with(monkeypatch, httpx.Response(503, json=body))

    with pytest.raises(ApiError, match="Please type your message"):
        transcribe_audio(BASE_URL, b"audio", "audio/wav")


def test_synthesize_speech_returns_audio_for_matching_message(monkeypatch) -> None:
    headers = {"content-type": "audio/mpeg", "X-Message-Id": "m1"}
    reply_with(monkeypatch, httpx.Response(200, content=b"mp3", headers=headers))

    assert synthesize_speech(BASE_URL, "m1", "Hello") == (b"mp3", "audio/mpeg")


@pytest.mark.parametrize(
    "headers, content, expected",
    [
        ({"content-type": "audio/mpeg", "X-Message-Id": "other"}, b"mp3", "doesn't match this message"),
        ({"content-type": "application/json", "X-Message-Id": "m1"}, b"{}", "invalid audio"),
        ({"content-type": "audio/mpeg", "X-Message-Id": "m1"}, b"", "invalid audio"),
    ],
)
def test_synthesize_speech_rejects_mismatched_or_invalid_audio(monkeypatch, headers, content, expected) -> None:
    reply_with(monkeypatch, httpx.Response(200, content=content, headers=headers))

    with pytest.raises(ApiError, match=expected):
        synthesize_speech(BASE_URL, "m1", "Hello")

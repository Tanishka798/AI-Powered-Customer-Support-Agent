import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.voice.pipeline import VoicePipeline
from tests.api.test_server_endpoints import install_pipeline
from tests.fakes import FakeSTT, FakeTTS, decision

WAV = ("recording.wav", b"RIFF-fake-audio", "audio/wav")


def install_voice(monkeypatch, stt=None, tts=None, ready: bool = True) -> VoicePipeline:
    """Swap the app's voice pipeline for one built from fakes (lifespan is not run)."""
    voice = VoicePipeline(stt or FakeSTT(), tts or FakeTTS())
    monkeypatch.setattr(server.app.state, "voice", voice)
    monkeypatch.setattr(server.app.state, "voice_ready", ready)
    return voice


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


# --- POST /voice/transcribe ---


def test_transcribe_returns_transcript_and_timing(monkeypatch, client) -> None:
    stt = FakeSTT()
    install_voice(monkeypatch, stt=stt)

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["transcript"] == "My payment was charged twice"
    assert isinstance(body["processing_time_ms"], int) and body["processing_time_ms"] >= 0
    assert stt.calls == [(b"RIFF-fake-audio", "audio/wav")]


def test_transcribe_rejects_empty_audio(monkeypatch, client) -> None:
    install_voice(monkeypatch)

    response = client.post("/voice/transcribe", files={"audio": ("empty.wav", b"", "audio/wav")})

    assert response.status_code == 422
    assert response.json() == {"success": False, "error": "Audio input is empty"}


def test_transcribe_reports_speech_free_audio(monkeypatch, client) -> None:
    install_voice(monkeypatch, stt=FakeSTT(transcript="  "))

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 422
    assert response.json()["error"] == "No understandable speech was detected"


def test_transcribe_requires_a_file(monkeypatch, client) -> None:
    install_voice(monkeypatch)

    response = client.post("/voice/transcribe")

    assert response.status_code == 422
    assert "detail" in response.json()


def test_transcribe_failure_is_generic_502(monkeypatch, client) -> None:
    install_voice(monkeypatch, stt=FakeSTT(error=RuntimeError("model crashed at /home/deploy/models")))

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 502
    assert response.json()["success"] is False
    assert "/home" not in response.text


def test_voice_unavailable_returns_503(monkeypatch, client) -> None:
    install_voice(monkeypatch, ready=False)

    assert client.post("/voice/transcribe", files={"audio": WAV}).status_code == 503
    assert client.post("/voice/synthesize", json={"message_id": "m1", "text": "Hi"}).status_code == 503


# --- POST /voice/synthesize ---


def test_synthesize_returns_audio_for_the_message(monkeypatch, client) -> None:
    install_voice(monkeypatch)

    response = client.post("/voice/synthesize", json={"message_id": "m1", "text": "Ticket created."})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-message-id"] == "m1"
    assert int(response.headers["x-processing-time-ms"]) >= 0
    assert response.content == b"audio:Ticket created."


def test_each_message_gets_its_own_audio(monkeypatch, client) -> None:
    install_voice(monkeypatch)

    first = client.post("/voice/synthesize", json={"message_id": "m1", "text": "First reply"})
    second = client.post("/voice/synthesize", json={"message_id": "m2", "text": "Second reply"})

    assert (first.headers["x-message-id"], first.content) == ("m1", b"audio:First reply")
    assert (second.headers["x-message-id"], second.content) == ("m2", b"audio:Second reply")


@pytest.mark.parametrize(
    "payload", [{"message_id": "m1"}, {"message_id": "", "text": "Hi"}, {"message_id": "m1", "text": ""}]
)
def test_synthesize_rejects_invalid_body(monkeypatch, client, payload) -> None:
    install_voice(monkeypatch)

    assert client.post("/voice/synthesize", json=payload).status_code == 422


def test_synthesize_rejects_blank_text_with_json_error(monkeypatch, client) -> None:
    install_voice(monkeypatch)

    response = client.post("/voice/synthesize", json={"message_id": "m1", "text": "   "})

    assert response.status_code == 422
    assert response.json() == {"success": False, "error": "Text input is empty"}


def test_synthesize_failure_is_generic_502(monkeypatch, client) -> None:
    install_voice(monkeypatch, tts=FakeTTS(error=ConnectionError("speech.platform.bing.com unreachable")))

    response = client.post("/voice/synthesize", json={"message_id": "m1", "text": "Hi"})

    assert response.status_code == 502
    assert response.json()["success"] is False
    assert "bing" not in response.text


# --- Typed chat is unaffected by voice ---


@pytest.mark.parametrize("voice_ready", [True, False])
def test_typed_chat_contract_unchanged(monkeypatch, tmp_path, client, voice_ready) -> None:
    install_pipeline(monkeypatch, tmp_path, decision("answer"), "Three to five business days.")
    install_voice(monkeypatch, tts=FakeTTS(error=RuntimeError("tts down")), ready=voice_ready)

    response = client.post("/chat", json={"session_id": "s1", "message": "How long does shipping take?"})

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "session_id": "s1",
        "response": "Three to five business days.",
        "sources": ["shipping.md"],
        "ticket_id": None,
    }

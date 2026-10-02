from types import SimpleNamespace

from src.ui.api_client import ApiError
from src.ui.voice_widgets import has_audio, resolve_submission


def recording(data: bytes = b"wav-bytes", media_type: str = "audio/wav"):
    return SimpleNamespace(getvalue=lambda: data, type=media_type)


def chat_value(text: str = "", audio=None):
    """Shaped like Streamlit's ChatInputValue (.text and .audio)."""
    return SimpleNamespace(text=text, audio=audio)


def fail_if_called(audio: bytes, media_type: str) -> str:
    raise AssertionError("typed text must not be transcribed")


def test_nothing_submitted_does_nothing() -> None:
    outcome = resolve_submission(None, fail_if_called)

    assert (outcome.send, outcome.prefill, outcome.error) == (None, None, None)


def test_typed_text_is_sent_directly() -> None:
    assert resolve_submission(chat_value("  How long is shipping?  "), fail_if_called).send == "How long is shipping?"
    assert resolve_submission("plain string", fail_if_called).send == "plain string"
    assert resolve_submission(chat_value("   "), fail_if_called).send is None


def test_recording_is_transcribed_into_the_box_not_sent() -> None:
    calls = []

    def transcribe(audio: bytes, media_type: str) -> str:
        calls.append((audio, media_type))
        return "My payment was charged twice"

    outcome = resolve_submission(chat_value(audio=recording()), transcribe)

    assert outcome.send is None
    assert outcome.prefill == "My payment was charged twice"
    assert calls == [(b"wav-bytes", "audio/wav")]


def test_typed_text_is_kept_before_the_transcript() -> None:
    outcome = resolve_submission(chat_value("Order 1234:", recording()), lambda audio, media_type: "it never arrived")

    assert outcome.prefill == "Order 1234: it never arrived"


def test_transcription_error_keeps_typed_text_and_reports() -> None:
    def transcribe(audio: bytes, media_type: str) -> str:
        raise ApiError("No understandable speech was detected")

    outcome = resolve_submission(chat_value("draft", recording()), transcribe)

    assert (outcome.send, outcome.prefill, outcome.error) == (None, "draft", "No understandable speech was detected")


def test_unexpected_error_is_customer_safe() -> None:
    def transcribe(audio: bytes, media_type: str) -> str:
        raise RuntimeError("socket closed at /internal/path")

    outcome = resolve_submission(chat_value(audio=recording()), transcribe)

    assert outcome.send is None
    assert "/internal" not in outcome.error


def test_has_audio() -> None:
    assert has_audio(chat_value(audio=recording()))
    assert not has_audio(chat_value("text"))
    assert not has_audio("text")
    assert not has_audio(None)

import pytest

from src.ui.speech_cache import SpeechCache


class RecordingSynthesizer:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.error = error

    def __call__(self, message_id: str, text: str) -> tuple[bytes, str]:
        self.calls.append((message_id, text))
        if self.error:
            raise self.error
        return f"{message_id}:{text}".encode(), "audio/mpeg"


def test_audio_is_stored_under_its_own_message_id() -> None:
    cache = SpeechCache({})
    synthesize = RecordingSynthesizer()

    first = cache.get_or_synthesize("m1", "First reply", synthesize)
    second = cache.get_or_synthesize("m2", "Second reply", synthesize)

    assert cache.get("m1") == first and first.audio == b"m1:First reply"
    assert cache.get("m2") == second and second.audio == b"m2:Second reply"


def test_repeat_clicks_reuse_existing_audio() -> None:
    cache = SpeechCache({})
    synthesize = RecordingSynthesizer()

    cache.get_or_synthesize("m1", "Reply", synthesize)
    again = cache.get_or_synthesize("m1", "Reply", synthesize)

    assert synthesize.calls == [("m1", "Reply")]
    assert again.audio == b"m1:Reply"


def test_changed_text_is_synthesized_again() -> None:
    cache = SpeechCache({})
    synthesize = RecordingSynthesizer()

    cache.get_or_synthesize("m1", "Old text", synthesize)
    updated = cache.get_or_synthesize("m1", "New text", synthesize)

    assert updated.audio == b"m1:New text"
    assert len(synthesize.calls) == 2


def test_failed_synthesis_stores_nothing() -> None:
    store: dict = {}

    with pytest.raises(RuntimeError):
        SpeechCache(store).get_or_synthesize("m1", "Reply", RecordingSynthesizer(error=RuntimeError("down")))

    assert store == {}

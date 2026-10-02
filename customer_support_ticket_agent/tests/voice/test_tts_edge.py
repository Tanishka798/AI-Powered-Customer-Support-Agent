import pytest

from src.voice import tts_edge
from src.voice.tts_edge import EdgeTTS


def fake_communicate(chunks=(), error: Exception | None = None):
    """Replacement for edge_tts.Communicate that streams the given chunks, then optionally fails."""

    class FakeCommunicate:
        calls: list[tuple[str, str]] = []

        def __init__(self, text: str, voice: str) -> None:
            FakeCommunicate.calls.append((text, voice))

        async def stream(self):
            for chunk in chunks:
                yield chunk
            if error:
                raise error

    return FakeCommunicate


@pytest.mark.asyncio
async def test_audio_chunks_are_joined_as_mp3(monkeypatch) -> None:
    communicate = fake_communicate(
        [{"type": "audio", "data": b"ab"}, {"type": "WordBoundary"}, {"type": "audio", "data": b"cd"}]
    )
    monkeypatch.setattr(tts_edge.edge_tts, "Communicate", communicate)

    audio, media_type = await EdgeTTS("en-US-AriaNeural").synthesize("Hello there")

    assert (audio, media_type) == (b"abcd", "audio/mpeg")
    assert communicate.calls == [("Hello there", "en-US-AriaNeural")]


@pytest.mark.asyncio
async def test_provider_error_becomes_runtime_error(monkeypatch) -> None:
    monkeypatch.setattr(tts_edge.edge_tts, "Communicate", fake_communicate(error=ConnectionError("offline")))

    with pytest.raises(RuntimeError, match="Edge TTS synthesis failed"):
        await EdgeTTS("en-US-AriaNeural").synthesize("Hello")


@pytest.mark.asyncio
async def test_no_audio_is_a_provider_failure(monkeypatch) -> None:
    monkeypatch.setattr(tts_edge.edge_tts, "Communicate", fake_communicate([{"type": "WordBoundary"}]))

    with pytest.raises(RuntimeError, match="returned no audio"):
        await EdgeTTS("en-US-AriaNeural").synthesize("Hello")

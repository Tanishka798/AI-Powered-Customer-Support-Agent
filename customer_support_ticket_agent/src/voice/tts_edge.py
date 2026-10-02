"""Text-to-speech with Microsoft Edge's online neural voices (edge-tts)."""

from __future__ import annotations

import edge_tts

from src.voice.contracts import TTSService

MEDIA_TYPE = "audio/mpeg"


class EdgeTTS(TTSService):
    """Streams MP3 audio for the given text; needs internet access but no API key."""

    def __init__(self, voice: str) -> None:
        self.voice = voice

    async def initialize(self) -> None:
        # Nothing to load; each request opens its own connection.
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        try:
            audio = b"".join(
                [
                    chunk["data"]
                    async for chunk in edge_tts.Communicate(text, self.voice).stream()
                    if chunk["type"] == "audio"
                ]
            )
        except Exception as exc:
            raise RuntimeError(f"Edge TTS synthesis failed: {exc}") from exc

        if not audio:
            raise RuntimeError("Edge TTS returned no audio")
        return audio, MEDIA_TYPE

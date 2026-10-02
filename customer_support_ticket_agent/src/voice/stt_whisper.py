"""Local speech-to-text with faster-whisper."""

from __future__ import annotations

import asyncio
from io import BytesIO

from faster_whisper import WhisperModel, decode_audio

from src.voice.contracts import STTService
from src.voice.errors import VoiceInputError

SUPPORTED_MEDIA_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/webm",
    "audio/ogg",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/flac",
}


def base_media_type(media_type: str) -> str:
    """"audio/webm;codecs=opus" -> "audio/webm"."""
    return media_type.split(";", 1)[0].strip().lower()


class FasterWhisperSTT(STTService):
    """Runs a Whisper model locally; no audio leaves the machine."""

    def __init__(self, model_name: str, device: str, compute_type: str, language: str) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: WhisperModel | None = None

    async def initialize(self) -> None:
        # Loading (and first-run download) blocks, so keep it off the event loop.
        self._model = await asyncio.to_thread(
            WhisperModel, self.model_name, device=self.device, compute_type=self.compute_type
        )

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        if self._model is None:
            raise RuntimeError("Speech-to-text model is not initialized")
        if base_media_type(media_type) not in SUPPORTED_MEDIA_TYPES:
            raise VoiceInputError(f"Unsupported audio format: {media_type or 'unknown'}")
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    async def cleanup(self) -> None:
        self._model = None

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        samples = decode_samples(audio_bytes)
        # The voice-activity filter drops silence, so noise-only audio yields no text.
        segments, _ = self._model.transcribe(samples, language=self.language, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments)


def decode_samples(audio_bytes: bytes):
    """Decode any supported container to 16 kHz mono samples, or raise VoiceInputError."""
    try:
        samples = decode_audio(BytesIO(audio_bytes))
    except Exception as exc:
        raise VoiceInputError("The audio could not be read. Please record again.") from exc
    if samples.size == 0:
        raise VoiceInputError("The recording contains no audio. Please record again.")
    return samples

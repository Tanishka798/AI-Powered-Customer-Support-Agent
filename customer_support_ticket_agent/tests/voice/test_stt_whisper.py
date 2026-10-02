import io
import wave
from types import SimpleNamespace

import pytest

from src.voice.errors import VoiceInputError
from src.voice.stt_whisper import FasterWhisperSTT, base_media_type


def silent_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakeWhisperModel:
    """Stands in for WhisperModel so no model download is needed."""

    def __init__(self, *texts: str) -> None:
        self.texts = texts
        self.sample_count = None

    def transcribe(self, samples, language, vad_filter):
        self.sample_count = samples.size
        return [SimpleNamespace(text=text) for text in self.texts], None


def stt_with(model: FakeWhisperModel) -> FasterWhisperSTT:
    stt = FasterWhisperSTT("base.en", "cpu", "int8", "en")
    stt._model = model
    return stt


@pytest.mark.parametrize(
    "media_type, expected",
    [("audio/webm;codecs=opus", "audio/webm"), (" Audio/WAV ", "audio/wav"), ("", "")],
)
def test_base_media_type(media_type: str, expected: str) -> None:
    assert base_media_type(media_type) == expected


@pytest.mark.asyncio
async def test_real_wav_is_decoded_and_segments_joined() -> None:
    model = FakeWhisperModel(" My payment ", " was charged twice ")

    transcript = await stt_with(model).transcribe(silent_wav(), "audio/wav")

    assert transcript == "My payment was charged twice"
    assert model.sample_count == 8000  # 0.5 s at 16 kHz


@pytest.mark.asyncio
async def test_unsupported_format_is_rejected() -> None:
    with pytest.raises(VoiceInputError, match="Unsupported audio format: text/plain"):
        await stt_with(FakeWhisperModel()).transcribe(b"hello", "text/plain")


@pytest.mark.asyncio
async def test_unreadable_audio_is_rejected() -> None:
    with pytest.raises(VoiceInputError, match="could not be read"):
        await stt_with(FakeWhisperModel()).transcribe(b"definitely not audio", "audio/wav")


@pytest.mark.asyncio
async def test_transcribe_before_initialize_fails() -> None:
    with pytest.raises(RuntimeError, match="not initialized"):
        await FasterWhisperSTT("base.en", "cpu", "int8", "en").transcribe(silent_wav(), "audio/wav")

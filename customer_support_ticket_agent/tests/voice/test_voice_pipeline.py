import asyncio

import pytest

from src.voice.contracts import STTService, TTSService
from src.voice.pipeline import VoicePipeline
from tests.fakes import FakeSTT as ScriptedSTT
from tests.fakes import FakeTTS as ScriptedTTS


# --- Supplied with the mid-session requirements (import path updated only) ---


class FakeSTT(STTService):
    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        return "My payment was charged twice"


class FakeTTS(TTSService):
    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        return b"fake-audio", "audio/mpeg"


@pytest.mark.asyncio
async def test_voice_pipeline_contracts() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())

    transcript, _ = await pipeline.transcribe(b"fake-input", "audio/wav")
    audio, media_type, _ = await pipeline.synthesize("Agent response")

    assert transcript == "My payment was charged twice"
    assert audio == b"fake-audio"
    assert media_type == "audio/mpeg"


# --- Additional pipeline behaviour ---


class SlowSTT(ScriptedSTT):
    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        await asyncio.sleep(0.05)
        return await super().transcribe(audio_bytes, media_type)


class SlowTTS(ScriptedTTS):
    async def synthesize(self, text: str) -> tuple[bytes, str]:
        await asyncio.sleep(0.05)
        return await super().synthesize(text)


@pytest.mark.asyncio
async def test_timing_reflects_adapter_duration() -> None:
    pipeline = VoicePipeline(SlowSTT(), SlowTTS())

    _, stt_ms = await pipeline.transcribe(b"audio", "audio/wav")
    _, media_type, tts_ms = await pipeline.synthesize("Hello")

    assert isinstance(stt_ms, int) and stt_ms >= 50
    assert isinstance(tts_ms, int) and tts_ms >= 50
    assert media_type == "audio/mpeg"


@pytest.mark.asyncio
async def test_transcript_is_trimmed_and_media_type_passed_through() -> None:
    stt = ScriptedSTT(transcript="  hello there  ")

    transcript, _ = await VoicePipeline(stt, ScriptedTTS()).transcribe(b"audio", "audio/webm")

    assert transcript == "hello there"
    assert stt.calls == [(b"audio", "audio/webm")]


@pytest.mark.asyncio
async def test_empty_audio_is_rejected_before_the_adapter() -> None:
    stt = ScriptedSTT()

    with pytest.raises(ValueError, match="Audio input is empty"):
        await VoicePipeline(stt, ScriptedTTS()).transcribe(b"", "audio/wav")
    assert stt.calls == []


@pytest.mark.asyncio
async def test_speech_free_audio_is_rejected() -> None:
    with pytest.raises(ValueError, match="No understandable speech"):
        await VoicePipeline(ScriptedSTT(transcript="   "), ScriptedTTS()).transcribe(b"audio", "audio/wav")


@pytest.mark.asyncio
async def test_blank_text_is_rejected_before_the_adapter() -> None:
    tts = ScriptedTTS()

    with pytest.raises(ValueError, match="Text input is empty"):
        await VoicePipeline(ScriptedSTT(), tts).synthesize("   ")
    assert tts.texts == []


@pytest.mark.asyncio
async def test_adapter_failures_propagate_unchanged() -> None:
    pipeline = VoicePipeline(ScriptedSTT(error=RuntimeError("stt down")), ScriptedTTS(error=RuntimeError("tts down")))

    with pytest.raises(RuntimeError, match="stt down"):
        await pipeline.transcribe(b"audio", "audio/wav")
    with pytest.raises(RuntimeError, match="tts down"):
        await pipeline.synthesize("Hello")

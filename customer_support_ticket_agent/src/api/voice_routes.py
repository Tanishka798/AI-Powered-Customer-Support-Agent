"""Voice endpoints: speech to text for input, text to speech for playback.

They sit beside ``POST /chat`` and never call the agent: a transcript only
reaches the agent when the customer confirms it and the UI sends it to /chat.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from src.voice.models import SynthesisRequest, TranscriptionResponse, VoiceErrorResponse
from src.voice.pipeline import VoicePipeline

logger = logging.getLogger(__name__)

MAX_AUDIO_BYTES = 10 * 1024 * 1024

router = APIRouter(prefix="/voice", tags=["voice"])

ERROR_RESPONSES = {
    status: {"model": VoiceErrorResponse} for status in (413, 422, 502, 503)
}


def voice_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=VoiceErrorResponse(error=message).model_dump())


def ready_voice_pipeline(request: Request) -> VoicePipeline | None:
    """The app's voice pipeline, or None if it failed to start."""
    if not getattr(request.app.state, "voice_ready", False):
        return None
    return request.app.state.voice


@router.post("/transcribe", response_model=TranscriptionResponse, responses=ERROR_RESPONSES)
async def transcribe(request: Request, audio: UploadFile = File(...)):
    voice = ready_voice_pipeline(request)
    if voice is None:
        return voice_error(503, "Voice input is unavailable right now. Please type your message.")

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return voice_error(413, "The recording is too long. Please keep it under a minute or two.")

    try:
        transcript, elapsed_ms = await voice.transcribe(audio_bytes, audio.content_type or "")
    except ValueError as exc:
        # Empty, unsupported, unreadable, or speech-free audio.
        return voice_error(422, str(exc))
    except Exception:
        logger.exception("Speech-to-text failed")
        return voice_error(502, "Transcription failed. Please try again or type your message.")

    return TranscriptionResponse(transcript=transcript, processing_time_ms=elapsed_ms)


@router.post("/synthesize", response_class=Response, responses=ERROR_RESPONSES)
async def synthesize(request: Request, body: SynthesisRequest):
    voice = ready_voice_pipeline(request)
    if voice is None:
        return voice_error(503, "Audio playback is unavailable right now.")

    try:
        audio, media_type, elapsed_ms = await voice.synthesize(body.text)
    except ValueError as exc:
        return voice_error(422, str(exc))
    except Exception:
        logger.exception("Text-to-speech failed for message %s", body.message_id)
        return voice_error(502, "Audio could not be generated. Please try again.")

    # Echo the message ID so the client can tie this audio to its chat message.
    headers = {"X-Message-Id": body.message_id, "X-Processing-Time-Ms": str(elapsed_ms)}
    return Response(content=audio, media_type=media_type, headers=headers)

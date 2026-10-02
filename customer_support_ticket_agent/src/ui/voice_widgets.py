"""Voice in the chat box (record -> editable transcript) and per-response playback."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from src.ui.api_client import ApiError
from src.ui.speech_cache import SpeechCache, Synthesizer

Transcriber = Callable[[bytes, str], str]  # (audio bytes, media type) -> transcript


def init_voice_state() -> None:
    defaults = {
        "speech_store": {},  # message_id -> Speech
        "speech_errors": {},  # message_id -> error text
        "speaking_id": None,  # message being synthesized; disables every speaker button
        "selected_audio_id": None,  # the one response whose player is shown
        "autoplay_id": None,  # play once, right after the click
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------
# Speaker icon on each agent response
# ---------------------------------------------------------


def render_speaker(message: dict, synthesize: Synthesizer) -> None:
    """One speaker button per response; audio is generated only on click and reused afterwards."""
    message_id = message["id"]
    st.button(
        "🔊",
        key=f"speak-{message_id}",
        help="Listen to this response",
        disabled=st.session_state.speaking_id is not None,
        on_click=_start_speaking,
        args=(message_id,),
    )

    if st.session_state.speaking_id == message_id:
        _synthesize_now(message, synthesize)

    if error := st.session_state.speech_errors.get(message_id):
        st.warning(error)

    speech = SpeechCache(st.session_state.speech_store).get(message_id)
    if speech is not None and st.session_state.selected_audio_id == message_id:
        autoplay = st.session_state.autoplay_id == message_id
        st.session_state.autoplay_id = None
        st.audio(speech.audio, format=speech.media_type, autoplay=autoplay)


def _start_speaking(message_id: str) -> None:
    st.session_state.speaking_id = message_id
    st.session_state.speech_errors.pop(message_id, None)


def _synthesize_now(message: dict, synthesize: Synthesizer) -> None:
    message_id = message["id"]
    try:
        with st.spinner("Generating audio..."):
            SpeechCache(st.session_state.speech_store).get_or_synthesize(
                message_id, message["content"], synthesize
            )
    except ApiError as exc:
        st.session_state.speech_errors[message_id] = f"Couldn't play this response: {exc}"
    except Exception:
        st.session_state.speech_errors[message_id] = "Couldn't play this response. Please try again."
    else:
        st.session_state.selected_audio_id = message_id
        st.session_state.autoplay_id = message_id
    finally:
        st.session_state.speaking_id = None
    # Re-render so the speaker buttons are enabled again and the player appears.
    st.rerun()


# ---------------------------------------------------------
# Mic inside the chat box -> transcript placed back in the box -> user sends
# ---------------------------------------------------------


@dataclass(frozen=True)
class ChatSubmission:
    """What to do with one chat-box submission."""

    send: str | None = None  # text to send to the agent now
    prefill: str | None = None  # transcript to put back in the box for editing
    error: str | None = None  # customer-safe transcription error


def has_audio(value: object) -> bool:
    return getattr(value, "audio", None) is not None


def resolve_submission(value: object, transcribe: Transcriber) -> ChatSubmission:
    """Typed text is sent as-is; a recording is transcribed and returned for editing, never sent directly."""
    if value is None:
        return ChatSubmission()
    text = (value if isinstance(value, str) else getattr(value, "text", "") or "").strip()
    audio = getattr(value, "audio", None)

    if audio is None:
        return ChatSubmission(send=text or None)

    try:
        transcript = transcribe(audio.getvalue(), audio.type or "audio/wav")
    except ApiError as exc:
        return ChatSubmission(prefill=text or None, error=str(exc))
    except Exception:
        return ChatSubmission(
            prefill=text or None, error="Transcription failed. Please try again or type your message."
        )
    # Keep anything already typed, followed by the transcript.
    return ChatSubmission(prefill=" ".join(part for part in (text, transcript) if part))

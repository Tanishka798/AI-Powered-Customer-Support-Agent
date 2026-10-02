"""Per-message audio cache, so each chat response is synthesized at most once per session."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

Synthesizer = Callable[[str, str], tuple[bytes, str]]  # (message_id, text) -> (audio, media_type)


@dataclass(frozen=True)
class Speech:
    text: str
    audio: bytes
    media_type: str


class SpeechCache:
    """Audio keyed by chat-message ID, stored in a caller-owned mapping (e.g. Streamlit session state)."""

    def __init__(self, store: MutableMapping[str, Speech]) -> None:
        self._store = store

    def get(self, message_id: str) -> Speech | None:
        return self._store.get(message_id)

    def get_or_synthesize(self, message_id: str, text: str, synthesize: Synthesizer) -> Speech:
        """Reuse this message's audio, or synthesize its exact text once and remember it."""
        cached = self._store.get(message_id)
        if cached is not None and cached.text == text:
            return cached

        audio, media_type = synthesize(message_id, text)
        speech = Speech(text=text, audio=audio, media_type=media_type)
        self._store[message_id] = speech
        return speech

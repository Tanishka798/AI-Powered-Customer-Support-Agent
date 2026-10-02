"""Deterministic test doubles shared by the component test folders."""

from __future__ import annotations

import json

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from src.config import Settings
from src.llm.workflow import build_support_workflow
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.voice.contracts import STTService, TTSService


def make_settings(**overrides) -> Settings:
    values = dict(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=".data/test-vector-db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )
    values.update(overrides)
    return Settings(**values)


def decision(route: str, **fields: str) -> dict:
    """A decision-model reply: the route plus any extracted ticket fields."""
    return {"route": route, **fields}


class FakeChatModel:
    """Returns scripted replies in order and records every user prompt.

    A dict reply is sent as JSON, a list as content blocks, and an exception
    is raised instead of replying.
    """

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=json.dumps(reply) if isinstance(reply, dict) else reply)


class FakeRetriever:
    """Returns the shipping policy for shipping questions and nothing otherwise."""

    SHIPPING_CHUNK = {"content": "Standard delivery takes 3-5 business days.", "source": "shipping.md"}

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        return [dict(self.SHIPPING_CHUNK)] if "ship" in query.lower() else []


class KeywordEmbeddings(Embeddings):
    """Tiny deterministic embeddings: one normalized dimension per topic keyword."""

    TOPICS = ["ship", "deliver", "return", "refund", "password", "account", "card", "charge"]

    def _embed(self, text: str) -> list[float]:
        text = text.lower()
        vector = [float(topic in text) for topic in self.TOPICS] + [0.01]
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class WorkflowHarness:
    """Drives the compiled workflow one turn at a time, recording history like the pipeline."""

    def __init__(self, *replies) -> None:
        self.model = FakeChatModel(*replies)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = build_support_workflow(self.model, FakeRetriever(), self.sessions, self.tickets)

    async def turn(self, message: str, session_id: str = "session-1") -> dict:
        session = self.sessions.get_or_create(session_id)
        result = await self.workflow.ainvoke({"session_id": session_id, "customer_message": message})
        session.history.append({"role": "user", "content": message})
        session.history.append({"role": "assistant", "content": result["response_text"]})
        return result


class FakeSTT(STTService):
    """Returns a fixed transcript, or raises the given error."""

    def __init__(self, transcript: str = "My payment was charged twice", error: Exception | None = None) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[tuple[bytes, str]] = []

    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        self.calls.append((audio_bytes, media_type))
        if self.error:
            raise self.error
        return self.transcript


class FakeTTS(TTSService):
    """Returns audio derived from the text, so tests can tell which text produced which audio."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.texts: list[str] = []

    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        self.texts.append(text)
        if self.error:
            raise self.error
        return b"audio:" + text.encode(), "audio/mpeg"

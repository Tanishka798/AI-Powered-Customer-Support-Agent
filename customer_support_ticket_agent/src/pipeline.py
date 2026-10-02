from __future__ import annotations

import logging
from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError

logger = logging.getLogger(__name__)


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared components once during FastAPI startup."""
        # Retrieval must be ready before a workflow that accepts traffic exists.
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            model=self.model,
            retriever=self.retriever,
            sessions=self.sessions,
            tickets=self.tickets,
        )
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        """Process one complete customer turn."""
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        session_id, message = self._clean_input(session_id, message)
        session = self.sessions.get_or_create(session_id)

        result = await self._run_workflow(session_id, message, session.history)
        response = self._build_response(session_id, result)

        # Keep history atomic so a failed request cannot leave an orphaned turn.
        session.history.append({"role": "user", "content": message})
        session.history.append({"role": "assistant", "content": response.response})
        return response

    @staticmethod
    def _clean_input(session_id: str, message: str) -> tuple[str, str]:
        session_id, message = session_id.strip(), message.strip()
        if not session_id:
            raise AgentProcessingError("session_id must not be blank")
        if not message:
            raise AgentProcessingError("message must not be blank")
        return session_id, message

    async def _run_workflow(
        self, session_id: str, message: str, history: list[dict[str, str]]
    ) -> dict:
        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in history[-8:]
            if turn.get("role") in {"user", "assistant"}
        ]
        messages.append({"role": "user", "content": message})
        initial_state = {
            "session_id": session_id,
            "customer_message": message,
            "messages": messages,
            "retrieved_chunks": [],
            "route": "",
            "extracted_fields": {},
            "response_text": "",
            "sources": [],
            "ticket_id": None,
        }
        try:
            return await self.workflow.ainvoke(initial_state)
        except ComponentNotReadyError:
            raise
        except Exception as exc:
            # Log details server-side only; the client gets a generic message
            # so provider errors, URLs, and paths are never exposed.
            logger.exception("Support workflow failed for session %s", session_id)
            raise AgentProcessingError(
                "The support agent could not process this request. Please try again."
            ) from exc

    def _build_response(self, session_id: str, result: dict) -> ChatResponse:
        response_text = result.get("response_text")
        if not isinstance(response_text, str) or not response_text.strip():
            raise AgentProcessingError("The support agent returned an empty response.")

        # Never return a ticket ID unless it really exists in the repository.
        ticket_id = result.get("ticket_id")
        if ticket_id is not None and self.tickets.get(ticket_id) is None:
            ticket_id = None

        return ChatResponse(
            session_id=session_id,
            response=response_text.strip(),
            sources=result.get("sources", []),
            ticket_id=ticket_id,
        )

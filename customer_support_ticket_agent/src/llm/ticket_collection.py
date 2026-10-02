"""Collect ticket details over several turns and create the ticket once they validate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import TypeAdapter, ValidationError

from src.llm.prompts import FIELD_PROMPTS, INVALID_TICKET_TEXT, RETRY_PROMPTS
from src.models import TicketCreate
from src.sessions.store import ConversationState
from src.tools.ticket_tool import TicketRepository, create_ticket_tool

SUMMARY_MAX_LENGTH = 160

# One validator per ticket field, built from TicketCreate's own constraints,
# so a single value can be checked the moment the customer gives it.
FIELD_VALIDATORS = {
    name: TypeAdapter(Annotated[info.annotation, info])
    for name, info in TicketCreate.model_fields.items()
    if name in FIELD_PROMPTS
}


@dataclass(frozen=True)
class TicketReply:
    text: str
    ticket_id: str | None = None


def with_ticket_reminder(text: str, session: ConversationState) -> str:
    """Re-ask the pending ticket question after answering a side question."""
    if session.ticket_in_progress() and session.awaiting_field in FIELD_PROMPTS:
        return f"{text}\n\nTo continue with your support ticket: {FIELD_PROMPTS[session.awaiting_field]}"
    return text


class TicketCollector:
    """Runs one ticket-collection turn for a session."""

    def __init__(self, tickets: TicketRepository) -> None:
        self.tickets = tickets

    def handle_turn(
        self, session_id: str, session: ConversationState, fields: dict[str, str]
    ) -> TicketReply:
        if existing := self._existing_ticket_reply(session):
            return existing

        valid, invalid = self._split_by_validity(fields)
        self._commit_fields(session, valid)

        if invalid:
            session.awaiting_field = invalid[0]
            return TicketReply(RETRY_PROMPTS[invalid[0]])

        if question := self._ask_next_missing_field(session):
            return TicketReply(question)

        try:
            request = self._build_request(session)
        except ValidationError as exc:
            return TicketReply(self._reject_invalid_fields(session, exc))

        return self._create_ticket(session_id, session, request)

    def _existing_ticket_reply(self, session: ConversationState) -> TicketReply | None:
        """Never create a second ticket for a session that already has one."""
        if not session.ticket_id:
            return None
        if self.tickets.get(session.ticket_id) is None:
            # Recover if session state references a ticket the repository lacks.
            session.ticket_id = None
            return None
        return TicketReply(
            f"Your support ticket is already created. Your ticket ID is {session.ticket_id}.",
            session.ticket_id,
        )

    @staticmethod
    def _split_by_validity(fields: dict[str, str]) -> tuple[dict[str, str], list[str]]:
        """Check each new value against its ticket field (e.g. a too-short description)."""
        valid: dict[str, str] = {}
        invalid: list[str] = []
        for name, value in fields.items():
            try:
                FIELD_VALIDATORS[name].validate_python(value)
            except ValidationError:
                invalid.append(name)
            else:
                valid[name] = value
        return valid, invalid

    @staticmethod
    def _commit_fields(session: ConversationState, fields: dict[str, str]) -> None:
        """Store already-validated values supplied this turn."""
        for name, value in fields.items():
            setattr(session, name, value)

    @staticmethod
    def _ask_next_missing_field(session: ConversationState) -> str | None:
        missing = session.missing_ticket_fields()
        if not missing:
            return None
        session.awaiting_field = missing[0]
        return FIELD_PROMPTS[missing[0]]

    @staticmethod
    def _build_request(session: ConversationState) -> TicketCreate:
        """Final authoritative validation; raises ValidationError."""
        summary = session.issue_description.strip()
        if len(summary) > SUMMARY_MAX_LENGTH:
            summary = summary[: SUMMARY_MAX_LENGTH - 3].rstrip() + "..."
        return TicketCreate(
            customer_name=session.customer_name,
            customer_email=session.customer_email,
            issue_description=session.issue_description,
            category=session.category,
            summary=summary,
        )

    @staticmethod
    def _reject_invalid_fields(session: ConversationState, exc: ValidationError) -> str:
        """Clear every invalid value and ask for the first one again."""
        invalid = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
        if "summary" in invalid:  # The summary is derived from the description.
            invalid.add("issue_description")

        invalid_fields = [name for name in RETRY_PROMPTS if name in invalid]
        for name in invalid_fields:
            setattr(session, name, None)

        session.awaiting_field = invalid_fields[0] if invalid_fields else None
        return RETRY_PROMPTS.get(session.awaiting_field, INVALID_TICKET_TEXT)

    def _create_ticket(
        self, session_id: str, session: ConversationState, request: TicketCreate
    ) -> TicketReply:
        """Invoke the session-bound tool; only a returned ID is stored."""
        try:
            ticket_id = create_ticket_tool(self.tickets, session_id).invoke(request.model_dump(mode="json"))
        except Exception as exc:
            raise RuntimeError(f"Ticket creation failed: {exc}") from exc

        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise RuntimeError("Ticket tool returned an invalid ticket identifier")

        session.ticket_id = ticket_id
        session.awaiting_field = None
        return TicketReply(
            f"Your support ticket has been created successfully. Your ticket ID is {ticket_id}.",
            ticket_id,
        )

"""Decide a turn's route and extract validated ticket fields from the customer message."""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from src.llm.output_parsing import message_text, parse_json_object
from src.llm.prompts import DECISION_SYSTEM_PROMPT, DECISION_USER_TEMPLATE
from src.sessions.store import ConversationState

Route = Literal["greeting", "answer", "ticket", "ticket_lookup", "clarify"]

TICKET_FIELDS = ("customer_name", "customer_email", "issue_description", "category")
TICKET_CATEGORIES = {"order", "payment", "account", "technical", "other"}
# Fields the model may paraphrase; all others must appear verbatim in the message.
FREE_TEXT_FIELDS = {"issue_description"}
_EMAIL_PATTERN = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|howdy|good morning|good afternoon|good evening)(?: there)?[!. ]*$",
    re.IGNORECASE,
)
_QUESTION_WORDS = {
    "how", "what", "when", "where", "why", "who", "which",
    "can", "could", "do", "does", "is", "are", "will", "would", "should",
}


class DecisionResult(BaseModel):
    """Controlled model output: route for this turn plus any extracted fields."""

    route: Route
    customer_name: str | None = None
    customer_email: str | None = None
    issue_description: str | None = None
    category: str | None = None


async def decide_turn(
    model: BaseChatModel, customer_message: str, session: ConversationState
) -> tuple[Route, dict[str, str]]:
    """Return the route for this turn and the validated fields it supplies."""
    if _GREETING_PATTERN.fullmatch(customer_message.strip()):
        return "greeting", {}
    if session.ticket_id and "ticket" in customer_message.lower():
        return "ticket_lookup", {}

    result = await request_decision(model, customer_message, session)
    extracted = clean_extracted_fields(result.model_dump(include=set(TICKET_FIELDS)))
    extracted = new_grounded_fields(extracted, customer_message, session)

    if not session.ticket_in_progress():
        return result.route, extracted

    extracted = {**extract_pending_reply(session, customer_message), **extracted}
    # Supplying ticket details keeps the turn in collection mode;
    # otherwise a separate policy question may still be answered.
    return ("ticket" if extracted else result.route), extracted


async def request_decision(
    model: BaseChatModel, customer_message: str, session: ConversationState
) -> DecisionResult:
    """Ask the model for a small controlled decision object.

    The model only interprets the current customer turn. It does not create
    tickets or decide repository-issued IDs.
    """
    user_prompt = DECISION_USER_TEMPLATE.format(
        session=json.dumps(_session_snapshot(session), indent=2),
        previous_reply=session.last_assistant_message() or "(none)",
        conversation_history=_format_history(session.history),
        message=customer_message,
    )
    try:
        response = await model.ainvoke(
            [SystemMessage(content=DECISION_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
        )
    except Exception as exc:
        raise RuntimeError(f"Decision model invocation failed: {exc}") from exc

    try:
        return DecisionResult.model_validate(parse_json_object(message_text(response)))
    except (ValueError, ValidationError) as exc:
        raise RuntimeError("Decision model returned an invalid routing response") from exc


def clean_extracted_fields(fields: dict[str, str | None]) -> dict[str, str]:
    """Normalize model strings while leaving authoritative validation to the collector."""
    cleaned: dict[str, str] = {}

    if name := _clean(fields.get("customer_name")):
        cleaned["customer_name"] = name

    email_value = fields.get("customer_email")
    if isinstance(email_value, str):
        cleaned["customer_email"] = email_value.strip()

    if description := _clean(fields.get("issue_description")):
        cleaned["issue_description"] = description

    if category := _clean(fields.get("category")):
        cleaned["category"] = category.lower() if category.lower() in TICKET_CATEGORIES else category

    return cleaned


def new_grounded_fields(
    fields: dict[str, str], customer_message: str, session: ConversationState
) -> dict[str, str]:
    """Drop values the model copied or invented.

    Small models often echo the session snapshot or guess a category, so a
    name, email, or category must appear in the customer's own message, and
    any value already stored in the session is not new information.
    """
    return {
        name: value
        for name, value in fields.items()
        if getattr(session, name) != value and _is_grounded(name, value, customer_message, session)
    }


def _is_grounded(name: str, value: str, message: str, session: ConversationState) -> bool:
    if name in FREE_TEXT_FIELDS:
        return True
    if name == "customer_email" and not value:
        return "email" in message.lower()
    if name == "category" and session.awaiting_field != "category":
        # "my order" mentions a category word without choosing one.
        return _states_category(message, value)
    return _mentions(message, value)


def extract_pending_reply(session: ConversationState, customer_message: str) -> dict[str, str]:
    """Deterministically read a bare email, category, or description reply the model may miss."""
    if session.awaiting_field == "customer_email":
        if match := _EMAIL_PATTERN.search(customer_message):
            return clean_extracted_fields({"customer_email": match.group(0).rstrip(".,;")})
        if not looks_like_question(customer_message) and customer_message.strip():
            return {"customer_email": customer_message.strip()}
    if session.awaiting_field == "category":
        return clean_extracted_fields({"category": customer_message})
    if session.awaiting_field == "issue_description" and not looks_like_question(customer_message):
        # Small models route vague answers like "the product was slow" to "answer".
        return clean_extracted_fields({"issue_description": customer_message})
    return {}


def looks_like_question(message: str) -> bool:
    """A side question ("How long do refunds take?") rather than a reply to ours."""
    text = message.strip().lower()
    return text.endswith("?") or text.split(" ", 1)[0] in _QUESTION_WORDS


def _session_snapshot(session: ConversationState) -> dict:
    return {
        **{name: getattr(session, name) for name in TICKET_FIELDS},
        "ticket_id": session.ticket_id,
        "ticket_in_progress": session.ticket_in_progress(),
        "awaiting_field": session.awaiting_field,
    }


def _mentions(message: str, value: str) -> bool:
    """Case-insensitive whole-word match, so "other" doesn't match "another"."""
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", message, flags=re.IGNORECASE) is not None


def _states_category(message: str, category: str) -> bool:
    """True for explicit choices like "a technical issue" or "category: payment"."""
    word = re.escape(category)
    pattern = rf"(?<!\w){word}\s+(issue|problem|question)(?!\w)|(?<!\w)category(?!\w)[^.?!]*(?<!\w){word}(?!\w)"
    return re.search(pattern, message, flags=re.IGNORECASE) is not None


def _clean(value: object) -> str | None:
    return (value.strip() or None) if isinstance(value, str) else None


def _format_history(history: list[dict[str, str]], limit: int = 8) -> str:
    recent = history[-limit:]
    if not recent:
        return "(none)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)

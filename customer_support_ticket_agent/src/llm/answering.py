"""Generate customer answers grounded only in retrieved knowledge chunks."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.output_parsing import message_text
from src.llm.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    CLARIFICATION_SYSTEM_PROMPT,
    CONVERSATION_USER_TEMPLATE,
    GREETING_SYSTEM_PROMPT,
)


def format_context(chunks: list[dict[str, str]]) -> str:
    return "\n\n---\n\n".join(
        f"Source: {chunk['source']}\nContent:\n{chunk['content']}" for chunk in chunks
    )


async def generate_answer(
    model: BaseChatModel,
    chunks: list[dict[str, str]],
    customer_message: str,
    history: list[dict[str, str]] | None = None,
    mode: str = "answer",
) -> str:
    """Generate a grounded answer or a conversational greeting/clarification."""
    history_text = _format_history(history or [])
    if mode == "answer":
        system_prompt = ANSWER_SYSTEM_PROMPT
        user_prompt = ANSWER_USER_TEMPLATE.format(
            conversation_history=history_text,
            context=format_context(chunks),
            message=customer_message,
        )
    elif mode == "greeting":
        system_prompt = GREETING_SYSTEM_PROMPT
        user_prompt = CONVERSATION_USER_TEMPLATE.format(
            conversation_history=history_text, message=customer_message
        )
    elif mode == "clarify":
        system_prompt = CLARIFICATION_SYSTEM_PROMPT
        user_prompt = CONVERSATION_USER_TEMPLATE.format(
            conversation_history=history_text, message=customer_message
        )
    else:
        raise ValueError(f"Unsupported response mode: {mode}")

    try:
        response = await model.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except Exception as exc:
        raise RuntimeError(f"Answer model invocation failed: {exc}") from exc

    answer = message_text(response)
    if not answer:
        raise RuntimeError("Answer model returned an empty response")
    return answer


def _format_history(history: list[dict[str, str]], limit: int = 8) -> str:
    recent = history[-limit:]
    if not recent:
        return "(none)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)

"""Turn raw chat-model output into text or JSON objects."""

from __future__ import annotations

import json
import re

from langchain_core.messages import BaseMessage

_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)


def message_text(message: BaseMessage) -> str:
    """Flatten model output, including content-block lists, into stripped text."""
    content = message.content
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        raise RuntimeError("Model returned unsupported content")
    return content.strip()


def parse_json_object(text: str) -> dict:
    """Parse a JSON object even when a local model wraps it in markdown or prose."""
    for candidate in _json_candidates(text.strip()):
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    raise ValueError("Model did not return a valid JSON object")


def _json_candidates(text: str) -> list[str]:
    """The whole text, a fenced block, then the outermost braces, in that order."""
    candidates = [text]
    if fenced := _FENCED_JSON.search(text):
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    return candidates

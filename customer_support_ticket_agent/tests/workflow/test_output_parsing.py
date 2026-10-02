import pytest
from langchain_core.messages import AIMessage

from src.llm.output_parsing import message_text, parse_json_object


@pytest.mark.parametrize(
    "text",
    [
        '{"route": "answer"}',
        '```json\n{"route": "answer"}\n```',
        'Sure! Here it is: {"route": "answer"} Hope that helps.',
    ],
)
def test_parse_json_accepts_plain_fenced_and_wrapped_output(text: str) -> None:
    assert parse_json_object(text) == {"route": "answer"}


@pytest.mark.parametrize("text", ["no json here", "[1, 2, 3]", "{broken"])
def test_parse_json_rejects_non_objects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_message_text_strips_plain_content() -> None:
    assert message_text(AIMessage(content="  Hello  ")) == "Hello"


def test_message_text_flattens_content_blocks() -> None:
    message = AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}])

    assert message_text(message) == "Hello there"

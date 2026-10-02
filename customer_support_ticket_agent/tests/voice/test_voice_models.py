import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.voice.models import SynthesisRequest, TranscriptionResponse, VoiceErrorResponse

EXAMPLES = json.loads((Path(__file__).parent / "example_responses.json").read_text())


def test_supplied_transcription_example_matches_the_model() -> None:
    example = EXAMPLES["transcription"]

    assert TranscriptionResponse(**example).model_dump() == example


def test_supplied_error_example_matches_the_model() -> None:
    example = EXAMPLES["transcription_error"]

    assert VoiceErrorResponse(**example).model_dump() == example


@pytest.mark.parametrize(
    "payload",
    [{"message_id": "", "text": "Hi"}, {"message_id": "m1", "text": ""}, {"message_id": "m1", "text": "x" * 4001}],
)
def test_synthesis_request_rejects_invalid_input(payload: dict) -> None:
    with pytest.raises(ValidationError):
        SynthesisRequest(**payload)

"""HTTP client for the support API that returns plain data or customer-safe errors."""

from urllib.parse import quote

import httpx

REQUEST_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)


class ApiError(Exception):
    """A failure with a message that is safe and useful to show the customer."""


def send_message(base_url: str, session_id: str, message: str) -> dict:
    """POST one chat turn and return the assistant message, or raise ApiError."""
    response = _request("POST", f"{base_url}/chat", json={"session_id": session_id, "message": message})
    if response.status_code != 200:
        raise ApiError(error_for_status(response))
    return parse_reply(response)


def fetch_ticket(base_url: str, ticket_id: str) -> dict:
    """GET one stored ticket by ID, or raise ApiError."""
    ticket_id = ticket_id.strip()
    if not ticket_id:
        raise ApiError("Please enter a ticket ID.")

    response = _request("GET", f"{base_url}/tickets/{quote(ticket_id, safe='')}")
    if response.status_code == 404:
        raise ApiError(f"No ticket found with ID {ticket_id}.")
    if response.status_code != 200:
        raise ApiError(error_for_status(response))
    return _json_body(response)


def transcribe_audio(base_url: str, audio_bytes: bytes, media_type: str) -> str:
    """Send a recording to speech-to-text and return the transcript, or raise ApiError."""
    files = {"audio": ("recording", audio_bytes, media_type or "application/octet-stream")}
    response = _request("POST", f"{base_url}/voice/transcribe", files=files)
    if response.status_code != 200:
        raise ApiError(error_for_status(response))

    transcript = _json_body(response).get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ApiError("No speech was recognized. Please record again or type your message.")
    return transcript.strip()


def synthesize_speech(base_url: str, message_id: str, text: str) -> tuple[bytes, str]:
    """Turn one agent response into audio; returns (audio bytes, media type) or raises ApiError."""
    response = _request(
        "POST", f"{base_url}/voice/synthesize", json={"message_id": message_id, "text": text}
    )
    if response.status_code != 200:
        raise ApiError(error_for_status(response))
    if response.headers.get("X-Message-Id") != message_id:
        raise ApiError("The audio returned doesn't match this message. Please try again.")
    media_type = response.headers.get("content-type", "").split(";", 1)[0]
    if not response.content or not media_type.startswith("audio/"):
        raise ApiError("The support service returned invalid audio. Please try again.")
    return response.content, media_type


def error_for_status(response: httpx.Response) -> str:
    """Map a non-200 API response to a customer-facing message."""
    body = _json_or_empty(response)
    # Voice endpoints already send customer-safe text as {"success": false, "error": ...}.
    if isinstance(body.get("error"), str):
        return body["error"]

    if response.status_code == 503:
        return "The support service is temporarily unavailable. A required backend component is not ready."

    if body:
        detail = body.get("detail", "Unknown backend error")
    else:
        detail = response.text or "The backend returned an invalid response."

    if response.status_code == 422:
        return f"Invalid request: {detail}"
    return f"The support service returned an error ({response.status_code}): {detail}"


def parse_reply(response: httpx.Response) -> dict:
    """Validate a successful chat body into an assistant chat message."""
    data = _json_body(response)

    text = data.get("response")
    if not isinstance(text, str) or not text.strip():
        raise ApiError("The support service returned an empty response. Please try again.")

    # The API returns filename-only sources; just guard the types.
    sources = data.get("sources")
    ticket_id = data.get("ticket_id")
    return {
        "role": "assistant",
        "content": text,
        "sources": [s for s in sources if isinstance(s, str)] if isinstance(sources, list) else [],
        "ticket_id": ticket_id if isinstance(ticket_id, str) else None,
    }


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    """Send a request, turning transport failures into ApiError."""
    try:
        return httpx.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except httpx.ConnectError as exc:
        raise ApiError(
            "Could not connect to the support backend. Please make sure the FastAPI server is running."
        ) from exc
    except httpx.TimeoutException as exc:
        raise ApiError("The support service took too long to respond. Please try again.") from exc
    except httpx.HTTPError as exc:
        raise ApiError(
            "A communication error occurred while contacting the support service. Please try again."
        ) from exc


def _json_or_empty(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _json_body(response: httpx.Response) -> dict:
    try:
        data = response.json()
    except ValueError as exc:
        raise ApiError("The support service returned an invalid response. Please try again.") from exc
    if not isinstance(data, dict):
        raise ApiError("The support service returned an invalid response. Please try again.")
    return data

import os
import uuid
from contextlib import nullcontext

import streamlit as st

from src.ui.api_client import ApiError, fetch_ticket, send_message, synthesize_speech, transcribe_audio
from src.ui.voice_widgets import has_audio, init_voice_state, render_speaker, resolve_submission

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CHAT_BOX_KEY = "chat_box"


# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))
        if message.get("ticket_id"):
            st.success(f"Support ticket created: {message['ticket_id']}")
        if message["role"] == "assistant" and message.get("id"):
            render_speaker(message, lambda message_id, text: synthesize_speech(API_BASE_URL, message_id, text))


def submit_message(text: str) -> None:
    """Send one customer message through the existing /chat workflow and render the result."""
    user_message = {"role": "user", "content": text}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        with st.spinner("Thinking..."):
            reply = send_message(API_BASE_URL, st.session_state.session_id, text)
    except ApiError as exc:
        with st.chat_message("assistant"):
            st.error(str(exc))
    except Exception:
        with st.chat_message("assistant"):
            st.error("Something went wrong while communicating with the support service. Please try again.")
    else:
        # Store everything needed to survive Streamlit reruns; the ID ties playback audio to this reply.
        reply["id"] = uuid.uuid4().hex
        st.session_state.messages.append(reply)
        render_message(reply)


def handle_chat_box(value) -> None:
    """Send typed text; for a recording, put its transcript back in the box to edit and send."""
    def transcribe(audio: bytes, media_type: str) -> str:
        return transcribe_audio(API_BASE_URL, audio, media_type)

    with st.spinner("Transcribing...") if has_audio(value) else nullcontext():
        outcome = resolve_submission(value, transcribe)

    if outcome.send:
        submit_message(outcome.send)
        return
    if outcome.prefill is None and outcome.error is None:
        return

    # The box can only be filled before it is drawn, so hand the transcript to the next run.
    st.session_state.chat_box_prefill = outcome.prefill
    st.session_state.voice_notice = (
        ("warning", outcome.error)
        if outcome.error
        else ("info", "Your transcript is in the message box. Check it, edit if needed, then press Enter to send.")
    )
    st.rerun()


def render_voice_notice() -> None:
    if notice := st.session_state.pop("voice_notice", None):
        kind, text = notice
        (st.warning if kind == "warning" else st.info)(text)


def render_ticket_lookup() -> None:
    """Sidebar form that shows a stored ticket by ID."""
    with st.sidebar.form("ticket_lookup"):
        st.subheader("Look up ticket")
        ticket_id = st.text_input("Ticket ID", placeholder="CST-2026-0001")
        if not st.form_submit_button("Look up"):
            return
        try:
            ticket = fetch_ticket(API_BASE_URL, ticket_id)
        except ApiError as exc:
            st.error(str(exc))
            return
        st.markdown(
            f"**{ticket.get('ticket_id')}** · {ticket.get('status')} · {ticket.get('category')}\n\n"
            f"**Name:** {ticket.get('customer_name')}  \n"
            f"**Email:** {ticket.get('customer_email')}  \n"
            f"**Issue:** {ticket.get('issue_description')}  \n"
            f"**Created:** {ticket.get('created_at')}"
        )


def reset_conversation() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.speech_store = {}
    st.session_state.speech_errors = {}
    st.session_state.selected_audio_id = None


# ---------------------------------------------------------
# Page
# ---------------------------------------------------------

st.set_page_config(page_title="Customer Support", page_icon="🎧")
st.title("Customer Support")

init_voice_state()
if "session_id" not in st.session_state:
    reset_conversation()

if st.sidebar.button("New conversation"):
    reset_conversation()
    st.rerun()

render_ticket_lookup()

for past_message in st.session_state.messages:
    render_message(past_message)

render_voice_notice()

if (prefill := st.session_state.pop("chat_box_prefill", None)) is not None:
    st.session_state[CHAT_BOX_KEY] = prefill

handle_chat_box(st.chat_input("Type a message, or tap the mic to speak", key=CHAT_BOX_KEY, accept_audio=True))

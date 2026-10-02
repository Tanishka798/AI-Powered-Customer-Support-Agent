"""LangGraph wiring: typed state, thin nodes, and routing for one support turn."""

from __future__ import annotations

from typing import Annotated, Protocol, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.llm.answering import generate_answer
from src.llm.decision import decide_turn
from src.llm.prompts import NO_ANSWER_TEXT
from src.llm.ticket_collection import TicketCollector, with_ticket_reminder
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository


class KnowledgeSearch(Protocol):
    """What the workflow needs from retrieval: deduplicated, filename-only chunks."""

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]: ...


class SupportWorkflowState(TypedDict, total=False):
    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None


def source_names(chunks: list[dict[str, str]]) -> list[str]:
    """Unique source filenames in retrieval order."""
    return list(dict.fromkeys(chunk["source"] for chunk in chunks))


def build_support_workflow(
    model: BaseChatModel,
    retriever: KnowledgeSearch,
    sessions: SessionStore,
    tickets: TicketRepository,
):
    """Build the LangGraph support workflow with explicitly injected dependencies."""
    collector = TicketCollector(tickets)

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        customer_message = state.get("customer_message", "").strip()
        session = sessions.get_or_create(state["session_id"])
        prior_user_turns = [
            turn["content"] for turn in session.history[-6:] if turn.get("role") == "user"
        ]
        search_query = "\n".join([*prior_user_turns, customer_message])
        chunks = await retriever.search(search_query) if customer_message else []
        return {"retrieved_chunks": chunks, "sources": source_names(chunks)}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        session = sessions.get_or_create(state["session_id"])
        route, fields = await decide_turn(model, state["customer_message"], session)
        return {"route": route, "extracted_fields": fields}

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        chunks = state.get("retrieved_chunks", [])
        session = sessions.get_or_create(state["session_id"])
        text = (
            await generate_answer(model, chunks, state["customer_message"], session.history)
            if chunks
            else NO_ANSWER_TEXT
        )
        return {
            "response_text": with_ticket_reminder(text, session),
            "sources": source_names(chunks),
            "ticket_id": None,
        }

    async def greet(state: SupportWorkflowState) -> SupportWorkflowState:
        session = sessions.get_or_create(state["session_id"])
        text = await generate_answer(
            model, [], state["customer_message"], session.history, mode="greeting"
        )
        return {
            "response_text": with_ticket_reminder(text, session),
            "sources": [],
            "ticket_id": None,
        }

    async def clarify(state: SupportWorkflowState) -> SupportWorkflowState:
        session = sessions.get_or_create(state["session_id"])
        text = await generate_answer(
            model, [], state["customer_message"], session.history, mode="clarify"
        )
        return {
            "response_text": with_ticket_reminder(text, session),
            "sources": [],
            "ticket_id": None,
        }

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        session_id = state["session_id"]
        reply = collector.handle_turn(
            session_id, sessions.get_or_create(session_id), state.get("extracted_fields", {})
        )
        return {"response_text": reply.text, "sources": [], "ticket_id": reply.ticket_id}

    async def lookup_ticket(state: SupportWorkflowState) -> SupportWorkflowState:
        session = sessions.get_or_create(state["session_id"])
        ticket = tickets.get(session.ticket_id) if session.ticket_id else None
        if ticket is None:
            return {
                "response_text": "I couldn't find an existing ticket for this conversation.",
                "sources": [],
                "ticket_id": None,
            }
        return {
            "response_text": f"Your ticket {ticket.ticket_id} is currently {ticket.status}.",
            "sources": [],
            "ticket_id": ticket.ticket_id,
        }

    def select_route(state: SupportWorkflowState) -> str:
        route = state.get("route")
        if route not in ("greeting", "answer", "ticket", "ticket_lookup", "clarify"):
            raise RuntimeError(f"Unexpected workflow route: {route!r}")
        return route

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("greeting", greet)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)
    graph.add_node("ticket_lookup", lookup_ticket)
    graph.add_node("clarify", clarify)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges(
        "decide",
        select_route,
        {
            "greeting": "greeting",
            "answer": "answer",
            "ticket": "ticket",
            "ticket_lookup": "ticket_lookup",
            "clarify": "clarify",
        },
    )
    graph.add_edge("greeting", END)
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)
    graph.add_edge("ticket_lookup", END)
    graph.add_edge("clarify", END)
    return graph.compile()

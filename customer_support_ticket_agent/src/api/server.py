import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.api.voice_routes import router as voice_router
from src.config import load_settings
from src.llm.model_check import ensure_llm_model
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.factory import build_voice_pipeline
from src.voice.pipeline import VoicePipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per LLM call is noise
logger = logging.getLogger(__name__)

settings = load_settings()
pipeline = SupportPipeline(settings, Path(__file__).resolve().parents[2] / "knowledge_base")


async def start_voice(voice: VoicePipeline) -> bool:
    """Voice is optional: a failure here must never stop the text agent from starting."""
    try:
        await voice.initialize()
    except Exception:
        logger.exception("Voice pipeline failed to start; voice endpoints will return 503")
        return False
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup owns expensive shared initialization. Request handlers reuse the
    # resulting components, while shutdown makes readiness false immediately.
    await ensure_llm_model(settings)
    await pipeline.initialize()
    app.state.voice_ready = await start_voice(app.state.voice)
    yield
    pipeline.ready = False
    app.state.voice_ready = False
    await app.state.voice.cleanup()


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)
app.state.voice = build_voice_pipeline(settings)
app.state.voice_ready = False
app.include_router(voice_router)


@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tickets", response_model=list[Ticket])
async def list_tickets() -> list[Ticket]:
    # Read-only view of every repository record, in creation order.
    return list(pipeline.tickets.all())


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    # Keep this endpoint read-only. It should return the exact repository record,
    # not ask the model to reconstruct ticket details. Test both the successful
    # lookup and unknown-ID response through FastAPI's test client.
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

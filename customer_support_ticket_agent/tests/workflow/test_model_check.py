import httpx
import pytest

from src.llm.model_check import ensure_llm_model, ollama_root
from tests.fakes import make_settings


def ollama(models: list[str], pull_status: int = 200, reachable: bool = True):
    """A mock Ollama server; records the paths it was asked for."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if not reachable:
            raise httpx.ConnectError("refused")
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in models]})
        if request.url.path == "/api/pull":
            return httpx.Response(pull_status, json={"status": "success"})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


@pytest.mark.parametrize(
    "url, root",
    [("http://localhost:11434/v1", "http://localhost:11434"), ("http://host:11434/v1/", "http://host:11434")],
)
def test_ollama_root_strips_openai_suffix(url: str, root: str) -> None:
    assert ollama_root(url) == root


@pytest.mark.asyncio
async def test_present_model_is_not_pulled() -> None:
    client, calls = ollama(["qwen2.5:3b"])

    assert await ensure_llm_model(make_settings(llm_model="qwen2.5:3b"), client) is True
    assert calls == ["/api/tags"]


@pytest.mark.asyncio
async def test_missing_model_is_pulled() -> None:
    client, calls = ollama(["llama3:latest"])

    assert await ensure_llm_model(make_settings(llm_model="qwen2.5:3b"), client) is True
    assert calls == ["/api/tags", "/api/pull"]


@pytest.mark.asyncio
async def test_bare_name_matches_latest_tag() -> None:
    client, calls = ollama(["llama3:latest"])

    assert await ensure_llm_model(make_settings(llm_model="llama3"), client) is True
    assert calls == ["/api/tags"]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_args", [dict(models=[], reachable=False), dict(models=[], pull_status=500)])
async def test_failures_are_reported_not_raised(client_args: dict) -> None:
    client, _ = ollama(**client_args)

    assert await ensure_llm_model(make_settings(), client) is False


@pytest.mark.asyncio
async def test_auto_pull_can_be_disabled() -> None:
    client, calls = ollama([])

    assert await ensure_llm_model(make_settings(llm_auto_pull=False), client) is None
    assert calls == []


@pytest.mark.asyncio
async def test_non_ollama_endpoint_is_skipped() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))

    assert await ensure_llm_model(make_settings(), client) is None

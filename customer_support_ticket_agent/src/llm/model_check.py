"""Detect whether the configured LLM is available, and pull it automatically from Ollama."""

from __future__ import annotations

import logging

import httpx

from src.config import Settings

logger = logging.getLogger(__name__)

PULL_TIMEOUT_SECONDS = 1800  # a first pull downloads a few GB


def ollama_root(llm_base_url: str) -> str:
    """"http://localhost:11434/v1" -> "http://localhost:11434"."""
    root = llm_base_url.rstrip("/")
    return root[: -len("/v1")] if root.endswith("/v1") else root


async def ensure_llm_model(settings: Settings, client: httpx.AsyncClient | None = None) -> bool | None:
    """Pull the configured model if the endpoint is Ollama and the model is missing.

    Returns True when the model is available, False when it isn't, and None when
    it can't be checked (auto-pull disabled, or the endpoint isn't Ollama). Never
    raises: problems are logged so the API still starts and /chat reports them.
    """
    if not settings.llm_auto_pull:
        return None

    root = ollama_root(settings.llm_base_url)
    async with client or httpx.AsyncClient(timeout=PULL_TIMEOUT_SECONDS) as http:
        try:
            tags = await http.get(f"{root}/api/tags", timeout=5)
        except httpx.HTTPError:
            logger.warning("LLM endpoint %s is unreachable; start Ollama or check LLM_BASE_URL", root)
            return False
        if tags.status_code != 200:
            logger.info("LLM endpoint is not Ollama; skipping model auto-download")
            return None

        if _has_model(tags.json(), settings.llm_model):
            logger.info("LLM model %s is available", settings.llm_model)
            return True

        logger.warning("LLM model %s not found; downloading it now (first run only)...", settings.llm_model)
        try:
            pulled = await http.post(f"{root}/api/pull", json={"model": settings.llm_model, "stream": False})
        except httpx.HTTPError as exc:
            logger.error("Could not download %s: %s", settings.llm_model, exc)
            return False
        if pulled.status_code != 200:
            logger.error("Could not download %s: %s", settings.llm_model, pulled.text[:200])
            return False
        logger.info("Downloaded LLM model %s", settings.llm_model)
        return True


def _has_model(tags: dict, model: str) -> bool:
    names = {entry.get("name") for entry in tags.get("models", [])}
    # Ollama reports "qwen2.5:3b"; a bare "llama3" means "llama3:latest".
    return model in names or f"{model}:latest" in names

"""One-command setup: python scripts/setup.py

Detects what is already present and only installs or downloads what's missing:
  1. .venv virtual environment
  2. Python dependencies (CPU-only PyTorch by default: it only runs the small
     embedding model, and the CUDA build adds several GB for no real benefit)
  3. .env copied from .env.example
  4. Embedding and Whisper models (cached by Hugging Face, downloaded once)
  5. The Ollama LLM (pulled through Ollama's API if Ollama is running)

Uses only the standard library so it can run before anything is installed.
Options: --skip-models (skip steps 4 and 5), --gpu (install the default CUDA PyTorch build).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

PREFETCH_MODELS = """
from src.config import load_settings
from langchain_huggingface import HuggingFaceEmbeddings
from faster_whisper import WhisperModel

settings = load_settings()
print(f"  embeddings: {settings.embedding_model}")
HuggingFaceEmbeddings(model_name=settings.embedding_model).embed_query("warm up")
print(f"  whisper: {settings.stt_model}")
WhisperModel(settings.stt_model, device=settings.stt_device, compute_type=settings.stt_compute_type)
"""

ENSURE_LLM = """
import asyncio, logging, sys
logging.basicConfig(level=logging.INFO, format="  %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
from src.config import load_settings
from src.llm.model_check import ensure_llm_model
available = asyncio.run(ensure_llm_model(load_settings()))
sys.exit(1 if available is False else 0)
"""


def step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def ensure_venv() -> None:
    step("Virtual environment")
    if VENV_PYTHON.exists():
        print("  .venv already exists")
        return
    # Match `python -m venv`: symlink on POSIX. A copied interpreter from some
    # Python distributions (e.g. uv-managed) can't find its standard library.
    venv.create(VENV, with_pip=True, symlinks=os.name != "nt")
    print("  created .venv")


def install_dependencies(use_gpu: bool) -> None:
    step("Python dependencies")
    run(str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "--upgrade", "pip")
    if not use_gpu and sys.platform != "darwin":
        print("  installing CPU-only PyTorch (use --gpu for the CUDA build)")
        run(str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "torch", "--index-url", CPU_TORCH_INDEX)
    run(str(VENV_PYTHON), "-m", "pip", "install", "--quiet", "-r", "requirements.txt")
    print("  dependencies installed")


def ensure_env_file() -> None:
    step("Configuration")
    env_file = ROOT / ".env"
    if env_file.exists():
        print("  .env already exists (left unchanged)")
        return
    shutil.copy(ROOT / ".env.example", env_file)
    print("  created .env from .env.example")


def prefetch_models() -> None:
    step("Embedding and speech-to-text models (downloaded once, then cached)")
    run(str(VENV_PYTHON), "-c", PREFETCH_MODELS)


def ensure_llm() -> None:
    step("LLM (Ollama)")
    if subprocess.run([str(VENV_PYTHON), "-c", ENSURE_LLM], cwd=ROOT).returncode != 0:
        # Not fatal: the API pulls the model itself at startup once Ollama is running.
        print("  The LLM isn't available yet. Install Ollama from https://ollama.com/download,")
        print("  start it, then re-run this script or just start the API (it pulls the model).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-models", action="store_true", help="don't download models now")
    parser.add_argument("--gpu", action="store_true", help="install the default (CUDA) PyTorch build")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        sys.exit(f"Python 3.11+ is required (found {sys.version.split()[0]}).")

    try:
        ensure_venv()
        install_dependencies(use_gpu=args.gpu)
        ensure_env_file()
        if not args.skip_models:
            prefetch_models()
            ensure_llm()
    except (subprocess.CalledProcessError, OSError) as exc:
        sys.exit(
            f"\nSetup stopped: {exc}\n"
            "Fix the problem shown above (often a dropped download) and re-run the script;\n"
            "finished steps are detected and skipped."
        )

    activate = r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate"
    step("Done. Start the app (two terminals, from the project root):")
    print(f"  {activate}")
    print("  uvicorn src.api.server:app --host 127.0.0.1 --port 8000")
    print("  streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501")


if __name__ == "__main__":
    main()

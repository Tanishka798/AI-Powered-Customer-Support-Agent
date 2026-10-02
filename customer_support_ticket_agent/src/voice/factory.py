from src.config import Settings
from src.voice.pipeline import VoicePipeline
from src.voice.stt_whisper import FasterWhisperSTT
from src.voice.tts_edge import EdgeTTS


def build_voice_pipeline(settings: Settings) -> VoicePipeline:
    """Wire the configured STT and TTS adapters into the shared voice pipeline."""
    stt = FasterWhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        language=settings.stt_language,
    )
    return VoicePipeline(stt, EdgeTTS(settings.tts_voice))

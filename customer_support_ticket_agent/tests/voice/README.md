# Voice tests

Tests for `src/voice/`. They use fake STT and TTS adapters (`tests/fakes.py`) and a fake Whisper model, so no model download or network access is needed. `example_responses.json` is the example payload file supplied with the mid-session requirements.

| File | What it checks |
|---|---|
| `test_voice_pipeline.py` | The supplied contract test (kept as-is), plus: timing reflects adapter duration, media types pass through, empty audio and blank text are rejected before reaching an adapter, speech-free audio is reported, and adapter failures propagate. |
| `test_voice_models.py` | The supplied example transcription and error payloads match the models, and invalid synthesis requests are rejected. |
| `test_stt_whisper.py` | Real WAV audio is decoded and transcript segments are joined; unsupported formats, unreadable audio, and use before initialization fail clearly. |
| `test_tts_edge.py` | Edge TTS audio chunks are joined as MP3; provider errors and empty audio become provider failures. |

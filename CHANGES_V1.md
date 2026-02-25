# V1: Replace Local ML Backends with Configurable Backend Registry

## Overview

Replaced all local STT/TTS/chat model executors with a configurable backend registry that maps model IDs to remote endpoints. The initial backend is Qwen3-Omni served via vllm-omni's OpenAI-compatible API.

## Architecture

### Before
- Local Whisper (STT), Piper/Kokoro (TTS), Parakeet (ASR) executors loaded models into memory
- HuggingFace Hub used for model discovery and downloading
- Chat routed to a separately configured OpenAI-compatible endpoint
- Complex executor registry managed model lifecycle (load/unload/TTL)

### After
- Single `model_backends.yaml` config maps model IDs to remote API endpoints
- All STT/TTS/chat requests proxied to remote backends via `httpx` with basic auth
- Hot-reloadable config (re-read every ~10s) allows backend changes without pod restart
- Only Silero VAD retained locally (needed for realtime WebSocket/WebRTC)

## New Files

| File | Purpose |
|------|---------|
| `model_backends.yaml` | YAML config defining backends (mountable as K8s ConfigMap) |
| `src/speaches/backend_registry.py` | `BackendConfig` model + `BackendRegistry` with hot-reload, model lookup, capability filtering, cached HTTP clients |
| `src/speaches/qwen_client.py` | `chat_completion()`, `transcribe()`, `synthesize()` functions targeting vllm-omni's API |

## Modified Files

| File | Changes |
|------|---------|
| `config.py` | Added `backend_config_path`. Removed `WhisperConfig`, `OrtOptions`, `stt_model_ttl`, `tts_model_ttl`, `chat_completion_base_url`, `chat_completion_api_key`, `preload_models`, `_unstable_vad_filter`, `unstable_ort_opts` |
| `types/chat.py` | Added `AudioURL`, `VideoURL`, `ChatCompletionContentPartAudioUrlParam`, `ChatCompletionContentPartVideoUrlParam` to support multimodal inputs |
| `dependencies.py` | Replaced `ExecutorRegistryDependency` with `BackendRegistryDependency` and `VadModelManagerDependency`. Removed `CompletionClientDependency`, `SpeechClientDependency`. Replaced `faster_whisper.audio.decode_audio` with `soundfile` for audio decoding |
| `routers/stt.py` | Rewrote to use `qwen_client.transcribe()`. Removed VAD preprocessing, executor registry, translation endpoint. Dropped `srt`/`vtt` response formats |
| `routers/speech.py` | Rewrote to use `qwen_client.synthesize()`. No `voice` parameter. Converts 24kHz WAV output to requested format via `convert_audio_format` |
| `routers/chat.py` | Rewrote to proxy via backend registry. Converts OpenAI `input_audio` format to vllm `audio_url` format. Supports `audio_url`, `image_url`, `video_url` content parts. Passes `modalities` and `extra_body` through to backend |
| `routers/models.py` | Rewrote to list models from backend registry with capabilities. Removed HuggingFace registry queries, model download/delete, voice listing |
| `routers/misc.py` | Stripped to just `/health` endpoint. Removed `/api/ps` model load/unload routes |
| `routers/realtime_ws.py` | Replaced `ExecutorRegistryDependency` with `VadModelManagerDependency` |
| `routers/realtime_rtc.py` | Same as above |
| `executors/silero_vad_v5.py` | Removed dependencies on `ModelRegistry`, `HfModelFilter`, `OrtOptions`, `handler_protocol`. Simplified to standalone module |
| `executors/shared/base_model_manager.py` | Replaced `get_ort_providers_with_options(OrtOptions)` with simple `get_ort_providers()` |
| `main.py` | Removed vad, diarization, speech_embedding router registrations. Removed executor registry preloading from lifespan |
| `ui/tabs/stt.py` | Simplified model dropdown to query `/v1/models?task=stt`. Removed task selector (translate), stream checkbox |
| `ui/tabs/tts.py` | Removed voice dropdown, speed slider, sample rate control. Uses `httpx` directly instead of OpenAI client |
| `ui/tabs/audio_chat.py` | Added output modality selector (text/audio). Supports multimodal file uploads (audio, image, video). Uses `httpx` + SSE directly instead of OpenAI client |
| `pyproject.toml` | Added `pyyaml`. Removed `ctranslate2`, `kokoro-onnx`, `piper-tts`, `onnx-asr`, `huggingface-hub`, `aiostream`, `sounddevice`, `onnx-diarization`. Cleaned up uv overrides |
| `tests/conftest.py` | Updated for new `Config` shape. Removed executor registry patches |

## Deleted Files (19)

### Executors
- `executors/whisper.py` — Whisper STT executor
- `executors/piper.py` — Piper TTS executor
- `executors/kokoro.py` — Kokoro TTS executor
- `executors/parakeet.py` — Parakeet ASR executor
- `executors/wespeaker_speaker_embedding.py` — Speaker embedding executor
- `executors/pyannote_speaker_segmentation.py` — Speaker diarization executor
- `executors/vllm_tts.py` — vLLM TTS executor (untracked)
- `executors/shared/executor.py` — Base executor class
- `executors/shared/handler_protocol.py` — Handler protocol definitions
- `executors/shared/registry.py` — Executor registry

### Routers
- `routers/vad.py` — VAD endpoint
- `routers/diarization.py` — Diarization endpoint
- `routers/speech_embedding.py` — Speaker embedding endpoint
- `routers/voice_cloning.py` — Voice cloning endpoint (untracked)
- `routers/utils.py` — Router utilities (HuggingFace model card lookup)

### Utilities
- `model_registry.py` — Model registry base class
- `model_aliases.py` — Model ID alias resolution
- `hf_utils.py` — HuggingFace Hub utilities
- `diarization.py` — Diarization logic
- `voice_registry.py` — Voice registry (untracked)

## Backend Config Format

```yaml
backends:
  - id: Qwen/Qwen3-Omni-30B-A3B-Instruct
    base_url: http://100.99.37.118/omni/v1
    auth_user: tts-api
    auth_password: fEQ6E4ugEtJF3j9rH8H4DJyU
    capabilities: [stt, tts, chat]
    input_modalities: [text, audio, image, video]
    output_modalities: [text, audio]
```

## Qwen3-Omni Constraints

- Only ONE output modality per request: `["text"]` OR `["audio"]`, never both
- Audio output is 24kHz WAV, base64-encoded in `message.audio.data`
- Inputs support `text`, `audio_url`, `image_url`, `video_url` content parts
- Auth: HTTP Basic (`tts-api:fEQ6E4ugEtJF3j9rH8H4DJyU`)

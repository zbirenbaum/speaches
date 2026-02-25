import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from speaches.dependencies import ExecutorRegistryDependency
from speaches.voice_registry import VoiceRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voice-cloning"])


def _get_voice_registry(executor_registry: ExecutorRegistryDependency) -> VoiceRegistry:
    if executor_registry.voice_registry is None:
        raise HTTPException(status_code=501, detail="Voice registry is not configured")
    return executor_registry.voice_registry


@router.post("/v1/voice-cloning/voices")
async def create_voice(
    executor_registry: ExecutorRegistryDependency,
    file: Annotated[UploadFile, File(...)],
    name: Annotated[str, Form(...)],
    ref_text: Annotated[str, Form()] = "",
) -> JSONResponse:
    registry = _get_voice_registry(executor_registry)
    audio_bytes = await file.read()
    voice = registry.save_voice(name=name, ref_audio=audio_bytes, ref_text=ref_text)

    # Register with vLLM backend if available for embedding cache warm-up.
    if executor_registry.vllm_tts_proxy:
        try:
            executor_registry.vllm_tts_proxy.register_voice_with_backends(
                voice_id=voice.id,
                audio_bytes=audio_bytes,
                ref_text=ref_text,
                name=name,
            )
        except Exception:
            logger.exception("Failed to register voice with vLLM backends (voice saved locally)")

    return JSONResponse(content=voice.model_dump(), status_code=201)


@router.get("/v1/voice-cloning/voices")
def list_saved_voices(
    executor_registry: ExecutorRegistryDependency,
) -> JSONResponse:
    registry = _get_voice_registry(executor_registry)
    voices = registry.list_voices()
    return JSONResponse(content={"voices": [v.model_dump() for v in voices], "object": "list"})


@router.get("/v1/voice-cloning/voices/{voice_id}")
def get_saved_voice(
    executor_registry: ExecutorRegistryDependency,
    voice_id: str,
) -> JSONResponse:
    registry = _get_voice_registry(executor_registry)
    voice = registry.get_voice(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    return JSONResponse(content=voice.model_dump())


@router.delete("/v1/voice-cloning/voices/{voice_id}")
def delete_saved_voice(
    executor_registry: ExecutorRegistryDependency,
    voice_id: str,
) -> JSONResponse:
    registry = _get_voice_registry(executor_registry)
    if not registry.delete_voice(voice_id):
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    return JSONResponse(content={"deleted": True, "id": voice_id})

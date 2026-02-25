import base64
from collections.abc import AsyncGenerator
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from speaches.dependencies import BackendRegistryDependency
from speaches.qwen_client import chat_completion

logger = logging.getLogger(__name__)

router = APIRouter(tags=["text-to-speech"])


class CreateSpeechRequestBody(BaseModel):
    model: str
    input: str


async def _stream_pcm(
    async_chunks: AsyncGenerator[dict, None],
) -> AsyncGenerator[bytes]:
    async for chunk in async_chunks:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            yield base64.b64decode(content)


@router.post("/v1/audio/speech")
async def handle_speech(
    registry: BackendRegistryDependency,
    body: CreateSpeechRequestBody,
) -> StreamingResponse:
    backend = registry.get_backend(body.model)
    if backend is None or "tts" not in backend.capabilities:
        raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found or does not support TTS")

    client = registry.get_http_client(backend)

    messages = [{"role": "user", "content": [{"type": "text", "text": body.input}]}]

    result = await chat_completion(
        client=client,
        messages=messages,
        model=body.model,
        modalities=["audio"],
        stream=True,
    )

    return StreamingResponse(_stream_pcm(result), media_type="audio/pcm")

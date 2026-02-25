import io
import logging
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Response,
)
import soundfile as sf

from speaches.dependencies import (
    AudioFileDependency,
    BackendRegistryDependency,
)
from speaches.qwen_client import transcribe

logger = logging.getLogger(__name__)

router = APIRouter(tags=["automatic-speech-recognition"])

type ResponseFormat = Literal["text", "json", "verbose_json"]
RESPONSE_FORMATS = ("text", "json", "verbose_json")
DEFAULT_RESPONSE_FORMAT: ResponseFormat = "json"


@router.post("/v1/audio/transcriptions")
async def transcribe_file(
    registry: BackendRegistryDependency,
    audio: AudioFileDependency,
    model: Annotated[str, Form()],
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = DEFAULT_RESPONSE_FORMAT,
    temperature: Annotated[float, Form()] = 0.0,  # noqa: ARG001
) -> Response:
    backend = registry.get_backend(model)
    if backend is None or "stt" not in backend.capabilities:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found or does not support STT")

    client = registry.get_http_client(backend)

    # Convert audio to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio.data, audio.sample_rate, format="WAV")
    audio_bytes = buf.getvalue()

    transcript_prompt = prompt
    if language:
        lang_prefix = f"Language: {language}. "
        transcript_prompt = f"{lang_prefix}{prompt}" if prompt else lang_prefix + "Transcribe this audio."

    text = await transcribe(client, audio_bytes, model, prompt=transcript_prompt)

    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    elif response_format == "json":
        import json

        return Response(content=json.dumps({"text": text}), media_type="application/json")
    elif response_format == "verbose_json":
        import json

        return Response(
            content=json.dumps({"text": text, "language": language or "unknown", "duration": audio.duration}),
            media_type="application/json",
        )
    raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")

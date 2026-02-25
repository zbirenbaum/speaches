from collections.abc import AsyncGenerator
import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Response
from fastapi.responses import StreamingResponse

from speaches.dependencies import BackendRegistryDependency
from speaches.qwen_client import chat_completion
from speaches.text_utils import format_as_sse
from speaches.types.chat import CompletionCreateParamsBase as OpenAICompletionCreateParamsBase

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice-chat"])


class CompletionCreateParamsBase(OpenAICompletionCreateParamsBase):
    stream: bool = False
    extra_body: dict[str, Any] | None = None


def _convert_input_audio_to_audio_url(content_parts: list[dict]) -> list[dict]:
    converted = []
    for part in content_parts:
        if part.get("type") == "input_audio":
            audio_data = part["input_audio"]["data"]
            audio_format = part["input_audio"].get("format", "wav")
            data_url = f"data:audio/{audio_format};base64,{audio_data}"
            converted.append({"type": "audio_url", "audio_url": {"url": data_url}})
        else:
            converted.append(part)
    return converted


def _prepare_messages(body: CompletionCreateParamsBase) -> list[dict]:
    messages = []
    for msg in body.messages:
        msg_dict = msg.model_dump(exclude_none=True)
        if msg_dict.get("role") == "user" and isinstance(msg_dict.get("content"), list):
            msg_dict["content"] = _convert_input_audio_to_audio_url(msg_dict["content"])
        messages.append(msg_dict)
    return messages


@router.post("/v1/chat/completions", response_model=None)
async def handle_completions(
    registry: BackendRegistryDependency,
    body: Annotated[CompletionCreateParamsBase, Body()],
) -> Response | StreamingResponse:
    backend = registry.get_backend(body.model)
    if backend is None or "chat" not in backend.capabilities:
        raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found or does not support chat")

    client = registry.get_http_client(backend)
    messages = _prepare_messages(body)
    modalities = body.modalities or ["text"]

    extra = body.extra_body or {}

    try:
        result = await chat_completion(
            client=client,
            messages=messages,
            model=body.model,
            modalities=modalities,
            stream=body.stream,
            max_tokens=body.max_tokens or body.max_completion_tokens,
            extra_body=extra,
        )
    except Exception as e:
        logger.exception("Chat completion request failed")
        raise HTTPException(status_code=502, detail="Backend chat completion request failed") from e

    if body.stream:
        assert not isinstance(result, dict)

        async def stream_response() -> AsyncGenerator[str]:
            async for chunk in result:
                yield format_as_sse(json.dumps(chunk))
            yield format_as_sse("[DONE]")

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    assert isinstance(result, dict)
    return Response(content=json.dumps(result), media_type="application/json")

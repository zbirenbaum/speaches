import base64
from collections.abc import AsyncGenerator
import json
import logging

import httpx
from httpx_sse import aconnect_sse

logger = logging.getLogger(__name__)


async def chat_completion(
    client: httpx.AsyncClient,
    messages: list[dict],
    model: str,
    modalities: list[str],
    stream: bool = False,
    max_tokens: int | None = None,
    extra_body: dict | None = None,
) -> dict | AsyncGenerator[dict, None]:
    body: dict = {
        "model": model,
        "messages": messages,
        "modalities": modalities,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if extra_body:
        body.update(extra_body)

    if stream:
        return _stream_chat_completion(client, body)

    response = await client.post("/chat/completions", json=body)
    response.raise_for_status()
    return response.json()


async def _stream_chat_completion(
    client: httpx.AsyncClient,
    body: dict,
) -> AsyncGenerator[dict, None]:
    async with aconnect_sse(client, "POST", "/chat/completions", json=body) as event_source:
        async for sse in event_source.aiter_sse():
            data = sse.data
            if data == "[DONE]":
                return
            yield json.loads(data)


async def transcribe(
    client: httpx.AsyncClient,
    audio_bytes: bytes,
    model: str,
    prompt: str | None = None,
) -> str:
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_url = f"data:audio/wav;base64,{audio_b64}"

    content: list[dict] = [{"type": "audio_url", "audio_url": {"url": audio_url}}]
    if prompt:
        content.insert(0, {"type": "text", "text": prompt})
    else:
        content.insert(0, {"type": "text", "text": "Transcribe this audio."})

    messages = [{"role": "user", "content": content}]

    response = await client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "modalities": ["text"],
        },
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


async def synthesize(
    client: httpx.AsyncClient,
    text: str,
    model: str,
) -> bytes:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    response = await client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "modalities": ["audio"],
        },
    )
    response.raise_for_status()
    result = response.json()
    audio_data_b64 = result["choices"][0]["message"]["audio"]["data"]
    return base64.b64decode(audio_data_b64)

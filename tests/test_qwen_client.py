import base64
from collections.abc import Callable
import json

import httpx
import pytest

from speaches.qwen_client import chat_completion, synthesize, transcribe

type RequestHandler = Callable[[httpx.Request], httpx.Response]


class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: RequestHandler) -> None:
        self._handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._handler(request)


def make_client(handler: RequestHandler, base_url: str = "http://test/v1") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=MockTransport(handler), base_url=base_url)


class TestTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            assert body["modalities"] == ["text"]
            messages = body["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            content_parts = messages[0]["content"]
            assert any(p["type"] == "audio_url" for p in content_parts)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Hello world"}}],
                },
            )

        client = make_client(handler)
        result = await transcribe(client, b"fake-wav-bytes", "test-model")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_with_prompt(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            content_parts = body["messages"][0]["content"]
            text_parts = [p for p in content_parts if p["type"] == "text"]
            assert len(text_parts) == 1
            assert text_parts[0]["text"] == "Custom prompt"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Transcribed text"}}]},
            )

        client = make_client(handler)
        result = await transcribe(client, b"audio", "test-model", prompt="Custom prompt")
        assert result == "Transcribed text"

    @pytest.mark.asyncio
    async def test_transcribe_audio_is_base64_encoded(self) -> None:
        audio_bytes = b"test-audio-data"

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            audio_parts = [p for p in body["messages"][0]["content"] if p["type"] == "audio_url"]
            assert len(audio_parts) == 1
            url = audio_parts[0]["audio_url"]["url"]
            assert url.startswith("data:audio/wav;base64,")
            decoded = base64.b64decode(url.split(",", 1)[1])
            assert decoded == audio_bytes
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        client = make_client(handler)
        await transcribe(client, audio_bytes, "model")

    @pytest.mark.asyncio
    async def test_transcribe_raises_on_http_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal error"})

        client = make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await transcribe(client, b"audio", "model")


class TestSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_returns_wav_bytes(self) -> None:
        wav_data = b"RIFF" + b"\x00" * 100
        b64_wav = base64.b64encode(wav_data).decode()

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "tts-model"
            assert body["modalities"] == ["audio"]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"audio": {"data": b64_wav}}}],
                },
            )

        client = make_client(handler)
        result = await synthesize(client, "Hello world", "tts-model")
        assert result == wav_data

    @pytest.mark.asyncio
    async def test_synthesize_sends_text_as_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            messages = body["messages"]
            assert len(messages) == 1
            content = messages[0]["content"]
            assert any(p.get("text") == "Speak this" for p in content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"audio": {"data": base64.b64encode(b"wav").decode()}}}],
                },
            )

        client = make_client(handler)
        await synthesize(client, "Speak this", "model")

    @pytest.mark.asyncio
    async def test_synthesize_raises_on_http_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, json={"error": "bad gateway"})

        client = make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await synthesize(client, "text", "model")


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming_returns_dict(self) -> None:
        response_body = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop", "index": 0}],
            "model": "test-model",
            "object": "chat.completion",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is False
            assert body["modalities"] == ["text"]
            return httpx.Response(200, json=response_body)

        client = make_client(handler)
        result = await chat_completion(
            client, messages=[{"role": "user", "content": "Hello"}], model="test-model", modalities=["text"]
        )
        assert isinstance(result, dict)
        assert result["choices"][0]["message"]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_non_streaming_passes_max_tokens(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["max_tokens"] == 512
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        client = make_client(handler)
        await chat_completion(
            client,
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            modalities=["text"],
            max_tokens=512,
        )

    @pytest.mark.asyncio
    async def test_non_streaming_passes_extra_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["custom_param"] == "custom_value"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        client = make_client(handler)
        await chat_completion(
            client,
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            modalities=["text"],
            extra_body={"custom_param": "custom_value"},
        )

    @pytest.mark.asyncio
    async def test_streaming_returns_async_generator(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}, "index": 0}]},
            {"choices": [{"delta": {"content": " world"}, "index": 0}]},
        ]
        sse_body = ""
        for chunk in chunks:
            sse_body += f"data: {json.dumps(chunk)}\n\n"
        sse_body += "data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is True
            return httpx.Response(
                200,
                content=sse_body.encode(),
                headers={"content-type": "text/event-stream"},
            )

        client = make_client(handler)
        result = await chat_completion(
            client,
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            modalities=["text"],
            stream=True,
        )
        collected = [chunk async for chunk in result]
        assert len(collected) == 2
        assert collected[0]["choices"][0]["delta"]["content"] == "Hello"
        assert collected[1]["choices"][0]["delta"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_non_streaming_raises_on_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "fail"})

        client = make_client(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await chat_completion(
                client,
                messages=[{"role": "user", "content": "Hi"}],
                model="m",
                modalities=["text"],
            )

    @pytest.mark.asyncio
    async def test_audio_modality_request(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["modalities"] == ["audio"]
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"audio": {"data": base64.b64encode(b"wav").decode(), "transcript": "hi"}}}]
                },
            )

        client = make_client(handler)
        result = await chat_completion(
            client,
            messages=[{"role": "user", "content": "Hi"}],
            model="m",
            modalities=["audio"],
        )
        assert isinstance(result, dict)
        assert "audio" in result["choices"][0]["message"]

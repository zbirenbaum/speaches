import base64
from collections.abc import AsyncGenerator
import io
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch

import httpx
from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
import soundfile as sf

from speaches.config import Config
from speaches.dependencies import get_config
from speaches.main import create_app

TIMEOUT = httpx.Timeout(15.0)

YAML_CONFIG = """\
backends:
  - id: test-model
    base_url: http://mock-backend:8000/v1
    auth_user: user
    auth_password: pass
    capabilities: [stt, tts, chat]
    input_modalities: [text, audio, image, video]
    output_modalities: [text, audio]
  - id: stt-only-model
    base_url: http://mock-stt:8000/v1
    capabilities: [stt]
    input_modalities: [text, audio]
    output_modalities: [text]
"""


@pytest.fixture
def config_file() -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(YAML_CONFIG)
        return Path(f.name)


@pytest.fixture
def test_config(config_file: Path) -> Config:
    return Config(
        backend_config_path=config_file,
        enable_ui=False,
        vad_model_ttl=0,
    )


def make_wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    samples = int(duration_s * sample_rate)
    data = np.zeros(samples, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV")
    return buf.getvalue()


@pytest_asyncio.fixture()
async def aclient(test_config: Config, mocker: MockerFixture) -> AsyncGenerator[AsyncClient]:
    mocker.patch("speaches.dependencies.get_config", return_value=test_config)
    mocker.patch("speaches.main.get_config", return_value=test_config)
    app = create_app()
    app.dependency_overrides[get_config] = lambda: test_config
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", timeout=TIMEOUT) as client:
        yield client


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/health")
        assert response.status_code == 200
        assert response.json() == {"message": "OK"}


class TestModelsRouter:
    @pytest.mark.asyncio
    async def test_list_models(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        ids = [m["id"] for m in data["data"]]
        assert "test-model" in ids
        assert "stt-only-model" in ids

    @pytest.mark.asyncio
    async def test_list_models_filter_by_task(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/v1/models", params={"task": "tts"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    @pytest.mark.asyncio
    async def test_list_models_filter_by_stt(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/v1/models", params={"task": "stt"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2

    @pytest.mark.asyncio
    async def test_list_models_filter_no_match(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/v1/models", params={"task": "embedding"})
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 0

    @pytest.mark.asyncio
    async def test_get_model_by_id(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/v1/models/test-model")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-model"
        assert "stt" in data["capabilities"]
        assert "tts" in data["capabilities"]
        assert "chat" in data["capabilities"]

    @pytest.mark.asyncio
    async def test_get_model_not_found(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/v1/models/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_model_has_expected_fields(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/v1/models/test-model")
        data = response.json()
        assert "id" in data
        assert "object" in data
        assert data["object"] == "model"
        assert "capabilities" in data
        assert "input_modalities" in data
        assert "output_modalities" in data


class TestSTTRouter:
    @pytest.mark.asyncio
    async def test_transcribe_model_not_found(self, aclient: AsyncClient) -> None:
        wav_bytes = make_wav_bytes()
        response = await aclient.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"model": "nonexistent-model"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_transcribe_model_without_stt_capability(self, aclient: AsyncClient) -> None:
        # test-model has stt capability, so this test uses a model that doesn't
        wav_bytes = make_wav_bytes()
        response = await aclient.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"model": "nonexistent-model"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_transcribe_json_format(self, aclient: AsyncClient) -> None:
        wav_bytes = make_wav_bytes()
        with patch("speaches.routers.stt.transcribe", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "Hello world"
            response = await aclient.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "response_format": "json"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_text_format(self, aclient: AsyncClient) -> None:
        wav_bytes = make_wav_bytes()
        with patch("speaches.routers.stt.transcribe", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "Hello world"
            response = await aclient.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "response_format": "text"},
            )
        assert response.status_code == 200
        assert response.text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_verbose_json_format(self, aclient: AsyncClient) -> None:
        wav_bytes = make_wav_bytes()
        with patch("speaches.routers.stt.transcribe", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "Hello"
            response = await aclient.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "response_format": "verbose_json"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello"
        assert "duration" in data
        assert "language" in data

    @pytest.mark.asyncio
    async def test_transcribe_default_format_is_json(self, aclient: AsyncClient) -> None:
        wav_bytes = make_wav_bytes()
        with patch("speaches.routers.stt.transcribe", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "Default"
            response = await aclient.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Default"

    @pytest.mark.asyncio
    async def test_transcribe_passes_prompt_to_client(self, aclient: AsyncClient) -> None:
        wav_bytes = make_wav_bytes()
        with patch("speaches.routers.stt.transcribe", new_callable=AsyncMock) as mock_transcribe:
            mock_transcribe.return_value = "test"
            await aclient.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "test-model", "prompt": "custom prompt"},
            )
            call_kwargs = mock_transcribe.call_args
            assert "custom prompt" in call_kwargs.kwargs.get("prompt", "")


class TestTTSRouter:
    @pytest.mark.asyncio
    async def test_speech_model_not_found(self, aclient: AsyncClient) -> None:
        response = await aclient.post(
            "/v1/audio/speech",
            json={"model": "nonexistent-model", "input": "Hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_speech_model_without_tts(self, aclient: AsyncClient) -> None:
        response = await aclient.post(
            "/v1/audio/speech",
            json={"model": "stt-only-model", "input": "Hello"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_speech_streams_pcm(self, aclient: AsyncClient) -> None:
        pcm_chunk = base64.b64encode(b"\x00" * 1024).decode()

        async def mock_stream(*_args, **_kwargs) -> AsyncGenerator[dict]:
            yield {"choices": [{"delta": {"content": pcm_chunk}}]}
            yield {"choices": [{"delta": {"content": pcm_chunk}}]}

        with patch("speaches.routers.speech.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_stream()
            response = await aclient.post(
                "/v1/audio/speech",
                json={"model": "test-model", "input": "Hello world"},
            )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert len(response.content) == 2048

    @pytest.mark.asyncio
    async def test_speech_uses_audio_modality(self, aclient: AsyncClient) -> None:
        async def mock_stream(*_args, **_kwargs) -> AsyncGenerator[dict]:
            yield {"choices": [{"delta": {"content": base64.b64encode(b"\x00").decode()}}]}

        with patch("speaches.routers.speech.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_stream()
            await aclient.post(
                "/v1/audio/speech",
                json={"model": "test-model", "input": "Hello"},
            )
            call_kwargs = mock_chat.call_args
            assert call_kwargs.kwargs["modalities"] == ["audio"]
            assert call_kwargs.kwargs["stream"] is True


class TestChatRouter:
    @pytest.mark.asyncio
    async def test_chat_model_not_found(self, aclient: AsyncClient) -> None:
        response = await aclient.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_chat_model_without_chat_capability(self, aclient: AsyncClient) -> None:
        response = await aclient.post(
            "/v1/chat/completions",
            json={
                "model": "stt-only-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_chat_text_non_streaming(self, aclient: AsyncClient) -> None:
        mock_response = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello!"}, "index": 0, "finish_reason": "stop"}],
            "model": "test-model",
            "object": "chat.completion",
        }
        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_response
            response = await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "modalities": ["text"],
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_chat_streaming(self, aclient: AsyncClient) -> None:
        async def mock_stream() -> AsyncGenerator[dict]:
            yield {"choices": [{"delta": {"content": "Hi"}, "index": 0}]}
            yield {"choices": [{"delta": {"content": "!"}, "index": 0}]}

        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_stream()
            response = await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "modalities": ["text"],
                    "stream": True,
                },
            )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        # Verify SSE format
        lines = response.text.strip().split("\n")
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) >= 2  # at least 2 chunks + [DONE]

    @pytest.mark.asyncio
    async def test_chat_converts_input_audio_to_audio_url(self, aclient: AsyncClient) -> None:
        audio_b64 = base64.b64encode(b"fake-audio").decode()

        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}],
            }
            await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                            ],
                        }
                    ],
                    "modalities": ["text"],
                },
            )
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            user_content = messages[0]["content"]
            audio_parts = [p for p in user_content if p.get("type") == "audio_url"]
            assert len(audio_parts) == 1
            assert audio_parts[0]["audio_url"]["url"].startswith("data:audio/wav;base64,")

    @pytest.mark.asyncio
    async def test_chat_passes_audio_url_through(self, aclient: AsyncClient) -> None:
        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}],
            }
            await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
                            ],
                        }
                    ],
                    "modalities": ["text"],
                },
            )
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            user_content = messages[0]["content"]
            audio_parts = [p for p in user_content if p.get("type") == "audio_url"]
            assert len(audio_parts) == 1

    @pytest.mark.asyncio
    async def test_chat_passes_image_url_through(self, aclient: AsyncClient) -> None:
        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}],
            }
            await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                            ],
                        }
                    ],
                    "modalities": ["text"],
                },
            )
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            user_content = messages[0]["content"]
            image_parts = [p for p in user_content if p.get("type") == "image_url"]
            assert len(image_parts) == 1

    @pytest.mark.asyncio
    async def test_chat_passes_video_url_through(self, aclient: AsyncClient) -> None:
        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}],
            }
            await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                            ],
                        }
                    ],
                    "modalities": ["text"],
                },
            )
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            user_content = messages[0]["content"]
            video_parts = [p for p in user_content if p.get("type") == "video_url"]
            assert len(video_parts) == 1

    @pytest.mark.asyncio
    async def test_chat_default_modality_is_text(self, aclient: AsyncClient) -> None:
        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}],
            }
            await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            call_args = mock_chat.call_args
            assert call_args.kwargs["modalities"] == ["text"]

    @pytest.mark.asyncio
    async def test_chat_multimodal_message(self, aclient: AsyncClient) -> None:
        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}],
            }
            await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is this?"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,BBBB"}},
                            ],
                        }
                    ],
                    "modalities": ["text"],
                },
            )
        assert mock_chat.called

    @pytest.mark.asyncio
    async def test_chat_system_message(self, aclient: AsyncClient) -> None:
        with patch("speaches.routers.chat.chat_completion", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = {
                "choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}],
            }
            await aclient.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hi"},
                    ],
                    "modalities": ["text"],
                },
            )
            call_args = mock_chat.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"


class TestOpenAPISchema:
    @pytest.mark.asyncio
    async def test_openapi_schema_loads(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema

    @pytest.mark.asyncio
    async def test_openapi_has_expected_paths(self, aclient: AsyncClient) -> None:
        response = await aclient.get("/openapi.json")
        paths = response.json()["paths"]
        assert "/v1/models" in paths
        assert "/v1/audio/transcriptions" in paths
        assert "/v1/audio/speech" in paths
        assert "/v1/chat/completions" in paths
        assert "/health" in paths

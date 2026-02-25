import logging

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from speaches.api_types import Model
from speaches.config import VllmTtsEndpoint

logger = logging.getLogger(__name__)


class VllmTtsProxy:
    def __init__(self, endpoints: list[VllmTtsEndpoint]) -> None:
        self._endpoints: dict[str, str] = {e.model_id: e.base_url.rstrip("/") for e in endpoints}

    def can_handle(self, model_id: str) -> bool:
        return model_id in self._endpoints

    def list_models(self) -> list[Model]:
        return [
            Model(
                id=model_id,
                owned_by="vllm",
                task="text-to-speech",
            )
            for model_id in self._endpoints
        ]

    def list_voices(self, model_id: str) -> list[dict]:
        base_url = self._endpoints.get(model_id)
        if base_url is None:
            return []
        try:
            response = httpx.get(f"{base_url}/v1/audio/voices", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data.get("voices", [])
        except Exception:
            logger.exception(f"Failed to fetch voices from {base_url}")
            return []

    def list_all_voices(self) -> list[dict]:
        voices: list[dict] = []
        for model_id in self._endpoints:
            voices.extend(self.list_voices(model_id))
        return voices

    def proxy_speech_request(
        self,
        *,
        model: str,
        input_text: str,
        voice: str,
        response_format: str = "wav",
        speed: float = 1.0,
        extra_body: dict | None = None,
    ) -> StreamingResponse:
        base_url = self._endpoints.get(model)
        if base_url is None:
            raise HTTPException(status_code=404, detail=f"vLLM model '{model}' not configured")

        payload: dict = {
            "model": model,
            "input": input_text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        if extra_body:
            payload.update(extra_body)

        url = f"{base_url}/v1/audio/speech"
        logger.info(f"Proxying TTS request to {url} for model {model}")

        try:
            client = httpx.Client(timeout=300.0)
            response = client.send(
                client.build_request("POST", url, json=payload),
                stream=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.exception(f"vLLM returned error for model {model}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e)) from e
        except httpx.ConnectError as e:
            logger.exception(f"Failed to connect to vLLM at {base_url}")
            raise HTTPException(status_code=502, detail=f"Cannot reach vLLM backend at {base_url}") from e

        content_type = response.headers.get("content-type", "audio/wav")

        def stream_response():
            try:
                for chunk in response.iter_bytes(chunk_size=4096):
                    yield chunk
            finally:
                response.close()
                client.close()

        return StreamingResponse(stream_response(), media_type=content_type)

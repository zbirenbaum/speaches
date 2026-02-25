from functools import lru_cache
import logging
from pathlib import Path
import time
from typing import Annotated

from fastapi import (
    Depends,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from httpx import ASGITransport, AsyncClient
import numpy as np
from numpy import float32
from openai import AsyncOpenAI
from openai.resources.audio import AsyncTranscriptions
import soundfile as sf

from speaches.audio import Audio
from speaches.backend_registry import BackendRegistry
from speaches.config import Config
from speaches.executors.silero_vad_v5 import SileroVADModelManager

logger = logging.getLogger(__name__)


@lru_cache
def get_config() -> Config:
    return Config()


async def get_config_async() -> Config:
    return get_config()


ConfigDependency = Annotated[Config, Depends(get_config_async)]


@lru_cache
def get_backend_registry() -> BackendRegistry:
    config = get_config()
    return BackendRegistry(config.backend_config_path)


async def get_backend_registry_async() -> BackendRegistry:
    return get_backend_registry()


BackendRegistryDependency = Annotated[BackendRegistry, Depends(get_backend_registry_async)]


@lru_cache
def get_vad_model_manager() -> SileroVADModelManager:
    config = get_config()
    return SileroVADModelManager(ttl=config.vad_model_ttl)


async def get_vad_model_manager_async() -> SileroVADModelManager:
    return get_vad_model_manager()


VadModelManagerDependency = Annotated[SileroVADModelManager, Depends(get_vad_model_manager_async)]


@lru_cache
def get_transcription_client() -> AsyncTranscriptions:
    config = get_config()
    if config.loopback_host_url is None:
        from speaches.routers.stt import router as stt_router

        http_client = AsyncClient(
            transport=ASGITransport(stt_router),
            base_url="http://test/v1",
        )
    else:
        http_client = AsyncClient(
            base_url=f"{config.loopback_host_url}/v1",
        )
    oai_client = AsyncOpenAI(
        http_client=http_client,
        api_key=config.api_key.get_secret_value() if config.api_key else "cant-be-empty",
        max_retries=0,
        base_url=f"{config.loopback_host_url}/v1" if config.loopback_host_url else None,
    )
    return oai_client.audio.transcriptions


async def get_transcription_client_async() -> AsyncTranscriptions:
    return get_transcription_client()


TranscriptionClientDependency = Annotated[AsyncTranscriptions, Depends(get_transcription_client_async)]


security = HTTPBearer(auto_error=False)


async def verify_api_key(
    config: ConfigDependency, credentials: HTTPAuthorizationCredentials | None = Depends(security)
) -> None:
    assert config.api_key is not None
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key required. Please provide an API key using the Authorization header with Bearer scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != config.api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key. The provided API key is incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        )


ApiKeyDependency = Depends(verify_api_key)


def _decode_audio_file(file: UploadFile) -> np.typing.NDArray[float32]:
    if file.content_type in ("audio/pcm", "audio/raw"):
        logger.debug(f"Detected {file.content_type}, parsing as s16le monochannel")
        raw_bytes = file.file.read()
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    data, _ = sf.read(file.file, dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data


def audio_file_dependency(
    file: Annotated[UploadFile, Form()],
) -> Audio:
    try:
        logger.debug(
            f"Decoding audio file: {file.filename}, content_type: {file.content_type}, header: {file.headers}, size: {file.size}"
        )
        start = time.perf_counter()
        audio_data = _decode_audio_file(file)
        elapsed = time.perf_counter() - start
        audio = Audio(audio_data, sample_rate=16000, name=Path(file.filename).stem if file.filename else None)
        logger.debug(f"Decoded {audio.duration}s of audio in {elapsed:.5f}s (RTF: {elapsed / audio.duration})")
        return audio
    except Exception as e:
        logger.exception("Failed to decode audio")
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}") from e


AudioFileDependency = Annotated[Audio, Depends(audio_file_dependency)]

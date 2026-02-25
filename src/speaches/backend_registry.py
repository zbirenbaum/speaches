import json
import logging
import os
from pathlib import Path
import time

import httpx
from pydantic import BaseModel
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CAPABILITIES = ["stt", "tts", "chat"]
DEFAULT_INPUT_MODALITIES = ["text", "audio", "image", "video"]
DEFAULT_OUTPUT_MODALITIES = ["text", "audio"]


class BackendConfig(BaseModel):
    id: str
    base_url: str
    alias: str | None = None
    auth_user: str | None = None
    auth_password: str | None = None
    capabilities: list[str] = DEFAULT_CAPABILITIES
    input_modalities: list[str] = DEFAULT_INPUT_MODALITIES
    output_modalities: list[str] = DEFAULT_OUTPUT_MODALITIES


class BackendRegistry:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._backends: dict[str, BackendConfig] = {}
        self._last_loaded: float = 0
        self._reload_interval: float = 10.0
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._load()

    def _load(self) -> None:
        backends: dict[str, BackendConfig] = {}

        # Load from YAML file
        try:
            if self._config_path.exists():
                raw = yaml.safe_load(self._config_path.read_text())
                for entry in raw.get("backends", []):
                    config = BackendConfig(**entry)
                    lookup_key = config.alias or config.id
                    backends[lookup_key] = config
                logger.info(f"Loaded {len(backends)} backend(s) from {self._config_path}")
        except Exception:
            logger.exception(f"Failed to load backend config from {self._config_path}")

        # Load from VLLM_TTS_ENDPOINTS env var
        endpoints_json = os.environ.get("VLLM_TTS_ENDPOINTS")
        if endpoints_json:
            try:
                endpoints = json.loads(endpoints_json)
                for entry in endpoints:
                    config = BackendConfig(
                        id=entry["model_id"],
                        base_url=entry["base_url"] + "/v1",
                        alias=entry.get("alias"),
                        capabilities=entry.get("capabilities", DEFAULT_CAPABILITIES),
                        input_modalities=entry.get("input_modalities", DEFAULT_INPUT_MODALITIES),
                        output_modalities=entry.get("output_modalities", DEFAULT_OUTPUT_MODALITIES),
                    )
                    lookup_key = config.alias or config.id
                    backends[lookup_key] = config
                logger.info(f"Loaded {len(endpoints)} backend(s) from VLLM_TTS_ENDPOINTS")
            except Exception:
                logger.exception("Failed to parse VLLM_TTS_ENDPOINTS")

        self._backends = backends
        self._last_loaded = time.monotonic()

    def _maybe_reload(self) -> None:
        if time.monotonic() - self._last_loaded > self._reload_interval:
            self._load()

    def get_backend(self, model_id: str) -> BackendConfig | None:
        self._maybe_reload()
        return self._backends.get(model_id)

    def get_backends_for_capability(self, capability: str) -> list[BackendConfig]:
        self._maybe_reload()
        return [b for b in self._backends.values() if capability in b.capabilities]

    def list_models(self) -> list[BackendConfig]:
        self._maybe_reload()
        return list(self._backends.values())

    def get_http_client(self, backend: BackendConfig) -> httpx.AsyncClient:
        key = backend.base_url
        if key not in self._clients:
            auth = None
            if backend.auth_user and backend.auth_password:
                auth = httpx.BasicAuth(backend.auth_user, backend.auth_password)
            self._clients[key] = httpx.AsyncClient(
                base_url=backend.base_url,
                auth=auth,
                timeout=httpx.Timeout(timeout=180.0),
            )
        return self._clients[key]

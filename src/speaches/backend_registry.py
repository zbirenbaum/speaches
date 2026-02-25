import logging
from pathlib import Path
import time

import httpx
from pydantic import BaseModel
import yaml

logger = logging.getLogger(__name__)


class BackendConfig(BaseModel):
    id: str
    base_url: str
    auth_user: str | None = None
    auth_password: str | None = None
    capabilities: list[str]
    input_modalities: list[str]
    output_modalities: list[str]


class BackendRegistry:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._backends: dict[str, BackendConfig] = {}
        self._last_loaded: float = 0
        self._reload_interval: float = 10.0
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = yaml.safe_load(self._config_path.read_text())
            backends = {}
            for entry in raw.get("backends", []):
                config = BackendConfig(**entry)
                backends[config.id] = config
            self._backends = backends
            self._last_loaded = time.monotonic()
            logger.info(f"Loaded {len(self._backends)} backend(s) from {self._config_path}")
        except Exception:
            logger.exception(f"Failed to load backend config from {self._config_path}")

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

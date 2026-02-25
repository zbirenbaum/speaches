from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")

    backend_config_path: Path = Path("model_backends.yaml")

    vad_model_ttl: int = Field(default=-1, ge=-1)

    api_key: SecretStr | None = None
    log_level: str = "debug"
    host: str = Field(alias="UVICORN_HOST", default="0.0.0.0")
    port: int = Field(alias="UVICORN_PORT", default=8000)
    allow_origins: list[str] | None = None

    enable_ui: bool = True

    loopback_host_url: str | None = None

    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "speaches"

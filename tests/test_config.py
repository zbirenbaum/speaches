from pathlib import Path

from pydantic import SecretStr, ValidationError
import pytest

from speaches.config import Config


class TestConfigDefaults:
    def test_default_config_creates(self) -> None:
        config = Config(enable_ui=False)
        assert config.backend_config_path == Path("model_backends.yaml")
        assert config.vad_model_ttl == -1
        assert config.api_key is None
        assert config.log_level == "debug"
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.allow_origins is None
        assert config.enable_ui is False
        assert config.loopback_host_url is None
        assert config.otel_exporter_otlp_endpoint is None
        assert config.otel_service_name == "speaches"

    def test_custom_backend_config_path(self) -> None:
        config = Config(backend_config_path=Path("/custom/path.yaml"), enable_ui=False)
        assert config.backend_config_path == Path("/custom/path.yaml")

    def test_api_key_set(self) -> None:
        config = Config(api_key=SecretStr("my-key"), enable_ui=False)
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "my-key"

    def test_vad_model_ttl_never_unload(self) -> None:
        config = Config(vad_model_ttl=-1, enable_ui=False)
        assert config.vad_model_ttl == -1

    def test_vad_model_ttl_immediate_unload(self) -> None:
        config = Config(vad_model_ttl=0, enable_ui=False)
        assert config.vad_model_ttl == 0

    def test_vad_model_ttl_custom(self) -> None:
        config = Config(vad_model_ttl=300, enable_ui=False)
        assert config.vad_model_ttl == 300

    def test_vad_model_ttl_rejects_below_minus_one(self) -> None:
        with pytest.raises(ValidationError):
            Config(vad_model_ttl=-2, enable_ui=False)

    def test_allow_origins_list(self) -> None:
        config = Config(allow_origins=["http://localhost:3000", "http://localhost:3001"], enable_ui=False)
        assert config.allow_origins == ["http://localhost:3000", "http://localhost:3001"]

    def test_otel_config(self) -> None:
        config = Config(
            otel_exporter_otlp_endpoint="http://localhost:4317",
            otel_service_name="my-service",
            enable_ui=False,
        )
        assert config.otel_exporter_otlp_endpoint == "http://localhost:4317"
        assert config.otel_service_name == "my-service"


class TestConfigRemovedFields:
    def test_no_whisper_config(self) -> None:
        assert not hasattr(Config, "whisper")

    def test_no_stt_model_ttl(self) -> None:
        config = Config(enable_ui=False)
        assert not hasattr(config, "stt_model_ttl")

    def test_no_tts_model_ttl(self) -> None:
        config = Config(enable_ui=False)
        assert not hasattr(config, "tts_model_ttl")

    def test_no_chat_completion_base_url(self) -> None:
        config = Config(enable_ui=False)
        assert not hasattr(config, "chat_completion_base_url")

    def test_no_chat_completion_api_key(self) -> None:
        config = Config(enable_ui=False)
        assert not hasattr(config, "chat_completion_api_key")

    def test_no_preload_models(self) -> None:
        config = Config(enable_ui=False)
        assert not hasattr(config, "preload_models")

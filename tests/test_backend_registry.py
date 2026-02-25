from pathlib import Path
import tempfile
import time

import pytest

from speaches.backend_registry import BackendConfig, BackendRegistry


@pytest.fixture
def yaml_content() -> str:
    return """\
backends:
  - id: model-a
    base_url: http://localhost:8001/v1
    auth_user: user1
    auth_password: pass1
    capabilities: [stt, tts, chat]
    input_modalities: [text, audio]
    output_modalities: [text, audio]
  - id: model-b
    base_url: http://localhost:8002/v1
    capabilities: [stt]
    input_modalities: [text, audio]
    output_modalities: [text]
"""


@pytest.fixture
def config_file(yaml_content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        return Path(f.name)


@pytest.fixture
def registry(config_file: Path) -> BackendRegistry:
    return BackendRegistry(config_file)


class TestBackendConfigModel:
    def test_full_config(self) -> None:
        config = BackendConfig(
            id="test-model",
            base_url="http://localhost:8000/v1",
            auth_user="user",
            auth_password="pass",  # noqa: S106
            capabilities=["stt", "tts"],
            input_modalities=["text", "audio"],
            output_modalities=["text"],
        )
        assert config.id == "test-model"
        assert config.auth_user == "user"
        assert config.auth_password == "pass"  # noqa: S105

    def test_config_without_auth(self) -> None:
        config = BackendConfig(
            id="test-model",
            base_url="http://localhost:8000/v1",
            capabilities=["chat"],
            input_modalities=["text"],
            output_modalities=["text"],
        )
        assert config.auth_user is None
        assert config.auth_password is None

    def test_config_empty_capabilities(self) -> None:
        config = BackendConfig(
            id="test-model",
            base_url="http://localhost:8000/v1",
            capabilities=[],
            input_modalities=[],
            output_modalities=[],
        )
        assert config.capabilities == []


class TestBackendRegistryLoading:
    def test_loads_backends_from_yaml(self, registry: BackendRegistry) -> None:
        models = registry.list_models()
        assert len(models) == 2

    def test_get_backend_by_id(self, registry: BackendRegistry) -> None:
        backend = registry.get_backend("model-a")
        assert backend is not None
        assert backend.id == "model-a"
        assert backend.base_url == "http://localhost:8001/v1"
        assert backend.auth_user == "user1"
        assert backend.auth_password == "pass1"  # noqa: S105

    def test_get_backend_not_found(self, registry: BackendRegistry) -> None:
        backend = registry.get_backend("nonexistent-model")
        assert backend is None

    def test_get_backend_second_model(self, registry: BackendRegistry) -> None:
        backend = registry.get_backend("model-b")
        assert backend is not None
        assert backend.id == "model-b"
        assert backend.auth_user is None

    def test_list_models_returns_all(self, registry: BackendRegistry) -> None:
        models = registry.list_models()
        ids = [m.id for m in models]
        assert "model-a" in ids
        assert "model-b" in ids


class TestBackendRegistryCapabilityFiltering:
    def test_filter_by_stt(self, registry: BackendRegistry) -> None:
        backends = registry.get_backends_for_capability("stt")
        assert len(backends) == 2

    def test_filter_by_tts(self, registry: BackendRegistry) -> None:
        backends = registry.get_backends_for_capability("tts")
        assert len(backends) == 1
        assert backends[0].id == "model-a"

    def test_filter_by_chat(self, registry: BackendRegistry) -> None:
        backends = registry.get_backends_for_capability("chat")
        assert len(backends) == 1
        assert backends[0].id == "model-a"

    def test_filter_by_nonexistent_capability(self, registry: BackendRegistry) -> None:
        backends = registry.get_backends_for_capability("embedding")
        assert len(backends) == 0


class TestBackendRegistryHotReload:
    def test_reload_picks_up_changes(self, config_file: Path) -> None:
        registry = BackendRegistry(config_file)
        registry._reload_interval = 0.0  # noqa: SLF001
        assert len(registry.list_models()) == 2

        config_file.write_text("""\
backends:
  - id: new-model
    base_url: http://localhost:9000/v1
    capabilities: [chat]
    input_modalities: [text]
    output_modalities: [text]
""")
        # Force reload by setting last_loaded to the past
        registry._last_loaded = 0  # noqa: SLF001
        models = registry.list_models()
        assert len(models) == 1
        assert models[0].id == "new-model"

    def test_no_reload_within_interval(self, config_file: Path) -> None:
        registry = BackendRegistry(config_file)
        registry._reload_interval = 100.0  # noqa: SLF001
        assert len(registry.list_models()) == 2

        config_file.write_text("backends: []")
        # Should not reload because interval hasn't passed
        models = registry.list_models()
        assert len(models) == 2

    def test_reload_after_interval_elapsed(self, config_file: Path) -> None:
        registry = BackendRegistry(config_file)
        registry._reload_interval = 0.1  # noqa: SLF001
        assert len(registry.list_models()) == 2

        config_file.write_text("backends: []")
        time.sleep(0.15)
        models = registry.list_models()
        assert len(models) == 0


class TestBackendRegistryHttpClient:
    def test_get_http_client_with_auth(self, registry: BackendRegistry) -> None:
        backend = registry.get_backend("model-a")
        assert backend is not None
        client = registry.get_http_client(backend)
        assert client is not None
        assert str(client.base_url).rstrip("/") == "http://localhost:8001/v1"

    def test_get_http_client_without_auth(self, registry: BackendRegistry) -> None:
        backend = registry.get_backend("model-b")
        assert backend is not None
        client = registry.get_http_client(backend)
        assert client is not None

    def test_http_client_is_cached(self, registry: BackendRegistry) -> None:
        backend = registry.get_backend("model-a")
        assert backend is not None
        client1 = registry.get_http_client(backend)
        client2 = registry.get_http_client(backend)
        assert client1 is client2

    def test_different_backends_get_different_clients(self, registry: BackendRegistry) -> None:
        backend_a = registry.get_backend("model-a")
        backend_b = registry.get_backend("model-b")
        assert backend_a is not None
        assert backend_b is not None
        client_a = registry.get_http_client(backend_a)
        client_b = registry.get_http_client(backend_b)
        assert client_a is not client_b


class TestBackendRegistryEdgeCases:
    def test_empty_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("backends: []\n")
            path = Path(f.name)
        registry = BackendRegistry(path)
        assert len(registry.list_models()) == 0

    def test_missing_file_does_not_crash(self) -> None:
        registry = BackendRegistry(Path("/nonexistent/path/config.yaml"))
        assert len(registry.list_models()) == 0

    def test_malformed_yaml_does_not_crash(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{{{{not yaml at all!!!!")
            path = Path(f.name)
        registry = BackendRegistry(path)
        assert len(registry.list_models()) == 0

    def test_yaml_missing_backends_key(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("something_else: true\n")
            path = Path(f.name)
        registry = BackendRegistry(path)
        assert len(registry.list_models()) == 0

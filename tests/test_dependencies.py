import io
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from speaches.backend_registry import BackendRegistry
from speaches.dependencies import _decode_audio_file


class TestDecodeAudioFile:
    def test_decode_wav_file(self) -> None:
        buf = io.BytesIO()
        data = np.zeros(8000, dtype=np.float32)
        sf.write(buf, data, 16000, format="WAV")
        buf.seek(0)

        mock_file = MagicMock()
        mock_file.file = buf
        mock_file.content_type = "audio/wav"

        result = _decode_audio_file(mock_file)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 8000

    def test_decode_pcm_raw_file(self) -> None:
        pcm_data = np.zeros(8000, dtype=np.int16).tobytes()

        mock_file = MagicMock()
        mock_file.file = io.BytesIO(pcm_data)
        mock_file.content_type = "audio/pcm"

        result = _decode_audio_file(mock_file)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 8000

    def test_decode_raw_content_type(self) -> None:
        pcm_data = np.zeros(4000, dtype=np.int16).tobytes()

        mock_file = MagicMock()
        mock_file.file = io.BytesIO(pcm_data)
        mock_file.content_type = "audio/raw"

        result = _decode_audio_file(mock_file)
        assert isinstance(result, np.ndarray)
        assert len(result) == 4000

    def test_decode_stereo_to_mono(self) -> None:
        buf = io.BytesIO()
        data = np.zeros((8000, 2), dtype=np.float32)
        sf.write(buf, data, 16000, format="WAV")
        buf.seek(0)

        mock_file = MagicMock()
        mock_file.file = buf
        mock_file.content_type = "audio/wav"

        result = _decode_audio_file(mock_file)
        assert result.ndim == 1


class TestBackendRegistryDependency:
    def test_registry_created_from_config(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                "backends:\n  - id: m\n    base_url: http://x\n    capabilities: [stt]\n    input_modalities: [text]\n    output_modalities: [text]\n"
            )
            path = Path(f.name)
        registry = BackendRegistry(path)
        assert len(registry.list_models()) == 1

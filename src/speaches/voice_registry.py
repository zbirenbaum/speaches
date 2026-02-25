import base64
import logging
from pathlib import Path
import re
import shutil
import time

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SavedVoice(BaseModel):
    id: str
    name: str
    ref_text: str
    created_at: int


class VoiceRegistry:
    def __init__(self, voice_dir: Path) -> None:
        self._voice_dir = voice_dir
        self._voice_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify(name: str) -> str:
        slug = name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        slug = slug.strip("-")
        return slug or "voice"

    def _voice_path(self, voice_id: str) -> Path:
        return self._voice_dir / voice_id

    def save_voice(self, name: str, ref_audio: bytes, ref_text: str) -> SavedVoice:
        voice_id = self._slugify(name)
        voice_path = self._voice_path(voice_id)

        if voice_path.exists():
            i = 2
            while (self._voice_dir / f"{voice_id}-{i}").exists():
                i += 1
            voice_id = f"{voice_id}-{i}"
            voice_path = self._voice_path(voice_id)

        voice_path.mkdir(parents=True)

        voice = SavedVoice(
            id=voice_id,
            name=name,
            ref_text=ref_text,
            created_at=int(time.time()),
        )

        (voice_path / "metadata.json").write_text(voice.model_dump_json(indent=2))
        (voice_path / "ref_audio.wav").write_bytes(ref_audio)

        logger.info(f"Saved voice '{name}' as '{voice_id}'")
        return voice

    def get_voice(self, voice_id: str) -> SavedVoice | None:
        metadata_path = self._voice_path(voice_id) / "metadata.json"
        if not metadata_path.exists():
            return None
        return SavedVoice.model_validate_json(metadata_path.read_text())

    def get_voice_audio_path(self, voice_id: str) -> Path | None:
        audio_path = self._voice_path(voice_id) / "ref_audio.wav"
        if not audio_path.exists():
            return None
        return audio_path

    def get_voice_audio_base64(self, voice_id: str) -> str | None:
        audio_path = self.get_voice_audio_path(voice_id)
        if audio_path is None:
            return None
        return base64.b64encode(audio_path.read_bytes()).decode()

    def list_voices(self) -> list[SavedVoice]:
        voices: list[SavedVoice] = []
        if not self._voice_dir.exists():
            return voices
        for entry in sorted(self._voice_dir.iterdir()):
            if entry.is_dir():
                metadata_path = entry / "metadata.json"
                if metadata_path.exists():
                    try:
                        voices.append(SavedVoice.model_validate_json(metadata_path.read_text()))
                    except Exception:
                        logger.exception(f"Failed to load voice from {entry}")
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        voice_path = self._voice_path(voice_id)
        if not voice_path.exists():
            return False
        shutil.rmtree(voice_path)
        logger.info(f"Deleted voice '{voice_id}'")
        return True

    def has_voice(self, voice_id: str) -> bool:
        return (self._voice_path(voice_id) / "metadata.json").exists()

"""Module Audio (étape 2) : fichier audio -> texte avec openai-whisper (reconnaissance
vocale, ASR). Le modèle n'est chargé qu'à la première utilisation (chargement lent)."""
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from . import config


class ASRUnavailableError(RuntimeError):
    """openai-whisper (ou ffmpeg) n'est pas installé."""


class NoSpeechFoundError(ValueError):
    """Aucune parole détectée dans l'audio."""


@lru_cache(maxsize=1)
def _load_model():
    try:
        import whisper  # import tardif : le reste de l'API fonctionne sans whisper
    except ImportError as exc:
        raise ASRUnavailableError("openai-whisper n'est pas installé (voir README).") from exc
    return whisper.load_model(os.getenv("WHISPER_MODEL", config.WHISPER_MODEL))


def audio_path_to_text(path) -> str:
    model = _load_model()
    try:
        result = model.transcribe(str(path), language=config.ASR_LANGUAGE, fp16=False)
    except FileNotFoundError as exc:  # ffmpeg absent
        raise ASRUnavailableError("ffmpeg est requis par whisper (voir README).") from exc
    text = result.get("text", "").strip()
    if not text:
        raise NoSpeechFoundError("Aucune parole détectée dans l'audio.")
    return text


def transcribe_audio(data: bytes, suffix: str = ".wav") -> str:
    """Transcrit des octets audio (whisper lit un fichier : on passe par un temporaire)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return audio_path_to_text(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

"""API REST FastAPI : classifieur de sentiment multi-source (texte / image / audio).

Lancer :  uvicorn main:app --reload      puis ouvrir http://127.0.0.1:8000/docs

Architecture en cascade : image --OCR--> texte ; audio --ASR--> texte ;
puis UN SEUL classifieur textuel (src/predictor.py) prend la décision finale.

Les routes sont des `def` (et non `async def`) volontairement : OCR et Whisper sont
des calculs bloquants ; FastAPI exécute alors chaque requête dans un thread séparé
et le serveur reste réactif.
"""
import io
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from src import config
from src.asr import ASRUnavailableError, NoSpeechFoundError, transcribe_audio
from src.ocr import NoTextFoundError, OCRUnavailableError, image_bytes_to_text
from src.predictor import get_predictor
from src.schemas import PredictionResponse, TextRequest

app = FastAPI(
    title="Classifieur de sentiment multi-source",
    description="Prédit le sentiment (positif / négatif / neutre) d'un avis client "
                "fourni sous forme de texte, d'image (OCR) ou d'audio (ASR).",
    version="1.0.0",
)


# --------------------------------------------------------------------------- utilitaires
def _extension(filename: Optional[str]) -> str:
    return os.path.splitext(filename or "")[1].lower()


def _is_allowed(file: UploadFile, mimes: set, exts: set) -> bool:
    """Type MIME déclaré valide, ou MIME générique + extension valide (cas de curl/navigateurs)."""
    ctype = (file.content_type or "").lower().split(";")[0].strip()
    if ctype in mimes:
        return True
    return ctype in {"", "application/octet-stream"} and _extension(file.filename) in exts


def _read_upload(file: UploadFile, mimes: set, exts: set, max_bytes: int, kind: str) -> bytes:
    """Valide le type MIME et la taille, puis renvoie le contenu."""
    if not _is_allowed(file, mimes, exts):
        raise HTTPException(
            status_code=422,
            detail=f"Type de fichier non supporté pour {kind} : '{file.content_type}'. "
                   f"Extensions acceptées : {sorted(exts)}.",
        )
    data = file.file.read(max_bytes + 1)  # on lit 1 octet de trop pour détecter un dépassement
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (max {max_bytes // (1024 * 1024)} Mo).")
    if not data:
        raise HTTPException(status_code=422, detail="Le fichier envoyé est vide.")
    return data


def _classify(source: str, text: str) -> PredictionResponse:
    sentiment, confiance = get_predictor().predict(text)
    return PredictionResponse(source=source, texte_extrait=text, sentiment_predit=sentiment, confiance=confiance)


def _process_image(file: UploadFile) -> PredictionResponse:
    data = _read_upload(file, config.ALLOWED_IMAGE_MIMES, config.ALLOWED_IMAGE_EXT, config.MAX_IMAGE_BYTES, "une image")
    try:  # le MIME peut mentir : on vérifie que les octets sont bien une image
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(status_code=422, detail="Le fichier n'est pas une image valide ou est corrompu.")
    try:
        text = image_bytes_to_text(data)
    except NoTextFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OCRUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _classify("image", text)


def _process_audio(file: UploadFile) -> PredictionResponse:
    data = _read_upload(file, config.ALLOWED_AUDIO_MIMES, config.ALLOWED_AUDIO_EXT, config.MAX_AUDIO_BYTES, "un audio")
    try:
        text = transcribe_audio(data, suffix=_extension(file.filename) or ".wav")
    except NoSpeechFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ASRUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:  # fichier audio illisible par ffmpeg/whisper
        raise HTTPException(status_code=422, detail="Le fichier audio est illisible ou corrompu.")
    return _classify("audio", text)


# --------------------------------------------------------------------------- routes
@app.get("/health", tags=["monitoring"])
def health():
    """Vérifie que l'API tourne et que le modèle est chargeable."""
    get_predictor()
    return {"status": "ok"}


@app.post("/predict/text", response_model=PredictionResponse, tags=["prédiction"])
def predict_text(payload: TextRequest):
    """Texte (JSON) -> sentiment."""
    return _classify("texte", payload.texte)


@app.post("/predict/image", response_model=PredictionResponse, tags=["prédiction"])
def predict_image(file: UploadFile = File(..., description="Capture d'écran (PNG/JPEG, max 5 Mo)")):
    """Image (multipart) -> OCR -> sentiment."""
    return _process_image(file)


@app.post("/predict/audio", response_model=PredictionResponse, tags=["prédiction"])
def predict_audio(file: UploadFile = File(..., description="Message vocal (WAV/MP3, max 10 Mo)")):
    """Audio (multipart) -> transcription -> sentiment."""
    return _process_audio(file)


@app.post("/predict", response_model=PredictionResponse, tags=["prédiction"])
def predict_auto(
    text: Optional[str] = Form(None, description="Avis en texte (champ de formulaire)"),
    file: Optional[UploadFile] = File(None, description="OU un fichier image/audio"),
):
    """Détection automatique : envoyer SOIT `text`, SOIT `file` (image ou audio)."""
    has_text = text is not None and text.strip() != ""
    if has_text == (file is not None):
        raise HTTPException(status_code=422, detail="Envoyez exactement un champ : 'text' OU 'file'.")
    if has_text:
        try:  # mêmes règles de validation que /predict/text
            payload = TextRequest(texte=text)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_context=False))
        return _classify("texte", payload.texte)
    if _is_allowed(file, config.ALLOWED_IMAGE_MIMES, config.ALLOWED_IMAGE_EXT):
        return _process_image(file)
    if _is_allowed(file, config.ALLOWED_AUDIO_MIMES, config.ALLOWED_AUDIO_EXT):
        return _process_audio(file)
    raise HTTPException(status_code=422, detail="Type de fichier non reconnu (image PNG/JPEG ou audio WAV/MP3 attendus).")

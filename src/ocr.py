"""Module Image (étape 2) : image -> texte brut avec pytesseract + Pillow."""
import io
import re

import pytesseract
from PIL import Image, ImageOps

from . import config


class OCRUnavailableError(RuntimeError):
    """Tesseract n'est pas installé sur la machine."""


class NoTextFoundError(ValueError):
    """L'OCR n'a trouvé aucun texte exploitable dans l'image."""


def _available_langs() -> set:
    try:
        return set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError("Tesseract n'est pas installé (voir README).") from exc


def _pick_lang() -> str:
    """'fra+eng' si possible, sinon on se replie sur les langues disponibles."""
    wanted = [l for l in config.OCR_LANGS.split("+")]
    available = _available_langs()
    usable = [l for l in wanted if l in available]
    return "+".join(usable) if usable else "eng"


def preprocess_image(image: Image.Image) -> Image.Image:
    """Niveaux de gris + autocontraste + agrandissement x2 : améliore nettement la
    lecture des petits caractères par Tesseract."""
    gray = ImageOps.autocontrast(image.convert("L"))
    return gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)


def _drop_banner(text: str) -> str:
    """Retire le bandeau 'CAPTURE AVIS CLIENT — Ixx.png' propre au dataset fourni."""
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and re.search(r"capture\s+avis\s+client", lines[0], flags=re.I):
        lines = lines[1:]
    return "\n".join(lines)


def image_to_text(image: Image.Image) -> str:
    lang = _pick_lang()
    try:
        raw = pytesseract.image_to_string(preprocess_image(image), lang=lang, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError("Tesseract n'est pas installé (voir README).") from exc
    if config.OCR_DROP_BANNER:
        raw = _drop_banner(raw)
    text = re.sub(r"\s+", " ", raw).strip()
    if not text:
        raise NoTextFoundError("Aucun texte détecté dans l'image.")
    return text


def image_bytes_to_text(data: bytes) -> str:
    return image_to_text(Image.open(io.BytesIO(data)))


def image_path_to_text(path) -> str:
    with Image.open(path) as img:
        return image_to_text(img)

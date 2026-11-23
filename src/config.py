"""Configuration centrale du projet (chemins, limites de validation, constantes)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "dataset_sentiment_ecommerce"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TEXT_CSV = DATA_DIR / "textes.csv"
IMAGES_CSV = DATA_DIR / "images.csv"
AUDIOS_CSV = DATA_DIR / "audios.csv"

MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

LABELS = ["negatif", "neutre", "positif"]  # mêmes libellés que le dataset

# --- Limites de validation des charges utiles --------------------------------
MAX_TEXT_CHARS = 5_000
MAX_IMAGE_BYTES = 5 * 1024 * 1024   # 5 Mo
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 Mo

ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
# Le sujet parle de .wav, mais les fichiers fournis dans le dataset sont des .mp3 :
# on accepte donc les deux formats.
ALLOWED_AUDIO_MIMES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
    "audio/mpeg", "audio/mp3",
}
ALLOWED_AUDIO_EXT = {".wav", ".mp3"}

# --- OCR / ASR ----------------------------------------------------------------
OCR_LANGS = "fra+eng"        # repli automatique sur "eng" si le pack français est absent
WHISPER_MODEL = "base"       # modifiable via la variable d'environnement WHISPER_MODEL
ASR_LANGUAGE = "fr"
# Les images du dataset contiennent un bandeau "CAPTURE AVIS CLIENT — Ixx.png"
# qui n'a rien à voir avec l'avis : on l'ignore (heuristique propre à ce dataset).
OCR_DROP_BANNER = True

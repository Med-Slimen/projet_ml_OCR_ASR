"""Tests d'intégration de l'API (étape 5) : 200 + structure JSON, et 422 sur charges invalides.

Lancer :  pytest -v
Le TestClient de FastAPI repose sur httpx. L'ASR (Whisper, lent) est remplacé par un
faux dans les tests rapides ; un test réel est exécuté seulement si whisper est installé.
"""
import shutil

import pytest
from fastapi.testclient import TestClient

import main
from src import config

client = TestClient(main.app)
VALID_LABELS = {"positif", "negatif", "neutre"}


def assert_valid_prediction(body: dict, source: str):
    assert set(body) == {"source", "texte_extrait", "sentiment_predit", "confiance"}
    assert body["source"] == source
    assert isinstance(body["texte_extrait"], str) and body["texte_extrait"]
    assert body["sentiment_predit"] in VALID_LABELS
    assert 0.0 <= body["confiance"] <= 1.0


def read_dataset(folder: str, name: str) -> bytes:
    return (config.DATA_DIR / folder / name).read_bytes()


# ------------------------------------------------------------------ /health
def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


# ------------------------------------------------------------------ /predict/text
@pytest.mark.parametrize("text,expected", [
    ("Excellent produit, livraison rapide, je recommande !", "positif"),
    ("Article cassé, service client injoignable, très déçu.", "negatif"),
    ("Le colis a été livré mardi à l'adresse indiquée.", "neutre"),
])
def test_predict_text_ok(text, expected):
    r = client.post("/predict/text", json={"texte": text})
    assert r.status_code == 200
    body = r.json()
    assert_valid_prediction(body, "texte")
    assert body["sentiment_predit"] == expected


def test_predict_text_accepts_english_key():
    assert client.post("/predict/text", json={"text": "Super produit, bravo"}).status_code == 200


@pytest.mark.parametrize("payload", [
    {},                                  # champ manquant
    {"texte": ""},                       # vide
    {"texte": "     "},                  # espaces seulement
    {"texte": 12345},                    # mauvais type
    {"texte": None},                     # null
    {"texte": "x" * (config.MAX_TEXT_CHARS + 1)},  # trop long
])
def test_predict_text_invalid_payload_422(payload):
    assert client.post("/predict/text", json=payload).status_code == 422


def test_predict_text_not_json_422():
    r = client.post("/predict/text", content="pas du json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ------------------------------------------------------------------ /predict/image
@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract non installé")
def test_predict_image_ok():
    files = {"file": ("I02.png", read_dataset("images", "I02.png"), "image/png")}
    r = client.post("/predict/image", files=files)
    assert r.status_code == 200
    body = r.json()
    assert_valid_prediction(body, "image")
    assert body["sentiment_predit"] == "negatif"  # I02 = réclamation


def test_predict_image_wrong_mime_422():
    files = {"file": ("note.txt", b"du texte", "text/plain")}
    assert client.post("/predict/image", files=files).status_code == 422


def test_predict_image_corrupted_422():
    files = {"file": ("fake.png", b"ceci n'est pas une image", "image/png")}
    assert client.post("/predict/image", files=files).status_code == 422


def test_predict_image_empty_422():
    files = {"file": ("empty.png", b"", "image/png")}
    assert client.post("/predict/image", files=files).status_code == 422


def test_predict_image_too_large_413():
    files = {"file": ("big.png", b"0" * (config.MAX_IMAGE_BYTES + 1), "image/png")}
    assert client.post("/predict/image", files=files).status_code == 413


def test_predict_image_missing_file_422():
    assert client.post("/predict/image").status_code == 422


# ------------------------------------------------------------------ /predict/audio
@pytest.fixture
def fake_asr(monkeypatch):
    monkeypatch.setattr(main, "transcribe_audio",
                        lambda data, suffix=".wav": "Très mécontent de la livraison, le produit est inutilisable.")


def test_predict_audio_ok_with_mocked_asr(fake_asr):
    files = {"file": ("msg.wav", b"RIFF....fake", "audio/wav")}
    r = client.post("/predict/audio", files=files)
    assert r.status_code == 200
    body = r.json()
    assert_valid_prediction(body, "audio")
    assert body["sentiment_predit"] == "negatif"


def test_predict_audio_wrong_mime_422(fake_asr):
    files = {"file": ("msg.pdf", b"%PDF-1.4", "application/pdf")}
    assert client.post("/predict/audio", files=files).status_code == 422


def test_predict_audio_too_large_413(fake_asr):
    files = {"file": ("big.wav", b"0" * (config.MAX_AUDIO_BYTES + 1), "audio/wav")}
    assert client.post("/predict/audio", files=files).status_code == 413


def test_predict_audio_missing_file_422():
    assert client.post("/predict/audio").status_code == 422


def test_predict_audio_real_whisper():
    pytest.importorskip("whisper")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg non installé")
    files = {"file": ("A02.mp3", read_dataset("audios", "A02.mp3"), "audio/mpeg")}
    r = client.post("/predict/audio", files=files)
    assert r.status_code == 200
    assert_valid_prediction(r.json(), "audio")


# ------------------------------------------------------------------ /predict (auto)
def test_auto_text():
    r = client.post("/predict", data={"text": "Très satisfait, produit parfait !"})
    assert r.status_code == 200
    assert_valid_prediction(r.json(), "texte")


def test_auto_audio(fake_asr):
    files = {"file": ("msg.mp3", b"ID3fake", "audio/mpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    assert_valid_prediction(r.json(), "audio")


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract non installé")
def test_auto_image():
    files = {"file": ("I01.png", read_dataset("images", "I01.png"), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    assert_valid_prediction(r.json(), "image")


def test_auto_nothing_sent_422():
    assert client.post("/predict").status_code == 422


def test_auto_both_sent_422():
    r = client.post("/predict", data={"text": "bonjour"}, files={"file": ("a.png", b"x", "image/png")})
    assert r.status_code == 422


def test_auto_unknown_file_type_422():
    assert client.post("/predict", files={"file": ("a.zip", b"PK", "application/zip")}).status_code == 422

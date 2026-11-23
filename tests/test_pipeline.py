"""Tests unitaires des briques ML (sans FastAPI) : nettoyage, lexique, prédicteur, OCR."""
import shutil

import pytest

from src import config
from src.features import lexicon_features
from src.preprocessing import clean_text, normalize
from src.predictor import get_predictor


def test_normalize_removes_accents_and_punctuation():
    assert normalize("Très déçu !!! 100% d'accord") == "tres decu d accord"


def test_clean_text_keeps_negations_and_drops_stopwords():
    out = clean_text("Ce n'est pas une réussite, le produit")
    assert "pas" in out.split()                      # la négation est conservée
    assert "le" not in out.split() and "une" not in out.split()  # les stop-words partent


def test_lexicon_detects_polarity_and_negation():
    pos, neg, *_ = lexicon_features("Produit excellent, je recommande")
    assert pos > 0 and neg == 0
    pos, neg, *_ = lexicon_features("Je ne suis pas satisfait")
    assert neg > 0 and pos == 0  # "pas satisfait" est inversé


@pytest.mark.parametrize("text,expected", [
    ("Produit excellent, livraison rapide, je recommande vivement !", "positif"),
    ("Colis cassé, service client injoignable, très déçu.", "negatif"),
    ("Le colis a été livré mardi à l'adresse indiquée.", "neutre"),
])
def test_predictor_on_clear_examples(text, expected):
    label, conf = get_predictor().predict(text)
    assert label == expected
    assert 0.0 <= conf <= 1.0


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract non installé")
def test_ocr_reads_dataset_image():
    from src.ocr import image_path_to_text
    text = image_path_to_text(config.DATA_DIR / "images" / "I02.png").lower()
    assert "colis" in text and "client" in text

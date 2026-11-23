"""Classifieur de sentiment final : texte brut -> (sentiment, confiance).
C'est LA brique commune aux trois modalités (texte, image, audio)."""
import joblib
from functools import lru_cache

from . import config
from .features import featurize
from .preprocessing import clean_text


class SentimentPredictor:
    def __init__(self, model, vectorizer, lexicon_weight: float = 3.0, use_lexicon: bool = True):
        self.model, self.vectorizer = model, vectorizer
        self.lexicon_weight, self.use_lexicon = lexicon_weight, use_lexicon

    def predict(self, text: str):
        X = featurize([text], self.vectorizer, clean_text, self.lexicon_weight, self.use_lexicon)
        proba = self.model.predict_proba(X)[0]
        best = int(proba.argmax())
        return str(self.model.classes_[best]), round(float(proba[best]), 4)


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    import json
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl introuvable : lancez d'abord `python -m src.train`.")
    meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8")) if config.METADATA_PATH.exists() else {}
    return SentimentPredictor(
        joblib.load(config.MODEL_PATH), joblib.load(config.VECTORIZER_PATH),
        meta.get("lexicon_weight", 3.0), meta.get("use_lexicon", True),
    )

"""Schémas Pydantic : validation stricte des entrées et de la réponse unifiée."""
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from . import config


class TextRequest(BaseModel):
    """Corps JSON de POST /predict/text : {"texte": "..."} (la clé "text" est aussi acceptée)."""
    model_config = ConfigDict(json_schema_extra={"example": {"texte": "Livraison rapide, produit parfait !"}})

    texte: str = Field(
        validation_alias=AliasChoices("texte", "text"),
        min_length=1,
        max_length=config.MAX_TEXT_CHARS,
        description="Avis client à analyser (1 à 5000 caractères).",
    )

    @field_validator("texte")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Le texte ne peut pas être vide ou composé uniquement d'espaces.")
        return v


class PredictionResponse(BaseModel):
    """Réponse JSON unifiée, identique pour les trois modalités."""
    source: Literal["texte", "image", "audio"]
    texte_extrait: str = Field(description="Texte analysé (saisi, lu par OCR ou transcrit par ASR).")
    sentiment_predit: Literal["positif", "negatif", "neutre"]
    confiance: float = Field(ge=0.0, le=1.0, description="Probabilité de la classe prédite.")

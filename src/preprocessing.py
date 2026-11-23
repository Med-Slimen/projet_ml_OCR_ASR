"""Nettoyage minimal du texte (étape 1 du sujet).

Choix importants, à pouvoir expliquer à l'oral :
- minuscules + suppression des accents : l'OCR (surtout sans pack de langue
  français) et l'ASR perdent ou changent souvent les accents ; normaliser rend
  le modèle insensible à ces différences ;
- suppression de la ponctuation et des chiffres ;
- suppression des stop-words FRANÇAIS, mais on GARDE les négations
  (ne, pas, jamais, plus, sans, rien...) : "pas satisfait" != "satisfait".
"""
import re
import unicodedata


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """minuscules, sans accents, sans ponctuation/chiffres, espaces nettoyés."""
    text = strip_accents(str(text).lower())
    text = re.sub(r"[^a-z\s]", " ", text)   # apostrophes, chiffres, symboles -> espace
    return re.sub(r"\s+", " ", text).strip()


# Liste volontairement courte et SANS négations (pas de dépendance à nltk).
_STOPWORDS_RAW = """
le la les un une des du de d l et ou a au aux en dans sur sous avec pour par
ce cet cette ces mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs
je tu il elle on nous vous ils elles me te se lui y qui que quoi dont
est sont etait ete etre ai as avons avez ont suis es sommes
c s j m n t qu ca cela ceci comme tout tous toute toutes meme aussi mais donc or ni car si alors
"""
STOPWORDS = frozenset(strip_accents(_STOPWORDS_RAW).split())
NEGATIONS = frozenset({"ne", "pas", "non", "jamais", "rien", "aucun", "aucune",
                       "sans", "plus", "impossible"})
assert not (STOPWORDS & NEGATIONS)  # garde-fou : on ne supprime jamais une négation


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Pipeline de nettoyage complet utilisé à l'entraînement ET en production."""
    tokens = normalize(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

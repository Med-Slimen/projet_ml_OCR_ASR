"""Vectorisation : TF-IDF + petit lexique de sentiment français.

POURQUOI un lexique en plus du TF-IDF ?
Le dataset ne contient que 24 textes : presque chaque mot n'apparaît qu'une
fois. Un TF-IDF seul apprend donc "par cœur" des mots isolés et ne généralise
pas à des phrases nouvelles (voir reports/model_comparison.csv : ~33 % de
précision en validation croisée, soit le hasard pour 3 classes).
Le lexique injecte une connaissance générale de la langue (mots d'appréciation,
de plainte, formules factuelles) sous forme de 5 nombres par texte, sans ajouter
aucune donnée d'entraînement.

Les entrées sont des préfixes de mots SANS accents (ex. "recommand" couvre
recommande, recommandé, recommanderai...).
"""
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack

from .preprocessing import NEGATIONS, normalize

POSITIVE_STEMS = (
    "excellen", "parfait", "super", "magnifi", "genial", "formidable", "fantastique",
    "ravi", "satisf", "content", "heureu", "bravo", "felicit", "merci", "recommand",
    "impecc", "exemplair", "incroyable", "bluff", "superbe", "agreable", "efficace",
    "rapide", "rapidite", "reactiv", "gentil", "soigne", "elegant", "fiable", "imbattable",
    "apprecie", "meritee", "meilleur", "top", "adore", "aime", "qualite superieure",
    "intuitif", "fluide", "etoiles", "cinq etoiles", "bonne surprise", "attention",
)
NEGATIVE_STEMS = (
    "decu", "mecontent", "scandaleu", "arnaque", "inadmissible", "inutilisable",
    "defectueu", "casse", "cassee", "fragile", "fissur", "mauvais", "mauvaise", "nul",
    "horrible", "catastroph", "deplorable", "decevant", "deception", "retard", "injoignable",
    "manquant", "endommag", "ecrase", "abime", "erreur", "panne", "probleme", "incompatib",
    "rembours", "fuyez", "honte", "pire", "lent", "complique", "insatisf", "furieu",
    "inacceptable", "regrette", "jamais recu", "pas recu", "attente", "litige", "plainte",
    "reclamation", "double", "debite", "impossible",
)
NEUTRAL_STEMS = (
    "conforme", "correct", "standard", "indique", "annonce", "prevu", "correspond",
    "specification", "dimension", "notice", "statut", "confirm", "recept", "livree",
    "livre", "recupere", "installation", "sans plus", "rien de particulier", "habituel",
    "moyen", "ordinaire", "classique", "fonctionne comme",
)

# Verbes de "dysfonctionnement" : "ne fonctionne pas", "ne s'allume plus", etc.
_FAILURE_PATTERN = re.compile(
    r"\bne (?:fonctionne|marche|repond|tient|s allume|demarre|charge|s active|se charge)\b"
    r"|\bn (?:a|ai|est) (?:pas|jamais) (?:fonctionne|marche|repondu|ete|arrive)\b"
    r"|\bplus de (?:reponse|nouvelles)\b|\bpersonne ne repond\b|\bsans (?:aucune )?(?:explication|reponse|nouvelle)\b"
)

FEATURE_NAMES = ["n_pos", "n_neg", "n_neutral", "has_negation", "no_polarity"]


def _count(tokens_text: str, stems) -> int:
    n = 0
    for stem in stems:
        # préfixe de mot : " stem..." (le texte est entouré d'espaces)
        n += len(re.findall(r"(?<![a-z])" + re.escape(stem), tokens_text))
    return n


def lexicon_features(text: str) -> list:
    """5 nombres décrivant un texte (comptes dans le lexique + négation)."""
    norm = " " + normalize(text) + " "
    tokens = norm.split()
    n_pos, n_neg, n_neu = (_count(norm, s) for s in (POSITIVE_STEMS, NEGATIVE_STEMS, NEUTRAL_STEMS))

    n_neg += len(_FAILURE_PATTERN.findall(norm))

    # Négation devant un mot positif ("pas satisfait") : on inverse sa polarité.
    for i, tok in enumerate(tokens):
        if any(tok.startswith(s) for s in POSITIVE_STEMS):
            window = tokens[max(0, i - 3):i]
            if any(w in NEGATIONS and w not in {"plus", "sans"} for w in window):
                n_pos -= 1
                n_neg += 1

    n_pos = max(n_pos, 0)
    has_neg = int(any(t in {"pas", "jamais", "rien", "aucun", "aucune", "impossible"} for t in tokens))
    no_pol = int(n_pos == 0 and n_neg == 0)
    return [min(n_pos, 3), min(n_neg, 3), min(n_neu, 3), has_neg, no_pol]


def featurize(raw_texts, vectorizer, clean_fn, lexicon_weight: float = 1.0, use_lexicon: bool = True):
    """Matrice finale = [TF-IDF | lexique]. Fonction UNIQUE utilisée à l'entraînement
    et à la prédiction (évite tout décalage entre les deux)."""
    cleaned = [clean_fn(t) for t in raw_texts]
    tfidf = vectorizer.transform(cleaned)
    if not use_lexicon:
        return tfidf
    lex = np.array([lexicon_features(t) for t in raw_texts], dtype=float) * lexicon_weight
    return hstack([tfidf, csr_matrix(lex)]).tocsr()

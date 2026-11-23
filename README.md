# Classifieur de sentiment multi-source (E-commerce Customer Voice)

Un avis client peut arriver en **texte**, en **image** (capture d'écran) ou en **audio** (message vocal).
Architecture **en cascade** : image → OCR → texte ; audio → ASR → texte ; puis **un seul classifieur
textuel** prédit `positif`, `negatif` ou `neutre`. Le tout est exposé par une API FastAPI et testable
via Swagger ou une mini-app Streamlit.

```
Texte ─► /predict/text ───────────────────────┐
Image ─► /predict/image : OCR (pytesseract) ──┼─► classifieur texte ─► {source, texte_extrait,
Audio ─► /predict/audio : ASR (whisper) ──────┤                         sentiment_predit, confiance}
Auto  ─► /predict (détecte le type) ──────────┘
```

## Structure
```
main.py                  API FastAPI (routes, validation des fichiers)
streamlit_app.py         mini-application web de test
src/
  config.py              chemins, limites (taille, types MIME)
  preprocessing.py       nettoyage du texte
  features.py            TF-IDF + lexique de sentiment
  train.py               entraînement, évaluation, sauvegarde des .pkl
  ocr.py / asr.py        image→texte (Tesseract) / audio→texte (Whisper)
  predictor.py           texte → (sentiment, confiance)
  schemas.py             schémas Pydantic
  evaluate.py            évaluation de bout en bout
models/                  model.pkl, vectorizer.pkl, metadata.json
reports/                 matrices de confusion, rapports, comparaison des modèles
data/                    dataset fourni (inchangé) + sanity_check.csv (21 phrases de contrôle)
tests/                   test_api.py, test_pipeline.py
EXPLICATIONS.md          rôle de chaque étape + résultats + limites (pour la soutenance)
```

## Installation
Python 3.10+ recommandé.
```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```
Deux programmes **externes** sont nécessaires en plus des paquets Python :

| Programme | Pour | Installation |
|---|---|---|
| **Tesseract** | OCR | Windows : installateur UB-Mannheim (cocher le pack *French*) · Ubuntu : `sudo apt install tesseract-ocr tesseract-ocr-fra` · macOS : `brew install tesseract tesseract-lang` |
| **ffmpeg** | lecture audio par Whisper | Windows : `winget install ffmpeg` · Ubuntu : `sudo apt install ffmpeg` · macOS : `brew install ffmpeg` |

Sous Windows, si Tesseract n'est pas dans le PATH, ajoutez en haut de `src/ocr.py` :
`pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"`.
Sans le pack français, l'OCR se replie automatiquement sur l'anglais (accents mal lus, sans gravité car le texte est normalisé).

## Utilisation
```bash
python -m src.train          # (ré)entraîne le modèle et écrit models/*.pkl + reports/*   (modèles déjà fournis)
python -m src.evaluate       # évaluation de bout en bout (OCR + ASR + phrases nouvelles)
uvicorn main:app --reload    # lance l'API  → http://127.0.0.1:8000/docs  (Swagger)  et /redoc
streamlit run streamlit_app.py   # dans un 2e terminal : mini-app → http://localhost:8501
pytest -v                    # tests automatisés
```
La première transcription audio télécharge le modèle Whisper `base` (~140 Mo) : prévoir une connexion internet
et quelques secondes. Autre taille : `WHISPER_MODEL=small uvicorn main:app`.

### Exemples d'appels
```bash
curl -X POST http://127.0.0.1:8000/predict/text -H "Content-Type: application/json" \
     -d '{"texte": "Livraison rapide, produit parfait !"}'
curl -X POST http://127.0.0.1:8000/predict/image -F "file=@data/dataset_sentiment_ecommerce/images/I02.png"
curl -X POST http://127.0.0.1:8000/predict/audio -F "file=@data/dataset_sentiment_ecommerce/audios/A02.mp3"
curl -X POST http://127.0.0.1:8000/predict -F "text=Très déçu de mon achat"
```
Réponse (identique pour toutes les routes) :
```json
{"source": "texte", "texte_extrait": "Livraison rapide, produit parfait !", "sentiment_predit": "positif", "confiance": 0.99}
```
Codes d'erreur : `422` (champ manquant/vide/trop long, type de fichier non supporté, image corrompue, aucun texte détecté),
`413` (fichier trop volumineux : 5 Mo image, 10 Mo audio), `503` (Tesseract/Whisper non installé).

## Résultats (détails et limites dans `EXPLICATIONS.md`)
| Évaluation | Résultat |
|---|---|
| TF-IDF seul, validation croisée (24 textes) | ≈ 27 % (hasard = 33 %) |
| TF-IDF + lexique, hold-out (6 textes) | 100 % |
| Références OCR+ASR (18 phrases hors entraînement) | 100 % |
| 21 phrases nouvelles écrites pour le contrôle | 100 % |
| OCR sur les 9 images → sentiment | 9/9, similarité de texte 0,98 |

⚠️ Dataset très petit (24 textes) et lexique écrit en connaissant ces textes : ces 100 % sont **optimistes**.
Voir la section « Limites » de `EXPLICATIONS.md`.

## Captures d'écran à ajouter (livrable demandé)
Placez vos captures dans `docs/` et référencez-les ici :
- `docs/swagger.png` : page `/docs` avec une requête exécutée ;
- `docs/streamlit_texte.png`, `docs/streamlit_image.png`, `docs/streamlit_audio.png` : l'application en fonctionnement.

## Notes sur le dataset fourni
- `audios.csv` référence des `.wav` alors que les fichiers sont des `.mp3` : l'API accepte les deux formats.
- Les images contiennent un bandeau « CAPTURE AVIS CLIENT — Ixx.png » ignoré par l'OCR (`OCR_DROP_BANNER` dans `src/config.py`).
"# projet_ml_OCR_ASR" 

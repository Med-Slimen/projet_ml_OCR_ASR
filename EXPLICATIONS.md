# Explications du projet (guide pour la soutenance)

> L'enseignant évalue la **compréhension** : le rôle de chaque étape et les résultats obtenus.
> Pour chaque étape : *à quoi elle sert → ce qui a été fait → résultat → limite.*

## 0. Idée générale : l'architecture en cascade
Trois formats d'entrée, un seul problème : « quel est le sentiment de cet avis ? ». Plutôt que d'entraîner trois
modèles, on **convertit tout en texte** (OCR pour l'image, ASR pour l'audio) puis un **unique classifieur textuel**
décide. Avantages : un seul modèle à maintenir, une seule API de décision, une réponse JSON identique.
Inconvénient : les erreurs de l'OCR/ASR se propagent au classifieur (« garbage in, garbage out »).

## Étape 1 – Classifieur texte (`src/preprocessing.py`, `features.py`, `train.py`)
**Rôle** : apprendre à associer un texte à `positif`, `negatif` ou `neutre`.

**Données** : `textes.csv`, 24 avis, 8 par classe (jeu parfaitement équilibré → l'accuracy est interprétable, le hasard vaut 33 %).

**Nettoyage** : minuscules, suppression des accents et de la ponctuation, suppression des *stop-words* français
(« le », « de »…). Deux choix à justifier :
- **les négations sont conservées** (« pas », « ne », « jamais », « plus », « sans ») car « pas satisfait » ≠ « satisfait » ;
- **les accents sont supprimés** car l'OCR (surtout sans pack français) et l'ASR les altèrent (on l'a observé : « aprés », « Modeéele »). Le modèle devient insensible à ces défauts.

**Vectorisation** : `TfidfVectorizer` (unigrammes + bigrammes, `sublinear_tf`). TF-IDF transforme un texte en vecteur où
chaque mot pèse d'autant plus qu'il est fréquent dans le texte mais rare dans le corpus.

**Modèle** : régression logistique (choisie plutôt que le SVM : elle donne des **probabilités** → champ `confiance`).

### Le résultat central (et la décision qu'il a imposée)
| Configuration (validation croisée 4 plis × 10 répétitions) | Accuracy |
|---|---|
| A. TF-IDF + régression logistique (baseline du sujet) | **0,27 ± 0,14** |
| B. TF-IDF + SVM | 0,27 ± 0,13 |
| C. TF-IDF + **lexique** + régression logistique (retenue) | 1,00 |
| D. TF-IDF + lexique + SVM | 1,00 |

La baseline classique est **au niveau du hasard**. Explication : avec 24 phrases, presque chaque mot n'apparaît qu'une fois ;
quand on teste sur un texte jamais vu, il partage très peu de vocabulaire avec l'entraînement, et les mots communs
(« livraison », « client », « produit ») apparaissent dans toutes les classes. Le modèle mémorise au lieu de généraliser.

**Solution retenue** : ajouter à chaque texte 5 nombres issus d'un **petit lexique de sentiment français** (`features.py`) :
nombre de mots positifs, négatifs, « factuels » (conforme, indiqué, dimensions…), présence d'une négation,
absence de toute polarité. Une négation devant un mot positif l'inverse (« pas satisfait » → négatif).
Cela apporte une connaissance générale de la langue **sans ajouter de données d'entraînement** (le dataset reste tel quel).
La classe *neutre* est apprise comme « avis factuel, sans mot de plainte ni d'éloge ».

### Évaluation (`reports/`)
- Hold-out stratifié (6 textes) : 100 % ; matrice de confusion `confusion_matrix_holdout.png`.
- Test hors échantillon : les 18 phrases de référence de `images.csv` et `audios.csv` (jamais utilisées pour entraîner) : 100 %.
- 21 phrases nouvelles (`data/sanity_check.csv`) : 100 %.
- Modèle final **ré-entraîné sur les 24 textes** puis sauvegardé (`model.pkl`, `vectorizer.pkl`).

### ⚠️ Limites à reconnaître honnêtement (très apprécié par un jury)
1. **Le lexique a été écrit après avoir lu le dataset** : les 100 % de la validation croisée et du hold-out sont donc optimistes
   (le lexique « connaît » déjà ces textes). Les 21 phrases de contrôle n'ont pas servi à régler quoi que ce soit,
   **mais** elles ont été écrites par la même personne, dans un style simple et canonique.
2. Un vrai jeu d'avis (ironie, avis mitigés « bien mais trop cher », argot, fautes) donnerait un score nettement plus bas.
3. 24 textes ne permettent pas de mesurer finement la performance : un texte faux = −4 points.
4. Le lexique est fixe et limité ; il est à enrichir (ou à remplacer par des *embeddings* / un modèle pré-entraîné) avec plus de données.
5. La confiance (≈ 0,8 à 1,0) reflète la certitude du modèle, pas la probabilité réelle d'avoir raison.

## Étape 2 – Prétraitement multimodal
### Image → OCR (`src/ocr.py`)
**Rôle** : lire le texte d'une capture d'écran. **Outil** : Tesseract via `pytesseract`, image préparée avec Pillow
(niveaux de gris, autocontraste, agrandissement ×2 : Tesseract lit mieux les caractères plus grands et contrastés).
`--psm 6` = « un bloc de texte uniforme ». Le bandeau « CAPTURE AVIS CLIENT — Ixx.png » du dataset est ignoré (il ne fait pas partie de l'avis).
**Résultat** (`python -m src.evaluate`, pack français absent) : similarité moyenne texte lu / texte attendu **0,98**, sentiment correct **9/9**.
**Limite** : photo floue, écriture manuscrite ou police fantaisie dégraderaient fortement l'OCR.

### Audio → ASR (`src/asr.py`)
**Rôle** : transcrire la parole. **Outil** : `openai-whisper`, modèle `base`, langue forcée `fr`, chargé une seule fois (chargement lent).
Whisper s'appuie sur **ffmpeg** pour lire les fichiers ; l'API écrit l'audio reçu dans un fichier temporaire puis le supprime.
Le sujet évoque des `.wav` mais le dataset contient des `.mp3` : les deux sont acceptés.
**Résultat** : ⚠️ l'ASR n'a **pas pu être exécuté** dans l'environnement où le projet a été préparé (pas d'accès réseau
pour installer Whisper). **Lancez `python -m src.evaluate` chez vous** et reportez ici la similarité et l'accuracy obtenues.
**Limite** : un modèle `base` fait plus d'erreurs que `small`/`medium` sur du français bruité.

## Étape 3 – API REST (`main.py`, `src/schemas.py`)
- Routes `POST /predict/text` (JSON), `/predict/image` et `/predict/audio` (multipart), `/predict` (détection automatique : on envoie soit `text`, soit `file`), et `GET /health`.
- **Pydantic** : `TextRequest` impose 1 à 5000 caractères non vides (clé `texte`, ou `text`) ; `PredictionResponse` impose les trois valeurs possibles du sentiment et une confiance entre 0 et 1 → la réponse est **garantie** conforme.
- **Validation des fichiers** : type MIME *ou* extension (certains clients envoient `application/octet-stream`), taille maximale (5 Mo / 10 Mo → `413`), fichier non vide, et pour les images vérification que les octets sont réellement une image (un MIME peut mentir). Erreurs de charge utile → `422`.
- Routes écrites en `def` : OCR et Whisper sont bloquants ; FastAPI les exécute dans des threads, le serveur reste disponible.
- Modèle chargé **une seule fois** (`lru_cache`), pas à chaque requête.

## Étape 4 – Application de test
Option A : Swagger (`/docs`) fourni par FastAPI. Option B (réalisée) : `streamlit_app.py`, trois onglets (texte / image / audio),
appelle l'API avec `requests`, affiche le sentiment en couleur, une barre de confiance et le texte extrait par OCR/ASR.

## Étape 5 – Tests (`tests/`)
`test_api.py` (httpx via `TestClient`) : pour chaque route, un cas `200` avec contrôle de la **structure JSON**
(4 clés, valeurs autorisées, confiance ∈ [0,1]) et des cas d'erreur `422` (champ absent, vide, mauvais type, trop long, MIME
non supporté, image corrompue) et `413` (fichier trop gros). L'ASR est **simulé** (mock) dans les tests rapides pour ne pas
dépendre d'un modèle de 140 Mo ; un test avec le vrai Whisper s'exécute seulement s'il est installé. Les tests OCR sont ignorés si Tesseract est absent.
`test_pipeline.py` : tests unitaires du nettoyage, du lexique (dont la négation) et du prédicteur.

## Questions probables à l'oral
- **Pourquoi garder « pas » et « ne » ?** Parce qu'ils inversent le sens.
- **Pourquoi supprimer les accents ?** L'OCR et l'ASR les abîment ; on normalise pour être robuste.
- **Pourquoi le TF-IDF seul échoue-t-il ?** 24 textes, vocabulaire quasi disjoint → mémorisation, pas de généralisation (0,27 en CV).
- **Pourquoi un lexique, n'est-ce pas de la triche ?** C'est de la connaissance du domaine injectée sous forme de variables, pas des données ajoutées ; mais écrit après lecture du dataset, il rend les scores optimistes — d'où le jeu de contrôle et la section limites.
- **Pourquoi la régression logistique ?** Simple, rapide, et fournit des probabilités pour la confiance.
- **Que faire pour améliorer ?** Plus d'avis réels annotés, validation croisée sérieuse, embeddings de phrases ou modèle français pré-entraîné (CamemBERT), post-traitement de l'OCR.
- **Pourquoi `def` et pas `async def` ?** Calculs bloquants (OCR/ASR) → threads.
- **Pourquoi des mocks dans les tests ?** Rapidité, indépendance vis-à-vis d'un gros modèle et du réseau.

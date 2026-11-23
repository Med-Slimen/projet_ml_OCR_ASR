"""Étape 1 : entraînement, évaluation et sauvegarde du classifieur texte.

Lancer :  python -m src.train

Déroulé (chaque bloc = une explication à donner à l'oral) :
 1. Chargement de textes.csv (24 avis, 8 par classe : jeu parfaitement équilibré).
 2. Nettoyage (src/preprocessing.py).
 3. COMPARAISON de plusieurs configurations en validation croisée répétée
    (4 plis x 10 répétitions) : avec 24 textes, un seul découpage train/test
    donnerait un score très instable, donc on moyenne.
 4. Évaluation détaillée de la config retenue sur un hold-out stratifié
    (matrice de confusion + classification_report).
 5. Test "hors échantillon" : les textes de référence d'images.csv et audios.csv
    (18 phrases étiquetées, jamais utilisées pour entraîner) simulent ce que le
    classifieur recevra après OCR / ASR parfaits.
 6. Ré-entraînement sur les 24 textes et sauvegarde (model.pkl, vectorizer.pkl).
"""
import json
import warnings
from datetime import date

import joblib
import matplotlib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

from . import config
from .features import featurize
from .preprocessing import clean_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def make_classifier(kind="logreg", C=10.0):
    if kind == "svm":
        return LinearSVC(C=C)
    return LogisticRegression(C=C, max_iter=5000)


def fit_eval(train_texts, train_y, test_texts, test_y, *, kind, use_lexicon, weight, C):
    vec = make_vectorizer().fit([clean_text(t) for t in train_texts])
    Xtr = featurize(train_texts, vec, clean_text, weight, use_lexicon)
    Xte = featurize(test_texts, vec, clean_text, weight, use_lexicon)
    clf = make_classifier(kind, C).fit(Xtr, train_y)
    return clf.predict(Xte)


def cross_validate(texts, y, **cfg):
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=RANDOM_STATE)
    scores = []
    for tr, te in cv.split(texts, y):
        pred = fit_eval([texts[i] for i in tr], [y[i] for i in tr],
                        [texts[i] for i in te], [y[i] for i in te], **cfg)
        scores.append(accuracy_score([y[i] for i in te], pred))
    return pd.Series(scores)


def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(config.TEXT_CSV)
    texts, y = df["texte"].tolist(), df["sentiment_ref"].tolist()
    print(f"[1] {len(df)} textes chargés ; répartition :\n{df['sentiment_ref'].value_counts().to_string()}\n")

    # --- 3. Comparaison de configurations ------------------------------------
    configs = {
        "A. TF-IDF seul + LogReg (baseline du sujet)": dict(kind="logreg", use_lexicon=False, weight=1.0, C=10.0),
        "B. TF-IDF seul + SVM": dict(kind="svm", use_lexicon=False, weight=1.0, C=1.0),
        "C. TF-IDF + lexique + LogReg": dict(kind="logreg", use_lexicon=True, weight=3.0, C=10.0),
        "D. TF-IDF + lexique + SVM": dict(kind="svm", use_lexicon=True, weight=3.0, C=1.0),
    }
    rows = []
    for name, cfg in configs.items():
        s = cross_validate(texts, y, **cfg)
        rows.append({"configuration": name, "accuracy_moyenne_CV": round(s.mean(), 3),
                     "ecart_type": round(s.std(), 3)})
        print(f"[3] {name:48s} accuracy CV = {s.mean():.3f} ± {s.std():.3f}")
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print("    (hasard pour 3 classes équilibrées = 0.333)\n")

    final_cfg = configs["C. TF-IDF + lexique + LogReg"]  # LogReg : fournit predict_proba -> "confiance"

    # --- 4. Hold-out stratifié -------------------------------------------------
    tr_x, te_x, tr_y, te_y = train_test_split(texts, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    pred = fit_eval(tr_x, tr_y, te_x, te_y, **final_cfg)
    report = classification_report(te_y, pred, labels=config.LABELS, zero_division=0)
    print(f"[4] Hold-out ({len(te_x)} textes de test)\n{report}")
    cm = confusion_matrix(te_y, pred, labels=config.LABELS)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=config.LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice de confusion (hold-out)")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix_holdout.png", dpi=150)
    plt.close(fig)

    # --- 5. Test hors échantillon : références OCR + ASR ---------------------
    ocr_ref = pd.read_csv(config.IMAGES_CSV).rename(columns={"texte_attendu_ocr": "texte"})
    asr_ref = pd.read_csv(config.AUDIOS_CSV).rename(columns={"transcription_reference": "texte"})
    ext = pd.concat([ocr_ref[["texte", "sentiment_ref"]], asr_ref[["texte", "sentiment_ref"]]], ignore_index=True)
    pred_ext = fit_eval(texts, y, ext["texte"].tolist(), ext["sentiment_ref"].tolist(), **final_cfg)
    report_ext = classification_report(ext["sentiment_ref"], pred_ext, labels=config.LABELS, zero_division=0)
    print(f"[5] Test hors échantillon ({len(ext)} phrases images+audios, modèle entraîné sur les 24 textes)\n{report_ext}")
    cm_ext = confusion_matrix(ext["sentiment_ref"], pred_ext, labels=config.LABELS)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm_ext, display_labels=config.LABELS).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title("Matrice de confusion (références OCR+ASR)")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix_external.png", dpi=150)
    plt.close(fig)

    (config.REPORTS_DIR / "classification_report.txt").write_text(
        "=== Hold-out (6 textes) ===\n" + report + "\n=== Références OCR + ASR (18 phrases) ===\n" + report_ext,
        encoding="utf-8")

    # --- 6. Ré-entraînement final sur les 24 textes et sauvegarde ------------
    vec = make_vectorizer().fit([clean_text(t) for t in texts])
    X = featurize(texts, vec, clean_text, final_cfg["weight"], final_cfg["use_lexicon"])
    clf = make_classifier(final_cfg["kind"], final_cfg["C"]).fit(X, y)
    joblib.dump(clf, config.MODEL_PATH)
    joblib.dump(vec, config.VECTORIZER_PATH)
    metadata = {
        "date_entrainement": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "classes": list(clf.classes_),
        "n_textes_entrainement": len(texts),
        "lexicon_weight": final_cfg["weight"],
        "use_lexicon": final_cfg["use_lexicon"],
        "accuracy_holdout": round(accuracy_score(te_y, pred), 3),
        "f1_macro_holdout": round(f1_score(te_y, pred, average="macro"), 3),
        "accuracy_externe_ocr_asr": round(accuracy_score(ext["sentiment_ref"], pred_ext), 3),
        "comparaison_CV": comparison.to_dict(orient="records"),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[6] Modèle sauvegardé : {config.MODEL_PATH.name}, {config.VECTORIZER_PATH.name}")


if __name__ == "__main__":
    main()

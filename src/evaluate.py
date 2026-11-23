"""Évaluation de bout en bout (à lancer après l'entraînement) :  python -m src.evaluate

 1. Jeu "sanity check" : 21 phrases NOUVELLES (data/sanity_check.csv), jamais vues
    à l'entraînement -> estimation honnête de la généralisation du classifieur.
 2. Images : OCR réel des 9 images -> classifieur ; qualité de l'OCR (similarité
    avec le texte attendu) + exactitude finale.
 3. Audios : transcription Whisper réelle des 9 fichiers -> classifieur
    (ignorée automatiquement si whisper / ffmpeg ne sont pas installés).
Le rapport est écrit dans reports/end_to_end_report.txt.
"""
import difflib
import shutil

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from . import config
from .predictor import get_predictor


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    pred = get_predictor()
    out = []

    d = pd.read_csv(config.ROOT / "data" / "sanity_check.csv")
    p = [pred.predict(t)[0] for t in d["texte"]]
    out.append(f"== 1. Phrases nouvelles ({len(d)}) : accuracy = {accuracy_score(d['sentiment_ref'], p):.3f}")
    out.append(classification_report(d["sentiment_ref"], p, zero_division=0))

    from .ocr import image_path_to_text
    im = pd.read_csv(config.IMAGES_CSV)
    preds, sims = [], []
    for _, r in im.iterrows():
        text = image_path_to_text(config.DATA_DIR / "images" / r["fichier"])
        preds.append(pred.predict(text)[0])
        sims.append(sim(text, r["texte_attendu_ocr"]))
    out.append(f"== 2. Images ({len(im)}) : similarité OCR moyenne = {sum(sims) / len(sims):.3f} ; "
               f"accuracy sentiment = {accuracy_score(im['sentiment_ref'], preds):.3f}")

    try:
        import whisper  # noqa: F401
        assert shutil.which("ffmpeg")
        from .asr import audio_path_to_text
        au = pd.read_csv(config.AUDIOS_CSV)
        preds, sims = [], []
        for _, r in au.iterrows():
            f = config.DATA_DIR / "audios" / r["fichier"].replace(".wav", ".mp3")
            text = audio_path_to_text(f)
            preds.append(pred.predict(text)[0])
            sims.append(sim(text, r["transcription_reference"]))
        out.append(f"== 3. Audios ({len(au)}) : similarité ASR moyenne = {sum(sims) / len(sims):.3f} ; "
                   f"accuracy sentiment = {accuracy_score(au['sentiment_ref'], preds):.3f}")
    except (ImportError, AssertionError):
        out.append("== 3. Audios : ignoré (installez openai-whisper + ffmpeg pour l'évaluer)")

    report = "\n".join(out)
    print(report)
    (config.REPORTS_DIR / "end_to_end_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

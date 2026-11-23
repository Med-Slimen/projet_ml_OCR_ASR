"""Mini-application web de test (étape 4, option B) : interface Streamlit qui interroge l'API.

Lancer l'API d'abord :  uvicorn main:app --reload
Puis :                  streamlit run streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
COLORS = {"positif": "#2e9e4f", "negatif": "#d64545", "neutre": "#7a7f87"}
EMOJI = {"positif": "😊", "negatif": "😠", "neutre": "😐"}

st.set_page_config(page_title="Sentiment multi-source", page_icon="💬")
st.title("💬 Classifieur de sentiment multi-source")
st.caption("Texte, image (OCR) ou audio (ASR) → un seul classifieur textuel en cascade.")

with st.sidebar:
    st.header("API")
    api_url = st.text_input("URL de l'API", API_URL)
    try:
        ok = requests.get(f"{api_url}/health", timeout=3).ok
    except requests.RequestException:
        ok = False
    st.success("API connectée") if ok else st.error("API injoignable : lancez `uvicorn main:app --reload`")


def show_result(resp: requests.Response):
    if resp.status_code != 200:
        st.error(f"Erreur {resp.status_code} : {resp.json().get('detail', resp.text)}")
        return
    r = resp.json()
    label, conf = r["sentiment_predit"], r["confiance"]
    st.markdown(
        f"<h2 style='color:{COLORS[label]}'>{EMOJI[label]} {label.upper()}</h2>", unsafe_allow_html=True
    )
    st.progress(min(max(conf, 0.0), 1.0), text=f"Confiance : {conf:.0%}")
    st.markdown(f"**Source :** {r['source']}")
    st.text_area("Texte analysé (extrait / transcrit)", r["texte_extrait"], height=100, disabled=True)


tab_text, tab_img, tab_audio = st.tabs(["📝 Texte", "🖼️ Image", "🎙️ Audio"])

with tab_text:
    txt = st.text_area("Saisissez un avis client", height=120)
    if st.button("Analyser le texte", type="primary"):
        if not txt.strip():
            st.warning("Saisissez un texte.")
        else:
            show_result(requests.post(f"{api_url}/predict/text", json={"texte": txt}, timeout=30))

with tab_img:
    img = st.file_uploader("Capture d'écran d'un avis", type=["png", "jpg", "jpeg"])
    if img:
        st.image(img, use_container_width=True)
        if st.button("Analyser l'image", type="primary"):
            files = {"file": (img.name, img.getvalue(), img.type)}
            with st.spinner("OCR en cours..."):
                show_result(requests.post(f"{api_url}/predict/image", files=files, timeout=120))

with tab_audio:
    aud = st.file_uploader("Message vocal", type=["wav", "mp3"])
    if aud:
        st.audio(aud)
        if st.button("Analyser l'audio", type="primary"):
            files = {"file": (aud.name, aud.getvalue(), aud.type or "audio/mpeg")}
            with st.spinner("Transcription en cours (la 1re fois, le modèle Whisper se charge)..."):
                show_result(requests.post(f"{api_url}/predict/audio", files=files, timeout=300))

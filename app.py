import tempfile
from pathlib import Path

import streamlit as st

from core.extract_text import extract_text
from core.tts import tts_to_mp3_file

st.set_page_config(page_title="Text to Speech for my Peach", layout="centered")
st.title("Text to Speech")

voice = st.selectbox("Voice", ["marin", "cedar", "alloy", "nova"], index=0)

f = st.file_uploader("Upload a .txt, .docx, or .pdf", type=["txt", "docx", "pdf"])

if f:
    data = f.read()
    text = extract_text(f.name, data).strip()

    st.text_area("Extracted text (preview)", text[:4000], height=200)

    if st.button("Read it"):
        # keep it simple: cap length for MVP
        text_for_tts = text[:4000]

        tmp = Path(tempfile.gettempdir()) / "readaloud.mp3"
        mp3_path = tts_to_mp3_file(text_for_tts, str(tmp), voice=voice)

        st.audio(str(mp3_path), format="audio/mp3")

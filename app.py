import json
import time
import uuid
from pathlib import Path

import streamlit as st

from core.extract_text import extract_text
from core.tts import tts_to_mp3_file

# ---- simple local storage ----
DATA_DIR = Path.cwd() / "data"
AUDIO_DIR = DATA_DIR / "audio"
TEXT_DIR = DATA_DIR / "text"
HISTORY_PATH = DATA_DIR / "history.json"

for d in (AUDIO_DIR, TEXT_DIR):
    d.mkdir(parents=True, exist_ok=True)

def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return []

def save_history(items: list[dict]) -> None:
    HISTORY_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")

# ---- UI ----
st.set_page_config(page_title="TextToVoice", layout="centered")
st.title("TextToVoice")

history = load_history()
history_sorted = sorted(history, key=lambda x: x["created_at"], reverse=True)

# Sidebar: history picker
st.sidebar.header("History")

selected = None
if history_sorted:
    ids = [item["id"] for item in history_sorted]

    def label_for(item_id: str) -> str:
        item = next(x for x in history_sorted if x["id"] == item_id)
        return f'{item["created_at"]} — {item["title"]} ({item["voice"]})'

    selected_id = st.sidebar.selectbox(
        "Pick something to play",
        ids,
        format_func=label_for,
    )

    selected = next(x for x in history_sorted if x["id"] == selected_id)
else:
    st.sidebar.caption("No history yet.")

# Main: show selected history item
if selected:
    st.subheader("Selected from history")
    st.write(f'**Title:** {selected["title"]}  |  **Voice:** {selected["voice"]}  |  **Created:** {selected["created_at"]}')

    audio_path = Path(selected["audio_path"])
    text_path = Path(selected["text_path"])

    if audio_path.exists():
        st.audio(str(audio_path), format="audio/mp3")
    else:
        st.warning("Audio file missing.")

    if text_path.exists():
        st.text_area("Saved text", text_path.read_text(encoding="utf-8", errors="ignore"), height=250)
    else:
        st.warning("Text file missing.")

st.divider()

# Upload + generate
voice = st.selectbox("Voice", ["marin", "cedar", "alloy", "nova"], index=0)
f = st.file_uploader("Upload a .txt, .docx, or .pdf", type=["txt", "docx", "pdf"])

if f:
    raw = f.read()  # this is NOT saved anywhere
    text = extract_text(f.name, raw).strip()

    st.text_area("Extracted text (preview)", text[:4000], height=200)

    if st.button("Read it"):
        item_id = uuid.uuid4().hex
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        text_path = TEXT_DIR / f"{item_id}.txt"
        audio_path = AUDIO_DIR / f"{item_id}.mp3"

        # save extracted text (not the original uploaded file)
        text_path.write_text(text, encoding="utf-8")

        # generate audio (MVP cap)
        tts_to_mp3_file(text[:4000], str(audio_path), voice=voice)

        # append to history.json
        history.append(
            {
                "id": item_id,
                "created_at": created_at,
                "title": f.name,
                "voice": voice,
                "text_path": str(text_path),
                "audio_path": str(audio_path),
            }
        )
        save_history(history)

        st.rerun()
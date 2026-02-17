# app.py
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

from core.extract_text import extract_text
from core.tts import tts_to_mp3_file
from core.stt import stt_mini_transcribe_to_json

# ----------------------------
# Local storage
# ----------------------------
DATA_DIR = Path.cwd() / "data"
ITEMS_DIR = DATA_DIR / "items"
HISTORY_PATH = DATA_DIR / "history.json"
ITEMS_DIR.mkdir(parents=True, exist_ok=True)

TTS_MAX_CHARS = 4096  # OpenAI Speech max input length :contentReference[oaicite:0]{index=0}


def load_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_history() -> list[dict]:
    return load_json(HISTORY_PATH, [])


def save_history(items: list[dict]) -> None:
    save_json(HISTORY_PATH, items)


def now_pt_string() -> str:
    dt = datetime.now(ZoneInfo("America/Los_Angeles"))
    return dt.strftime("%Y-%m-%d %I:%M %p PT")  # no seconds, AM/PM PT


def audio_autoplay(mp3_path: Path, key: str) -> None:
    if not mp3_path.exists():
        st.warning("Audio missing.")
        return
    b64 = base64.b64encode(mp3_path.read_bytes()).decode("utf-8")
    components.html(
        f"""
        <audio id="peachy" controls autoplay style="width:100%">
          <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg" />
        </audio>
        <script>
          const a = document.getElementById("peachy");
          a.load();
          a.play().catch(()=>{{}});
        </script>
        """,
        height=70,
        key=key,
    )


def scroll_box(text: str, height_px: int = 360) -> None:
    safe = (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    st.markdown(
        f"""
        <style>
          .box {{
            height: {height_px}px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 12px;
            border-radius: 12px;
            line-height: 1.35;
            font-size: 16px;
          }}
        </style>
        <div class="box">{safe}</div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Peachy", layout="centered")
st.title("Peachy 🍑")

history = load_history()
history_sorted = sorted(history, key=lambda x: x["created_at"], reverse=True)
ids = [x["id"] for x in history_sorted]

# session state
if "mode" not in st.session_state:
    st.session_state.mode = "playback"  # "new" | "playback"
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "pending_doc" not in st.session_state:
    st.session_state.pending_doc = None
if "force_history_select" not in st.session_state:
    st.session_state.force_history_select = None

# default selection (BEFORE widgets)
if ids and st.session_state.selected_id is None:
    st.session_state.selected_id = ids[0]

if "history_select" not in st.session_state and st.session_state.selected_id:
    st.session_state.history_select = st.session_state.selected_id

# if we just generated a new item, sync the dropdown on the next run (before widget)
if st.session_state.force_history_select and st.session_state.force_history_select in ids:
    st.session_state.history_select = st.session_state.force_history_select
    st.session_state.force_history_select = None


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    if st.button("New Peachy", use_container_width=True):
        st.session_state.mode = "new"
        st.session_state.pending_doc = None
        st.rerun()

    st.divider()
    st.header("History")

    if not ids:
        st.caption("No history yet.")
    else:
        def fmt(item_id: str) -> str:
            it = next(x for x in history_sorted if x["id"] == item_id)
            return f'{it["created_at"]} — {it["title"]}'

        picked = st.selectbox("Select", ids, format_func=fmt, key="history_select")

        if picked != st.session_state.selected_id:
            st.session_state.selected_id = picked
            st.session_state.mode = "playback"
            st.rerun()


# ----------------------------
# Main: New view
# ----------------------------
if st.session_state.mode == "new":
    st.subheader("Create a new Peachy :)")

    voice = st.selectbox(
        "Voice",
        ["marin", "cedar", "alloy", "nova", "coral", "sage", "shimmer", "verse", "ash", "ballad", "echo", "fable", "onyx"],
        index=0,
        key="new_voice",
    )
    speed = st.slider("Speed", 0.70, 1.30, 1.00, 0.05, key="new_speed")

    up = st.file_uploader("Upload .txt / .docx / .pdf", type=["txt", "docx", "pdf"], key="new_upload_main")

    if up:
        raw = up.read()
        text = extract_text(up.name, raw).strip()  # preview immediately (no OpenAI)
        st.session_state.pending_doc = {"title": up.name, "text": text}
    else:
        st.session_state.pending_doc = None

    if st.session_state.pending_doc:
        st.caption(f'Preview: {st.session_state.pending_doc["title"]}')
        scroll_box(st.session_state.pending_doc["text"], height_px=360)

        if st.button("Generate audio", use_container_width=True):
            item_id = uuid.uuid4().hex
            created_at = now_pt_string()

            item_dir = ITEMS_DIR / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            title = st.session_state.pending_doc["title"]
            full_text = st.session_state.pending_doc["text"]

            # Save extracted text (not the original upload)
            (item_dir / "full.txt").write_text(full_text, encoding="utf-8")

            # 1) TTS (single call, max 4096 chars)
            audio_path = item_dir / "audio.mp3"
            tts_input = full_text[:TTS_MAX_CHARS]
            tts_to_mp3_file(tts_input, str(audio_path), voice=voice, speed=speed)

            # 2) STT mini (transcribe the generated audio)
            # Note: gpt-4o-mini-transcribe is the STT mini model :contentReference[oaicite:1]{index=1}
            with st.spinner("Creating transcript…"):
                stt_json = stt_mini_transcribe_to_json(str(audio_path))
                save_json(item_dir / "stt.json", stt_json)
                (item_dir / "stt.txt").write_text(stt_json.get("text", ""), encoding="utf-8")

            history.append(
                {
                    "id": item_id,
                    "created_at": created_at,
                    "title": title,
                    "voice": voice,
                    "speed": speed,
                    "item_dir": str(item_dir),
                }
            )
            save_history(history)

            st.session_state.selected_id = item_id
            st.session_state.force_history_select = item_id
            st.session_state.mode = "playback"
            st.session_state.pending_doc = None
            st.rerun()

    else:
        st.caption("Upload a file to see a preview, then generate audio.")


# ----------------------------
# Main: Playback view
# ----------------------------
else:
    if not ids:
        st.subheader("Create a new Peachy :)")
    else:
        selected = next((x for x in history_sorted if x["id"] == st.session_state.selected_id), None)
        if not selected:
            st.subheader("Create a new Peachy :)")
        else:
            item_dir = Path(selected["item_dir"])
            audio_path = item_dir / "audio.mp3"
            text_path = item_dir / "full.txt"
            stt_path = item_dir / "stt.txt"

            st.subheader("Playback")
            st.write(
                f'**Title:** {selected["title"]}  |  '
                f'**Voice:** {selected.get("voice","")}  |  '
                f'**Speed:** {selected.get("speed", 1.0)}  |  '
                f'**Created:** {selected["created_at"]}'
            )

            audio_autoplay(audio_path, key=f"player_{selected['id']}")

            if text_path.exists():
                st.markdown("#### Extracted text")
                scroll_box(text_path.read_text(encoding="utf-8", errors="ignore"), height_px=420)

            if stt_path.exists():
                with st.expander("Transcript (STT mini)"):
                    st.write(stt_path.read_text(encoding="utf-8", errors="ignore"))

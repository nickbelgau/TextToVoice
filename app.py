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
from core.stt import whisper_segments_verbose_json, extract_segments

# ----------------------------
# Config
# ----------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
TTS_MAX_CHARS = 4096

DATA_DIR = Path.cwd() / "data"
ITEMS_DIR = DATA_DIR / "items"
HISTORY_PATH = DATA_DIR / "history.json"
ITEMS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
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
    return dt.strftime("%Y-%m-%d %I:%M %p PT")


def fmt_mmss(seconds: float) -> str:
    s = int(max(0, seconds))
    m = s // 60
    ss = s % 60
    return f"{m}:{ss:02d}"


def scroll_box(text: str, height_px: int) -> None:
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


def audio_player_seek(mp3_path: Path, seek_to: float, marker: str) -> None:
    """
    Reliable seek+play: render in an iframe (components.html) so the script always runs.
    `marker` should change when selection/seek changes to force a fresh DOM id.
    """
    if not mp3_path.exists():
        st.warning("Audio missing.")
        return

    b64 = base64.b64encode(mp3_path.read_bytes()).decode("utf-8")
    dom_id = f"peachy_{marker}_{uuid.uuid4().hex}"
    seek = float(seek_to or 0.0)

    components.html(
        f"""
        <div style="width:100%;">
          <audio id="{dom_id}" controls style="width:100%;">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg" />
          </audio>
        </div>
        <script>
          (function() {{
            const a = document.getElementById("{dom_id}");
            const seekTo = {seek};

            const start = () => {{
              try {{
                a.currentTime = Math.max(0, seekTo);
              }} catch(e) {{}}
              a.play().catch(()=>{{}});
            }};

            if (a.readyState >= 1) {{
              start();
            }} else {{
              a.addEventListener("loadedmetadata", start, {{ once: true }});
              a.load();
            }}
          }})();
        </script>
        """,
        height=90,
    )


# ----------------------------
# App state
# ----------------------------
st.set_page_config(page_title="Peachy", layout="centered")

history = load_history()
history_sorted = sorted(history, key=lambda x: x["created_at"], reverse=True)
ids = [x["id"] for x in history_sorted]

if "mode" not in st.session_state:
    st.session_state.mode = "playback"  # "new" | "playback"
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "pending_doc" not in st.session_state:
    st.session_state.pending_doc = None
if "force_select_id" not in st.session_state:
    st.session_state.force_select_id = None
if "seek_to" not in st.session_state:
    st.session_state.seek_to = 0.0
if "seek_nonce" not in st.session_state:
    st.session_state.seek_nonce = 0

# Defaults BEFORE widgets
if ids and st.session_state.selected_id is None:
    st.session_state.selected_id = ids[0]

if "history_select" not in st.session_state and st.session_state.selected_id:
    st.session_state.history_select = st.session_state.selected_id

if st.session_state.force_select_id and st.session_state.force_select_id in ids:
    st.session_state.history_select = st.session_state.force_select_id
    st.session_state.force_select_id = None


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("Peachy 🍑")

    if st.button("New Peachy", use_container_width=True):
        st.session_state.mode = "new"
        st.session_state.pending_doc = None
        st.session_state.seek_to = 0.0
        st.session_state.seek_nonce += 1
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
            st.session_state.seek_to = 0.0
            st.session_state.seek_nonce += 1
            st.rerun()


# ----------------------------
# Main: New view
# ----------------------------
if st.session_state.mode == "new":
    st.subheader("Create a new Peachy :)")

    voice = st.selectbox("Voice", ["marin", "cedar", "alloy", "nova"], index=0, key="new_voice")
    speed = st.slider("Speed", 0.70, 1.30, 1.00, 0.05, key="new_speed")

    up = st.file_uploader(
        "Upload .txt / .docx / .pdf (max 10 MB)",
        type=["txt", "docx", "pdf"],
        key="new_upload_main",
    )

    st.session_state.pending_doc = None
    if up:
        # enforce 10MB
        size = getattr(up, "size", None)
        raw = up.read()
        if size is None:
            size = len(raw)

        if size > MAX_UPLOAD_BYTES:
            st.error("File too large. Max is 10 MB.")
        else:
            text = extract_text(up.name, raw).strip()  # preview immediately (no OpenAI)
            st.session_state.pending_doc = {"title": up.name, "text": text}

    if st.session_state.pending_doc:
        st.caption(f'Preview: {st.session_state.pending_doc["title"]}')
        scroll_box(st.session_state.pending_doc["text"], height_px=180)  # smaller preview

        if st.button("Generate audio + timestamped transcript", use_container_width=True):
            item_id = uuid.uuid4().hex
            created_at = now_pt_string()

            item_dir = ITEMS_DIR / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            title = st.session_state.pending_doc["title"]
            full_text = st.session_state.pending_doc["text"]

            (item_dir / "full.txt").write_text(full_text, encoding="utf-8")

            audio_path = item_dir / "audio.mp3"
            segments_path = item_dir / "segments.json"
            stt_verbose_path = item_dir / "stt_verbose.json"

            status = st.status("Working…", expanded=True)

            try:
                status.write("Step 1/2: Generating audio…")
                tts_to_mp3_file(full_text[:TTS_MAX_CHARS], str(audio_path), voice=voice, speed=speed)
                status.write("✅ Audio generated")

                status.write("Step 2/2: Transcribing with Whisper (segments)…")
                stt_verbose = whisper_segments_verbose_json(str(audio_path))
                segments = extract_segments(stt_verbose)
                save_json(stt_verbose_path, stt_verbose)
                save_json(segments_path, segments)
                status.write("✅ Transcript created")

                status.update(label="Done", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Failed", state="error", expanded=True)
                st.exception(e)
                st.stop()

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
            st.session_state.force_select_id = item_id
            st.session_state.mode = "playback"
            st.session_state.pending_doc = None
            st.session_state.seek_to = 0.0
            st.session_state.seek_nonce += 1
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
            segments = load_json(item_dir / "segments.json", [])
            full_text_path = item_dir / "full.txt"

            st.subheader("Playback")
            st.write(
                f'**Title:** {selected["title"]}  |  '
                f'**Voice:** {selected.get("voice","")}  |  '
                f'**Speed:** {selected.get("speed", 1.0)}  |  '
                f'**Created:** {selected["created_at"]}'
            )

            # This is the only widget that can reliably seek-to-time in Streamlit.
            # st.audio cannot be programmatically seeked.
            audio_player_seek(
                audio_path,
                seek_to=st.session_state.seek_to,
                marker=f"{selected['id']}_{st.session_state.seek_nonce}",
            )

            st.markdown("### Transcript (tap to jump)")
            if not segments:
                st.caption("No segments found.")
            else:
                for i, seg in enumerate(segments):
                    start = float(seg.get("start", 0.0))
                    txt = (seg.get("text") or "").strip()
                    label = f"{fmt_mmss(start)}  {txt}"

                    if st.button(label, key=f"seg_{selected['id']}_{i}", use_container_width=True):
                        st.session_state.seek_to = start
                        st.session_state.seek_nonce += 1
                        st.rerun()

            if full_text_path.exists():
                with st.expander("Extracted text"):
                    scroll_box(full_text_path.read_text(encoding="utf-8", errors="ignore"), height_px=360)

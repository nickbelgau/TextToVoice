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
from core.stt import whisper_segments_verbose_json, extract_segments, merge_segments

# ----------------------------
# Config
# ----------------------------
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
TTS_MAX_CHARS = 4096

PREVIEW_MAX_CHARS = 1200
PREVIEW_HEIGHT_PX = 160

DATA_DIR = Path.cwd() / "data"
ITEMS_DIR = DATA_DIR / "items"
HISTORY_PATH = DATA_DIR / "history.json"
ITEMS_DIR.mkdir(parents=True, exist_ok=True)

VOICES = ["cedar", "nova", "alloy", "marin"]
VOICE_PREVIEW_TEXT = "Hi! I'm Peachy. This is a quick preview of the voice you're about to choose."
VOICE_PREVIEW_DIR = DATA_DIR / "voice_previews"
VOICE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_SEG_MAX_SECONDS = 10.0
DISPLAY_SEG_MAX_CHARS = 520


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
    return dt.strftime("%Y-%m-%d %I:%M %p")  # PT removed


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


def make_preview(text: str, max_chars: int = PREVIEW_MAX_CHARS) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= max_chars:
        return t

    cut = t.rfind("\n\n", 0, max_chars)
    if cut < 200:
        cut = max_chars

    shown = t[:cut].rstrip()
    marker = (
        "\n\n--- PREVIEW TRUNCATED ---\n"
        f"(Showing first ~{len(shown):,} characters of {len(t):,})\n"
    )
    return shown + marker


@st.cache_data(show_spinner=False)
def voice_preview_bytes(voice: str, speed: float) -> bytes:
    speed_key = f"{float(speed):.2f}".replace(".", "_")
    out = VOICE_PREVIEW_DIR / f"{voice}_s{speed_key}.mp3"
    if not out.exists():
        tts_to_mp3_file(VOICE_PREVIEW_TEXT, str(out), voice=voice, speed=float(speed))
    return out.read_bytes()


def audio_player(mp3_path: Path, seek_to: float, should_play: bool, marker: str) -> None:
    """
    Does NOT autoplay on initial render or history change.
    Only seeks+plays when should_play=True (i.e., user clicked a timestamp).
    """
    if not mp3_path.exists():
        st.warning("Audio missing.")
        return

    b64 = base64.b64encode(mp3_path.read_bytes()).decode("utf-8")
    dom_id = f"peachy_{marker}_{uuid.uuid4().hex}"
    seek = float(seek_to or 0.0)
    play_flag = "true" if should_play else "false"

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
            const shouldPlay = {play_flag};

            if (!shouldPlay) return;

            const start = () => {{
              try {{ a.currentTime = Math.max(0, seekTo); }} catch(e) {{}}
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
if "pending_play" not in st.session_state:
    st.session_state.pending_play = False

if "new_voice" not in st.session_state:
    st.session_state.new_voice = VOICES[0]

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
    # darker / less bright button styling (scoped to sidebar primary buttons)
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] button[data-testid="baseButton-primary"]{
            background-color: #b75b55 !important;  /* muted brick-peach */
            border-color: #b75b55 !important;
            color: #ffffff !important;
            font-size: 1.05rem !important;
            padding: 0.60rem 1.05rem !important;
            border-radius: 12px !important;
          }
          section[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover{
            background-color: #a9544f !important;
            border-color: #a9544f !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:
        if st.button("New Peachy 🍑", type="primary", use_container_width=True, key="btn_new_peachy"):
            st.session_state.mode = "new"
            st.session_state.pending_doc = None
            st.session_state.seek_to = 0.0
            st.session_state.pending_play = False
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
            st.session_state.pending_play = False
            st.session_state.seek_nonce += 1
            st.rerun()


# ----------------------------
# Main: New view
# ----------------------------
if st.session_state.mode == "new":
    st.subheader("Create a new Peachy :)")

    # narrower speed slider
    c_speed, _ = st.columns([2, 5])
    with c_speed:
        speed = st.slider("Speed", 0.70, 1.30, float(st.session_state.get("new_speed", 1.00)), 0.05, key="new_speed")

    st.markdown("**Voice**")
    voice = st.radio(
        label="",
        options=VOICES,
        index=VOICES.index(st.session_state.get("new_voice", VOICES[0])),
        horizontal=True,
        key="new_voice_radio",
    )
    st.session_state.new_voice = voice

    # single preview (changes with voice + speed)
    st.audio(voice_preview_bytes(voice, speed), format="audio/mp3")

    up = st.file_uploader(
        "Upload .txt / .docx / .pdf (max 2 MB)",
        type=["txt", "docx", "pdf"],
        key="new_upload_main",
    )

    st.session_state.pending_doc = None
    if up:
        size = getattr(up, "size", None)
        raw = up.read()
        if size is None:
            size = len(raw)

        if size > MAX_UPLOAD_BYTES:
            st.error("File too large. Max is 2 MB.")
        else:
            text = extract_text(up.name, raw).strip()
            st.session_state.pending_doc = {"title": up.name, "text": text}

    if st.session_state.pending_doc:
        st.caption(f'Preview: {st.session_state.pending_doc["title"]}')
        scroll_box(make_preview(st.session_state.pending_doc["text"]), height_px=PREVIEW_HEIGHT_PX)

        if st.button("Generate this mama jamma", use_container_width=True):
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
                tts_to_mp3_file(full_text[:TTS_MAX_CHARS], str(audio_path), voice=voice, speed=float(speed))
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
                    "speed": float(speed),
                    "item_dir": str(item_dir),
                }
            )
            save_history(history)

            st.session_state.selected_id = item_id
            st.session_state.force_select_id = item_id
            st.session_state.mode = "playback"
            st.session_state.pending_doc = None
            st.session_state.seek_to = 0.0
            st.session_state.pending_play = False
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

            st.subheader("Playback")
            st.write(
                f'**Title:** {selected["title"]}  |  '
                f'**Voice:** {selected.get("voice","")}  |  '
                f'**Speed:** {selected.get("speed", 1.0)}  |  '
                f'**Created:** {selected["created_at"]}'
            )

            audio_player(
                audio_path,
                seek_to=st.session_state.seek_to,
                should_play=st.session_state.pending_play,
                marker=f"{selected['id']}_{st.session_state.seek_nonce}",
            )
            st.session_state.pending_play = False

            display_segments = merge_segments(
                segments or [],
                max_seconds=DISPLAY_SEG_MAX_SECONDS,
                max_chars=DISPLAY_SEG_MAX_CHARS,
            )

            st.markdown("### Transcript")
            if not display_segments:
                st.caption("No segments found.")
            else:
                for i, seg in enumerate(display_segments):
                    start = float(seg.get("start", 0.0))
                    end = float(seg.get("end", start))
                    txt = (seg.get("text") or "").strip()
                    label = f"{fmt_mmss(start)}–{fmt_mmss(end)}  {txt}"

                    if st.button(label, key=f"seg_{selected['id']}_{i}", use_container_width=True):
                        st.session_state.seek_to = start
                        st.session_state.pending_play = True
                        st.session_state.seek_nonce += 1
                        st.rerun()

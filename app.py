# app.py
import base64
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

from core.extract_text import extract_text
from core.tts import tts_to_mp3_file
from core.stt import whisper_segments_verbose_json, extract_segments, merge_segments
from core.storage import LocalStorage, B2Storage, Storage

# ----------------------------
# Config
# ----------------------------
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
TTS_MAX_CHARS = 4096

PREVIEW_MAX_CHARS = 1200
PREVIEW_HEIGHT_PX = 160

DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_KEY = "history.json"

VOICES = ["cedar", "nova", "alloy", "marin"]
VOICE_PREVIEW_TEXT = "Hi! I'm Peachy. This is a quick preview of the voice you're about to choose."

DISPLAY_SEG_MAX_SECONDS = 10.0
DISPLAY_SEG_MAX_CHARS = 520

TRANSCRIPT_HEIGHT_PX = 620


# ----------------------------
# Storage selection (local vs B2)
# ----------------------------
def get_storage() -> Storage:
    backend = (st.secrets.get("STORAGE_BACKEND") or "local").lower()

    if backend == "b2":
        return B2Storage(
            endpoint_url=st.secrets["B2_S3_ENDPOINT"],
            bucket=st.secrets["B2_BUCKET"],
            access_key_id=st.secrets["B2_ACCESS_KEY_ID"],
            secret_access_key=st.secrets["B2_SECRET_APPL_KEY"],
            prefix=st.secrets.get("B2_PREFIX", ""),  # optional
        )

    return LocalStorage(root=DATA_DIR)


storage = get_storage()


# ----------------------------
# Helpers
# ----------------------------
def load_history() -> list[dict]:
    return storage.read_json(HISTORY_KEY, [])


def save_history(items: list[dict]) -> None:
    storage.write_json(HISTORY_KEY, items)


def now_pt_string() -> str:
    dt = datetime.now(ZoneInfo("America/Los_Angeles"))
    return dt.strftime("%Y-%m-%d %I:%M %p")


def fmt_mmss(seconds: float) -> str:
    s = int(max(0, seconds))
    m = s // 60
    ss = s % 60
    return f"{m}:{ss:02d}"


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


def html_escape(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def voice_preview_bytes(voice: str, speed: float) -> bytes:
    speed_key = f"{float(speed):.2f}".replace(".", "_")
    key = f"voice_previews/{voice}_s{speed_key}.mp3"

    if storage.exists(key):
        return storage.read_bytes(key)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "preview.mp3"
        tts_to_mp3_file(VOICE_PREVIEW_TEXT, str(out), voice=voice, speed=float(speed))
        b = out.read_bytes()
        storage.write_bytes(key, b, content_type="audio/mpeg")
        return b


def try_close_sidebar_once() -> None:
    """
    Best-effort: on mobile, if the sidebar overlay is open, click the "Close sidebar" control.
    This is UI/DOM dependent but works in current Streamlit builds.
    """
    components.html(
        """
        <script>
          (function() {
            try {
              const doc = window.parent.document;
              const btn =
                doc.querySelector('button[aria-label="Close sidebar"]') ||
                doc.querySelector('button[title="Close sidebar"]') ||
                doc.querySelector('button[data-testid="collapsedControl"]'); // fallback (sometimes used)
              if (btn) btn.click();
            } catch (e) {}
          })();
        </script>
        """,
        height=0,
    )


def render_audio_and_transcript(mp3_bytes: bytes, segments: list[dict], marker: str) -> None:
    """
    Option B: continuous, paragraph-like transcript where each segment span is clickable to seek+play.
    This happens purely in the embedded HTML (no Streamlit rerun needed).
    """
    if not mp3_bytes:
        st.warning("Audio missing.")
        return

    audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")
    audio_id = f"peachy_audio_{marker}_{uuid.uuid4().hex}"
    transcript_id = f"peachy_tx_{marker}_{uuid.uuid4().hex}"

    # Build continuous text with clickable spans
    spans = []
    for seg in segments or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue

        # keep it paragraph-like: just spaces between spans
        # add a subtle timestamp tooltip on hover
        tooltip = f"{fmt_mmss(start)}–{fmt_mmss(end)}"
        spans.append(
            f'<span class="seg" data-start="{start:.3f}" title="{html_escape(tooltip)}">{html_escape(txt)}</span>'
        )

    transcript_html = " ".join(spans) if spans else "<em>No segments found.</em>"

    components.html(
        f"""
        <div class="wrap">
          <audio id="{audio_id}" controls style="width:100%;">
            <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg" />
          </audio>

          <div id="{transcript_id}" class="transcript">
            {transcript_html}
          </div>
        </div>

        <style>
          .wrap {{
            width: 100%;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          }}
          .transcript {{
            margin-top: 12px;
            height: {TRANSCRIPT_HEIGHT_PX}px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 14px;
            border-radius: 12px;
            line-height: 1.55;
            font-size: 16px;
            background: #fff;
          }}
          .seg {{
            cursor: pointer;
            border-radius: 6px;
            padding: 1px 2px;
          }}
          .seg:hover {{
            background: rgba(183, 91, 85, 0.12);
          }}
          .seg.active {{
            background: rgba(183, 91, 85, 0.22);
          }}
        </style>

        <script>
          (function() {{
            const audio = document.getElementById("{audio_id}");
            const box = document.getElementById("{transcript_id}");
            if (!audio || !box) return;

            box.addEventListener("click", (e) => {{
              const el = e.target.closest(".seg");
              if (!el) return;

              // visual active state
              try {{
                box.querySelectorAll(".seg.active").forEach(x => x.classList.remove("active"));
                el.classList.add("active");
              }} catch (err) {{}}

              const t = parseFloat(el.dataset.start || "0");
              try {{ audio.currentTime = Math.max(0, t); }} catch (err) {{}}
              audio.play().catch(() => {{}});
            }});
          }})();
        </script>
        """,
        height=90 + TRANSCRIPT_HEIGHT_PX + 40,
        scrolling=False,
    )


# ----------------------------
# App state
# ----------------------------
st.set_page_config(page_title="Peachy", layout="wide")

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

if "new_voice" not in st.session_state:
    st.session_state.new_voice = VOICES[0]

# for mobile UX: close sidebar overlay after selecting New Peachy
if "request_close_sidebar" not in st.session_state:
    st.session_state.request_close_sidebar = False

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
    st.markdown(
        """
        <style>
          section[data-testid="stSidebar"] button[data-testid="baseButton-primary"]{
            background-color: #b75b55 !important;
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
            st.session_state.request_close_sidebar = True
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


# If requested, close sidebar overlay (mobile UX)
if st.session_state.request_close_sidebar:
    try_close_sidebar_once()
    st.session_state.request_close_sidebar = False


# ----------------------------
# Main: New view
# ----------------------------
if st.session_state.mode == "new":
    st.subheader("Create a new Peachy :)")

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

            title = st.session_state.pending_doc["title"]
            full_text = st.session_state.pending_doc["text"]

            item_prefix = f"items/{item_id}"
            audio_key = f"{item_prefix}/audio.mp3"
            segments_key = f"{item_prefix}/segments.json"
            stt_verbose_key = f"{item_prefix}/stt_verbose.json"
            full_text_key = f"{item_prefix}/full.txt"

            status = st.status("Working…", expanded=True)
            try:
                status.write("Step 1/2: Generating audio…")

                with tempfile.TemporaryDirectory() as td:
                    tmp_audio = Path(td) / "audio.mp3"
                    tts_to_mp3_file(full_text[:TTS_MAX_CHARS], str(tmp_audio), voice=voice, speed=float(speed))

                    audio_bytes = tmp_audio.read_bytes()
                    storage.write_bytes(audio_key, audio_bytes, content_type="audio/mpeg")

                    status.write("✅ Audio generated")
                    status.write("Step 2/2: Transcribing with Whisper (segments)…")

                    stt_verbose = whisper_segments_verbose_json(str(tmp_audio))
                    segments = extract_segments(stt_verbose)

                storage.write_text(full_text_key, full_text)
                storage.write_json(stt_verbose_key, stt_verbose)
                storage.write_json(segments_key, segments)

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
                    "item_dir": item_prefix,  # storage prefix, not a local path
                }
            )
            save_history(history)

            st.session_state.selected_id = item_id
            st.session_state.force_select_id = item_id
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
            item_id = selected["id"]
            item_prefix = selected.get("item_dir") or f"items/{item_id}"

            audio_key = f"{item_prefix}/audio.mp3"
            segments_key = f"{item_prefix}/segments.json"

            audio_bytes = storage.read_bytes(audio_key) if storage.exists(audio_key) else b""
            segments = storage.read_json(segments_key, [])

            st.subheader("Playback")
            st.write(
                f'**Title:** {selected["title"]}  |  '
                f'**Voice:** {selected.get("voice","")}  |  '
                f'**Speed:** {selected.get("speed", 1.0)}  |  '
                f'**Created:** {selected["created_at"]}'
            )

            display_segments = merge_segments(
                segments or [],
                max_seconds=DISPLAY_SEG_MAX_SECONDS,
                max_chars=DISPLAY_SEG_MAX_CHARS,
            )

            st.markdown("### Transcript (tap to jump)")
            render_audio_and_transcript(
                mp3_bytes=audio_bytes,
                segments=display_segments,
                marker=f"{item_id}",
            )

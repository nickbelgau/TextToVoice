# app.py
import base64
import json
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

from core.extract_text import extract_text
from core.tts import tts_to_wav_file, tts_to_mp3_file
from core.stt import whisper_segments_verbose_json, extract_segments, merge_segments
from core.storage import LocalStorage, B2Storage, Storage
from core.audio_utils import wav_duration_seconds, stitch_wavs, convert_wav_to_mp3
from core.alignment import align_segments_to_text, AlignConfig

# ----------------------------
# Config
# ----------------------------
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB

# TTS: stay below the per-request limit with buffer
TTS_CHUNK_MAX_CHARS = 3600

PREVIEW_MAX_CHARS = 1200
PREVIEW_HEIGHT_PX = 160

DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_KEY = "history.json"

VOICES = ["cedar", "nova", "alloy", "marin"]
VOICE_PREVIEW_TEXT = "Hi! I'm Peachy. This is a quick preview of the voice you're about to choose."

# Merge for a nicer UI + better alignment anchors
DISPLAY_SEG_MAX_SECONDS = 10.0
DISPLAY_SEG_MAX_CHARS = 520

READING_HEIGHT_PX = 700

TZ = ZoneInfo("America/Los_Angeles")


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
            prefix=st.secrets.get("B2_PREFIX", ""),
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
    dt = datetime.now(TZ)
    return dt.strftime("%Y-%m-%d %I:%M %p")


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


def split_text_into_chunks_with_offsets(text: str, max_chars: int) -> list[dict]:
    """
    Returns list of:
      { "text": <exact substring>, "orig_start": int, "orig_end": int }
    Splits on paragraph boundaries when possible, else splits inside long paragraphs.
    """
    if not text:
        return []

    # paragraph separators: 2+ newlines
    sep_pat = re.compile(r"\n{2,}")
    spans: list[tuple[int, int]] = []

    start = 0
    for m in sep_pat.finditer(text):
        end = m.end()  # include separator in span
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))

    chunks: list[dict] = []
    cur_start = None
    cur_end = None

    def flush():
        nonlocal cur_start, cur_end
        if cur_start is None or cur_end is None or cur_end <= cur_start:
            cur_start, cur_end = None, None
            return
        chunk_text = text[cur_start:cur_end]
        chunks.append({"text": chunk_text, "orig_start": cur_start, "orig_end": cur_end})
        cur_start, cur_end = None, None

    for (p_start, p_end) in spans:
        p_len = p_end - p_start

        # If paragraph span itself is huge, split inside it
        if p_len > max_chars:
            flush()
            i = p_start
            while i < p_end:
                j = min(p_end, i + max_chars)

                # try to cut at sentence boundary
                slice_ = text[i:j]
                cut = slice_.rfind(". ")
                if cut < int(max_chars * 0.55):
                    cut = slice_.rfind(" ")
                if cut <= 0:
                    cut = len(slice_)

                piece_end = i + cut
                if piece_end <= i:
                    piece_end = j

                chunks.append({"text": text[i:piece_end], "orig_start": i, "orig_end": piece_end})
                i = piece_end
            continue

        # Normal paragraph: try to add into current chunk
        if cur_start is None:
            cur_start, cur_end = p_start, p_end
        else:
            if (p_end - cur_start) <= max_chars:
                cur_end = p_end
            else:
                flush()
                cur_start, cur_end = p_start, p_end

    flush()
    # filter empty-ish
    return [c for c in chunks if (c["text"] or "").strip()]


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
    components.html(
        """
        <script>
          (function() {
            try {
              const doc = window.parent.document;
              const btn =
                doc.querySelector('button[aria-label="Close sidebar"]') ||
                doc.querySelector('button[title="Close sidebar"]') ||
                doc.querySelector('button[data-testid="collapsedControl"]');
              if (btn) btn.click();
            } catch (e) {}
          })();
        </script>
        """,
        height=0,
    )


def render_audio_and_clickable_doc(audio_bytes: bytes, mime: str, full_text: str, segments: list[dict], marker: str) -> None:
    """
    Shows original extracted text (format preserved via pre-wrap).
    Click highlighted spans to seek+play audio.
    """
    if not audio_bytes:
        st.warning("Audio missing.")
        return

    segs = [
        s for s in (segments or [])
        if isinstance(s, dict) and "orig_char_start" in s and "orig_char_end" in s and "start" in s
    ]
    segs.sort(key=lambda x: int(x["orig_char_start"]))

    if not segs:
        st.caption("No aligned spans were found. (Transcript still exists, but can’t map to original text yet.)")
        return

    # Build HTML with spans inserted into original text
    pos = 0
    parts: list[str] = []
    n = len(full_text)

    for s in segs:
        a = int(s["orig_char_start"])
        b = int(s["orig_char_end"])
        t = float(s["start"])

        if a < pos:
            continue
        if b <= a:
            continue
        if a > n:
            break
        b = min(b, n)

        if pos < a:
            parts.append(html_escape(full_text[pos:a]))

        span_text = full_text[a:b]
        parts.append(
            f'<span class="seg" data-start="{t:.3f}">{html_escape(span_text)}</span>'
        )
        pos = b

    if pos < n:
        parts.append(html_escape(full_text[pos:]))

    doc_html = "".join(parts)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_id = f"peachy_audio_{marker}_{uuid.uuid4().hex}"
    doc_id = f"peachy_doc_{marker}_{uuid.uuid4().hex}"

    components.html(
        f"""
        <div class="wrap">
          <audio id="{audio_id}" controls style="width:100%;">
            <source src="data:{mime};base64,{audio_b64}" type="{mime}" />
          </audio>

          <div id="{doc_id}" class="doc">
            {doc_html}
          </div>
        </div>

        <style>
          .wrap {{
            width: 100%;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          }}
          .doc {{
            margin-top: 12px;
            height: {READING_HEIGHT_PX}px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 14px;
            border-radius: 12px;
            line-height: 1.55;
            font-size: 16px;
            background: #fff;
            white-space: pre-wrap;
          }}
          .seg {{
            cursor: pointer;
            border-radius: 6px;
            padding: 0px 2px;
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
            const box = document.getElementById("{doc_id}");
            if (!audio || !box) return;

            box.addEventListener("click", (e) => {{
              const el = e.target.closest(".seg");
              if (!el) return;

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
        height=90 + READING_HEIGHT_PX + 40,
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
if "request_close_sidebar" not in st.session_state:
    st.session_state.request_close_sidebar = False

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
            text = extract_text(up.name, raw)
            st.session_state.pending_doc = {"title": up.name, "text": text}

    if st.session_state.pending_doc:
        st.caption(f'Preview: {st.session_state.pending_doc["title"]}')
        scroll_box(make_preview(st.session_state.pending_doc["text"]), height_px=PREVIEW_HEIGHT_PX)

        if st.button("Generate this mama jamma", use_container_width=True):
            item_id = uuid.uuid4().hex
            created_at = now_pt_string()

            title = st.session_state.pending_doc["title"]
            full_text = st.session_state.pending_doc["text"] or ""

            chunks = split_text_into_chunks_with_offsets(full_text, max_chars=TTS_CHUNK_MAX_CHARS)
            if not chunks:
                st.error("No text extracted from this file.")
                st.stop()

            item_prefix = f"items/{item_id}"
            full_text_key = f"{item_prefix}/full.txt"
            segments_key = f"{item_prefix}/segments.json"
            manifest_key = f"{item_prefix}/manifest.json"

            audio_mp3_key = f"{item_prefix}/audio.mp3"
            audio_wav_key = f"{item_prefix}/audio.wav"

            status = st.status("Working…", expanded=True)

            try:
                status.write(f"Step 1/3: Generating {len(chunks)} audio chunks (WAV) + STT…")

                all_segments: list[dict] = []
                manifest = {
                    "id": item_id,
                    "title": title,
                    "created_at": created_at,
                    "voice": voice,
                    "speed": float(speed),
                    "tts_chunk_max_chars": TTS_CHUNK_MAX_CHARS,
                    "chunks": [],
                }

                align_cfg = AlignConfig(lookback=250, ahead=9000, threshold=78, min_query_len=10)

                with tempfile.TemporaryDirectory() as td:
                    td = Path(td)
                    wav_paths: list[Path] = []
                    audio_offset = 0.0

                    for i, ch in enumerate(chunks):
                        chunk_text = ch["text"]
                        chunk_orig_start = int(ch["orig_start"])
                        chunk_orig_end = int(ch["orig_end"])

                        status.write(f"Chunk {i+1}/{len(chunks)}: TTS → WAV")
                        wav_path = td / f"chunk_{i:04d}.wav"
                        tts_to_wav_file(chunk_text, str(wav_path), voice=voice, speed=float(speed))

                        dur = wav_duration_seconds(wav_path)
                        wav_paths.append(wav_path)

                        status.write(f"Chunk {i+1}/{len(chunks)}: STT (segments)")
                        stt_verbose = whisper_segments_verbose_json(str(wav_path))
                        segs = extract_segments(stt_verbose)

                        # merge for readability + better alignment anchors
                        segs = merge_segments(segs, max_seconds=DISPLAY_SEG_MAX_SECONDS, max_chars=DISPLAY_SEG_MAX_CHARS)

                        # align merged segments back to THIS chunk's original text slice
                        align_segments_to_text(chunk_text, segs, cfg=align_cfg)

                        # finalize: offset time + convert local orig spans to global orig spans
                        for s in segs:
                            s["start"] = float(s.get("start", 0.0)) + audio_offset
                            s["end"] = float(s.get("end", 0.0)) + audio_offset

                            if "orig_char_start_local" in s and "orig_char_end_local" in s:
                                local_a = int(s["orig_char_start_local"])
                                local_b = int(s["orig_char_end_local"])

                                s["orig_char_start"] = chunk_orig_start + local_a
                                s["orig_char_end"] = chunk_orig_start + local_b

                                # cleanup locals if you want
                                del s["orig_char_start_local"]
                                del s["orig_char_end_local"]

                            all_segments.append(s)

                        manifest["chunks"].append(
                            {
                                "index": i,
                                "orig_char_start": chunk_orig_start,
                                "orig_char_end": chunk_orig_end,
                                "chars": len(chunk_text),
                                "duration_seconds": dur,
                                "audio_offset_seconds": audio_offset,
                            }
                        )

                        audio_offset += dur

                    status.write("Step 2/3: Stitching WAV chunks → master.wav")
                    master_wav = td / "master.wav"
                    stitch_wavs(wav_paths, master_wav)

                    # store full original extracted text (for reading view)
                    storage.write_text(full_text_key, full_text)

                    status.write("Step 3/3: Convert master.wav → audio.mp3 (single encode)")
                    master_mp3 = td / "audio.mp3"

                    audio_key = audio_mp3_key
                    audio_mime = "audio/mpeg"

                    try:
                        convert_wav_to_mp3(master_wav, master_mp3, bitrate_kbps=64)
                        audio_bytes = master_mp3.read_bytes()
                        storage.write_bytes(audio_key, audio_bytes, content_type="audio/mpeg")
                        manifest["audio"] = {"format": "mp3", "key": audio_key, "mime": audio_mime}
                    except Exception:
                        # fallback: store wav if ffmpeg not present
                        audio_key = audio_wav_key
                        audio_mime = "audio/wav"
                        audio_bytes = master_wav.read_bytes()
                        storage.write_bytes(audio_key, audio_bytes, content_type="audio/wav")
                        manifest["audio"] = {"format": "wav", "key": audio_key, "mime": audio_mime}

                # persist aligned segments + manifest
                storage.write_json(segments_key, all_segments)
                storage.write_json(manifest_key, manifest)

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
                    "item_dir": item_prefix,
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

            full_text_key = f"{item_prefix}/full.txt"
            segments_key = f"{item_prefix}/segments.json"
            manifest_key = f"{item_prefix}/manifest.json"

            audio_mp3_key = f"{item_prefix}/audio.mp3"
            audio_wav_key = f"{item_prefix}/audio.wav"

            full_text = storage.read_text(full_text_key) if storage.exists(full_text_key) else ""
            segments = storage.read_json(segments_key, [])
            manifest = storage.read_json(manifest_key, {})

            # prefer mp3
            if storage.exists(audio_mp3_key):
                audio_key = audio_mp3_key
                audio_mime = "audio/mpeg"
            elif storage.exists(audio_wav_key):
                audio_key = audio_wav_key
                audio_mime = "audio/wav"
            else:
                ak = (manifest.get("audio") or {}).get("key")
                am = (manifest.get("audio") or {}).get("mime")
                if ak and storage.exists(ak):
                    audio_key = ak
                    audio_mime = am or "audio/mpeg"
                else:
                    audio_key = ""
                    audio_mime = "audio/mpeg"

            audio_bytes = storage.read_bytes(audio_key) if audio_key else b""

            st.subheader("Playback")
            st.write(
                f'**Title:** {selected["title"]}  |  '
                f'**Voice:** {selected.get("voice","")}  |  '
                f'**Speed:** {selected.get("speed", 1.0)}  |  '
                f'**Created:** {selected["created_at"]}'
            )

            st.markdown("### Reading view (tap highlighted text to jump)")
            render_audio_and_clickable_doc(
                audio_bytes=audio_bytes,
                mime=audio_mime,
                full_text=full_text,
                segments=segments,
                marker=item_id,
            )

import json
import re
import time
import uuid
from pathlib import Path

import streamlit as st

from core.extract_text import extract_text
from core.tts import tts_to_mp3_file

# ----------------------------
# Storage (local)
# ----------------------------
DATA_DIR = Path.cwd() / "data"
ITEMS_DIR = DATA_DIR / "items"
HISTORY_PATH = DATA_DIR / "history.json"

ITEMS_DIR.mkdir(parents=True, exist_ok=True)


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


# ----------------------------
# Chunking (simple + readable)
# ----------------------------
def chunk_text(text: str, target_chars: int = 900) -> list[str]:
    """
    Chunk by paragraphs; if a paragraph is too long, split by sentences,
    falling back to hard splits. Keeps chunks comfortably under the 4096 limit.
    """
    t = (text or "").strip()
    if not t:
        return []

    # Normalize whitespace a bit
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)

    paras = [p.strip() for p in t.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for p in paras:
        if len(p) <= target_chars:
            chunks.append(p)
            continue

        # sentence split
        sentences = re.split(r"(?<=[.!?])\s+", p)
        cur = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(cur) + len(s) + 1 <= target_chars:
                cur = (cur + " " + s).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s

        if cur:
            chunks.append(cur)

    # Hard cap any weird huge chunk
    final: list[str] = []
    for c in chunks:
        c = c.strip()
        if len(c) <= 1200:
            final.append(c)
        else:
            for i in range(0, len(c), 1200):
                final.append(c[i : i + 1200].strip())

    return [c for c in final if c]


def render_followalong(chunks: list[str], current_idx: int) -> None:
    # Compact scroll box + highlight current chunk
    html_parts = [
        """
        <style>
          .box { height: 320px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 8px; }
          .p { margin: 0 0 10px 0; padding: 6px 8px; border-radius: 6px; }
          .cur { background: rgba(255, 235, 59, 0.35); border: 1px solid rgba(255, 235, 59, 0.7); }
          .idx { opacity: 0.55; font-size: 12px; margin-right: 6px; }
        </style>
        <div class="box">
        """
    ]

    for i, c in enumerate(chunks):
        cls = "p cur" if i == current_idx else "p"
        safe = (
            c.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        html_parts.append(f'<div class="{cls}"><span class="idx">{i+1}.</span>{safe}</div>')

    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="TextToVoice", layout="centered")
st.title("TextToVoice")

history = load_history()
history_sorted = sorted(history, key=lambda x: x["created_at"], reverse=True)

# Session state
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "chunk_idx" not in st.session_state:
    st.session_state.chunk_idx = 0

# Sidebar: pick history item
st.sidebar.header("History")
if history_sorted:
    ids = [x["id"] for x in history_sorted]

    def fmt(item_id: str) -> str:
        it = next(x for x in history_sorted if x["id"] == item_id)
        return f'{it["created_at"]} — {it["title"]} ({it["voice"]})'

    picked = st.sidebar.selectbox("Select", ids, format_func=fmt)
    if picked != st.session_state.selected_id:
        st.session_state.selected_id = picked
        st.session_state.chunk_idx = 0
else:
    st.sidebar.caption("No history yet.")

# Load selected item (if any)
selected = None
chunks = []
audio_dir = None
if st.session_state.selected_id:
    selected = next((x for x in history_sorted if x["id"] == st.session_state.selected_id), None)
    if selected:
        item_dir = Path(selected["item_dir"])
        chunks = load_json(item_dir / "chunks.json", [])
        audio_dir = item_dir / "audio"

# Main: show selected + follow-along + controls
if selected:
    st.subheader("Follow along")
    st.write(f'**Title:** {selected["title"]}  |  **Voice:** {selected["voice"]}  |  **Chunks:** {selected["chunk_count"]}')

    if chunks:
        # clamp index
        st.session_state.chunk_idx = max(0, min(st.session_state.chunk_idx, len(chunks) - 1))
        render_followalong(chunks, st.session_state.chunk_idx)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Prev", use_container_width=True):
                st.session_state.chunk_idx = max(0, st.session_state.chunk_idx - 1)
                st.rerun()
        with col2:
            if st.button("Play chunk", use_container_width=True):
                pass  # audio render below
        with col3:
            if st.button("Next", use_container_width=True):
                st.session_state.chunk_idx = min(len(chunks) - 1, st.session_state.chunk_idx + 1)
                st.rerun()

        # Always show the current chunk audio player (so "Play" works reliably)
        cur_mp3 = audio_dir / f"chunk_{st.session_state.chunk_idx:03d}.mp3"
        if cur_mp3.exists():
            st.audio(str(cur_mp3), format="audio/mp3")
        else:
            st.warning("Audio for this chunk is missing. (Regenerate from upload or add on-demand generation.)")

        st.caption("Tip: audio won’t auto-advance to the next chunk yet (manual Next).")
    else:
        st.warning("No chunks found for this history item.")

st.divider()

# Upload + generate new item
st.subheader("New document")
voice = st.selectbox("Voice", ["marin", "cedar", "alloy", "nova"], index=0)
f = st.file_uploader("Upload a .txt, .docx, or .pdf", type=["txt", "docx", "pdf"])

if f:
    raw = f.read()  # not persisted
    text = extract_text(f.name, raw).strip()

    # chunk + show compact
    new_chunks = chunk_text(text, target_chars=900)
    st.write(f"Chunks: **{len(new_chunks)}**")
    if new_chunks:
        render_followalong(new_chunks, 0)

    if st.button("Save + Generate audio"):
        item_id = uuid.uuid4().hex
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        item_dir = ITEMS_DIR / item_id
        audio_dir = item_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # persist extracted text + chunks (not original file)
        (item_dir / "full.txt").write_text(text, encoding="utf-8")
        save_json(item_dir / "chunks.json", new_chunks)

        # generate mp3 per chunk
        prog = st.progress(0)
        total = max(1, len(new_chunks))
        for i, c in enumerate(new_chunks):
            mp3_path = audio_dir / f"chunk_{i:03d}.mp3"
            # keep safely under limit for now
            tts_to_mp3_file(c[:4000], str(mp3_path), voice=voice)
            prog.progress(int(((i + 1) / total) * 100))

        # update history index
        history.append(
            {
                "id": item_id,
                "created_at": created_at,
                "title": f.name,  # document title = filename for now
                "voice": voice,
                "chunk_count": len(new_chunks),
                "item_dir": str(item_dir),
            }
        )
        save_history(history)

        # select the new item
        st.session_state.selected_id = item_id
        st.session_state.chunk_idx = 0
        st.rerun()
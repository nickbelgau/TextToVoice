from pathlib import Path
from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

def whisper_segments_verbose_json(mp3_path: str) -> dict:
    """
    Whisper with segment timestamps (when supported by SDK).
    """
    p = Path(mp3_path)
    with p.open("rb") as f:
        try:
            r = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        except TypeError:
            # Older SDK fallback (verbose_json still typically includes segments)
            r = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )

    if hasattr(r, "model_dump"):
        return r.model_dump()
    if hasattr(r, "to_dict"):
        return r.to_dict()
    return dict(r)

def extract_segments(stt_verbose_json: dict) -> list[dict]:
    segs = stt_verbose_json.get("segments") or []
    out = []
    for s in segs:
        out.append(
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": (s.get("text") or "").strip(),
            }
        )
    return out

def merge_segments(segments: list[dict], max_seconds: float = 12.0, max_chars: int = 420) -> list[dict]:
    """
    Make segments less granular by merging adjacent ones.
    Keeps 'start' at first start, updates 'end' to last end, concatenates text.
    """
    merged = []
    cur = None

    for s in segments:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue

        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))

        if cur is None:
            cur = {"start": start, "end": end, "text": txt}
            continue

        candidate_text = (cur["text"] + " " + txt).strip()
        candidate_end = end
        duration = candidate_end - float(cur["start"])

        if duration <= max_seconds and len(candidate_text) <= max_chars:
            cur["end"] = candidate_end
            cur["text"] = candidate_text
        else:
            merged.append(cur)
            cur = {"start": start, "end": end, "text": txt}

    if cur is not None:
        merged.append(cur)

    return merged

from pathlib import Path
from openai import OpenAI

key = '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'
client = OpenAI(api_key=key)

def whisper_segments_verbose_json(mp3_path: str) -> dict:
    """
    Returns Whisper verbose_json which includes segments with start/end timestamps.
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
            # fallback if your installed SDK doesn't accept timestamp_granularities
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

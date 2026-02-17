from pathlib import Path
from openai import OpenAI

key = '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'
client = OpenAI(api_key=key)

def stt_mini_transcribe_to_json(mp3_path: str) -> dict:
    """
    Speech-to-text using STT mini.
    Model: gpt-4o-mini-transcribe :contentReference[oaicite:2]{index=2}
    """
    p = Path(mp3_path)
    with p.open("rb") as f:
        # Keep it simple: default json output, grab .text plus any metadata
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )

    # transcription is an SDK object; convert to plain dict safely
    # (OpenAI SDK objects support model_dump() in newer versions)
    if hasattr(transcription, "model_dump"):
        return transcription.model_dump()
    if hasattr(transcription, "to_dict"):
        return transcription.to_dict()
    return dict(transcription)

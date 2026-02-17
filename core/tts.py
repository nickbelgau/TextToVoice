from pathlib import Path
from openai import OpenAI

key = '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'
client = OpenAI(api_key=key)

def tts_to_mp3_file(
    text: str,
    out_path: str,
    voice: str = "nova",
    speed: float = 1.0,
    model: str = "gpt-4o-mini-tts",
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        speed=speed,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out)

    return out
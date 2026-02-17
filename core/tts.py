from pathlib import Path
from openai import OpenAI

client = OpenAI()

def tts_to_mp3_file(text: str, out_path: str, voice: str = "marin", speed: float = 1.0) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        speed=speed,
        instructions="Read naturally, like a helpful assistant.",
        response_format="mp3",
    ) as response:
        response.stream_to_file(out)

    return out
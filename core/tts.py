from pathlib import Path
from openai import OpenAI

key = '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'
client = OpenAI(api_key=key)

DEFAULT_INSTRUCTIONS = (
    "Speak like a calm, confident interviewer. "
    "Use a steady pace and short, clear sentences. "
    "Pause briefly between ideas. "
    "When reading Q&A prep, emphasize key phrases and outcomes. "
    "Do not sound robotic."
)

def tts_to_mp3_file(
    text: str,
    out_path: str,
    voice: str = "nova",
    speed: float = 1.0,
    model: str = "gpt-4o-mini-tts",
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        speed=speed,
        instructions=instructions,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out)

    return out

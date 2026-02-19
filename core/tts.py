from pathlib import Path
from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_KEY"])

DEFAULT_INSTRUCTIONS = (
    "Speak like a calm, confident interviewer. "
    "Use a steady pace and short, clear sentences. "
    "Pause briefly between ideas. "
    "When reading Q&A prep, emphasize key phrases and outcomes. "
    "Do not sound robotic."
)

def _tts_to_file(
    text: str,
    out_path: str,
    voice: str,
    speed: float,
    model: str,
    instructions: str,
    response_format: str,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        speed=speed,
        instructions=instructions,
        response_format=response_format,
    ) as response:
        response.stream_to_file(out)

    return out

def tts_to_mp3_file(
    text: str,
    out_path: str,
    voice: str = "nova",
    speed: float = 1.0,
    model: str = "gpt-4o-mini-tts",
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> Path:
    return _tts_to_file(
        text=text,
        out_path=out_path,
        voice=voice,
        speed=speed,
        model=model,
        instructions=instructions,
        response_format="mp3",
    )

def tts_to_wav_file(
    text: str,
    out_path: str,
    voice: str = "nova",
    speed: float = 1.0,
    model: str = "gpt-4o-mini-tts",
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> Path:
    return _tts_to_file(
        text=text,
        out_path=out_path,
        voice=voice,
        speed=speed,
        model=model,
        instructions=instructions,
        response_format="wav",
    )

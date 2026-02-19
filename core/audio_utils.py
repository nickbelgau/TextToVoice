# core/audio_utils.py
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate()
        if rate <= 0:
            return 0.0
        return frames / float(rate)


def _wav_spec(wav_path: Path) -> tuple[int, int, int]:
    """
    Returns (channels, sampwidth_bytes, framerate_hz)
    Validates basic ranges to avoid wave 'argument out of range' errors.
    """
    with wave.open(str(wav_path), "rb") as w:
        ch = int(w.getnchannels())
        sw = int(w.getsampwidth())
        fr = int(w.getframerate())
        ct = w.getcomptype()

    # Python wave only supports uncompressed ('NONE') for writing
    if ct != "NONE":
        raise ValueError(f"WAV not PCM/uncompressed: {wav_path} comptype={ct}")

    if not (1 <= ch <= 8):
        raise ValueError(f"Invalid WAV channels: {ch} in {wav_path}")
    if sw not in (1, 2, 3, 4):
        raise ValueError(f"Invalid WAV sample width: {sw} bytes in {wav_path}")
    if not (1 <= fr <= 384000):
        raise ValueError(f"Invalid WAV framerate: {fr} Hz in {wav_path}")

    return ch, sw, fr


def stitch_wavs(wav_paths: list[Path], out_wav: Path) -> None:
    """
    Stitch WAV chunks into a single WAV.
    This avoids wave.setparams(params) (which can trigger 'argument out of range' on some inputs).
    """
    if not wav_paths:
        raise ValueError("No wavs to stitch")

    ch0, sw0, fr0 = _wav_spec(wav_paths[0])

    # Validate all chunks match format
    for p in wav_paths[1:]:
        ch, sw, fr = _wav_spec(p)
        if (ch, sw, fr) != (ch0, sw0, fr0):
            raise ValueError(
                "WAV chunk format mismatch:\n"
                f"  first: channels={ch0} sampwidth={sw0} fr={fr0} ({wav_paths[0]})\n"
                f"  this : channels={ch}  sampwidth={sw}  fr={fr}  ({p})\n"
                "Fix: re-encode chunks to a common format (ffmpeg) before stitching."
            )

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(out_wav), "wb") as out:
        out.setnchannels(ch0)
        out.setsampwidth(sw0)
        out.setframerate(fr0)

        # Stream frames to avoid loading everything into memory
        for p in wav_paths:
            with wave.open(str(p), "rb") as w:
                frames_left = w.getnframes()
                # read in blocks
                block = 65536
                while frames_left > 0:
                    n = min(block, frames_left)
                    data = w.readframes(n)
                    if not data:
                        break
                    out.writeframesraw(data)
                    frames_left -= n

        # finalize header
        out.writeframes(b"")


def convert_wav_to_mp3(in_wav: Path, out_mp3: Path, bitrate_kbps: int = 64) -> None:
    """
    Requires ffmpeg available on PATH.
    Streamlit Community Cloud: add packages.txt with 'ffmpeg'
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(in_wav),
        "-vn",
        "-b:a",
        f"{bitrate_kbps}k",
        str(out_mp3),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[:600]}")

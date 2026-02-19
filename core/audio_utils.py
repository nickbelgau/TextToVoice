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


def stitch_wavs(wav_paths: list[Path], out_wav: Path) -> None:
    if not wav_paths:
        raise ValueError("No wavs to stitch")

    with wave.open(str(wav_paths[0]), "rb") as w0:
        params = w0.getparams()

    with wave.open(str(out_wav), "wb") as out:
        out.setparams(params)

        for p in wav_paths:
            with wave.open(str(p), "rb") as w:
                if w.getparams()[:4] != params[:4]:
                    # nchannels, sampwidth, framerate, nframes (nframes can differ; compare first 3)
                    if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (
                        params.nchannels,
                        params.sampwidth,
                        params.framerate,
                    ):
                        raise ValueError("WAV chunk format mismatch (channels/samplewidth/framerate)")
                out.writeframes(w.readframes(w.getnframes()))


def convert_wav_to_mp3(in_wav: Path, out_mp3: Path, bitrate_kbps: int = 64) -> None:
    """
    Requires ffmpeg available on PATH.
    Streamlit Community Cloud can install via packages.txt containing 'ffmpeg'.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

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
        raise RuntimeError(f"ffmpeg failed: {p.stderr[:400]}")

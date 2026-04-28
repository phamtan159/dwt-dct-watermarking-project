"""
Extract and normalize media files into 16kHz mono WAV audio.

Input:
    data/raw/<name>.(wav|mp3|flac|m4a|mp4|mov|mkv|avi)

Output:
    data/audio/<name>.wav
"""

import argparse
import subprocess
from pathlib import Path

import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcript"

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
}
AUDIO_ONLY_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_waveform(waveform, sample_rate):
    """Convert input audio into mono 16kHz for the training pipeline."""
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    return waveform.to(torch.float32)


def convert_with_torchaudio(input_path, output_path):
    waveform, sample_rate = torchaudio.load(str(input_path))
    waveform = normalize_waveform(waveform, sample_rate)
    torchaudio.save(str(output_path), waveform, 16000)


def convert_with_ffmpeg(input_path, output_path):
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def prepare_one_file(input_path):
    output_path = AUDIO_DIR / f"{input_path.stem}.wav"

    try:
        if input_path.suffix.lower() in AUDIO_ONLY_EXTENSIONS:
            convert_with_torchaudio(input_path, output_path)
        else:
            convert_with_ffmpeg(input_path, output_path)
    except (RuntimeError, OSError):
        convert_with_ffmpeg(input_path, output_path)

    transcript_path = TRANSCRIPT_DIR / f"{input_path.stem}.txt"
    if not transcript_path.exists():
        print(f"  Warning: transcript missing for {input_path.name} -> {transcript_path.name}")

    return output_path


def main():
    global RAW_DIR, AUDIO_DIR

    parser = argparse.ArgumentParser(description="Prepare raw media into 16kHz mono WAV audio")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR))
    args = parser.parse_args()

    RAW_DIR = Path(args.raw_dir).resolve()
    AUDIO_DIR = Path(args.audio_dir).resolve()

    ensure_directories()

    inputs = sorted(
        path for path in RAW_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not inputs:
        print(f"No supported media files found in {RAW_DIR}")
        return

    print(f"Preparing {len(inputs)} file(s) from {RAW_DIR}")
    for input_path in inputs:
        output_path = prepare_one_file(input_path)
        print(f"  {input_path.name} -> {output_path.name}")

    print(f"Done. Normalized WAV files are in {AUDIO_DIR}")


if __name__ == "__main__":
    main()

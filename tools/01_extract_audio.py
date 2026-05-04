import subprocess
from pathlib import Path


RAW_DIR = Path("data/raw")
AUDIO_DIR = Path("data/audio")
TARGET_SAMPLE_RATE = 16000

# Text/metadata files are allowed to live beside media in data/raw, but they
# should not be sent through ffmpeg.
SKIP_EXTENSIONS = {
    ".json",
    ".lab",
    ".srt",
    ".textgrid",
    ".tsv",
    ".txt",
    ".vtt",
}


def extract_audio(raw_dir=RAW_DIR, audio_dir=AUDIO_DIR):
    audio_dir.mkdir(parents=True, exist_ok=True)

    media_files = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and path.suffix.lower() not in SKIP_EXTENSIONS
    ]

    if not media_files:
        print(f"No media files found in {raw_dir}")
        return

    for media_path in media_files:
        output_path = audio_dir / f"{media_path.stem}.wav"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-ac",
            "1",
            str(output_path),
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Converted {media_path} -> {output_path}")
        except subprocess.CalledProcessError as exc:
            print(f"Could not convert {media_path}: ffmpeg exited with {exc.returncode}")


if __name__ == "__main__":
    extract_audio()

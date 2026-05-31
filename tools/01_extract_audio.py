import subprocess
from pathlib import Path

from speaker_paths import iter_speaker_files, relative_id, relative_output_path, warn_root_level_files


RAW_DIR = Path("data/raw")
AUDIO_DIR = Path("data/audio")
TARGET_SAMPLE_RATE = 16000

# Only these media files should be sent through ffmpeg. Placeholder files such
# as .gitkeep can live beside media in data/raw.
MEDIA_EXTENSIONS = {
    ".aac",
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def extract_audio(raw_dir=RAW_DIR, audio_dir=AUDIO_DIR):
    audio_dir.mkdir(parents=True, exist_ok=True)
    warn_root_level_files(raw_dir, MEDIA_EXTENSIONS, "raw media")

    media_files = iter_speaker_files(raw_dir, MEDIA_EXTENSIONS)

    if not media_files:
        print(f"No speaker media files found in {raw_dir}")
        return

    for media_path in media_files:
        rel_id = relative_id(media_path, raw_dir)
        output_path = relative_output_path(audio_dir, rel_id, ".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
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

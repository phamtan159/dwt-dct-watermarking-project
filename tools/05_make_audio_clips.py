import json
import subprocess
from pathlib import Path


AUDIO_DIR = Path("data/audio")
ANNOTATION_DIR = Path("data/annotations/auto")
CLIPS_DIR = Path("data/processed/clips")


def make_audio_clips(audio_dir=AUDIO_DIR, annotation_dir=ANNOTATION_DIR, clips_dir=CLIPS_DIR):
    clips_dir.mkdir(parents=True, exist_ok=True)

    for annotation_path in sorted(annotation_dir.glob("*.json")):
        name = annotation_path.stem
        audio_path = audio_dir / f"{name}.wav"

        if not audio_path.exists():
            print(f"Skip {name}: missing {audio_path}")
            continue

        with annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)

        out_dir = clips_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for segment in annotation.get("segments", []):
            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(0.0, end - start)

            if duration <= 0:
                print(f"Skip {name}/{segment.get('id', '<unknown>')}: invalid duration")
                continue

            clip_path = out_dir / f"{segment['id']}.wav"
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.6f}",
                "-i",
                str(audio_path),
                "-t",
                f"{duration:.6f}",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(clip_path),
            ]

            try:
                subprocess.run(command, check=True)
                count += 1
            except subprocess.CalledProcessError as exc:
                print(f"Could not create {clip_path}: ffmpeg exited with {exc.returncode}")

        print(f"Created {count} audio clips in {out_dir}")


if __name__ == "__main__":
    make_audio_clips()

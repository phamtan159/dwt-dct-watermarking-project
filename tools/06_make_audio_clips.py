import json
import argparse
import os
import subprocess
from pathlib import Path

from speaker_paths import find_matching_file, iter_speaker_files, relative_id, warn_root_level_files


AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "data/audio"))
ANNOTATION_DIR = Path(os.environ.get("ANNOTATION_DIR", "data/annotations/auto"))
CLIPS_DIR = Path(os.environ.get("AUDIO_CLIP_ROOT", os.environ.get("CLIP_ROOT", "data/processed/clips")))


def make_audio_clips(audio_dir=AUDIO_DIR, annotation_dir=ANNOTATION_DIR, clips_dir=CLIPS_DIR):
    clips_dir.mkdir(parents=True, exist_ok=True)

    warn_root_level_files(annotation_dir, {".json"}, "annotation")
    for annotation_path in iter_speaker_files(annotation_dir, {".json"}):
        rel_id = relative_id(annotation_path, annotation_dir)
        audio_path = find_matching_file(audio_dir, rel_id, ".wav")

        if audio_path is None or not audio_path.exists():
            print(f"Skip {rel_id}: missing audio")
            continue

        with annotation_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)

        out_dir = clips_dir / rel_id
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for segment in annotation.get("segments", []):
            start = float(segment["start"])
            end = float(segment["end"])
            duration = max(0.0, end - start)

            if duration <= 0:
                print(f"Skip {rel_id}/{segment.get('id', '<unknown>')}: invalid duration")
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
    parser = argparse.ArgumentParser(description="Cut per-phoneme audio clips from MFA timing.")
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR))
    parser.add_argument("--annotation-dir", default=str(ANNOTATION_DIR))
    parser.add_argument("--clips-dir", default=str(CLIPS_DIR))
    args = parser.parse_args()
    make_audio_clips(Path(args.audio_dir), Path(args.annotation_dir), Path(args.clips_dir))

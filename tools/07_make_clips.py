import json
import os
import shutil
from pathlib import Path

from speaker_paths import iter_speaker_files, relative_id


AUTO_DIR = Path("data/annotations/auto")
MOUTH_ROOT = Path("data/processed/mouth")
META_DIR = Path("data/meta")
CLIP_ROOT = Path("data/processed/clips")
PAD = 3


def make_visual_clips(
    auto_dir=AUTO_DIR,
    mouth_root=MOUTH_ROOT,
    meta_dir=META_DIR,
    clip_root=CLIP_ROOT,
):
    annotation_paths = iter_speaker_files(auto_dir, {".json"})
    if not annotation_paths:
        print(f"No speaker annotation JSON files found in {auto_dir}")
        return

    for annotation_path in annotation_paths:
        rel_id = relative_id(annotation_path, auto_dir)
        frames_dir = mouth_root / rel_id
        if not frames_dir.exists():
            print(f"Skip {rel_id}: missing {frames_dir}")
            continue

        frames = sorted(
            f for f in os.listdir(frames_dir)
            if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not frames:
            print(f"Skip {rel_id}: no mouth frames")
            continue

        meta_path = meta_dir / Path(rel_id).with_suffix(".json")
        if not meta_path.exists():
            print(f"Skip {rel_id}: missing {meta_path}")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ann = json.loads(annotation_path.read_text(encoding="utf-8"))
        fps = float(meta["fps"])

        out_dir = clip_root / rel_id
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for seg in ann.get("segments", []):
            start_f = max(0, round(float(seg["start"]) * fps) - PAD)
            end_f = round(float(seg["end"]) * fps) + PAD

            clip_dir = out_dir / seg["id"]
            clip_dir.mkdir(parents=True, exist_ok=True)

            for i in range(start_f, min(end_f, len(frames))):
                shutil.copy(frames_dir / frames[i], clip_dir / frames[i])
            count += 1

        print(f"Created {count} visual clips in {out_dir}")


if __name__ == "__main__":
    make_visual_clips()

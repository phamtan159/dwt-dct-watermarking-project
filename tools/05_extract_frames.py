import json
from pathlib import Path

import cv2

from speaker_paths import iter_speaker_files, relative_id, relative_output_path, warn_root_level_files


RAW_DIR = Path("data/raw")
FRAME_ROOT = Path("data/processed/frames")
META_DIR = Path("data/meta")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def write_image(path: Path, image) -> bool:
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True


def extract_frames(raw_dir=RAW_DIR, frame_root=FRAME_ROOT, meta_dir=META_DIR):
    warn_root_level_files(raw_dir, VIDEO_EXTENSIONS, "video")
    videos = iter_speaker_files(raw_dir, VIDEO_EXTENSIONS)
    if not videos:
        print(f"No speaker videos found in {raw_dir}")
        return

    for video_path in videos:
        rel_id = relative_id(video_path, raw_dir)
        cap = cv2.VideoCapture(str(video_path))
        out_dir = frame_root / rel_id
        out_dir.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if write_image(out_dir / f"{frame_count:04d}.jpg", frame):
                frame_count += 1

        cap.release()
        meta_path = relative_output_path(meta_dir, rel_id, ".json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "id": rel_id,
                    "speaker_id": Path(rel_id).parts[0],
                    "sample_id": Path(rel_id).name,
                    "fps": fps,
                    "frames": frame_count,
                    "source_video": str(video_path).replace("\\", "/"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {frame_count} frames to {out_dir}")


if __name__ == "__main__":
    extract_frames()

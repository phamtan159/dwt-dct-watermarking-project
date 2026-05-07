"""
06_build_dataset.py

Build a compact JSON dataset from compared pronunciation annotations.

Input annotation format:
  data/annotations/compare/*.json

Environment override:
  $env:ANNOTATION_DIR="data/annotations/compare"

Output:
  data/final/dataset.json
  data/final/label_map.json
"""

import json
import os
from pathlib import Path


ANNOTATION_DIR = os.environ.get("ANNOTATION_DIR", "data/annotations/compare")
CLIP_ROOT = os.environ.get("CLIP_ROOT", "data/processed/clips")
OUT_DIR = os.environ.get("OUT_DIR", "data/final")
SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>", "<sil>"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_silence_phone(phone):
    if phone is None:
        return True
    return str(phone).strip().lower() in SILENCE_PHONES


def should_skip_segment(seg):
    if "phoneme_standard" in seg:
        return is_silence_phone(seg.get("phoneme_standard"))
    if "standard_phone" in seg:
        return is_silence_phone(seg.get("standard_phone"))
    phone = seg.get("phoneme") or seg.get("phone") or seg.get("phoneme_real")
    return is_silence_phone(phone)


def collect_labels(annotation_paths):
    labels = {"no_error"}
    for path in annotation_paths:
        data = load_json(path)
        for seg in data.get("segments", []):
            if should_skip_segment(seg):
                continue
            labels.add(seg.get("error_id") or seg.get("error") or "no_error")
    return {label: idx for idx, label in enumerate(sorted(labels))}


def build_sample(path, label_map):
    video_id = path.stem
    data = load_json(path)
    segments = []

    for seg in data.get("segments", []):
        if should_skip_segment(seg):
            continue
        label = seg.get("error_id") or seg.get("error") or "no_error"
        real_phone = seg.get("phoneme_real") if "phoneme_real" in seg else seg.get("phoneme")
        standard_phone = seg.get("phoneme_standard")
        phone = standard_phone if is_silence_phone(real_phone) else real_phone
        segment_id = seg.get("id")

        segments.append(
            {
                "id": segment_id,
                "phone": phone,
                "phoneme_real": real_phone,
                "standard_phone": standard_phone,
                "phoneme_standard": standard_phone,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "label": label,
                "label_id": label_map[label],
                "error_id": label,
                "error_code": seg.get("error_code"),
                "clip_dir": os.path.join(CLIP_ROOT, video_id, segment_id).replace("\\", "/") if segment_id else None,
            }
        )

    return {
        "id": video_id,
        "video_id": video_id,
        "annotation_path": str(path).replace("\\", "/"),
        "segments": segments,
    }


def main():
    annotation_dir = Path(ANNOTATION_DIR)
    annotation_paths = sorted(annotation_dir.glob("*.json"))
    if not annotation_paths:
        print(f"No annotation JSON files found in {annotation_dir}")
        return

    label_map = collect_labels(annotation_paths)
    samples = [build_sample(path, label_map) for path in annotation_paths]

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "dataset.json"
    label_map_path = out_dir / "label_map.json"

    dataset = {
        "annotation_dir": str(annotation_dir).replace("\\", "/"),
        "clip_root": CLIP_ROOT.replace("\\", "/"),
        "num_samples": len(samples),
        "num_segments": sum(len(sample["segments"]) for sample in samples),
        "label_map": label_map,
        "samples": samples,
    }

    dataset_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Built {dataset_path} with {len(samples)} samples")
    print(f"Wrote {label_map_path}")


if __name__ == "__main__":
    main()


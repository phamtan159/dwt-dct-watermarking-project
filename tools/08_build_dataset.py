"""
Build the unified audio-only pronunciation dataset.

Inputs:
  data/annotations/compare/<speaker>/<sample>.json
  data/processed/clips/<speaker>/<sample>/<segment_id>.wav     optional audio segment clips

Outputs:
  data/final/dataset.json        audio samples and attributes
  data/final/label_map.json
  data/final/audio_dataset.json  compatibility view for the audio trainer
  data/final/speakers/<speaker>/ per-speaker dataset views
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from speaker_paths import (
    iter_speaker_files,
    relative_id,
    sample_id_from_relative,
    speaker_id_from_relative,
    warn_root_level_files,
)


SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>", "<sil>"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_sample_metadata(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            if isinstance(data.get("samples"), list):
                rows = data["samples"]
            else:
                return {str(key): value for key, value in data.items() if isinstance(value, dict)}
        elif isinstance(data, list):
            rows = data
        else:
            return {}
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        return {}

    metadata = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_id = row.get("sample_id") or row.get("id") or row.get("audio_id") or row.get("video_id")
        if sample_id:
            metadata[str(sample_id)] = dict(row)
    return metadata


def to_posix(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def sample_metadata_for(sample_id: str, annotation: dict, metadata: dict[str, dict], rel_id: str | None = None) -> dict:
    keys = [
        rel_id,
        sample_id,
        annotation.get("audio_id"),
        annotation.get("video_id"),
        Path(str(annotation.get("audio_path", ""))).stem if annotation.get("audio_path") else None,
        Path(str(annotation.get("video_path", ""))).stem if annotation.get("video_path") else None,
    ]
    for key in keys:
        if key is not None and str(key) in metadata:
            return metadata[str(key)]
    return {}


def first_present(*values):
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def is_silence_phone(phone) -> bool:
    if phone is None:
        return True
    return str(phone).strip().lower() in SILENCE_PHONES


def should_skip_segment(seg: dict) -> bool:
    standard = seg.get("phoneme_standard", seg.get("standard_phone"))
    if standard is not None:
        return is_silence_phone(standard)
    return is_silence_phone(seg.get("phoneme") or seg.get("phone") or seg.get("phoneme_real"))


def normalize_label_id(value) -> str:
    if value is None or str(value).strip() == "":
        return "OK"
    value = str(value).strip()
    if value.upper() == "OK" or value == "no_error" or value == "0":
        return "OK"
    return value


def segment_label_id(seg: dict) -> str:
    for key in ("final_error_label", "error_code", "error_id", "label_id", "label", "error"):
        value = seg.get(key)
        if value is not None and str(value).strip() != "":
            return normalize_label_id(value)
    return "OK"


def load_label_map(path: Path, annotation_paths: list[Path]) -> dict[str, int]:
    label_ids: list[str] = []

    if path.exists():
        data = load_json(path)
        if isinstance(data, dict):
            indexed: list[tuple[int, str]] = []
            other: list[str] = []

            def add_label(raw_index, raw_id):
                label_id = normalize_label_id(raw_id)
                if label_id == "OTHER":
                    other.append(label_id)
                elif label_id.isdigit():
                    indexed.append((int(label_id), label_id))
                elif label_id == "OK":
                    indexed.append((0, label_id))
                else:
                    try:
                        indexed.append((int(raw_index), label_id))
                    except (TypeError, ValueError):
                        other.append(label_id)

            if isinstance(data.get("no_error"), dict):
                item = data["no_error"]
                add_label(item.get("index", item.get("id", 0)), item.get("code", "OK"))
            if isinstance(data.get("other"), dict):
                item = data["other"]
                add_label(item.get("index", item.get("id", "OTHER")), item.get("code", "OTHER"))
            for phoneme_group in (data.get("phonemes") or {}).values():
                if not isinstance(phoneme_group, dict):
                    continue
                for category_key, category in (phoneme_group.get("categories") or {}).items():
                    if not isinstance(category, dict):
                        continue
                    add_label(
                        category.get("index", category.get("id")),
                        category.get("code", category_key),
                    )
                    for fine_key, fine_label in (category.get("labels") or {}).items():
                        if isinstance(fine_label, dict):
                            add_label(fine_label.get("index", fine_label.get("id")), fine_label.get("code", fine_key))

            for key, item in data.items():
                if key in {"schema_version", "description", "phonemes", "no_error", "other"}:
                    continue
                if isinstance(item, dict):
                    raw_index = item.get("index", item.get("id"))
                    if key == "no_error" or str(raw_index) == "0":
                        raw_id = "OK"
                    else:
                        raw_id = item.get("code", key)
                else:
                    raw_id = key

                label_id = normalize_label_id(raw_id)
                if label_id == "OTHER":
                    other.append(label_id)
                elif label_id.isdigit():
                    indexed.append((int(label_id), label_id))
                elif label_id == "OK":
                    indexed.append((0, label_id))
                else:
                    other.append(label_id)

            for _, label_id in sorted(indexed):
                if label_id not in label_ids:
                    label_ids.append(label_id)
            for label_id in other:
                if label_id not in label_ids:
                    label_ids.append(label_id)

    observed_label_ids: list[str] = []
    for annotation_path in annotation_paths:
        data = load_json(annotation_path)
        for seg in data.get("segments", []):
            if should_skip_segment(seg):
                continue
            label_id = segment_label_id(seg)
            if label_id not in observed_label_ids:
                observed_label_ids.append(label_id)

    observed = set(observed_label_ids)
    label_ids = [label_id for label_id in label_ids if label_id in observed]
    for label_id in observed_label_ids:
        if label_id not in label_ids:
            label_ids.append(label_id)

    if "0" in label_ids and "OK" not in label_ids:
        label_ids[label_ids.index("0")] = "OK"
    if "OK" not in label_ids:
        label_ids.insert(0, "OK")
    return {label_id: index for index, label_id in enumerate(label_ids)}


def audio_attributes(seg: dict) -> dict | None:
    standard = seg.get("wavlm_standard_attributes")
    real = seg.get("wavlm_real_attributes")
    if standard is None and real is None:
        return None
    return {
        "model": "pretrained/microsoft-wavlm-large",
        "standard": standard,
        "real": real,
    }


def build_segment(seg: dict, rel_id: str, paths: dict[str, Path], label_map: dict[str, int]) -> dict:
    segment_id = seg.get("id")
    audio_clip_path = paths["audio_clip_root"] / rel_id / f"{segment_id}.wav" if segment_id else None
    label_id = segment_label_id(seg)
    real_phone = seg.get("phoneme_real") if "phoneme_real" in seg else seg.get("phoneme")
    standard_phone = seg.get("phoneme_standard", seg.get("standard_phone"))
    phone = standard_phone if is_silence_phone(real_phone) else real_phone
    audio_attrs = audio_attributes(seg)

    return {
        "id": segment_id,
        "phone": phone,
        "phoneme": phone,
        "phoneme_real": real_phone,
        "standard_phone": standard_phone,
        "phoneme_standard": standard_phone,
        "start": seg.get("start"),
        "end": seg.get("end"),
        "label": label_id,
        "label_id": label_map.get(label_id, label_map["OK"]),
        "error_id": label_id,
        "error": seg.get("error"),
        "error_code": seg.get("error_code"),
        "alignment_op": seg.get("alignment_op"),
        "alignment_cost": seg.get("alignment_cost"),
        "acoustic_support": seg.get("acoustic_support"),
        "word": seg.get("word"),
        "word_index": seg.get("word_index"),
        "phoneme_index_in_word": seg.get("phoneme_index_in_word"),
        "raw_start": seg.get("raw_start"),
        "raw_end": seg.get("raw_end"),
        "raw_segment_ids": seg.get("raw_segment_ids", []),
        "phoneme_source_model": seg.get("phoneme_source_model"),
        "audio_clip_path": to_posix(audio_clip_path),
        "audio_clip_exists": bool(audio_clip_path and audio_clip_path.exists()),
        "audio_attributes": audio_attrs,
        "wavlm_standard_attributes": seg.get("wavlm_standard_attributes"),
        "wavlm_real_attributes": seg.get("wavlm_real_attributes"),
    }


def build_sample(
    annotation_path: Path,
    annotation_dir: Path,
    paths: dict[str, Path],
    label_map: dict[str, int],
    metadata: dict[str, dict],
) -> dict:
    rel_id = relative_id(annotation_path, annotation_dir)
    sample_id = sample_id_from_relative(rel_id)
    annotation = load_json(annotation_path)
    speaker_from_path = speaker_id_from_relative(rel_id)
    audio_path = paths["audio_dir"] / Path(rel_id).with_suffix(".wav")
    sample_meta = sample_metadata_for(sample_id, annotation, metadata, rel_id)
    speaker_id = first_present(
        annotation.get("speaker_id"),
        annotation.get("speaker"),
        speaker_from_path,
        sample_meta.get("speaker_id"),
        sample_meta.get("speaker"),
    )
    take_id = first_present(annotation.get("take_id"), sample_meta.get("take_id"), sample_meta.get("take"))
    read_style = first_present(annotation.get("read_style"), sample_meta.get("read_style"), sample_meta.get("style"))

    segments = []
    for seg in annotation.get("segments", []):
        if should_skip_segment(seg):
            continue
        segments.append(build_segment(seg, rel_id, paths, label_map))

    return {
        "id": rel_id,
        "audio_id": annotation.get("audio_id", sample_id),
        "sample_id": annotation.get("sample_id", sample_id),
        "video_id": rel_id,
        "speaker_id": str(speaker_id) if speaker_id is not None else None,
        "take_id": str(take_id) if take_id is not None else None,
        "read_style": str(read_style) if read_style is not None else None,
        "transcript": annotation.get("transcript"),
        "audio": to_posix(Path("../data/audio") / Path(rel_id).with_suffix(".wav")),
        "audio_path": to_posix(audio_path),
        "audio_exists": audio_path.exists(),
        "annotation_path": to_posix(annotation_path),
        "metadata": sample_meta,
        "alignment_method": annotation.get("alignment_method"),
        "alignment_score": annotation.get("alignment_score"),
        "phoneme_model": annotation.get("phoneme_model"),
        "phoneme_decoder": annotation.get("phoneme_decoder"),
        "audio_attribute_model": annotation.get("attribute_model"),
        "segments": segments,
    }


def build_audio_training_view(samples: list[dict]) -> list[dict]:
    items = []
    for sample in samples:
        phonemes = []
        labels = []
        for seg in sample["segments"]:
            phonemes.append(
                {
                    "s": seg["start"],
                    "e": seg["end"],
                    "phone": seg.get("phoneme_real") or seg.get("phone"),
                    "standard_phone": seg.get("phoneme_standard"),
                }
            )
            labels.append(seg["label"])

        if phonemes:
            items.append(
                {
                    "audio": sample["audio"],
                    "audio_path": sample["audio_path"],
                    "audio_id": sample.get("audio_id"),
                    "sample_id": sample.get("sample_id") or sample.get("id"),
                    "speaker_id": sample.get("speaker_id") or "unknown_speaker",
                    "phonemes": phonemes,
                    "labels": labels,
                }
            )
    return items


def write_speaker_final_views(out_dir: Path, dataset: dict, samples: list[dict], label_map: dict[str, int]) -> None:
    speakers: dict[str, list[dict]] = {}
    for sample in samples:
        speaker_id = sample.get("speaker_id") or "unknown_speaker"
        speakers.setdefault(str(speaker_id), []).append(sample)

    speaker_root = out_dir / "speakers"
    for speaker_id, speaker_samples in sorted(speakers.items()):
        speaker_dir = speaker_root / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        speaker_dataset = {
            **{key: value for key, value in dataset.items() if key != "samples"},
            "speaker_id": speaker_id,
            "num_samples": len(speaker_samples),
            "num_segments": sum(len(sample["segments"]) for sample in speaker_samples),
            "label_map": label_map,
            "samples": speaker_samples,
        }
        write_json(speaker_dir / "dataset.json", speaker_dataset)
        write_json(speaker_dir / "audio_dataset.json", build_audio_training_view(speaker_samples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified audio-only dataset.")
    parser.add_argument("--annotation-dir", default=os.environ.get("ANNOTATION_DIR", "data/annotations/compare"))
    parser.add_argument(
        "--audio-clip-root",
        default=os.environ.get("AUDIO_CLIP_ROOT", os.environ.get("CLIP_ROOT", "data/processed/clips")),
    )
    parser.add_argument("--audio-dir", default=os.environ.get("AUDIO_DIR", "data/audio"))
    parser.add_argument("--sample-metadata", default=os.environ.get("SAMPLE_METADATA", "data/sample_metadata.csv"))
    parser.add_argument("--label-map", default=os.environ.get("LABEL_MAP", "data/label_map.json"))
    parser.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "data/final"))
    args = parser.parse_args()

    annotation_dir = Path(args.annotation_dir)
    warn_root_level_files(annotation_dir, {".json"}, "compare annotation")
    annotation_paths = iter_speaker_files(annotation_dir, {".json"})
    if not annotation_paths:
        print(f"No speaker annotation JSON files found in {annotation_dir}")
        return

    label_map = load_label_map(Path(args.label_map), annotation_paths)
    paths = {
        "audio_clip_root": Path(args.audio_clip_root),
        "audio_dir": Path(args.audio_dir),
    }

    metadata = load_sample_metadata(Path(args.sample_metadata))
    samples = [build_sample(path, annotation_dir, paths, label_map, metadata) for path in annotation_paths]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = {
        "schema_version": "audio_only_v1",
        "annotation_dir": to_posix(annotation_dir),
        "audio_clip_root": to_posix(paths["audio_clip_root"]),
        "audio_dir": to_posix(paths["audio_dir"]),
        "sample_metadata": to_posix(args.sample_metadata),
        "num_samples": len(samples),
        "num_segments": sum(len(sample["segments"]) for sample in samples),
        "label_map": label_map,
        "samples": samples,
    }

    write_json(out_dir / "dataset.json", dataset)
    write_json(out_dir / "label_map.json", label_map)
    write_json(out_dir / "audio_dataset.json", build_audio_training_view(samples))
    write_speaker_final_views(out_dir, dataset, samples, label_map)

    print(f"Built {out_dir / 'dataset.json'} with {len(samples)} samples")
    print(f"Wrote {out_dir / 'label_map.json'}")
    print(f"Wrote {out_dir / 'audio_dataset.json'}")
    print(f"Wrote speaker datasets under {out_dir / 'speakers'}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


DEFAULT_INPUT = Path("data/final/segment_attributes.json")
DEFAULT_OUTPUT_DIR = Path("data/labels/review")
DEFAULT_LABEL_MAP = Path("data/label_map.json")


def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def feature_error_summary(errors):
    if not errors:
        return ""
    parts = []
    for item in errors:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        attr = item.get("attribute", "")
        category = item.get("category", "")
        expected = item.get("expected", "")
        observed = item.get("observed", "")
        parts.append(f"{attr}:{category}:{expected}->{observed}")
    return "; ".join(parts)


def sample_rel_path(sample, suffix=".json"):
    sample_id = sample.get("id") or sample.get("video_id") or sample.get("sample_id")
    if not sample_id:
        speaker = sample.get("speaker_id", "unknown-speaker")
        metadata = sample.get("metadata") or {}
        phone = metadata.get("phone_folder", "unknown-phone")
        mode = metadata.get("mode", "unknown-mode")
        stem = sample.get("sample_id", "unknown-sample")
        return Path(speaker) / phone / mode / f"{stem}{suffix}"
    return Path(*str(sample_id).replace("\\", "/").split("/")).with_suffix(suffix)


def row_for_segment(sample, segment):
    metadata = sample.get("metadata") or {}
    return {
        "sample_id": sample.get("id") or sample.get("sample_id") or "",
        "speaker_id": sample.get("speaker_id") or metadata.get("speaker_id") or "",
        "phone_folder": metadata.get("phone_folder") or "",
        "mode": metadata.get("mode") or "",
        "take_id": sample.get("take_id") or metadata.get("take_id") or "",
        "read_style": sample.get("read_style") or metadata.get("read_style") or "",
        "target_word": metadata.get("target_word") or "",
        "transcript": sample.get("transcript") or "",
        "word": segment.get("word") or "",
        "segment_id": segment.get("id") or "",
        "start": segment.get("start") or "",
        "end": segment.get("end") or "",
        "expected_phoneme": segment.get("target_phoneme")
        or segment.get("phoneme_standard")
        or segment.get("standard_phone")
        or "",
        "observed_phoneme": segment.get("observed_phoneme")
        or segment.get("phoneme_real")
        or "",
        "auto_label": segment.get("label") or "",
        "auto_error_code": segment.get("error_code") or segment.get("error_id") or "",
        "alignment_op": segment.get("alignment_op") or "",
        "feature_error_categories": ";".join(segment.get("feature_error_categories") or []),
        "feature_errors_summary": feature_error_summary(segment.get("feature_errors") or []),
        "audio_clip_path": segment.get("audio_clip_path") or "",
        "visual_clip_dir": segment.get("visual_clip_dir") or segment.get("clip_dir") or "",
        "human_label": "",
        "severity": "",
        "primary_evidence": "",
        "note": "",
    }


def compact_features(features):
    if not isinstance(features, dict):
        return {}
    return {key: value for key, value in features.items() if value not in (0, 0.0, None, "")}


def load_label_taxonomy(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def label_options_for_sample(sample, taxonomy):
    metadata = sample.get("metadata") or {}
    phoneme_key = metadata.get("phone_folder") or metadata.get("target_group") or metadata.get("target_phoneme")
    phonemes = taxonomy.get("phonemes") if isinstance(taxonomy, dict) else None
    if not isinstance(phonemes, dict):
        return {}
    options = phonemes.get(str(phoneme_key))
    if options:
        return options

    target = metadata.get("target_phoneme")
    for group in phonemes.values():
        if isinstance(group, dict) and group.get("phoneme") == target:
            return group
    return {}


def segment_for_json(segment):
    return {
        "segment_id": segment.get("id") or "",
        "word": segment.get("word") or "",
        "time": {
            "start": segment.get("start"),
            "end": segment.get("end"),
        },
        "phoneme": {
            "expected": segment.get("target_phoneme")
            or segment.get("phoneme_standard")
            or segment.get("standard_phone")
            or "",
            "observed": segment.get("observed_phoneme")
            or segment.get("phoneme_real")
            or "",
            "alignment_op": segment.get("alignment_op") or "",
        },
        "auto": {
            "label": segment.get("label") or "",
            "error_code": segment.get("error_code") or segment.get("error_id") or "",
            "feature_error_categories": segment.get("feature_error_categories") or [],
            "feature_errors": segment.get("feature_errors") or [],
        },
        "evidence": {
            "expected_features": compact_features(segment.get("expected_features")),
            "observed_features": compact_features(segment.get("observed_features")),
            "predicted_features": compact_features(segment.get("predicted_features")),
            "audio_clip_path": segment.get("audio_clip_path") or "",
            "visual_clip_dir": segment.get("visual_clip_dir")
            or segment.get("clip_dir")
            or "",
        },
        "human_label": {
            "category": "",
            "label": "",
            "severity": "",
            "primary_evidence": "",
            "note": "",
        },
    }


def sample_for_json(sample, label_taxonomy=None):
    metadata = sample.get("metadata") or {}
    label_options = label_options_for_sample(sample, label_taxonomy or {})
    return {
        "id": sample.get("id") or sample.get("sample_id") or "",
        "speaker_id": sample.get("speaker_id") or metadata.get("speaker_id") or "",
        "sample_id": sample.get("sample_id") or "",
        "phone_folder": metadata.get("phone_folder") or "",
        "mode": metadata.get("mode") or "",
        "take_id": sample.get("take_id") or metadata.get("take_id") or "",
        "read_style": sample.get("read_style") or metadata.get("read_style") or "",
        "target_word": metadata.get("target_word") or "",
        "target_words": metadata.get("target_words") or "",
        "transcript": sample.get("transcript") or "",
        "label_instructions": {
            "category": "Chọn category trong label_options, ví dụ th_to_t, dh_to_d, final_t_weak_or_omitted.",
            "label": "Chọn fine label trong category đó. Nếu đúng thì dùng OK.",
            "severity": "0=đúng, 1=nhẹ, 2=rõ, 3=nặng.",
            "primary_evidence": "audio, visual, hoặc audio_visual.",
            "note": "Ghi chú ngắn nếu cần.",
        },
        "label_options": label_options,
        "segments": [segment_for_json(segment) for segment in sample.get("segments", [])],
    }


def write_json(path, payload, overwrite=False):
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export small per-sample JSON files for human labeling.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument(
        "--sample-id",
        help="Export only one sample id, for example: speaker-01/θ/F/F_01",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    samples = data.get("samples", data if isinstance(data, list) else [])
    label_taxonomy = load_label_taxonomy(args.label_map)

    written = 0
    skipped = 0
    index_rows = []

    for sample in samples:
        sid = sample.get("id") or sample.get("video_id") or sample.get("sample_id")
        if args.sample_id and sid != args.sample_id:
            continue

        payload = sample_for_json(sample, label_taxonomy)
        rel_path = sample_rel_path(sample, suffix=".json")
        output_path = args.output_dir / rel_path
        did_write = write_json(output_path, payload, overwrite=args.overwrite)
        written += 1 if did_write else 0
        skipped += 0 if did_write else 1
        index_rows.append(
            {
                "sample_id": sid or "",
                "speaker_id": sample.get("speaker_id") or "",
                "transcript": sample.get("transcript") or "",
                "num_segments": len(payload["segments"]),
                "label_file": output_path.as_posix(),
            }
        )

    if not args.sample_id:
        write_json(
            args.output_dir / "index.json",
            index_rows,
            overwrite=True,
        )

    print(
        f"Done: wrote {written} label review JSON files, skipped {skipped} existing files."
    )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

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


def empty_human_label():
    return {
        "category": "",
        "label": "",
        "severity": "",
        "primary_evidence": "",
        "note": "",
    }


def existing_human_labels(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    labels = {}
    for segment in data.get("segments", []):
        phoneme = segment.get("phoneme") or {}
        segment_id = phoneme.get("segment_id") or segment.get("segment_id")
        human_label = segment.get("human_label")
        suggested_label = segment.get("suggested_label")
        if segment_id:
            labels[str(segment_id)] = {
                "human_label": human_label if isinstance(human_label, dict) else None,
                "suggested_label": suggested_label if isinstance(suggested_label, dict) else None,
            }
    return labels


def segment_for_json(segment, preserved_labels=None):
    segment_id = segment.get("id") or ""
    preserved = (preserved_labels or {}).get(segment_id, {})
    human_label = preserved.get("human_label") if isinstance(preserved, dict) else None
    suggested_label = preserved.get("suggested_label") if isinstance(preserved, dict) else None
    return {
        "phoneme": {
            "segment_id": segment_id,
            "word": segment.get("word") or "",
            "expected": segment.get("target_phoneme")
            or segment.get("phoneme_standard")
            or segment.get("standard_phone")
            or "",
            "observed": segment.get("observed_phoneme")
            or segment.get("phoneme_real")
            or "",
            "alignment_op": segment.get("alignment_op") or "",
        },
        "human_label": human_label or empty_human_label(),
        "suggested_label": suggested_label or empty_human_label(),
    }


def sample_for_json(sample, output_path=None, label_taxonomy=None, reset_labels=False):
    metadata = sample.get("metadata") or {}
    label_taxonomy = label_taxonomy or {}
    label_options = label_options_for_sample(sample, label_taxonomy)
    preserved_labels = {} if reset_labels else (existing_human_labels(output_path) if output_path else {})
    category_index_map = dict(label_options.get("category_index_map", {"0": "OK"}))
    label_index_map = dict(label_options.get("label_index_map", {"0": "OK"}))
    category_index_map.setdefault("99", "OTHER")
    label_index_map.setdefault("99", "OTHER")
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
            "category": "Dien code category hoac so trong category_index_map. Vi du: th_to_t hoac 1.",
            "label": "Dien code fine label hoac so trong label_index_map. Neu dung thi dung OK hoac 0.",
            "severity": "Dien so: 0=dung, 1=nhe, 2=ro, 3=nang.",
            "primary_evidence": "Dien code evidence hoac so trong primary_evidence_index_map.",
            "note": "Ghi chu ngan neu can.",
        },
        "category_index_map": category_index_map,
        "label_index_map": label_index_map,
        "severity_index_map": label_taxonomy.get("severity_index_map", {}),
        "primary_evidence_index_map": label_taxonomy.get("primary_evidence_index_map", {}),
        "segments": [
            segment_for_json(segment, preserved_labels=preserved_labels)
            for segment in sample.get("segments", [])
        ],
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
    parser.add_argument(
        "--reset-labels",
        action="store_true",
        help="Clear all human_label values instead of preserving existing manual labels.",
    )
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

        rel_path = sample_rel_path(sample, suffix=".json")
        output_path = args.output_dir / rel_path
        payload = sample_for_json(
            sample,
            output_path=output_path,
            label_taxonomy=label_taxonomy,
            reset_labels=args.reset_labels,
        )
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

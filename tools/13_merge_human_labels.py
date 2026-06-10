import argparse
import copy
import json
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

try:
    from rule_engine import default_primary_evidence
except Exception:  # pragma: no cover
    default_primary_evidence = None


DEFAULT_ATTRIBUTES = Path("data/final/segment_attributes.json")
DEFAULT_LABEL_REVIEW_DIR = Path("data/labels/review")
DEFAULT_OUTPUT = Path("data/final/segment_attributes_labeled.json")


EMPTY_VALUES = {"", None}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_value(value, mapping):
    if value in EMPTY_VALUES:
        return ""
    text = str(value).strip()
    return mapping.get(text, text)


def is_labeled(human_label):
    if not isinstance(human_label, dict):
        return False
    category = human_label.get("category")
    label = human_label.get("label")
    return category not in EMPTY_VALUES or label not in EMPTY_VALUES


def normalized_human_label(raw, review_file):
    category_map = review_file.get("category_index_map") or {}
    label_map = review_file.get("label_index_map") or {}
    severity_map = review_file.get("severity_index_map") or {}
    evidence_map = review_file.get("primary_evidence_index_map") or {}

    category = normalize_value(raw.get("category"), category_map)
    label = normalize_value(raw.get("label"), label_map)
    severity = normalize_value(raw.get("severity"), severity_map)
    primary_evidence = normalize_value(raw.get("primary_evidence"), evidence_map)

    if category == "" and label == "OK":
        category = "OK"
    if label == "" and category == "OK":
        label = "OK"

    return {
        "category": category,
        "label": label,
        "severity": severity,
        "primary_evidence": primary_evidence,
        "note": raw.get("note", ""),
    }


def apply_default_primary_evidence(human_label, segment):
    if human_label.get("primary_evidence"):
        return human_label

    updated = dict(human_label)
    category = str(updated.get("category") or "").strip().lower()
    label = str(updated.get("label") or "").strip().lower()
    if category in {"ok", "0"} or label in {"ok", "0"}:
        updated["primary_evidence"] = "none"
    elif default_primary_evidence is not None:
        temp_segment = dict(segment)
        temp_segment["final_error_label"] = updated.get("label")
        temp_segment["final_error_category"] = updated.get("category")
        updated["primary_evidence"] = default_primary_evidence(temp_segment)
    else:
        updated["primary_evidence"] = "audio"
    return updated


def collect_labels(review_dir):
    labels = {}
    for path in sorted(review_dir.rglob("*.json")):
        if path.name == "index.json":
            continue
        review = load_json(path)
        sample_id = review.get("id")
        if not sample_id:
            continue
        for segment in review.get("segments", []):
            phoneme = segment.get("phoneme") or {}
            segment_id = phoneme.get("segment_id")
            raw_human_label = segment.get("human_label") or {}
            if not segment_id or not is_labeled(raw_human_label):
                continue
            labels[(sample_id, segment_id)] = normalized_human_label(raw_human_label, review)
    return labels


def merge_labels(attributes, labels):
    output = copy.deepcopy(attributes)
    output["schema_version"] = "segment_attributes_labeled_v1"
    output["source"] = {
        "attributes": str(DEFAULT_ATTRIBUTES).replace("\\", "/"),
        "labels": str(DEFAULT_LABEL_REVIEW_DIR).replace("\\", "/"),
        "filter": "only segments with non-empty human_label.category or human_label.label",
    }

    labeled_samples = []
    labeled_segment_count = 0
    for sample in attributes.get("samples", []):
        sample_id = sample.get("id") or sample.get("video_id") or sample.get("sample_id")
        new_sample = copy.deepcopy(sample)
        new_segments = []
        for segment in sample.get("segments", []):
            segment_id = segment.get("id")
            human_label = labels.get((sample_id, segment_id))
            if not human_label:
                continue

            human_label = apply_default_primary_evidence(human_label, segment)
            new_segment = copy.deepcopy(segment)
            new_segment["human_label"] = human_label
            new_segment["final_error_category"] = human_label["category"]
            new_segment["final_error_label"] = human_label["label"]
            new_segment["final_error_severity"] = human_label["severity"]
            new_segment["final_error_primary_evidence"] = human_label["primary_evidence"]
            new_segment["final_error_note"] = human_label["note"]
            new_segments.append(new_segment)

        if new_segments:
            new_sample["segments"] = new_segments
            labeled_segment_count += len(new_segments)
            labeled_samples.append(new_sample)

    output["samples"] = labeled_samples
    output["num_samples"] = len(labeled_samples)
    output["num_segments"] = labeled_segment_count
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Merge human labels into segment attributes and keep only labeled segments."
    )
    parser.add_argument("--attributes", type=Path, default=DEFAULT_ATTRIBUTES)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_LABEL_REVIEW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    attributes = load_json(args.attributes)
    labels = collect_labels(args.review_dir)
    output = merge_labels(attributes, labels)
    write_json(args.output, output)

    print(f"Human-labeled segments found: {len(labels)}")
    print(f"Output samples: {output['num_samples']}")
    print(f"Output segments: {output['num_segments']}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

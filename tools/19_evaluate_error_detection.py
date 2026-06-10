from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


OK_VALUES = {"", "0", "ok", "no_error", "correct", "none", "null"}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def round4(value: float) -> float:
    return round(float(value), 4)


def sample_id(sample: dict) -> str:
    return str(sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or sample.get("video_id") or "")


def segment_id(segment: dict) -> str:
    return str(segment.get("id") or segment.get("segment_id") or ((segment.get("phoneme") or {}).get("segment_id")) or "")


def normalize_label(value) -> str:
    text = str(value if value is not None else "").strip()
    if text.lower() in OK_VALUES:
        return "OK"
    return text


def to_binary_label(value) -> str:
    return "OK" if normalize_label(value) == "OK" else "ERROR"


def choose_label(payload: dict | None, target: str) -> str:
    if not isinstance(payload, dict):
        return ""
    if target == "binary":
        for key in ("label", "category", "severity"):
            value = payload.get(key)
            if value not in (None, ""):
                return to_binary_label(value)
        return "OK"
    return normalize_label(payload.get(target))


def human_label_from_segment(segment: dict, target: str) -> str:
    if target == "label":
        value = segment.get("final_error_label")
        if value not in (None, ""):
            return normalize_label(value)
    if target == "category":
        value = segment.get("final_error_category")
        if value not in (None, ""):
            return normalize_label(value)
    if target == "severity":
        value = segment.get("final_error_severity")
        if value not in (None, ""):
            return normalize_label(value)
    if target == "binary":
        value = segment.get("final_error_label") or segment.get("final_error_category")
        if value not in (None, ""):
            return to_binary_label(value)

    human = segment.get("human_label") or {}
    return choose_label(human, target)


def predicted_label_from_segment(segment: dict, target: str) -> str:
    prediction = segment.get("predicted_human_label") or segment.get("recommended_human_label") or {}
    value = choose_label(prediction, target)
    if value:
        return value

    if target == "label":
        return normalize_label(segment.get("predicted_label") or segment.get("final_error_label"))
    if target == "category":
        return normalize_label(segment.get("predicted_category") or segment.get("final_error_category"))
    if target == "severity":
        return normalize_label(segment.get("predicted_severity") or segment.get("final_error_severity"))
    if target == "binary":
        return to_binary_label(
            segment.get("predicted_label")
            or segment.get("predicted_category")
            or segment.get("final_error_label")
            or segment.get("final_error_category")
        )
    return ""


def collect_ground_truth(dataset: dict, target: str) -> dict[tuple[str, str], str]:
    labels = {}
    for sample in dataset.get("samples", []):
        sid = sample_id(sample)
        for segment in sample.get("segments", []):
            seg_id = segment_id(segment)
            label = human_label_from_segment(segment, target)
            if not sid or not seg_id or label == "":
                continue
            labels[(sid, seg_id)] = label
    return labels


def collect_predictions(dataset: dict, target: str) -> dict[tuple[str, str], str]:
    labels = {}
    for sample in dataset.get("samples", []):
        sid = sample_id(sample)
        for segment in sample.get("segments", []):
            seg_id = segment_id(segment)
            label = predicted_label_from_segment(segment, target)
            if not sid or not seg_id or label == "":
                continue
            labels[(sid, seg_id)] = label
    return labels


def evaluate_labels(
    truth: dict[tuple[str, str], str],
    predictions: dict[tuple[str, str], str],
    *,
    include_ok: bool,
) -> dict:
    keys = sorted(set(truth) & set(predictions))
    missing_predictions = sorted(set(truth) - set(predictions))
    extra_predictions = sorted(set(predictions) - set(truth))
    classes = sorted(set(truth.values()) | set(predictions.values()))
    eval_classes = classes if include_ok else [item for item in classes if item != "OK"]

    confusion: dict[str, Counter] = defaultdict(Counter)
    for key in keys:
        confusion[truth[key]][predictions[key]] += 1

    per_class = {}
    for klass in eval_classes:
        tp = sum(1 for key in keys if truth[key] == klass and predictions[key] == klass)
        fp = sum(1 for key in keys if truth[key] != klass and predictions[key] == klass)
        fn = sum(1 for key in keys if truth[key] == klass and predictions[key] != klass)
        support = sum(1 for key in keys if truth[key] == klass)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[klass] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": round4(precision),
            "recall": round4(recall),
            "f1": round4(f1),
        }

    correct = sum(1 for key in keys if truth[key] == predictions[key])
    total = len(keys)
    tp_total = sum(metrics["tp"] for metrics in per_class.values())
    fp_total = sum(metrics["fp"] for metrics in per_class.values())
    fn_total = sum(metrics["fn"] for metrics in per_class.values())
    micro_precision = safe_div(tp_total, tp_total + fp_total)
    micro_recall = safe_div(tp_total, tp_total + fn_total)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    macro_f1 = safe_div(sum(metrics["f1"] for metrics in per_class.values()), len(per_class))
    macro_precision = safe_div(sum(metrics["precision"] for metrics in per_class.values()), len(per_class))
    macro_recall = safe_div(sum(metrics["recall"] for metrics in per_class.values()), len(per_class))
    support_total = sum(metrics["support"] for metrics in per_class.values())
    weighted_f1 = safe_div(
        sum(metrics["f1"] * metrics["support"] for metrics in per_class.values()),
        support_total,
    )

    return {
        "schema_version": "error_detection_eval_v1",
        "num_ground_truth": len(truth),
        "num_predictions": len(predictions),
        "num_matched_segments": total,
        "num_missing_predictions": len(missing_predictions),
        "num_extra_predictions": len(extra_predictions),
        "accuracy": round4(safe_div(correct, total)),
        "micro_precision": round4(micro_precision),
        "micro_recall": round4(micro_recall),
        "micro_f1": round4(micro_f1),
        "macro_precision": round4(macro_precision),
        "macro_recall": round4(macro_recall),
        "macro_f1": round4(macro_f1),
        "weighted_f1": round4(weighted_f1),
        "classes": eval_classes,
        "per_class": per_class,
        "confusion_matrix": {truth_label: dict(pred_counts) for truth_label, pred_counts in sorted(confusion.items())},
        "missing_prediction_keys": [{"sample_id": key[0], "segment_id": key[1]} for key in missing_predictions[:50]],
        "extra_prediction_keys": [{"sample_id": key[0], "segment_id": key[1]} for key in extra_predictions[:50]],
    }


def write_per_class_csv(path: Path, per_class: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class", "support", "tp", "fp", "fn", "precision", "recall", "f1"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for klass, metrics in per_class.items():
            row = {"class": klass}
            row.update({field: metrics.get(field, 0) for field in fields if field != "class"})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pronunciation error detection with Precision/Recall/F1."
    )
    parser.add_argument("--ground-truth", default="data/final/segment_attributes_labeled.json")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--target", choices=["label", "category", "severity", "binary"], default="label")
    parser.add_argument("--include-ok", action="store_true", help="Include OK/no_error as a metric class.")
    parser.add_argument("--output", default="data/final/eval_error_detection.json")
    parser.add_argument("--per-class-csv", default="data/final/eval_error_detection_per_class.csv")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    pred_path = Path(args.predictions)
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {gt_path}. Run tools/13_merge_human_labels.py first."
        )
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    truth = collect_ground_truth(load_json(gt_path), args.target)
    predictions = collect_predictions(load_json(pred_path), args.target)
    result = evaluate_labels(truth, predictions, include_ok=args.include_ok)
    result["target"] = args.target
    result["ground_truth"] = str(gt_path).replace("\\", "/")
    result["predictions"] = str(pred_path).replace("\\", "/")

    write_json(Path(args.output), result)
    write_per_class_csv(Path(args.per_class_csv), result["per_class"])

    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.per_class_csv}")
    print(
        "Overall: "
        f"target={args.target} "
        f"matched={result['num_matched_segments']} "
        f"accuracy={result['accuracy']} "
        f"macro_f1={result['macro_f1']} "
        f"micro_f1={result['micro_f1']}"
    )


if __name__ == "__main__":
    main()

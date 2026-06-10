from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "train"
TOOLS_DIR = PROJECT_ROOT / "tools"
for path in (TRAIN_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from attribute_classifier import (  # noqa: E402
    AttributeSoftmaxClassifier,
    build_examples,
    feature_evidence,
    load_json,
    vectorize,
    write_json,
)

try:
    from rule_engine import default_primary_evidence  # noqa: E402
except Exception:  # pragma: no cover
    default_primary_evidence = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_checkpoint(path: Path):
    model, metadata = AttributeSoftmaxClassifier.load(path)
    return model, metadata


def predict_target(dataset: dict, checkpoint_path: Path) -> dict[tuple[str, str], dict]:
    model, metadata = load_checkpoint(checkpoint_path)
    rows, _, _ = build_examples(
        dataset,
        include_rule=bool(metadata.get("include_rule_features", True)),
        include_wavlm=bool(metadata.get("include_wavlm_features", True)),
        include_phoneme_identity=bool(metadata.get("include_phoneme_identity_features", True)),
        include_phonetic_features=bool(metadata.get("include_phonetic_features", True)),
        target=str(metadata.get("target", "label")),
        feature_set=str(metadata.get("feature_set", "full")),
        feature_names=metadata["feature_names"],
        label_names=metadata["label_names"],
    )

    output = {}
    for row in rows:
        x = vectorize(row["features"], metadata["feature_names"]).reshape(1, -1)
        probs = model.predict_proba(x).reshape(-1)
        pred_index = int(np.argmax(probs))
        top_indices = np.argsort(probs)[::-1][: min(3, len(metadata["label_names"]))]
        key = (row["sample_id"], row["segment_id"])
        output[key] = {
            "target": metadata.get("target", "label"),
            "label": metadata["label_names"][pred_index],
            "confidence": round(float(probs[pred_index]), 4),
            "topk": [
                {
                    "label": metadata["label_names"][int(index)],
                    "confidence": round(float(probs[int(index)]), 4),
                }
                for index in top_indices
            ],
            "evidence": feature_evidence(row["features"], limit=10),
        }
    return output


def primary_evidence_for_segment(seg: dict) -> str:
    if default_primary_evidence is None:
        target = str(seg.get("target_phoneme") or seg.get("phoneme_standard") or "")
        return "audio_visual" if target in {"θ", "ð"} else "audio"
    return default_primary_evidence(seg)


def reconcile(category_pred: dict, label_pred: dict, severity_pred: dict) -> dict:
    category = category_pred.get("label", "OK")
    fine_label = label_pred.get("label", "OK")
    severity = severity_pred.get("label", "OK")

    if category == "OK":
        fine_label = "OK"
        severity = "OK"
    elif fine_label == "OK":
        fine_label = category

    return {
        "category": category,
        "label": fine_label,
        "severity": severity,
    }


def compact_phoneme(seg: dict) -> dict:
    return {
        "segment_id": seg.get("id") or seg.get("segment_id"),
        "word": seg.get("word"),
        "target_phoneme": seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone"),
        "phoneme_index_in_word": seg.get("phoneme_index_in_word"),
        "word_index": seg.get("word_index"),
        "start": seg.get("start"),
        "end": seg.get("end"),
    }


def build_payload(
    dataset: dict,
    category_predictions: dict,
    label_predictions: dict,
    severity_predictions: dict,
) -> dict:
    samples = []
    for sample in dataset.get("samples", []):
        sample_id = sample.get("id") or sample.get("video_id") or sample.get("audio_id")
        out_sample = {
            "id": sample_id,
            "audio_id": sample.get("audio_id"),
            "video_id": sample.get("video_id"),
            "sample_id": sample.get("sample_id"),
            "speaker_id": sample.get("speaker_id"),
            "segments": [],
        }

        for seg in sample.get("segments", []):
            segment_id = seg.get("id") or seg.get("segment_id")
            key = (sample_id, segment_id)
            category_pred = category_predictions.get(key, {})
            label_pred = label_predictions.get(key, {})
            severity_pred = severity_predictions.get(key, {})
            prediction = reconcile(category_pred, label_pred, severity_pred)
            prediction["primary_evidence"] = primary_evidence_for_segment(seg)
            prediction["note"] = ""

            out_sample["segments"].append(
                {
                    "id": segment_id,
                    "phoneme": compact_phoneme(seg),
                    "predicted_human_label": prediction,
                    "raw_model_predictions": {
                        "category": category_pred,
                        "label": label_pred,
                        "severity": severity_pred,
                    },
                    "note_input": {
                        "instruction": (
                            "Use predicted_human_label plus evidence to write short formative "
                            "pronunciation feedback. Do not change the label."
                        ),
                        "predicted_human_label": prediction,
                        "evidence": {
                            "category": category_pred.get("evidence", []),
                            "label": label_pred.get("evidence", []),
                            "severity": severity_pred.get("evidence", []),
                        },
                    },
                }
            )

        samples.append(out_sample)

    return {
        "schema_version": "recommended_human_label_predictions_v1",
        "description": (
            "Classifier predictions shaped like human_label. The note field is intentionally "
            "left empty; note_input is for an LLM to generate formative feedback from labels and evidence."
        ),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict category/label/severity and fuse them into human_label-shaped output."
    )
    parser.add_argument("--dataset", default="data/final/segment_attributes.json")
    parser.add_argument("--category-checkpoint", default="train/final_pronunciation_category_recommended_wavlm.npz")
    parser.add_argument("--label-checkpoint", default="train/final_pronunciation_label_recommended_wavlm.npz")
    parser.add_argument("--severity-checkpoint", default="train/final_pronunciation_severity_recommended_wavlm.npz")
    parser.add_argument("--output", default="data/final/recommended_human_label_predictions.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_json(dataset_path)
    category_predictions = predict_target(dataset, Path(args.category_checkpoint))
    label_predictions = predict_target(dataset, Path(args.label_checkpoint))
    severity_predictions = predict_target(dataset, Path(args.severity_checkpoint))

    payload = build_payload(dataset, category_predictions, label_predictions, severity_predictions)
    write_json(Path(args.output), payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

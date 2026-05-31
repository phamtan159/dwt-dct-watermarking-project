"""
Fuse rule-engine and AI-classifier decisions into final pronunciation labels.

Input:
  data/final/dataset.json
  optional classifier predictions JSON

Output:
  data/final/diagnosis.json
  data/final/speakers/<speaker>/diagnosis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rule_engine import rule_segment


def normalize_label(value):
    if value is None or str(value).strip() == "":
        return None
    label = str(value).strip()
    if label.upper() == "OK" or label in {"0", "no_error"}:
        return "OK"
    return label


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def segment_key(sample_id, segment_id):
    return f"{sample_id}/{segment_id}"


def normalize_classifier_item(item):
    if item is None:
        return None
    label = normalize_label(item.get("label") or item.get("predicted_label") or item.get("final_error_label"))
    if not label:
        return None
    confidence = item.get("confidence", item.get("score", 0.0))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "label": str(label),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "evidence": item.get("evidence", []),
    }


def load_classifier_predictions(path: Path | None):
    if path is None or not path.exists():
        return {}

    payload = load_json(path)
    items = {}

    if isinstance(payload, dict) and "samples" in payload:
        for sample in payload.get("samples", []):
            sample_id = sample.get("id") or sample.get("video_id") or sample.get("audio_id")
            for seg in sample.get("segments", []):
                key = segment_key(sample_id, seg.get("id"))
                items[key] = normalize_classifier_item(seg)
    elif isinstance(payload, dict):
        for key, item in payload.items():
            items[key] = normalize_classifier_item(item)
    elif isinstance(payload, list):
        for item in payload:
            sample_id = item.get("sample_id") or item.get("video_id") or item.get("audio_id")
            segment_id = item.get("segment_id") or item.get("id")
            if sample_id and segment_id:
                items[segment_key(sample_id, segment_id)] = normalize_classifier_item(item)

    return {key: value for key, value in items.items() if value is not None}


def fuse(rule_result, classifier_result):
    if classifier_result is None:
        return {
            "label": rule_result["label"],
            "confidence": rule_result["confidence"],
            "strategy": "rule_only",
            "evidence": rule_result["evidence"],
        }

    if classifier_result["label"] == rule_result["label"]:
        confidence = 1.0 - ((1.0 - rule_result["confidence"]) * (1.0 - classifier_result["confidence"]))
        return {
            "label": rule_result["label"],
            "confidence": round(min(confidence, 0.98), 3),
            "strategy": "agreement",
            "evidence": rule_result["evidence"] + classifier_result.get("evidence", []),
        }

    if classifier_result["confidence"] >= rule_result["confidence"] + 0.12:
        return {
            "label": classifier_result["label"],
            "confidence": classifier_result["confidence"],
            "strategy": "classifier_overrode_rule",
            "evidence": classifier_result.get("evidence", []) + [
                {
                    "source": "fusion",
                    "reason": "classifier confidence exceeded rule confidence by at least 0.12",
                    "rule_label": rule_result["label"],
                    "rule_confidence": rule_result["confidence"],
                }
            ],
        }

    return {
        "label": rule_result["label"],
        "confidence": rule_result["confidence"],
        "strategy": "rule_preferred",
        "evidence": rule_result["evidence"] + [
            {
                "source": "fusion",
                "reason": "classifier disagreed but did not exceed the override margin",
                "classifier_label": classifier_result["label"],
                "classifier_confidence": classifier_result["confidence"],
            }
        ],
    }


def llm_feedback_input(sample, seg, final_result):
    return {
        "word": seg.get("word"),
        "target_phoneme": seg.get("phoneme_standard") or seg.get("standard_phone"),
        "observed_phoneme": seg.get("phoneme_real") or seg.get("phone"),
        "position": {
            "word_index": seg.get("word_index"),
            "phoneme_index_in_word": seg.get("phoneme_index_in_word"),
        },
        "final_error_label": final_result["label"],
        "confidence": final_result["confidence"],
        "evidence": final_result["evidence"],
        "audio_attributes": seg.get("audio_attributes"),
        "visual_attributes": seg.get("visual_attributes"),
        "expected_features": seg.get("expected_features"),
        "observed_features": seg.get("observed_features"),
        "observed_feature_source": seg.get("observed_feature_source"),
        "feature_errors": seg.get("feature_errors"),
        "feature_error_categories": seg.get("feature_error_categories"),
        "instruction": "Generate concise Vietnamese teacher-style feedback and one concrete correction drill.",
        "transcript": sample.get("transcript"),
    }


def diagnose_sample(sample, classifier_predictions):
    sample_id = sample.get("id") or sample.get("video_id") or sample.get("audio_id")
    speaker_id = sample.get("speaker_id") or "unknown_speaker"
    output_segments = []

    for seg in sample.get("segments", []):
        key = segment_key(sample_id, seg.get("id"))
        rule_result = rule_segment(seg)
        classifier_result = classifier_predictions.get(key)
        final_result = fuse(rule_result, classifier_result)

        output_segments.append(
            {
                "id": seg.get("id"),
                "word": seg.get("word"),
                "phoneme_standard": seg.get("phoneme_standard") or seg.get("standard_phone"),
                "phoneme_real": seg.get("phoneme_real") or seg.get("phone"),
                "final_error_label": final_result["label"],
                "confidence": final_result["confidence"],
                "fusion_strategy": final_result["strategy"],
                "evidence": final_result["evidence"],
                "rule_engine": rule_result,
                "ai_classifier": classifier_result,
                "expected_features": seg.get("expected_features"),
                "observed_features": seg.get("observed_features"),
                "observed_feature_source": seg.get("observed_feature_source"),
                "feature_errors": seg.get("feature_errors"),
                "feature_error_categories": seg.get("feature_error_categories"),
                "llm_feedback_input": llm_feedback_input(sample, seg, final_result),
            }
        )

    return {
        "id": sample_id,
        "audio_id": sample.get("audio_id"),
        "video_id": sample.get("video_id"),
        "sample_id": sample.get("sample_id"),
        "speaker_id": speaker_id,
        "transcript": sample.get("transcript"),
        "segments": output_segments,
    }


def write_speaker_diagnosis(output_path: Path, output: dict, samples: list[dict]) -> None:
    speakers: dict[str, list[dict]] = {}
    for sample in samples:
        speaker_id = sample.get("speaker_id") or "unknown_speaker"
        speakers.setdefault(str(speaker_id), []).append(sample)

    for speaker_id, speaker_samples in sorted(speakers.items()):
        speaker_output = {
            **{key: value for key, value in output.items() if key != "samples"},
            "speaker_id": speaker_id,
            "num_samples": len(speaker_samples),
            "num_segments": sum(len(sample["segments"]) for sample in speaker_samples),
            "samples": speaker_samples,
        }
        write_json(output_path.parent / "speakers" / speaker_id / output_path.name, speaker_output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse rule engine and classifier results.")
    parser.add_argument("--dataset", default="data/final/dataset.json")
    parser.add_argument("--classifier-predictions", default=None)
    parser.add_argument("--output", default="data/final/diagnosis.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    classifier_path = Path(args.classifier_predictions) if args.classifier_predictions else None
    classifier_predictions = load_classifier_predictions(classifier_path)
    dataset = load_json(dataset_path)

    samples = [
        diagnose_sample(sample, classifier_predictions)
        for sample in dataset.get("samples", [])
    ]
    output = {
        "schema_version": "diagnosis_v1",
        "source_dataset": str(dataset_path).replace("\\", "/"),
        "classifier_predictions": str(classifier_path).replace("\\", "/") if classifier_path else None,
        "num_samples": len(samples),
        "num_segments": sum(len(sample["segments"]) for sample in samples),
        "samples": samples,
    }
    output_path = Path(args.output)
    write_json(output_path, output)
    write_speaker_diagnosis(output_path, output, samples)
    print(f"Wrote {args.output} ({output['num_samples']} samples, {output['num_segments']} segments)")


if __name__ == "__main__":
    main()

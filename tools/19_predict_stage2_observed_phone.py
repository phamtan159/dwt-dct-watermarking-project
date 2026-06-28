from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class ObservedPhoneMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        hidden = max(64, min(256, input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_stage1_inference_module():
    path = TOOLS_DIR / "18_predict_mdd_classifier.py"
    spec = importlib.util.spec_from_file_location("stage1_inference", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_input_dim(checkpoint: dict) -> int:
    state = checkpoint.get("model") or {}
    weight = state.get("net.0.weight")
    if weight is None:
        raise ValueError("Checkpoint does not contain model['net.0.weight'].")
    return int(weight.shape[1])


def normalize_phone(value) -> str:
    phone = str(value or "").strip().lower()
    phone = phone.strip("`/[](){} ")
    phone = phone.replace("\u0261", "g")
    phone = phone.replace("\u0279", "r")
    phone = phone.replace("\u027e", "r")
    return phone


def sample_id(sample: dict) -> str:
    return str(sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or "")


def segment_id(seg: dict) -> str:
    return str(seg.get("id") or seg.get("segment_id") or "")


def global_segment_id(sample: dict, seg: dict) -> str:
    return f"{sample_id(sample)}/{segment_id(seg)}"


def load_mdd_predictions(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    out = {}
    for sample in payload.get("samples", []):
        sid = str(sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or "")
        for seg in sample.get("segments", []):
            seg_id = str(seg.get("id") or seg.get("segment_id") or "")
            if sid and seg_id:
                out[(sid, seg_id)] = seg.get("mdd_classifier") or {}
    return out


def standardize_vector(x: np.ndarray, checkpoint: dict) -> np.ndarray:
    mean = np.asarray(checkpoint.get("mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(checkpoint.get("std"), dtype=np.float32).reshape(-1)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    if x.shape[0] != mean.shape[0]:
        raise ValueError(f"Feature dimension mismatch: vector={x.shape[0]} checkpoint_mean={mean.shape[0]}")
    return (x - mean) / std


def topk_payload(probs: np.ndarray, labels: list[str], k: int = 3) -> list[dict]:
    if probs.size == 0:
        return []
    top_ids = np.argsort(-probs)[: min(k, len(labels))]
    return [
        {
            "phone": labels[int(idx)],
            "probability": round(float(probs[int(idx)]), 6),
        }
        for idx in top_ids
    ]


def predict_segment(seg: dict, checkpoint: dict, model: ObservedPhoneMLP, helper) -> tuple[dict, dict]:
    labels = list(checkpoint.get("id_to_label") or [])
    if not labels:
        raise ValueError("Stage 2 checkpoint does not contain id_to_label.")
    x, coverage = helper.vectorize_segment(seg, checkpoint)
    expected_dim = checkpoint_input_dim(checkpoint)
    if x.shape[0] != expected_dim:
        raise ValueError(f"Feature dimension mismatch: vector={x.shape[0]} checkpoint_model={expected_dim}")
    x = standardize_vector(x, checkpoint)
    with torch.no_grad():
        logits = model(torch.tensor(x.reshape(1, -1), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
    pred_id = int(np.argmax(probs))
    prediction = labels[pred_id]
    return {
        "prediction": prediction,
        "prediction_normalized": normalize_phone(prediction),
        "confidence": round(float(probs[pred_id]), 6),
        "top3": topk_payload(probs, labels, 3),
    }, coverage


def compact_segment_info(sample: dict, seg: dict) -> dict:
    return {
        "id": segment_id(seg),
        "global_id": global_segment_id(sample, seg),
        "word": seg.get("word"),
        "target_phoneme": seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone"),
        "wav2vec2_observed_phoneme": seg.get("observed_phoneme") or seg.get("phoneme_real") or seg.get("phone"),
        "alignment_op": seg.get("alignment_op"),
        "start": seg.get("start"),
        "end": seg.get("end"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Stage 2 observed-phone labels for segment_attributes.json.")
    parser.add_argument("--input", default="data/final/segment_attributes.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mdd-predictions", default="data/final/mdd_predictions.json")
    parser.add_argument("--output", default="data/final/stage2_observed_phone_predictions.json")
    parser.add_argument(
        "--require-posterior",
        action="store_true",
        help="Fail if any segment uses observed-phone posterior fallback instead of full Wav2Vec2 posterior.",
    )
    args = parser.parse_args()

    helper = load_stage1_inference_module()
    dataset = load_json(Path(args.input))
    checkpoint = load_checkpoint(Path(args.checkpoint))
    labels = list(checkpoint.get("id_to_label") or [])
    input_dim = checkpoint_input_dim(checkpoint)
    model = ObservedPhoneMLP(input_dim, len(labels))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mdd_predictions = load_mdd_predictions(Path(args.mdd_predictions)) if args.mdd_predictions else {}

    samples = []
    total = 0
    predicted_errors = 0
    fallback_posterior = 0
    failed_segments = 0
    for sample in dataset.get("samples", []):
        sid = sample_id(sample)
        out_segments = []
        for seg in sample.get("segments", []):
            total += 1
            seg_id = segment_id(seg)
            mdd = mdd_predictions.get((sid, seg_id), {})
            try:
                prediction, coverage = predict_segment(seg, checkpoint, model, helper)
                stage1_is_error = mdd.get("is_error")
                if stage1_is_error is True:
                    predicted_errors += 1
                if coverage.get("posterior_fallback"):
                    fallback_posterior += 1
                out_segments.append(
                    {
                        **compact_segment_info(sample, seg),
                        "mdd_classifier": mdd,
                        "stage2_observed_phone": {
                            **prediction,
                            "applicable": bool(stage1_is_error),
                            "decision_policy": (
                                "Use this as the primary observed-phone candidate when mdd_classifier.is_error is true; "
                                "cross-check it against Wav2Vec2 posterior, speech attributes, duration, and energy."
                            ),
                        },
                        "feature_coverage": coverage,
                    }
                )
            except Exception as exc:
                failed_segments += 1
                out_segments.append(
                    {
                        **compact_segment_info(sample, seg),
                        "mdd_classifier": mdd,
                        "stage2_observed_phone": {
                            "prediction": None,
                            "prediction_normalized": None,
                            "confidence": None,
                            "top3": [],
                            "applicable": bool(mdd.get("is_error")),
                            "error": str(exc),
                        },
                        "feature_coverage": {},
                    }
                )
        samples.append(
            {
                "id": sid,
                "speaker_id": sample.get("speaker_id"),
                "segments": out_segments,
            }
        )

    report = checkpoint.get("report") or {}
    payload = {
        "schema_version": "stage2_observed_phone_predictions_v1",
        "description": (
            "Observed-phone predictions from the trained Stage 2 diagnosis classifier. "
            "Stage 2 should be treated as the primary observed-phone candidate only when Stage 1 predicts an error."
        ),
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "mdd_predictions": str(args.mdd_predictions),
        "checkpoint_schema": report.get("schema_version"),
        "checkpoint_feature_set": checkpoint.get("feature_set") or report.get("feature_set"),
        "checkpoint_feature_dim": input_dim,
        "label_inventory": labels,
        "num_samples": len(samples),
        "num_segments": total,
        "num_stage1_predicted_errors": predicted_errors,
        "num_failed_segments": failed_segments,
        "num_segments_using_posterior_fallback": fallback_posterior,
        "posterior_fallback_rate": round(float(fallback_posterior / total), 6) if total else 0.0,
        "samples": samples,
    }
    if args.require_posterior and fallback_posterior:
        raise RuntimeError(
            "--require-posterior was set, but "
            f"{fallback_posterior}/{total} segments used observed-phone posterior fallback. "
            "Regenerate segment_attributes.json with full Wav2Vec2 posterior before Stage 2 inference."
        )
    write_json(Path(args.output), payload)
    print(f"Wrote {args.output}")
    print(f"Samples: {len(samples)}")
    print(f"Segments: {total}")
    print(f"Stage1 predicted errors: {predicted_errors}")
    print(f"Posterior fallback segments: {fallback_posterior}")
    if failed_segments:
        print(f"Failed segments: {failed_segments}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


FEATURE_SET_ALIASES = {
    "wav2vec2_only": "posterior_canonical",
    "no_sa": "posterior_canonical_duration_energy_wavlm",
    "with_sa": "posterior_canonical_duration_energy_wavlm_sa",
}


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 2),
        )

    def forward(self, x):
        return self.net(x)


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


def normalize_phone(value) -> str:
    phone = str(value or "").strip().lower()
    phone = phone.strip("`/[](){} ")
    phone = phone.replace("\u0279", "r")
    phone = phone.replace("\u0261", "g")
    return phone


def target_phone(seg: dict) -> str:
    return str(seg.get("target_phoneme") or seg.get("phoneme_standard") or seg.get("standard_phone") or "")


def observed_phone(seg: dict) -> str:
    return str(seg.get("observed_phoneme") or seg.get("phoneme_real") or seg.get("phone") or seg.get("phoneme") or "")


def segment_id(sample: dict, seg: dict) -> str:
    sample_id = sample.get("id") or sample.get("sample_id") or sample.get("audio_id")
    return f"{sample_id}/{seg.get('id') or seg.get('segment_id')}"


def checkpoint_input_dim(checkpoint: dict) -> int:
    state = checkpoint.get("model") or {}
    weight = state.get("net.0.weight")
    if weight is None:
        raise ValueError("Checkpoint does not contain model['net.0.weight'].")
    return int(weight.shape[1])


def load_checkpoint(path: Path) -> dict:
    # The checkpoints are generated locally by this project and include numpy
    # arrays. PyTorch 2.6+ defaults to weights_only=True, which rejects them.
    return torch.load(path, map_location="cpu", weights_only=False)


def feature_set_from_checkpoint(checkpoint: dict) -> str:
    report = checkpoint.get("report") or {}
    feature_set = str(report.get("feature_set") or "")
    if not checkpoint.get("feature_names"):
        return f"legacy_{feature_set or 'unknown'}"
    return FEATURE_SET_ALIASES.get(feature_set, feature_set or "unknown")


def posterior_from_segment(seg: dict, phones: list[str]) -> tuple[dict[str, float], dict]:
    for key in ("posterior", "wav2vec2_posterior", "phoneme_posterior"):
        payload = seg.get(key)
        if isinstance(payload, dict):
            return {str(k): safe_float(v) for k, v in payload.items()}, {
                "posterior_source": key,
                "posterior_available": True,
                "posterior_fallback": False,
            }

    observed = normalize_phone(observed_phone(seg))
    posterior = {phone: 0.0 for phone in phones}
    matched = None
    for phone in phones:
        if normalize_phone(phone) == observed:
            posterior[phone] = 1.0
            matched = phone
            break
    return posterior, {
        "posterior_source": "observed_phone_one_hot_fallback",
        "posterior_available": False,
        "posterior_fallback": True,
        "posterior_fallback_observed_phone": observed,
        "posterior_fallback_matched_phone": matched,
    }


def posterior_stats(seg: dict, posterior: dict[str, float], target: str) -> dict:
    values = sorted([safe_float(v) for v in posterior.values()], reverse=True)
    top_prob = safe_float(seg.get("top_prob"), values[0] if values else 0.0)
    second_prob = safe_float(seg.get("second_prob"), values[1] if len(values) > 1 else 0.0)
    entropy = seg.get("entropy")
    if entropy is None and values:
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in values if p > 0)
    margin = seg.get("margin")
    if margin is None:
        margin = top_prob - second_prob
    return {
        "expected": safe_float(posterior.get(target)),
        "top1": top_prob,
        "top2": second_prob,
        "top1_top2_ratio": top_prob / max(second_prob, 1e-6),
        "margin": safe_float(margin),
        "entropy": safe_float(entropy),
    }


def duration_values(seg: dict) -> dict:
    start = safe_float(seg.get("start"))
    end = safe_float(seg.get("end"))
    raw_start = safe_float(seg.get("raw_start"))
    raw_end = safe_float(seg.get("raw_end"))
    duration = safe_float(seg.get("duration_sec"))
    if duration <= 0 and end > start:
        duration = end - start
    raw_duration = max(0.0, raw_end - raw_start) if raw_end > raw_start else 0.0
    if raw_duration > 0 and (duration <= 0 or duration > raw_duration * 5):
        # The pipeline stores both MFA-aligned and raw Wav2Vec2 spans.
        # The raw span is often closer to the actual acoustic evidence window.
        duration = raw_duration
    return {
        "duration": max(duration, 0.0),
        "log_duration": math.log(max(duration, 1e-6)),
        "utterance_ratio": safe_float(seg.get("duration_utterance_ratio")),
    }


def energy_value(seg: dict) -> float:
    for key in ("energy_ratio", "energy", "rms_ratio"):
        if key in seg:
            return safe_float(seg.get(key))
    audio = seg.get("audio_attributes") or {}
    real = audio.get("real") or seg.get("wavlm_real_attributes") or {}
    standard = audio.get("standard") or seg.get("wavlm_standard_attributes") or {}
    real_norm = safe_float(real.get("embedding_norm"))
    standard_norm = safe_float(standard.get("embedding_norm"))
    if standard_norm > 0:
        return real_norm / standard_norm
    return 0.0


def wavlm_presence(seg: dict) -> float:
    if "wavlm_presence_score" in seg:
        return safe_float(seg.get("wavlm_presence_score"))
    audio = seg.get("audio_attributes") or {}
    real = audio.get("real") or seg.get("wavlm_real_attributes") or {}
    standard = audio.get("standard") or seg.get("wavlm_standard_attributes") or {}
    real_norm = safe_float(real.get("embedding_norm"))
    standard_norm = safe_float(standard.get("embedding_norm"))
    if standard_norm > 0:
        return real_norm / standard_norm
    return 0.0


def sa_probs_from_segment(seg: dict, sa_attrs: list[str]) -> tuple[dict[str, float], float]:
    confidence = ((seg.get("speech_attribute_prediction") or {}).get("feature_confidence") or {})
    probs = {attr: safe_float(confidence.get(attr), 0.5) for attr in sa_attrs}
    expected = seg.get("expected_features") or {}
    if not isinstance(expected, dict) or not expected:
        return probs, 0.5
    scores = []
    for attr, expected_value in expected.items():
        if attr not in probs:
            continue
        prob = probs[attr]
        scores.append(prob if int(safe_float(expected_value)) == 1 else 1.0 - prob)
    return probs, float(np.mean(scores)) if scores else 0.5


def train_stats(checkpoint: dict) -> dict:
    return checkpoint.get("train_feature_stats") or {}


def phone_mean(stats: dict, group: str, phone: str, fallback: float) -> float:
    values = stats.get(group) or {}
    return max(safe_float(values.get(phone), fallback), 1e-6)


def canonical_feature_value(name: str, seg: dict, checkpoint: dict, phones: list[str], sa_attrs: list[str]) -> tuple[float, dict]:
    target = target_phone(seg)
    normalized_target = normalize_phone(target)
    posterior, coverage = posterior_from_segment(seg, phones)
    stats = posterior_stats(seg, posterior, target)
    durations = duration_values(seg)
    model_stats = train_stats(checkpoint)
    global_duration = max(safe_float(model_stats.get("global_duration_mean"), 1.0), 1e-6)
    global_energy = max(safe_float(model_stats.get("global_energy_mean"), 1.0), 1e-6)
    phone_duration_mean = phone_mean(model_stats, "phone_duration_mean", target, global_duration)
    phone_energy_mean = phone_mean(model_stats, "phone_energy_mean", target, global_energy)
    energy = energy_value(seg)
    sa_probs, sa_expected_score = sa_probs_from_segment(seg, sa_attrs)

    if name.startswith("posterior:"):
        key = name.split(":", 1)[1]
        if key == "expected":
            return stats["expected"], coverage
        if key == "top1":
            return stats["top1"], coverage
        if key == "top2":
            return stats["top2"], coverage
        if key == "top1_top2_ratio":
            return stats["top1_top2_ratio"], coverage
        if key == "margin":
            return stats["margin"], coverage
        if key == "entropy":
            return stats["entropy"], coverage
        return safe_float(posterior.get(key)), coverage
    if name.startswith("canonical:"):
        key = name.split(":", 1)[1]
        return 1.0 if normalize_phone(key) == normalized_target else 0.0, coverage
    if name == "duration:sec":
        return durations["duration"], coverage
    if name == "duration:log_sec":
        return durations["log_duration"], coverage
    if name == "duration:utterance_ratio":
        return durations["utterance_ratio"], coverage
    if name == "duration:global_train_ratio":
        return durations["duration"] / global_duration, coverage
    if name == "duration:expected_phone_train_ratio":
        return durations["duration"] / phone_duration_mean, coverage
    if name == "duration:log_expected_phone_train_ratio":
        return math.log(max(durations["duration"] / phone_duration_mean, 1e-6)), coverage
    if name == "energy:utterance_rms_ratio":
        return energy, coverage
    if name == "energy:expected_phone_train_ratio":
        return energy / phone_energy_mean, coverage
    if name == "wavlm:presence_score":
        return wavlm_presence(seg), coverage
    if name == "sa:expected_match_score":
        return sa_expected_score, coverage
    if name.startswith("sa:"):
        return safe_float(sa_probs.get(name.split(":", 1)[1]), 0.5), coverage
    return 0.0, coverage


def legacy_vector(seg: dict, checkpoint: dict) -> tuple[np.ndarray, dict]:
    phones = checkpoint.get("phones") or []
    sa_attrs = checkpoint.get("sa_attrs") or []
    report = checkpoint.get("report") or {}
    feature_set = str(report.get("feature_set") or "with_sa")
    flags = {
        "posterior": True,
        "summary": True,
        "energy": feature_set in {"no_sa", "with_sa"},
        "wavlm": feature_set in {"no_sa", "with_sa"},
        "sa": feature_set == "with_sa",
    }
    target = target_phone(seg)
    posterior, coverage = posterior_from_segment(seg, phones)
    stats = posterior_stats(seg, posterior, target)
    vec = []
    if flags["posterior"]:
        vec.extend([safe_float(posterior.get(phone)) for phone in phones])
    if flags["summary"]:
        vec.extend([stats["expected"], stats["top1"], stats["top2"], stats["top1_top2_ratio"], stats["margin"], stats["entropy"]])
    if flags["energy"]:
        vec.append(energy_value(seg))
    if flags["wavlm"]:
        vec.append(wavlm_presence(seg))
    if flags["sa"]:
        sa_probs, sa_expected_score = sa_probs_from_segment(seg, sa_attrs)
        vec.append(sa_expected_score)
        vec.extend([safe_float(sa_probs.get(attr), 0.5) for attr in sa_attrs])
    return np.asarray(vec, dtype=np.float32), coverage


def vectorize_segment(seg: dict, checkpoint: dict) -> tuple[np.ndarray, dict]:
    feature_names = checkpoint.get("feature_names") or []
    phones = checkpoint.get("phones") or []
    sa_attrs = checkpoint.get("sa_attrs") or []
    if feature_names:
        values = []
        coverage = {}
        for name in feature_names:
            value, item_coverage = canonical_feature_value(name, seg, checkpoint, phones, sa_attrs)
            values.append(value)
            coverage.update(item_coverage)
        return np.asarray(values, dtype=np.float32), coverage
    return legacy_vector(seg, checkpoint)


def standardize_vector(x: np.ndarray, checkpoint: dict) -> np.ndarray:
    mean = np.asarray(checkpoint.get("mean"), dtype=np.float32).reshape(-1)
    std = np.asarray(checkpoint.get("std"), dtype=np.float32).reshape(-1)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    if x.shape[0] != mean.shape[0]:
        raise ValueError(f"Feature dimension mismatch: vector={x.shape[0]} checkpoint_mean={mean.shape[0]}")
    return (x - mean) / std


def predict_segment(seg: dict, checkpoint: dict, model: MLP, threshold: float) -> tuple[dict, dict]:
    x, coverage = vectorize_segment(seg, checkpoint)
    expected_dim = checkpoint_input_dim(checkpoint)
    if x.shape[0] != expected_dim:
        raise ValueError(f"Feature dimension mismatch: vector={x.shape[0]} checkpoint_model={expected_dim}")
    x = standardize_vector(x, checkpoint)
    with torch.no_grad():
        logits = model(torch.tensor(x.reshape(1, -1), dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy().reshape(-1)
    error_probability = float(probs[1])
    is_error = error_probability >= threshold
    return {
        "prediction": "incorrect" if is_error else "correct",
        "is_error": bool(is_error),
        "error_probability": round(error_probability, 6),
        "correct_probability": round(float(probs[0]), 6),
        "threshold": round(float(threshold), 6),
        "confidence": round(error_probability if is_error else float(probs[0]), 6),
    }, coverage


def compact_segment_info(sample: dict, seg: dict) -> dict:
    return {
        "id": seg.get("id") or seg.get("segment_id"),
        "global_id": segment_id(sample, seg),
        "word": seg.get("word"),
        "target_phoneme": target_phone(seg),
        "observed_phoneme": observed_phone(seg),
        "alignment_op": seg.get("alignment_op"),
        "start": seg.get("start"),
        "end": seg.get("end"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict correct/incorrect MDD labels for segment_attributes.json.")
    parser.add_argument("--input", default="data/final/segment_attributes.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="data/final/mdd_predictions.json")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    parser.add_argument(
        "--require-posterior",
        action="store_true",
        help="Fail instead of writing predictions if any segment uses observed-phone posterior fallback.",
    )
    args = parser.parse_args()

    dataset = load_json(Path(args.input))
    checkpoint = load_checkpoint(Path(args.checkpoint))
    threshold = float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5))
    input_dim = checkpoint_input_dim(checkpoint)
    model = MLP(input_dim)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    samples = []
    total = 0
    errors = 0
    fallback_posterior = 0
    failed_segments = 0
    for sample in dataset.get("samples", []):
        out_segments = []
        for seg in sample.get("segments", []):
            total += 1
            try:
                prediction, coverage = predict_segment(seg, checkpoint, model, threshold)
                if prediction["is_error"]:
                    errors += 1
                if coverage.get("posterior_fallback"):
                    fallback_posterior += 1
                out_segments.append(
                    {
                        **compact_segment_info(sample, seg),
                        "mdd_classifier": prediction,
                        "feature_coverage": coverage,
                    }
                )
            except Exception as exc:
                failed_segments += 1
                out_segments.append(
                    {
                        **compact_segment_info(sample, seg),
                        "mdd_classifier": {
                            "prediction": "unavailable",
                            "is_error": None,
                            "error_probability": None,
                            "correct_probability": None,
                            "threshold": round(float(threshold), 6),
                            "confidence": None,
                            "error": str(exc),
                        },
                        "feature_coverage": {},
                    }
                )
        samples.append(
            {
                "id": sample.get("id") or sample.get("sample_id") or sample.get("audio_id"),
                "speaker_id": sample.get("speaker_id"),
                "segments": out_segments,
            }
        )

    report = checkpoint.get("report") or {}
    payload = {
        "schema_version": "mdd_classifier_predictions_v1",
        "description": "Segment-level correct/incorrect predictions from the trained MDD meta-classifier.",
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "checkpoint_schema": report.get("schema_version"),
        "checkpoint_feature_set": feature_set_from_checkpoint(checkpoint),
        "checkpoint_feature_dim": input_dim,
        "threshold": round(float(threshold), 6),
        "num_samples": len(samples),
        "num_segments": total,
        "num_predicted_errors": errors,
        "num_failed_segments": failed_segments,
        "num_segments_using_posterior_fallback": fallback_posterior,
        "posterior_fallback_rate": round(float(fallback_posterior / total), 6) if total else 0.0,
        "warning": (
            "Posterior fallback means segment_attributes.json did not contain full Wav2Vec2 posterior; "
            "the tool used observed_phone as a one-hot approximation for compatibility. "
            "For research metrics, prefer a checkpoint/pipeline where full posterior is exported."
            if fallback_posterior
            else ""
        ),
        "samples": samples,
    }
    if args.require_posterior and fallback_posterior:
        raise RuntimeError(
            "--require-posterior was set, but "
            f"{fallback_posterior}/{total} segments used observed-phone posterior fallback. "
            "Regenerate segment_attributes.json with full Wav2Vec2 posterior before producing paper-grade MDD predictions."
        )
    write_json(Path(args.output), payload)
    print(f"Wrote {args.output}")
    print(f"Samples: {len(samples)}")
    print(f"Segments: {total}")
    print(f"Predicted errors: {errors}")
    print(f"Posterior fallback segments: {fallback_posterior}")
    if failed_segments:
        print(f"Failed segments: {failed_segments}")


if __name__ == "__main__":
    main()

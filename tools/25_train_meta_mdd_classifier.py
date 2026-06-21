from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = Path(r"D:\A Project YTB\L2artic")
DEFAULT_DETAILS = DEFAULT_CORPUS_ROOT / "eval_newfolder_wav2vec2_mdd_details.csv"
DEFAULT_FEATURE_CACHE = DEFAULT_CORPUS_ROOT / "meta_classifier_feature_cache.L2ARTIC_24.json"
CONSONANTS = {
    "b",
    "d",
    "dʒ",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "tʃ",
    "v",
    "w",
    "z",
    "ð",
    "ŋ",
    "ɾ",
    "ʃ",
    "ʒ",
    "θ",
}
FEATURE_SETS = {
    "posterior_canonical": {
        "posterior": True,
        "canonical": True,
        "summary": True,
        "duration": False,
        "energy": False,
        "wavlm": False,
        "sa": False,
    },
    "posterior_canonical_duration_energy": {
        "posterior": True,
        "canonical": True,
        "summary": True,
        "duration": True,
        "energy": True,
        "wavlm": False,
        "sa": False,
    },
    "posterior_canonical_duration_energy_wavlm": {
        "posterior": True,
        "canonical": True,
        "summary": True,
        "duration": True,
        "energy": True,
        "wavlm": True,
        "sa": False,
    },
    "posterior_canonical_duration_energy_wavlm_sa": {
        "posterior": True,
        "canonical": True,
        "summary": True,
        "duration": True,
        "energy": True,
        "wavlm": True,
        "sa": True,
    },
}
FEATURE_SETS["wav2vec2_only"] = FEATURE_SETS["posterior_canonical"]
FEATURE_SETS["no_sa"] = FEATURE_SETS["posterior_canonical_duration_energy_wavlm"]
FEATURE_SETS["with_sa"] = FEATURE_SETS["posterior_canonical_duration_energy_wavlm_sa"]
FEATURE_SET_DESCRIPTIONS = {
    "posterior_canonical": "Step 1: full Wav2Vec2 posterior distribution, canonical phone one-hot, and confidence summaries.",
    "posterior_canonical_duration_energy": "Step 2: Step 1 plus duration and energy normalization features.",
    "posterior_canonical_duration_energy_wavlm": "Step 3: Step 2 plus WavLM presence score.",
    "posterior_canonical_duration_energy_wavlm_sa": "Step 4: Step 3 plus Speech Attribute probabilities.",
    "wav2vec2_only": "Alias of posterior_canonical for backward compatibility.",
    "no_sa": "Alias of posterior_canonical_duration_energy_wavlm for backward compatibility.",
    "with_sa": "Alias of posterior_canonical_duration_energy_wavlm_sa for backward compatibility.",
}
EVALUATION_MODES = ["strict_all", "consonant_only"]
EVALUATION_MODE_DESCRIPTIONS = {
    "strict_all": "Evaluate every aligned phone segment; this is the strict reference score.",
    "consonant_only": "Evaluate consonant segments only, excluding vowels, schwa, and r-colored vowel cases.",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_speaker_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def validate_disjoint_splits(train_speakers: list[str], validation_speakers: list[str], test_speakers: list[str]) -> None:
    groups = {
        "train": set(train_speakers),
        "validation": set(validation_speakers),
        "test": set(test_speakers),
    }
    if not groups["train"]:
        raise ValueError("At least one train speaker is required.")
    if not groups["validation"]:
        raise ValueError("Validation speakers are required; never select thresholds on test.")
    if not groups["test"]:
        raise ValueError("At least one test speaker is required.")
    overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    bad = {name: speakers for name, speakers in overlaps.items() if speakers}
    if bad:
        raise ValueError(f"Speaker leakage detected between splits: {bad}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def load_details(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing details CSV: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["index"] = int(row["index"])
            row["y_error"] = 0 if row.get("human_op") == "match" else 1
            row["baseline_error"] = 0 if row.get("predicted_op") == "match" else 1
            rows.append(row)
    return rows


def build_cache_index(cache: dict) -> dict[str, str]:
    index = {}
    for key in cache:
        parts = key.split("/")
        if len(parts) >= 4:
            index.setdefault("/".join(parts[:3]), key)
            index.setdefault("/".join(parts[:4]), key)
    return index


def feature_cache_key(row: dict, cache_index: dict[str, str]) -> str | None:
    prefix = f"{row['speaker_id']}/{row['sample_id']}/{row['index']}/{row['expected']}/"
    exact = prefix.rstrip("/")
    if exact in cache_index:
        return cache_index[exact]
    # Some CSVs were exported through a non-UTF8 console; fall back to index-only.
    loose = f"{row['speaker_id']}/{row['sample_id']}/{row['index']}"
    return cache_index.get(loose)


def collect_rows(all_rows: list[dict], speakers: list[str]) -> list[dict]:
    wanted = set(speakers)
    return [row for row in all_rows if row.get("speaker_id") in wanted]


def build_feature_schema(rows: list[dict], cache: dict, cache_index: dict[str, str]) -> tuple[list[str], list[str]]:
    phones = set()
    sa_attrs = set()
    for row in rows:
        key = feature_cache_key(row, cache_index)
        if not key:
            continue
        item = cache.get(key) or {}
        phones.update((item.get("posterior") or {}).keys())
        sa_attrs.update((item.get("sa_positive_probs") or {}).keys())
    return sorted(phones), sorted(sa_attrs)


def row_duration(row: dict, item: dict) -> float:
    duration = safe_float(item.get("duration_sec"))
    if duration > 0:
        return duration
    start = safe_float(row.get("start"))
    end = safe_float(row.get("end"))
    return max(0.0, end - start)


def build_train_feature_stats(rows: list[dict], cache: dict, cache_index: dict[str, str]) -> dict:
    phone_durations = defaultdict(list)
    phone_energies = defaultdict(list)
    durations = []
    energies = []
    for row in rows:
        key = feature_cache_key(row, cache_index)
        if not key:
            continue
        item = cache.get(key) or {}
        duration = row_duration(row, item)
        energy = safe_float(item.get("energy_ratio"))
        expected = str(row.get("expected") or "")
        if duration > 0:
            durations.append(duration)
            phone_durations[expected].append(duration)
        if energy > 0:
            energies.append(energy)
            phone_energies[expected].append(energy)

    global_duration = float(np.mean(durations)) if durations else 1.0
    global_energy = float(np.mean(energies)) if energies else 1.0
    return {
        "global_duration_mean": global_duration,
        "global_energy_mean": global_energy,
        "phone_duration_mean": {phone: float(np.mean(values)) for phone, values in phone_durations.items() if values},
        "phone_energy_mean": {phone: float(np.mean(values)) for phone, values in phone_energies.items() if values},
    }


def feature_names(phones: list[str], sa_attrs: list[str], feature_set: str) -> list[str]:
    flags = FEATURE_SETS[feature_set]
    names = []
    if flags["posterior"]:
        names.extend([f"posterior:{phone}" for phone in phones])
    if flags["canonical"]:
        names.extend([f"canonical:{phone}" for phone in phones])
    if flags["summary"]:
        names.extend(
            [
                "posterior:expected",
                "posterior:top1",
                "posterior:top2",
                "posterior:top1_top2_ratio",
                "posterior:margin",
                "posterior:entropy",
            ]
        )
    if flags["duration"]:
        names.extend(
            [
                "duration:sec",
                "duration:log_sec",
                "duration:utterance_ratio",
                "duration:global_train_ratio",
                "duration:expected_phone_train_ratio",
                "duration:log_expected_phone_train_ratio",
            ]
        )
    if flags["energy"]:
        names.extend(["energy:utterance_rms_ratio", "energy:expected_phone_train_ratio"])
    if flags["wavlm"]:
        names.append("wavlm:presence_score")
    if flags["sa"]:
        names.append("sa:expected_match_score")
        names.extend([f"sa:{attr}" for attr in sa_attrs])
    return names


def vectorize(
    rows: list[dict],
    cache: dict,
    cache_index: dict[str, str],
    phones: list[str],
    sa_attrs: list[str],
    feature_set: str,
    train_stats: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    flags = FEATURE_SETS[feature_set]
    features = []
    labels = []
    meta = []
    missing = 0
    for row in rows:
        key = feature_cache_key(row, cache_index)
        if not key:
            missing += 1
            continue
        item = cache.get(key) or {}
        posterior = item.get("posterior") or {}
        sa_probs = item.get("sa_positive_probs") or {}
        top_prob = safe_float(item.get("top_prob"))
        second_prob = safe_float(item.get("second_prob"))
        vec = []
        if flags["posterior"]:
            vec.extend([safe_float(posterior.get(phone)) for phone in phones])
        if flags["canonical"]:
            expected = str(row.get("expected") or "")
            vec.extend([1.0 if expected == phone else 0.0 for phone in phones])
        if flags["summary"]:
            vec.extend(
                [
                    safe_float(posterior.get(row.get("expected"))),
                    top_prob,
                    second_prob,
                    top_prob / max(second_prob, 1e-6),
                    safe_float(item.get("margin")),
                    safe_float(item.get("entropy")),
                ]
            )
        if flags["duration"]:
            duration = row_duration(row, item)
            global_duration = max(safe_float(train_stats.get("global_duration_mean"), 1.0), 1e-6)
            phone_duration_mean = max(
                safe_float((train_stats.get("phone_duration_mean") or {}).get(row.get("expected")), global_duration),
                1e-6,
            )
            phone_duration_ratio = duration / phone_duration_mean
            vec.extend(
                [
                    duration,
                    math.log(max(duration, 1e-6)),
                    safe_float(item.get("duration_utterance_ratio")),
                    duration / global_duration,
                    phone_duration_ratio,
                    math.log(max(phone_duration_ratio, 1e-6)),
                ]
            )
        if flags["energy"]:
            energy = safe_float(item.get("energy_ratio"))
            global_energy = max(safe_float(train_stats.get("global_energy_mean"), 1.0), 1e-6)
            phone_energy_mean = max(
                safe_float((train_stats.get("phone_energy_mean") or {}).get(row.get("expected")), global_energy),
                1e-6,
            )
            vec.extend([energy, energy / phone_energy_mean])
        if flags["wavlm"]:
            vec.append(safe_float(item.get("wavlm_presence_score")))
        if flags["sa"]:
            vec.append(safe_float(item.get("sa_expected_match_score"), 0.5))
            vec.extend([safe_float(sa_probs.get(attr)) for attr in sa_attrs])
        features.append(vec)
        labels.append(int(row["y_error"]))
        meta.append(row)
    if missing:
        print(f"Warning: skipped {missing} rows because feature_cache entries were missing.")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), meta


def standardize(train_x: np.ndarray, other_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (other_x - mean) / std, mean, std


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


def train_model(x: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> MLP:
    torch.manual_seed(seed)
    model = MLP(int(x.shape[1]))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    counts = torch.bincount(ty, minlength=2).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    for _ in range(epochs):
        model.train()
        optim.zero_grad()
        loss = F.cross_entropy(model(tx), ty, weight=weights)
        loss.backward()
        optim.step()
    return model


def predict_probs(model: MLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        return torch.softmax(logits, dim=-1).cpu().numpy()


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    counts = Counter()
    for truth, pred in zip(y_true, y_pred):
        if truth == 1 and pred == 1:
            counts["tp"] += 1
        elif truth == 0 and pred == 1:
            counts["fp"] += 1
        elif truth == 1 and pred == 0:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def selected_indices(rows: list[dict], mode: str) -> list[int]:
    if mode == "strict_all":
        return list(range(len(rows)))
    if mode == "consonant_only":
        return [idx for idx, row in enumerate(rows) if str(row.get("expected") or "") in CONSONANTS]
    raise ValueError(f"Unknown evaluation mode: {mode}")


def evaluation_table(rows: list[dict], y_true: np.ndarray, classifier_pred: np.ndarray, baseline_pred: np.ndarray) -> dict:
    table = {}
    for mode in EVALUATION_MODES:
        indices = selected_indices(rows, mode)
        if not indices:
            table[mode] = {
                "num_segments": 0,
                "classifier": binary_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)),
                "wav2vec2_argmax_baseline": binary_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)),
            }
            continue
        idx = np.asarray(indices, dtype=np.int64)
        table[mode] = {
            "num_segments": int(len(indices)),
            "classifier": binary_metrics(y_true[idx], classifier_pred[idx]),
            "wav2vec2_argmax_baseline": binary_metrics(y_true[idx], baseline_pred[idx]),
        }
    return table


def choose_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict, list[dict]]:
    best = None
    rows = []
    for threshold in [round(x / 100, 2) for x in range(20, 81, 5)]:
        pred = (probs >= threshold).astype(np.int64)
        row = {"threshold": threshold, **binary_metrics(y_true, pred)}
        rows.append(row)
        if best is None or row["f1"] > best["f1"]:
            best = row
    assert best is not None
    return best, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe L2-ARCTIC source-to-Vietnamese binary MDD meta-classifier.")
    parser.add_argument("--details-csv", default=str(DEFAULT_DETAILS))
    parser.add_argument("--feature-cache", default=str(DEFAULT_FEATURE_CACHE))
    parser.add_argument("--train-speakers", required=True)
    parser.add_argument("--validation-speakers", required=True)
    parser.add_argument("--test-speakers", required=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "train" / "l2arctic_protocol" / "manual_run"))
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="with_sa")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    train_speakers = parse_speaker_list(args.train_speakers)
    validation_speakers = parse_speaker_list(args.validation_speakers)
    test_speakers = parse_speaker_list(args.test_speakers)
    validate_disjoint_splits(train_speakers, validation_speakers, test_speakers)

    all_rows = load_details(Path(args.details_csv))
    cache = load_json(Path(args.feature_cache))
    cache_index = build_cache_index(cache)
    train_rows = collect_rows(all_rows, train_speakers)
    validation_rows = collect_rows(all_rows, validation_speakers)
    test_rows = collect_rows(all_rows, test_speakers)
    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError(
            f"Empty split: train={len(train_rows)}, validation={len(validation_rows)}, test={len(test_rows)}. "
            "Check speaker IDs and details CSV."
        )

    phones, sa_attrs = build_feature_schema(train_rows + validation_rows, cache, cache_index)
    train_stats = build_train_feature_stats(train_rows, cache, cache_index)
    train_x, train_y, train_meta = vectorize(train_rows, cache, cache_index, phones, sa_attrs, args.feature_set, train_stats)
    validation_x, validation_y, validation_meta = vectorize(
        validation_rows, cache, cache_index, phones, sa_attrs, args.feature_set, train_stats
    )
    test_x, test_y, test_meta = vectorize(test_rows, cache, cache_index, phones, sa_attrs, args.feature_set, train_stats)
    if len(train_meta) == 0 or len(validation_meta) == 0 or len(test_meta) == 0:
        raise RuntimeError(
            "Feature cache does not cover one or more splits after speaker filtering: "
            f"train={len(train_meta)}, validation={len(validation_meta)}, test={len(test_meta)}. "
            "Use the full L2-ARCTIC feature cache or regenerate features for the selected speakers."
        )
    train_x, validation_x, mean, std = standardize(train_x, validation_x)
    test_x = (test_x - mean) / std

    model = train_model(train_x, train_y, args.epochs, args.seed)
    validation_probs = predict_probs(model, validation_x)[:, 1]
    best, threshold_sweep = choose_threshold(validation_y, validation_probs)
    test_probs = predict_probs(model, test_x)[:, 1]
    test_pred = (test_probs >= best["threshold"]).astype(np.int64)
    baseline_pred = np.asarray([int(row["baseline_error"]) for row in test_meta], dtype=np.int64)
    eval_table = evaluation_table(test_meta, test_y, test_pred, baseline_pred)

    output_dir = Path(args.output_dir)
    report = {
        "schema_version": "l2arctic_leakage_safe_meta_mdd_v2",
        "model_type": "torch_mlp_binary_error_detector",
        "train_speakers": train_speakers,
        "validation_speakers": validation_speakers,
        "test_speakers": test_speakers,
        "num_train_segments": int(len(train_meta)),
        "num_validation_segments": int(len(validation_meta)),
        "num_test_segments": int(len(test_meta)),
        "feature_set": args.feature_set,
        "feature_set_flags": FEATURE_SETS[args.feature_set],
        "feature_set_descriptions": FEATURE_SET_DESCRIPTIONS,
        "feature_dim": int(train_x.shape[1]),
        "feature_names": feature_names(phones, sa_attrs, args.feature_set),
        "phone_features": phones,
        "speech_attribute_features": sa_attrs,
        "train_feature_stats": train_stats,
        "evaluation_mode_descriptions": EVALUATION_MODE_DESCRIPTIONS,
        "leakage_controls": {
            "speaker_split": "train/validation/test are disjoint speaker sets",
            "threshold": "selected on validation only",
            "test": "used only for final metrics",
            "feature_schema": "built from train+validation only",
            "duration_energy_normalization": "phone/global duration-energy stats are fit on train only",
            "standardization": "fit on train only",
        },
        "threshold_from_validation": best,
        "test_classifier": binary_metrics(test_y, test_pred),
        "test_wav2vec2_argmax_baseline": binary_metrics(test_y, baseline_pred),
        "evaluation_modes": eval_table,
    }
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "threshold_sweep.validation.csv", threshold_sweep, ["threshold", "precision", "recall", "f1", "fpr", "accuracy", "tp", "fp", "fn", "tn"])
    detail_rows = []
    for row, truth, prob, pred in zip(test_meta, test_y, test_probs, test_pred):
        detail_rows.append(
            {
                "speaker_id": row["speaker_id"],
                "sample_id": row["sample_id"],
                "index": row["index"],
                "expected": row["expected"],
                "human_observed": row["human_observed"],
                "human_op": row["human_op"],
                "wav2vec2_predicted": row["predicted"],
                "wav2vec2_predicted_op": row["predicted_op"],
                "true_error": int(truth),
                "classifier_error_probability": round(float(prob), 6),
                "classifier_error": int(pred),
            }
        )
    write_csv(
        output_dir / "test_predictions.csv",
        detail_rows,
        [
            "speaker_id",
            "sample_id",
            "index",
            "expected",
            "human_observed",
            "human_op",
            "wav2vec2_predicted",
            "wav2vec2_predicted_op",
            "true_error",
            "classifier_error_probability",
            "classifier_error",
        ],
    )
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean,
            "std": std,
            "phones": phones,
            "sa_attrs": sa_attrs,
            "feature_names": feature_names(phones, sa_attrs, args.feature_set),
            "train_feature_stats": train_stats,
            "threshold": best["threshold"],
            "report": report,
        },
        output_dir / "meta_mdd_classifier.pt",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()

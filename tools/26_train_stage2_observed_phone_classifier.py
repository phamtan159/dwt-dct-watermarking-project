from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
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
DEFAULT_PROTOCOL = PROJECT_ROOT / "data" / "protocols" / "l2arctic_frozen_source_vietnamese_raw_cache_v1.json"
DEFAULT_L2_CACHE = Path(r"D:\A Project YTB\L2artic\regenerated_cache_v1_full\regenerated_cache_v1")
DEFAULT_DETAILS = DEFAULT_L2_CACHE / "eval_l2arctic_raw_regenerated_details.csv"
DEFAULT_FEATURE_CACHE = DEFAULT_L2_CACHE / "meta_classifier_feature_cache.raw_v1.json"
DEFAULT_STAGE1_ROOT = DEFAULT_L2_CACHE / "models"
DEFAULT_OUTPUT = DEFAULT_L2_CACHE / "stage2_observed_phone"

DELETE_LABEL = "<delete>"
CANONICAL_FEATURE_SETS = [
    "posterior_canonical",
    "posterior_canonical_duration_energy",
    "posterior_canonical_duration_energy_wavlm",
    "posterior_canonical_duration_energy_wavlm_sa",
]
FEATURE_SET_ALIASES = {
    "wav2vec2_only": "posterior_canonical",
    "no_sa": "posterior_canonical_duration_energy_wavlm",
    "with_sa": "posterior_canonical_duration_energy_wavlm_sa",
}
CONSONANTS = {
    "b",
    "d",
    "d\u0292",
    "f",
    "g",
    "\u0261",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "\u014b",
    "p",
    "r",
    "\u0279",
    "\u027e",
    "s",
    "\u0283",
    "t",
    "t\u0283",
    "v",
    "w",
    "z",
    "\u00f0",
    "\u03b8",
    "\u0292",
}
EVALUATION_MODES = ["strict_all", "consonant_only"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_stage1_module():
    path = PROJECT_ROOT / "tools" / "25_train_meta_mdd_classifier.py"
    spec = importlib.util.spec_from_file_location("stage1_mdd", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import Stage 1 helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def parse_speaker_list(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def canonical_feature_set(feature_set: str) -> str:
    return FEATURE_SET_ALIASES.get(feature_set, feature_set)


def expand_protocol_rows(rows: list[dict]) -> list[dict]:
    if any(row.get("feature_set") in CANONICAL_FEATURE_SETS for row in rows):
        return rows

    groups = {}
    for row in rows:
        if row.get("variant") != "source_only":
            continue
        output_dir = Path(row["output_dir"])
        base_output_dir = output_dir.parent if output_dir.name == row.get("feature_set") else output_dir
        key = (
            row.get("fold_id"),
            row.get("variant"),
            row.get("train_speakers"),
            row.get("validation_speakers"),
            row.get("test_speakers"),
            str(base_output_dir),
        )
        groups.setdefault(key, (row, base_output_dir))

    expanded = []
    for row, base_output_dir in groups.values():
        for feature_set in CANONICAL_FEATURE_SETS:
            item = dict(row)
            item["feature_set"] = feature_set
            item["output_dir"] = str(base_output_dir / feature_set)
            expanded.append(item)
    return expanded or rows


def normalize_phone(value: str | None) -> str:
    phone = str(value or "").strip()
    if not phone:
        return ""
    lower = phone.lower()
    if lower in {"delete", "deleted", "<deleted>", "<delete>", "deletion", "del", "\u2205", "eps", "<eps>"}:
        return DELETE_LABEL
    replacements = {
        "\u0261": "g",
        "\u0279": "r",
        "\u027e": "r",
        "\u025a": "\u0259r",
        "\u025d": "\u0259r",
    }
    return replacements.get(phone, phone)


def expected_is_consonant(row: dict) -> bool:
    return normalize_phone(row.get("expected")) in {normalize_phone(phone) for phone in CONSONANTS}


def observed_label(row: dict) -> str:
    op = str(row.get("human_op") or "")
    if op == "delete":
        return DELETE_LABEL
    label = normalize_phone(row.get("human_observed"))
    return label or DELETE_LABEL


def wav2vec2_observed_label(row: dict) -> str:
    op = str(row.get("predicted_op") or "")
    if op == "delete":
        return DELETE_LABEL
    label = normalize_phone(row.get("predicted"))
    return label or DELETE_LABEL


def row_key(row: dict) -> tuple[str, str, int]:
    return (str(row.get("speaker_id") or ""), str(row.get("sample_id") or ""), int(row.get("index") or 0))


def selected_indices(rows: list[dict], mode: str) -> list[int]:
    if mode == "strict_all":
        return list(range(len(rows)))
    if mode == "consonant_only":
        return [idx for idx, row in enumerate(rows) if expected_is_consonant(row)]
    raise ValueError(f"Unknown evaluation mode: {mode}")


def filter_error_rows(x: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, list[dict]]:
    indices = [idx for idx, row in enumerate(rows) if str(row.get("human_op") or "") != "match"]
    if not indices:
        return np.empty((0, x.shape[1]), dtype=np.float32), []
    idx = np.asarray(indices, dtype=np.int64)
    return x[idx], [rows[i] for i in indices]


def standardize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, [(x - mean) / std for x in others], mean, std


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


def multiclass_prf(y_true: list[str], y_pred: list[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "num_items": 0,
        }
    total = len(y_true)
    correct = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred)
    macro_p = []
    macro_r = []
    macro_f = []
    weighted_p = 0.0
    weighted_r = 0.0
    weighted_f = 0.0
    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        support = sum(1 for truth in y_true if truth == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_p.append(precision)
        macro_r.append(recall)
        macro_f.append(f1)
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f += f1 * support
    return {
        "accuracy": round(correct / total if total else 0.0, 4),
        "macro_precision": round(float(np.mean(macro_p)), 4),
        "macro_recall": round(float(np.mean(macro_r)), 4),
        "macro_f1": round(float(np.mean(macro_f)), 4),
        "weighted_precision": round(weighted_p / total if total else 0.0, 4),
        "weighted_recall": round(weighted_r / total if total else 0.0, 4),
        "weighted_f1": round(weighted_f / total if total else 0.0, 4),
        "num_items": total,
    }


def train_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_labels: list[str],
    id_to_label: list[str],
    epochs: int,
    seed: int,
) -> tuple[ObservedPhoneMLP, dict]:
    torch.manual_seed(seed)
    model = ObservedPhoneMLP(int(train_x.shape[1]), len(id_to_label))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tx = torch.tensor(train_x, dtype=torch.float32)
    ty = torch.tensor(train_y, dtype=torch.long)
    counts = torch.bincount(ty, minlength=len(id_to_label)).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_macro_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        loss = F.cross_entropy(model(tx), ty, weight=weights)
        loss.backward()
        optim.step()
        if validation_x.size == 0:
            continue
        validation_pred, _ = predict_labels(model, validation_x, id_to_label)
        metrics = multiclass_prf(validation_labels, validation_pred)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_macro_f1": round(best_macro_f1, 4)}


def predict_labels(model: ObservedPhoneMLP, x: np.ndarray, id_to_label: list[str]) -> tuple[list[str], np.ndarray]:
    model.eval()
    if x.size == 0:
        return [], np.empty((0, len(id_to_label)), dtype=np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_ids = probs.argmax(axis=1)
    return [id_to_label[int(idx)] for idx in pred_ids], probs


def topk_accuracy(y_true: list[str], probs: np.ndarray, label_to_id: dict[str, int], k: int = 3) -> float:
    if not y_true:
        return 0.0
    top = np.argsort(-probs, axis=1)[:, : min(k, probs.shape[1])]
    correct = 0
    for truth, row_top in zip(y_true, top):
        truth_id = label_to_id.get(truth)
        if truth_id is not None and truth_id in set(int(x) for x in row_top):
            correct += 1
    return round(correct / len(y_true), 4)


def oracle_metrics(rows: list[dict], pred: list[str], probs: np.ndarray, label_to_id: dict[str, int]) -> dict:
    truth = [observed_label(row) for row in rows]
    metrics = multiclass_prf(truth, pred)
    metrics["top3_accuracy"] = topk_accuracy(truth, probs, label_to_id, 3)
    metrics["oov_true_labels"] = sum(1 for label in truth if label not in label_to_id)
    return metrics


def end_to_end_counts(rows: list[dict], stage1_error: list[int], observed_pred: list[str]) -> dict:
    counts = Counter()
    for row, error_flag, pred_label in zip(rows, stage1_error, observed_pred):
        human_error = str(row.get("human_op") or "") != "match"
        if not human_error and not error_flag:
            counts["TA"] += 1
        elif not human_error and error_flag:
            counts["FR"] += 1
        elif human_error and not error_flag:
            counts["FA"] += 1
        else:
            truth_label = observed_label(row)
            if pred_label == truth_label:
                counts["CD"] += 1
            else:
                counts["DE"] += 1
    ta = counts["TA"]
    fr = counts["FR"]
    fa = counts["FA"]
    cd = counts["CD"]
    de = counts["DE"]
    detected_errors = cd + de
    true_errors = fa + cd + de
    predicted_errors = fr + cd + de
    detection_precision = detected_errors / predicted_errors if predicted_errors else 0.0
    detection_recall = detected_errors / true_errors if true_errors else 0.0
    detection_f1 = (
        2 * detection_precision * detection_recall / (detection_precision + detection_recall)
        if detection_precision + detection_recall
        else 0.0
    )
    diagnosis_precision = cd / predicted_errors if predicted_errors else 0.0
    diagnosis_recall = cd / true_errors if true_errors else 0.0
    diagnosis_f1 = (
        2 * diagnosis_precision * diagnosis_recall / (diagnosis_precision + diagnosis_recall)
        if diagnosis_precision + diagnosis_recall
        else 0.0
    )
    return {
        "TA": ta,
        "FR": fr,
        "FA": fa,
        "CD": cd,
        "DE": de,
        "FAR": round(fa / true_errors if true_errors else 0.0, 4),
        "FRR": round(fr / (fr + ta) if fr + ta else 0.0, 4),
        "DER": round(de / detected_errors if detected_errors else 0.0, 4),
        "detection_precision": round(detection_precision, 4),
        "detection_recall": round(detection_recall, 4),
        "detection_f1": round(detection_f1, 4),
        "diagnosis_precision": round(diagnosis_precision, 4),
        "diagnosis_recall": round(diagnosis_recall, 4),
        "diagnosis_f1": round(diagnosis_f1, 4),
        "num_segments": len(rows),
    }


def load_stage1_predictions(path: Path) -> dict[tuple[str, str, int], int]:
    if not path.exists():
        return {}
    result = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (str(row.get("speaker_id") or ""), str(row.get("sample_id") or ""), int(row.get("index") or 0))
            result[key] = int(float(row.get("classifier_error") or 0))
    return result


def subset_by_mode(
    rows: list[dict],
    stage1_error: list[int],
    observed_pred: list[str],
    mode: str,
) -> tuple[list[dict], list[int], list[str]]:
    indices = selected_indices(rows, mode)
    return [rows[i] for i in indices], [stage1_error[i] for i in indices], [observed_pred[i] for i in indices]


def confusion_rows(rows: list[dict], pred: list[str], limit: int = 30) -> list[dict]:
    counts = Counter()
    for row, guess in zip(rows, pred):
        truth = observed_label(row)
        if truth != guess:
            counts[(normalize_phone(row.get("expected")), truth, guess)] += 1
    out = []
    for (expected, truth, guess), count in counts.most_common(limit):
        out.append({"expected": expected, "human_observed": truth, "stage2_predicted": guess, "count": count})
    return out


def format_metric(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return ""
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_metric(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def aggregate_oracle(rows: list[dict], group_fields: tuple[str, str], buckets: dict, prefix: str) -> dict:
    bucket = buckets[group_fields]
    metrics = multiclass_prf(bucket[f"{prefix}_truth"], bucket[f"{prefix}_pred"])
    if prefix == "model":
        metrics["top3_accuracy"] = round(
            sum(bucket["model_top3_correct"]) / len(bucket["model_top3_correct"]) if bucket["model_top3_correct"] else 0.0,
            4,
        )
        metrics["oov_true_labels"] = bucket["model_oov"]
    metrics["num_items"] = len(bucket[f"{prefix}_truth"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 observed-phone diagnosis classifier on L2-ARCTIC frozen protocol.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--details-csv", default=str(DEFAULT_DETAILS))
    parser.add_argument("--feature-cache", default=str(DEFAULT_FEATURE_CACHE))
    parser.add_argument("--stage1-models-root", default=str(DEFAULT_STAGE1_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--feature-set", default="")
    parser.add_argument("--fold-id", default="")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    stage1 = load_stage1_module()
    protocol = load_json(Path(args.protocol))
    selected_feature_set = canonical_feature_set(args.feature_set)
    run_rows = []
    for row in expand_protocol_rows(protocol.get("run_table", [])):
        if row.get("variant") != "source_only":
            continue
        row = dict(row)
        row["feature_set"] = canonical_feature_set(row.get("feature_set", ""))
        if selected_feature_set and row["feature_set"] != selected_feature_set:
            continue
        if args.fold_id and row.get("fold_id") != args.fold_id:
            continue
        run_rows.append(row)
    if not run_rows:
        raise RuntimeError("No protocol runs selected.")

    details_path = Path(args.details_csv)
    feature_cache_path = Path(args.feature_cache)
    stage1_root = Path(args.stage1_models_root)
    output_root = Path(args.output_dir)
    all_rows = stage1.load_details(details_path)
    cache = stage1.load_json(feature_cache_path)
    cache_index = stage1.build_cache_index(cache)

    summary_rows = []
    confusion_summary = []
    aggregate = defaultdict(lambda: defaultdict(list))
    aggregate_counts = defaultdict(Counter)
    aggregate_oov = defaultdict(int)

    for run in run_rows:
        fold_id = run["fold_id"]
        feature_set = run["feature_set"]
        train_speakers = parse_speaker_list(run["train_speakers"])
        validation_speakers = parse_speaker_list(run["validation_speakers"])
        test_speakers = parse_speaker_list(run["test_speakers"])
        stage1.validate_disjoint_splits(train_speakers, validation_speakers, test_speakers)

        train_rows_all = stage1.collect_rows(all_rows, train_speakers)
        validation_rows_all = stage1.collect_rows(all_rows, validation_speakers)
        test_rows_all = stage1.collect_rows(all_rows, test_speakers)
        phones, sa_attrs = stage1.build_feature_schema(train_rows_all + validation_rows_all, cache, cache_index)
        train_stats = stage1.build_train_feature_stats(train_rows_all, cache, cache_index)
        train_x_all, _, train_meta_all = stage1.vectorize(
            train_rows_all, cache, cache_index, phones, sa_attrs, feature_set, train_stats
        )
        validation_x_all, _, validation_meta_all = stage1.vectorize(
            validation_rows_all, cache, cache_index, phones, sa_attrs, feature_set, train_stats
        )
        test_x_all, _, test_meta_all = stage1.vectorize(
            test_rows_all, cache, cache_index, phones, sa_attrs, feature_set, train_stats
        )
        train_x_error, train_rows_error = filter_error_rows(train_x_all, train_meta_all)
        validation_x_error, validation_rows_error = filter_error_rows(validation_x_all, validation_meta_all)
        test_x_error, test_rows_error = filter_error_rows(test_x_all, test_meta_all)
        if not train_rows_error or not validation_rows_error or not test_rows_error:
            raise RuntimeError(f"Empty Stage 2 error split for {fold_id}/{feature_set}.")

        id_to_label = sorted(set(observed_label(row) for row in train_rows_error))
        label_to_id = {label: idx for idx, label in enumerate(id_to_label)}
        train_y = np.asarray([label_to_id[observed_label(row)] for row in train_rows_error], dtype=np.int64)
        train_x, standardized, mean, std = standardize(train_x_error, validation_x_error, test_x_error, test_x_all)
        validation_x, test_x_error_std, test_x_all_std = standardized[0], standardized[1], standardized[2]
        validation_truth = [observed_label(row) for row in validation_rows_error]
        test_truth = [observed_label(row) for row in test_rows_error]

        model, train_info = train_model(
            train_x,
            train_y,
            validation_x,
            validation_truth,
            id_to_label,
            args.epochs,
            args.seed,
        )
        test_pred_error, test_probs_error = predict_labels(model, test_x_error_std, id_to_label)
        test_pred_all, test_probs_all = predict_labels(model, test_x_all_std, id_to_label)
        raw_pred_error = [wav2vec2_observed_label(row) for row in test_rows_error]
        raw_pred_all = [wav2vec2_observed_label(row) for row in test_meta_all]

        run_dir = output_root / fold_id / "source_only" / feature_set
        run_dir.mkdir(parents=True, exist_ok=True)
        stage1_pred_path = stage1_root / fold_id / "source_only" / feature_set / "test_predictions.csv"
        stage1_pred_map = load_stage1_predictions(stage1_pred_path)
        stage1_error_all = [stage1_pred_map.get(row_key(row), int(row.get("baseline_error") or 0)) for row in test_meta_all]
        raw_error_all = [int(row.get("baseline_error") or 0) for row in test_meta_all]

        run_report = {
            "schema_version": "l2arctic_stage2_observed_phone_v1",
            "model_type": "torch_mlp_observed_phone_classifier",
            "fold_id": fold_id,
            "feature_set": feature_set,
            "train_speakers": train_speakers,
            "validation_speakers": validation_speakers,
            "test_speakers": test_speakers,
            "num_train_error_segments": len(train_rows_error),
            "num_validation_error_segments": len(validation_rows_error),
            "num_test_error_segments": len(test_rows_error),
            "feature_dim": int(train_x.shape[1]),
            "label_inventory": id_to_label,
            "training": train_info,
            "stage1_prediction_file": str(stage1_pred_path),
            "leakage_controls": {
                "speaker_split": "train/validation/test speaker sets are disjoint",
                "stage2_training_rows": "only train-speaker human error rows are used to learn observed-phone labels",
                "label_inventory": "built from train-speaker observed-phone labels only",
                "validation": "used only to select the best Stage 2 epoch",
                "test": "used only after model selection",
                "feature_schema": "built from train+validation speakers only",
                "normalization": "duration/energy stats are fit on train speakers; feature standardization is fit on train error rows",
            },
            "evaluation_modes": {},
        }

        prediction_rows = []
        for row, truth, model_guess, raw_guess, probs in zip(
            test_rows_error, test_truth, test_pred_error, raw_pred_error, test_probs_error
        ):
            top_ids = np.argsort(-probs)[: min(3, len(id_to_label))]
            prediction_rows.append(
                {
                    "speaker_id": row["speaker_id"],
                    "sample_id": row["sample_id"],
                    "index": row["index"],
                    "expected": normalize_phone(row.get("expected")),
                    "human_observed": truth,
                    "human_op": row["human_op"],
                    "wav2vec2_predicted": raw_guess,
                    "stage2_predicted": model_guess,
                    "stage2_correct": int(model_guess == truth),
                    "stage2_top3": ";".join(id_to_label[int(i)] for i in top_ids),
                    "stage2_top_prob": round(float(probs[int(top_ids[0])]), 6) if len(top_ids) else 0.0,
                }
            )
        write_csv(
            run_dir / "test_error_predictions.csv",
            prediction_rows,
            [
                "speaker_id",
                "sample_id",
                "index",
                "expected",
                "human_observed",
                "human_op",
                "wav2vec2_predicted",
                "stage2_predicted",
                "stage2_correct",
                "stage2_top3",
                "stage2_top_prob",
            ],
        )

        for mode in EVALUATION_MODES:
            error_indices = selected_indices(test_rows_error, mode)
            mode_error_rows = [test_rows_error[i] for i in error_indices]
            mode_model_pred = [test_pred_error[i] for i in error_indices]
            mode_raw_pred = [raw_pred_error[i] for i in error_indices]
            mode_probs = test_probs_error[error_indices] if error_indices else np.empty((0, len(id_to_label)))

            model_oracle = oracle_metrics(mode_error_rows, mode_model_pred, mode_probs, label_to_id)
            raw_oracle = multiclass_prf([observed_label(row) for row in mode_error_rows], mode_raw_pred)
            raw_oracle["top3_accuracy"] = ""
            raw_oracle["oov_true_labels"] = ""

            mode_rows_all, mode_stage1_error, mode_model_all = subset_by_mode(
                test_meta_all, stage1_error_all, test_pred_all, mode
            )
            _, mode_raw_error, mode_raw_all = subset_by_mode(test_meta_all, raw_error_all, raw_pred_all, mode)
            e2e_model = end_to_end_counts(mode_rows_all, mode_stage1_error, mode_model_all)
            e2e_raw = end_to_end_counts(mode_rows_all, mode_raw_error, mode_raw_all)

            run_report["evaluation_modes"][mode] = {
                "oracle_on_true_error_segments": model_oracle,
                "wav2vec2_oracle_baseline_on_true_error_segments": raw_oracle,
                "end_to_end_stage1_stage2": e2e_model,
                "end_to_end_wav2vec2_baseline": e2e_raw,
            }
            for system_name, oracle, e2e in [
                ("stage2_model", model_oracle, e2e_model),
                ("wav2vec2_argmax", raw_oracle, e2e_raw),
            ]:
                summary_rows.append(
                    {
                        "fold_id": fold_id,
                        "feature_set": feature_set,
                        "mode": mode,
                        "system": system_name,
                        "test_speakers": ",".join(test_speakers),
                        "oracle_accuracy": oracle["accuracy"],
                        "oracle_macro_f1": oracle["macro_f1"],
                        "oracle_weighted_f1": oracle["weighted_f1"],
                        "oracle_top3_accuracy": oracle.get("top3_accuracy", ""),
                        "end_to_end_detection_precision": e2e["detection_precision"],
                        "end_to_end_detection_recall": e2e["detection_recall"],
                        "end_to_end_detection_f1": e2e["detection_f1"],
                        "diagnosis_precision": e2e["diagnosis_precision"],
                        "diagnosis_recall": e2e["diagnosis_recall"],
                        "diagnosis_f1": e2e["diagnosis_f1"],
                        "FAR": e2e["FAR"],
                        "FRR": e2e["FRR"],
                        "DER": e2e["DER"],
                        "TA": e2e["TA"],
                        "FR": e2e["FR"],
                        "FA": e2e["FA"],
                        "CD": e2e["CD"],
                        "DE": e2e["DE"],
                        "num_segments": e2e["num_segments"],
                        "num_error_segments": oracle["num_items"],
                        "oov_true_labels": oracle.get("oov_true_labels", ""),
                    }
                )

            key = (feature_set, mode)
            aggregate[key]["model_truth"].extend([observed_label(row) for row in mode_error_rows])
            aggregate[key]["model_pred"].extend(mode_model_pred)
            aggregate[key]["raw_truth"].extend([observed_label(row) for row in mode_error_rows])
            aggregate[key]["raw_pred"].extend(mode_raw_pred)
            top_ids = np.argsort(-mode_probs, axis=1)[:, : min(3, len(id_to_label))] if len(mode_error_rows) else []
            for truth, row_top in zip([observed_label(row) for row in mode_error_rows], top_ids):
                truth_id = label_to_id.get(truth)
                aggregate[key]["model_top3_correct"].append(1 if truth_id is not None and truth_id in set(int(x) for x in row_top) else 0)
                if truth_id is None:
                    aggregate_oov[key] += 1
            aggregate_counts[(key, "stage2_model")].update(e2e_model)
            aggregate_counts[(key, "wav2vec2_argmax")].update(e2e_raw)

        write_json(run_dir / "report.json", run_report)
        write_csv(
            run_dir / "confusions.top30.csv",
            confusion_rows(test_rows_error, test_pred_error, 30),
            ["expected", "human_observed", "stage2_predicted", "count"],
        )
        confusion_summary.extend(
            {
                "fold_id": fold_id,
                "feature_set": feature_set,
                **row,
            }
            for row in confusion_rows(test_rows_error, test_pred_error, 10)
        )
        torch.save(
            {
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "phones": phones,
                "sa_attrs": sa_attrs,
                "feature_set": feature_set,
                "feature_names": stage1.feature_names(phones, sa_attrs, feature_set),
                "train_feature_stats": train_stats,
                "id_to_label": id_to_label,
                "label_to_id": label_to_id,
                "report": run_report,
            },
            run_dir / "stage2_observed_phone_classifier.pt",
        )
        print(f"Done {fold_id}/{feature_set}: labels={len(id_to_label)} feature_dim={train_x.shape[1]}")

    aggregate_rows = []
    for (feature_set, mode), bucket in sorted(aggregate.items()):
        model_oracle = multiclass_prf(bucket["model_truth"], bucket["model_pred"])
        model_oracle["top3_accuracy"] = round(
            sum(bucket["model_top3_correct"]) / len(bucket["model_top3_correct"]) if bucket["model_top3_correct"] else 0.0,
            4,
        )
        model_oracle["oov_true_labels"] = aggregate_oov[(feature_set, mode)]
        raw_oracle = multiclass_prf(bucket["raw_truth"], bucket["raw_pred"])
        for system_name, oracle in [("stage2_model", model_oracle), ("wav2vec2_argmax", raw_oracle)]:
            e2e = end_to_end_counts_from_counter(aggregate_counts[((feature_set, mode), system_name)])
            aggregate_rows.append(
                {
                    "feature_set": feature_set,
                    "mode": mode,
                    "system": system_name,
                    "oracle_accuracy": oracle["accuracy"],
                    "oracle_macro_f1": oracle["macro_f1"],
                    "oracle_weighted_f1": oracle["weighted_f1"],
                    "oracle_top3_accuracy": oracle.get("top3_accuracy", ""),
                    "end_to_end_detection_precision": e2e["detection_precision"],
                    "end_to_end_detection_recall": e2e["detection_recall"],
                    "end_to_end_detection_f1": e2e["detection_f1"],
                    "diagnosis_precision": e2e["diagnosis_precision"],
                    "diagnosis_recall": e2e["diagnosis_recall"],
                    "diagnosis_f1": e2e["diagnosis_f1"],
                    "FAR": e2e["FAR"],
                    "FRR": e2e["FRR"],
                    "DER": e2e["DER"],
                    "TA": e2e["TA"],
                    "FR": e2e["FR"],
                    "FA": e2e["FA"],
                    "CD": e2e["CD"],
                    "DE": e2e["DE"],
                    "num_segments": e2e["num_segments"],
                    "num_error_segments": oracle["num_items"],
                    "oov_true_labels": oracle.get("oov_true_labels", ""),
                }
            )

    fields = [
        "fold_id",
        "feature_set",
        "mode",
        "system",
        "test_speakers",
        "oracle_accuracy",
        "oracle_macro_f1",
        "oracle_weighted_f1",
        "oracle_top3_accuracy",
        "end_to_end_detection_precision",
        "end_to_end_detection_recall",
        "end_to_end_detection_f1",
        "diagnosis_precision",
        "diagnosis_recall",
        "diagnosis_f1",
        "FAR",
        "FRR",
        "DER",
        "TA",
        "FR",
        "FA",
        "CD",
        "DE",
        "num_segments",
        "num_error_segments",
        "oov_true_labels",
    ]
    aggregate_fields = [field for field in fields if field not in {"fold_id", "test_speakers"}]
    write_csv(output_root / "stage2_summary_by_run.csv", summary_rows, fields)
    write_csv(output_root / "stage2_summary_aggregate_vietnamese.csv", aggregate_rows, aggregate_fields)
    write_csv(output_root / "stage2_confusions_top.csv", confusion_summary, ["fold_id", "feature_set", "expected", "human_observed", "stage2_predicted", "count"])
    write_json(
        output_root / "stage2_manifest.json",
        {
            "schema_version": "l2arctic_stage2_observed_phone_manifest_v1",
            "details_csv": str(details_path),
            "feature_cache": str(feature_cache_path),
            "stage1_models_root": str(stage1_root),
            "output_dir": str(output_root),
            "num_runs": len(run_rows),
            "epochs": args.epochs,
            "seed": args.seed,
            "evaluation_note": "Oracle metrics score observed-phone diagnosis on true human-error segments. End-to-end metrics combine Stage 1 error detection with Stage 2 observed-phone diagnosis.",
        },
    )

    md = [
        "# Stage 2 Observed-Phone Diagnosis Results",
        "",
        "## Aggregate Vietnamese Test Speakers",
        "",
        markdown_table(
            aggregate_rows,
            [
                "feature_set",
                "mode",
                "system",
                "oracle_accuracy",
                "oracle_macro_f1",
                "oracle_weighted_f1",
                "oracle_top3_accuracy",
                "diagnosis_precision",
                "diagnosis_recall",
                "diagnosis_f1",
                "FAR",
                "FRR",
                "DER",
                "CD",
                "DE",
            ],
        ),
        "",
        "## Per-Fold Results",
        "",
        markdown_table(
            summary_rows,
            [
                "fold_id",
                "feature_set",
                "mode",
                "system",
                "oracle_accuracy",
                "oracle_macro_f1",
                "oracle_weighted_f1",
                "diagnosis_f1",
                "FAR",
                "FRR",
                "DER",
            ],
        ),
        "",
    ]
    (output_root / "stage2_paper_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {output_root / 'stage2_summary_aggregate_vietnamese.csv'}")
    print(f"Wrote: {output_root / 'stage2_paper_tables.md'}")


def end_to_end_counts_from_counter(counts: Counter) -> dict:
    ta = counts["TA"]
    fr = counts["FR"]
    fa = counts["FA"]
    cd = counts["CD"]
    de = counts["DE"]
    detected_errors = cd + de
    true_errors = fa + cd + de
    predicted_errors = fr + cd + de
    detection_precision = detected_errors / predicted_errors if predicted_errors else 0.0
    detection_recall = detected_errors / true_errors if true_errors else 0.0
    detection_f1 = (
        2 * detection_precision * detection_recall / (detection_precision + detection_recall)
        if detection_precision + detection_recall
        else 0.0
    )
    diagnosis_precision = cd / predicted_errors if predicted_errors else 0.0
    diagnosis_recall = cd / true_errors if true_errors else 0.0
    diagnosis_f1 = (
        2 * diagnosis_precision * diagnosis_recall / (diagnosis_precision + diagnosis_recall)
        if diagnosis_precision + diagnosis_recall
        else 0.0
    )
    return {
        "TA": ta,
        "FR": fr,
        "FA": fa,
        "CD": cd,
        "DE": de,
        "FAR": round(fa / true_errors if true_errors else 0.0, 4),
        "FRR": round(fr / (fr + ta) if fr + ta else 0.0, 4),
        "DER": round(de / detected_errors if detected_errors else 0.0, 4),
        "detection_precision": round(detection_precision, 4),
        "detection_recall": round(detection_recall, 4),
        "detection_f1": round(detection_f1, 4),
        "diagnosis_precision": round(diagnosis_precision, 4),
        "diagnosis_recall": round(diagnosis_recall, 4),
        "diagnosis_f1": round(diagnosis_f1, 4),
        "num_segments": ta + fr + fa + cd + de,
    }


if __name__ == "__main__":
    main()

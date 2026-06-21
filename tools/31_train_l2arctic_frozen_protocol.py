from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "data" / "protocols" / "l2arctic_frozen_source_vietnamese_raw_cache_v1.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "train" / "l2arctic_raw_cache_v1" / "source_only_summary.csv"
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
FEATURE_SET_CHOICES = [
    "",
    *CANONICAL_FEATURE_SETS,
    "wav2vec2_only",
    "no_sa",
    "with_sa",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}). See {log_path}")


def canonical_feature_set(feature_set: str) -> str:
    return FEATURE_SET_ALIASES.get(feature_set, feature_set)


def expand_legacy_protocol_rows(rows: list[dict]) -> list[dict]:
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
            row.get("details_csv"),
            row.get("feature_cache"),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen L2-ARCTIC source-to-Vietnamese protocol from regenerated raw cache.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--feature-set", default="", choices=FEATURE_SET_CHOICES)
    parser.add_argument("--fold-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    protocol = load_json(Path(args.protocol))
    selected_feature_set = canonical_feature_set(args.feature_set)
    run_rows = []
    for row in expand_legacy_protocol_rows(protocol.get("run_table", [])):
        if row.get("variant") != "source_only":
            continue
        if selected_feature_set and canonical_feature_set(row.get("feature_set", "")) != selected_feature_set:
            continue
        if args.fold_id and row.get("fold_id") != args.fold_id:
            continue
        run_rows.append(row)
    if not run_rows:
        raise RuntimeError("No runs selected from protocol.")

    for row in run_rows:
        output_dir = Path(row["output_dir"])
        command = [
            sys.executable,
            "tools/25_train_meta_mdd_classifier.py",
            "--details-csv",
            row["details_csv"],
            "--feature-cache",
            row["feature_cache"],
            "--train-speakers",
            row["train_speakers"],
            "--validation-speakers",
            row["validation_speakers"],
            "--test-speakers",
            row["test_speakers"],
            "--feature-set",
            row["feature_set"],
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
        ]
        print("RUN", " ".join(command))
        if not args.dry_run:
            run_command(command, output_dir / "train.log")

    if args.dry_run:
        return

    summary_rows = []
    for row in run_rows:
        report_path = Path(row["output_dir"]) / "report.json"
        report = load_json(report_path)
        for mode, table in report["evaluation_modes"].items():
            cls = table["classifier"]
            base = table["wav2vec2_argmax_baseline"]
            summary_rows.append(
                {
                    "fold_id": row["fold_id"],
                    "variant": row["variant"],
                    "feature_set": row["feature_set"],
                    "mode": mode,
                    "test_speakers": row["test_speakers"],
                    "validation_speakers": row["validation_speakers"],
                    "num_segments": table["num_segments"],
                    "precision": cls["precision"],
                    "recall": cls["recall"],
                    "f1": cls["f1"],
                    "fpr": cls["fpr"],
                    "accuracy": cls["accuracy"],
                    "tp": cls["tp"],
                    "fp": cls["fp"],
                    "fn": cls["fn"],
                    "tn": cls["tn"],
                    "baseline_precision": base["precision"],
                    "baseline_recall": base["recall"],
                    "baseline_f1": base["f1"],
                    "threshold": report["threshold_from_validation"]["threshold"],
                    "feature_dim": report["feature_dim"],
                }
            )
    fields = [
        "fold_id",
        "variant",
        "feature_set",
        "mode",
        "test_speakers",
        "validation_speakers",
        "num_segments",
        "precision",
        "recall",
        "f1",
        "fpr",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "baseline_precision",
        "baseline_recall",
        "baseline_f1",
        "threshold",
        "feature_dim",
    ]
    write_csv(Path(args.summary), summary_rows, fields)
    print(f"Wrote: {args.summary}")


if __name__ == "__main__":
    main()

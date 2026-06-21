from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = Path(r"D:\A Project YTB\L2artic")

SPEAKERS_BY_L1 = {
    "Arabic": {"male": ["ABA", "YBAA"], "female": ["SKA", "ZHAA"]},
    "Chinese": {"male": ["BWC", "TXHC"], "female": ["LXC", "NCC"]},
    "Hindi": {"male": ["ASI", "RRBI"], "female": ["SVBI", "TNI"]},
    "Korean": {"male": ["HKK", "YKWK"], "female": ["HJK", "YDCK"]},
    "Spanish": {"male": ["EBVS", "ERMS"], "female": ["MBMPS", "NJS"]},
    "Vietnamese": {"male": ["HQTV", "TLV"], "female": ["PNV", "THV"]},
}

VIETNAMESE_FOLDS = [
    {"fold_id": "test_TLV", "vietnamese_adapt_train": ["HQTV", "PNV"], "vietnamese_dev": ["THV"], "vietnamese_test": ["TLV"]},
    {"fold_id": "test_HQTV", "vietnamese_adapt_train": ["TLV", "THV"], "vietnamese_dev": ["PNV"], "vietnamese_test": ["HQTV"]},
    {"fold_id": "test_THV", "vietnamese_adapt_train": ["HQTV", "PNV"], "vietnamese_dev": ["TLV"], "vietnamese_test": ["THV"]},
    {"fold_id": "test_PNV", "vietnamese_adapt_train": ["TLV", "THV"], "vietnamese_dev": ["HQTV"], "vietnamese_test": ["PNV"]},
]
FEATURE_SETS = ["wav2vec2_only", "no_sa", "with_sa"]
FEATURE_SET_DESCRIPTIONS = {
    "wav2vec2_only": "Wav2Vec2 posterior distribution plus confidence summaries only.",
    "no_sa": "All non-SA cues: Wav2Vec2 posterior, confidence summaries, energy ratio, and WavLM presence score.",
    "with_sa": "Full cue set: Wav2Vec2 posterior, confidence summaries, energy ratio, WavLM presence score, and Speech Attribute probabilities.",
}
EVALUATION_MODES = ["strict_all", "consonant_only"]
EVALUATION_MODE_DESCRIPTIONS = {
    "strict_all": "Evaluate every aligned phone segment; this is the strict reference score.",
    "consonant_only": "Evaluate consonant segments only, excluding vowels, schwa, and r-colored vowel cases.",
}


def flatten(group: dict[str, list[str]]) -> list[str]:
    return [speaker for gender in ["male", "female"] for speaker in group[gender]]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assert_disjoint(name: str, train: list[str], validation: list[str], test: list[str]) -> None:
    groups = {"train": set(train), "validation": set(validation), "test": set(test)}
    overlaps = {
        "train_validation": sorted(groups["train"] & groups["validation"]),
        "train_test": sorted(groups["train"] & groups["test"]),
        "validation_test": sorted(groups["validation"] & groups["test"]),
    }
    bad = {key: value for key, value in overlaps.items() if value}
    if bad:
        raise RuntimeError(f"Leakage in {name}: {bad}")


def command_for(row: dict) -> str:
    return (
        "python tools/25_train_meta_mdd_classifier.py "
        f"--details-csv \"{row['details_csv']}\" "
        f"--feature-cache \"{row['feature_cache']}\" "
        f"--train-speakers \"{row['train_speakers']}\" "
        f"--validation-speakers \"{row['validation_speakers']}\" "
        f"--test-speakers \"{row['test_speakers']}\" "
        f"--feature-set \"{row['feature_set']}\" "
        f"--output-dir \"{row['output_dir']}\""
    )


def build_protocol(corpus_root: Path) -> dict:
    source_speakers = [
        speaker
        for l1, group in SPEAKERS_BY_L1.items()
        if l1 != "Vietnamese"
        for speaker in flatten(group)
    ]
    vietnamese_speakers = flatten(SPEAKERS_BY_L1["Vietnamese"])
    details_csv = str(corpus_root / "eval_newfolder_wav2vec2_mdd_details.csv")
    feature_cache = str(corpus_root / "meta_classifier_feature_cache.L2ARTIC_24.json")
    run_rows = []
    folds = []

    for fold in VIETNAMESE_FOLDS:
        fold_rows = []
        for variant, train in [
            ("source_only", source_speakers),
            ("vietnamese_adapted", source_speakers + fold["vietnamese_adapt_train"]),
        ]:
            for feature_set in FEATURE_SETS:
                validation = fold["vietnamese_dev"]
                test = fold["vietnamese_test"]
                assert_disjoint(f"{fold['fold_id']}/{variant}/{feature_set}", train, validation, test)
                row = {
                    "fold_id": fold["fold_id"],
                    "variant": variant,
                    "feature_set": feature_set,
                    "train_speakers": ",".join(train),
                    "validation_speakers": ",".join(validation),
                    "test_speakers": ",".join(test),
                    "details_csv": details_csv,
                    "feature_cache": feature_cache,
                    "output_dir": str(PROJECT_ROOT / "train" / "l2arctic_protocol" / fold["fold_id"] / variant / feature_set),
                }
                row["command"] = command_for(row)
                run_rows.append(row)
                fold_rows.append(row)
        folds.append({**fold, "runs": fold_rows})

    return {
        "schema_version": "l2arctic_source_to_vietnamese_protocol_v1",
        "corpus_root": str(corpus_root),
        "speaker_inventory": SPEAKERS_BY_L1,
        "source_speakers": source_speakers,
        "vietnamese_speakers": vietnamese_speakers,
        "source_training_ablation": {
            "feature_sets": FEATURE_SETS,
            "feature_set_descriptions": FEATURE_SET_DESCRIPTIONS,
            "evaluation_modes": EVALUATION_MODES,
            "evaluation_mode_descriptions": EVALUATION_MODE_DESCRIPTIONS,
        },
        "folds": folds,
        "run_table": run_rows,
        "leakage_rules": [
            "No speaker may appear in train, validation, and test within the same run.",
            "Thresholds, feature choices, prompt variants, and ablations are selected using train/validation only.",
            "The Vietnamese test speaker is evaluated once for final reporting.",
            "Feature extraction may run on test audio only when it is deterministic and label-free.",
            "Human labels from test must never be used for model selection.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-safe L2-ARCTIC source-to-Vietnamese protocol manifest.")
    parser.add_argument("--corpus-root", default=str(DEFAULT_CORPUS_ROOT))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "protocols" / "l2arctic_source_vietnamese_protocol.json"))
    args = parser.parse_args()

    output = Path(args.output)
    protocol = build_protocol(Path(args.corpus_root))
    csv_path = output.with_suffix(".runs.csv")
    write_json(output, protocol)
    write_csv(
        csv_path,
        protocol["run_table"],
        ["fold_id", "variant", "feature_set", "train_speakers", "validation_speakers", "test_speakers", "details_csv", "feature_cache", "output_dir", "command"],
    )
    print(f"Wrote: {output}")
    print(f"Wrote: {csv_path}")
    print(f"Runs: {len(protocol['run_table'])}")


if __name__ == "__main__":
    main()

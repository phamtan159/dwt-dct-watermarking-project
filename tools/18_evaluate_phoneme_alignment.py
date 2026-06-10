from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


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


def target_phone(segment: dict) -> str:
    return str(
        segment.get("target_phoneme")
        or segment.get("phoneme_standard")
        or segment.get("standard_phone")
        or segment.get("expected")
        or ""
    )


def observed_phone(segment: dict) -> str:
    return str(
        segment.get("observed_phoneme")
        or segment.get("phoneme_real")
        or segment.get("phone")
        or segment.get("observed")
        or ""
    )


def normalize_op(value) -> str:
    op = str(value or "").strip().lower()
    aliases = {
        "sub": "substitute",
        "substitution": "substitute",
        "del": "delete",
        "deletion": "delete",
        "ins": "insert",
        "insertion": "insert",
        "equal": "match",
        "ok": "match",
    }
    return aliases.get(op, op or "unknown")


def empty_phone(value: str) -> bool:
    return value.strip() in {"", "-", "<eps>", "eps", "none", "null"}


def infer_op(segment: dict) -> str:
    op = normalize_op(segment.get("alignment_op"))
    if op != "unknown":
        return op

    expected = target_phone(segment)
    observed = observed_phone(segment)
    if empty_phone(expected) and not empty_phone(observed):
        return "insert"
    if not empty_phone(expected) and empty_phone(observed):
        return "delete"
    if expected == observed:
        return "match"
    return "substitute"


def segment_iter(dataset: dict):
    for sample in dataset.get("samples", []):
        sample_id = sample.get("id") or sample.get("sample_id") or sample.get("audio_id") or sample.get("video_id")
        speaker_id = sample.get("speaker_id") or "unknown"
        for segment in sample.get("segments", []):
            yield sample_id, speaker_id, segment


def metrics_from_counts(counts: Counter, denominator: int) -> dict:
    substitutions = counts.get("substitute", 0)
    deletions = counts.get("delete", 0)
    insertions = counts.get("insert", 0)
    matches = counts.get("match", 0)
    errors = substitutions + deletions + insertions
    return {
        "expected_phonemes": denominator,
        "matches": matches,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "other_ops": sum(value for key, value in counts.items() if key not in {"match", "substitute", "delete", "insert"}),
        "per": round4(safe_div(errors, denominator)),
        "match_rate": round4(safe_div(matches, denominator)),
        "substitution_rate": round4(safe_div(substitutions, denominator)),
        "deletion_rate": round4(safe_div(deletions, denominator)),
        "insertion_rate": round4(safe_div(insertions, denominator)),
    }


def evaluate(dataset: dict) -> dict:
    global_counts: Counter = Counter()
    by_phoneme: dict[str, Counter] = defaultdict(Counter)
    by_speaker: dict[str, Counter] = defaultdict(Counter)
    by_sample: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    expected_total = 0

    for sample_id, speaker_id, segment in segment_iter(dataset):
        expected = target_phone(segment)
        observed = observed_phone(segment)
        op = infer_op(segment)

        if op != "insert":
            expected_total += 1
            by_phoneme[expected][op] += 1
        else:
            by_phoneme["<inserted>"][op] += 1

        global_counts[op] += 1
        by_speaker[speaker_id][op] += 1
        by_sample[str(sample_id)][op] += 1
        confusion[expected][observed] += 1

    per_phoneme = {}
    for phoneme, counts in sorted(by_phoneme.items()):
        denominator = sum(counts.values()) - counts.get("insert", 0)
        if phoneme == "<inserted>":
            denominator = counts.get("insert", 0)
        per_phoneme[phoneme] = metrics_from_counts(counts, denominator)

    per_speaker = {
        speaker: metrics_from_counts(counts, sum(counts.values()) - counts.get("insert", 0))
        for speaker, counts in sorted(by_speaker.items())
    }
    per_sample = {
        sample: metrics_from_counts(counts, sum(counts.values()) - counts.get("insert", 0))
        for sample, counts in sorted(by_sample.items())
    }

    return {
        "schema_version": "phoneme_alignment_eval_v1",
        "description": "PER and substitution/deletion/insertion rates computed from segment alignment_op.",
        "overall": metrics_from_counts(global_counts, expected_total),
        "op_counts": dict(sorted(global_counts.items())),
        "per_phoneme": per_phoneme,
        "per_speaker": per_speaker,
        "per_sample": per_sample,
        "confusion_matrix": {truth: dict(preds) for truth, preds in sorted(confusion.items())},
    }


def write_per_phoneme_csv(path: Path, per_phoneme: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phoneme",
        "expected_phonemes",
        "matches",
        "substitutions",
        "deletions",
        "insertions",
        "per",
        "match_rate",
        "substitution_rate",
        "deletion_rate",
        "insertion_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for phoneme, metrics in per_phoneme.items():
            row = {"phoneme": phoneme}
            row.update({field: metrics.get(field, 0) for field in fields if field != "phoneme"})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate phoneme alignment with PER/S/D/I metrics.")
    parser.add_argument("--dataset", default="data/final/segment_attributes.json")
    parser.add_argument("--output", default="data/final/eval_phoneme_alignment.json")
    parser.add_argument("--per-phoneme-csv", default="data/final/eval_phoneme_alignment_per_phoneme.csv")
    args = parser.parse_args()

    dataset = load_json(Path(args.dataset))
    result = evaluate(dataset)
    write_json(Path(args.output), result)
    write_per_phoneme_csv(Path(args.per_phoneme_csv), result["per_phoneme"])

    overall = result["overall"]
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.per_phoneme_csv}")
    print(
        "Overall: "
        f"PER={overall['per']} "
        f"S={overall['substitutions']} "
        f"D={overall['deletions']} "
        f"I={overall['insertions']} "
        f"N={overall['expected_phonemes']}"
    )


if __name__ == "__main__":
    main()

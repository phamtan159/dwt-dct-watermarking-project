"""
Create a fixed stability benchmark split for pronunciation fine-tuning.

Input:
    data/final/dataset.json

Outputs:
    data/final/train_dataset.json
    data/final/stability_benchmark.json

The benchmark is stratified by label when possible, so common OK phonemes and
pronunciation-error labels are both represented. Keep this file fixed across
experiments to catch regressions/forgetting during fine-tuning.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def stable_key(item):
    """Return a stable key for deterministic sorting/splitting."""
    audio = str(item.get("audio", ""))
    labels = "|".join(item.get("labels", []))
    phones = "|".join(p.get("phone", "") for p in item.get("phonemes", []))
    raw = f"{audio}|{labels}|{phones}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def primary_label(item):
    """Prefer an error label; otherwise mark the sample as OK."""
    labels = item.get("labels", [])
    for label in labels:
        if label != "OK":
            return label
    return "OK"


def make_split(items, benchmark_ratio, min_per_label):
    groups = defaultdict(list)
    for item in items:
        groups[primary_label(item)].append(item)

    benchmark_ids = set()
    for label, group in groups.items():
        ordered = sorted(group, key=stable_key)
        quota = max(min_per_label, round(len(ordered) * benchmark_ratio))
        quota = min(quota, len(ordered))
        for item in ordered[:quota]:
            benchmark_ids.add(id(item))

    train = [item for item in items if id(item) not in benchmark_ids]
    benchmark = [item for item in items if id(item) in benchmark_ids]
    return train, benchmark


def main():
    parser = argparse.ArgumentParser(description="Create a fixed stability benchmark split.")
    parser.add_argument("--input", default="data/final/dataset.json")
    parser.add_argument("--train-output", default="data/final/train_dataset.json")
    parser.add_argument("--benchmark-output", default="data/final/stability_benchmark.json")
    parser.add_argument("--benchmark-ratio", type=float, default=0.2)
    parser.add_argument("--min-per-label", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    train_path = Path(args.train_output)
    benchmark_path = Path(args.benchmark_output)

    with input_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Expected dataset JSON to contain a list of samples.")

    train, benchmark = make_split(items, args.benchmark_ratio, args.min_per_label)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with benchmark_path.open("w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"Input samples: {len(items)}")
    print(f"Train samples: {len(train)} -> {train_path}")
    print(f"Benchmark samples: {len(benchmark)} -> {benchmark_path}")


if __name__ == "__main__":
    main()

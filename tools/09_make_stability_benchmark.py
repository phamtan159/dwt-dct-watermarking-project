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


def sample_segments(item):
    return item.get("segments", [])


def segment_label(seg):
    for key in ("final_error_label", "label", "error_code", "error_id"):
        value = seg.get(key)
        if value is not None and str(value).strip():
            value = str(value).strip()
            return "OK" if value in {"0", "no_error"} or value.upper() == "OK" else value
    return "OK"


def stable_key(item):
    """Return a stable key for deterministic sorting/splitting."""
    audio = str(item.get("audio", item.get("audio_path", item.get("id", ""))))
    if "segments" in item:
        labels = "|".join(segment_label(seg) for seg in item.get("segments", []))
        phones = "|".join(
            str(seg.get("phoneme_real") or seg.get("phone") or seg.get("phoneme") or "")
            for seg in item.get("segments", [])
        )
    else:
        labels = "|".join(item.get("labels", []))
        phones = "|".join(p.get("phone", "") for p in item.get("phonemes", []))
    raw = f"{audio}|{labels}|{phones}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def primary_label(item):
    """Prefer an error label; otherwise mark the sample as OK."""
    if "segments" in item:
        for seg in item.get("segments", []):
            label = segment_label(seg)
            if label != "OK":
                return label
        return "OK"

    labels = item.get("labels", [])
    for label in labels:
        if label not in {"OK", "0", "no_error"}:
            return label
    return "OK"


def with_samples(payload, samples, split_info=None):
    if isinstance(payload, dict) and "samples" in payload:
        output = dict(payload)
        output["samples"] = samples
        output["num_samples"] = len(samples)
        output["num_segments"] = sum(len(sample_segments(sample)) for sample in samples)
        if split_info is not None:
            output["split"] = split_info
        return output
    return samples


def speaker_id(item):
    value = item.get("speaker_id") or item.get("speaker")
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def speaker_label_set(samples):
    labels = set()
    for sample in samples:
        if "segments" in sample:
            for seg in sample.get("segments", []):
                labels.add(segment_label(seg))
        else:
            labels.update(sample.get("labels", []))
    labels.discard("OK")
    return labels or {"OK"}


def make_sample_split(items, benchmark_ratio, min_per_label):
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


def make_speaker_split(items, benchmark_ratio, min_speakers):
    missing = [item.get("id") or item.get("audio_id") or item.get("video_id") for item in items if speaker_id(item) is None]
    if missing:
        preview = ", ".join(str(item) for item in missing[:8])
        raise ValueError(
            "Speaker-aware split requires speaker_id for every sample. "
            f"Missing speaker_id in {len(missing)} sample(s): {preview}. "
            "Add data/sample_metadata.csv or speaker_id fields before splitting."
        )

    by_speaker = defaultdict(list)
    for item in items:
        by_speaker[speaker_id(item)].append(item)

    speakers = []
    for speaker, samples in by_speaker.items():
        labels = "|".join(sorted(speaker_label_set(samples)))
        key = hashlib.sha1(f"{speaker}|{labels}".encode("utf-8")).hexdigest()
        speakers.append((speaker, samples, labels, key))

    if len(speakers) < 2:
        raise ValueError("Speaker-aware split needs at least 2 different speaker_id values.")

    target_count = max(min_speakers, round(len(speakers) * benchmark_ratio))
    target_count = min(max(1, target_count), len(speakers) - 1)

    speakers_by_coverage = sorted(
        speakers,
        key=lambda item: (-len(speaker_label_set(item[1])), item[3]),
    )
    benchmark_speakers = {speaker for speaker, _, _, _ in speakers_by_coverage[:target_count]}

    train = [item for item in items if speaker_id(item) not in benchmark_speakers]
    benchmark = [item for item in items if speaker_id(item) in benchmark_speakers]
    return train, benchmark


def make_split(items, benchmark_ratio, min_per_label, split_by, min_speakers):
    if split_by == "sample":
        return make_sample_split(items, benchmark_ratio, min_per_label), "sample"
    if split_by == "speaker":
        return make_speaker_split(items, benchmark_ratio, min_speakers), "speaker"

    if all(speaker_id(item) is not None for item in items) and len({speaker_id(item) for item in items}) >= 2:
        return make_speaker_split(items, benchmark_ratio, min_speakers), "speaker"
    return make_sample_split(items, benchmark_ratio, min_per_label), "sample"


def main():
    parser = argparse.ArgumentParser(description="Create a fixed stability benchmark split.")
    parser.add_argument("--input", default="data/final/dataset.json")
    parser.add_argument("--train-output", default="data/final/train_dataset.json")
    parser.add_argument("--benchmark-output", default="data/final/stability_benchmark.json")
    parser.add_argument("--split-by", choices=["speaker", "sample", "auto"], default="speaker")
    parser.add_argument("--benchmark-ratio", type=float, default=0.2)
    parser.add_argument("--min-per-label", type=int, default=1)
    parser.add_argument("--min-speakers", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    train_path = Path(args.train_output)
    benchmark_path = Path(args.benchmark_output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {input_path}. "
            "Run tools/08_build_dataset.py first."
        )

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        items = payload["samples"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Expected dataset JSON to contain a list or an audio_visual_v1 object.")

    (train, benchmark), split_method = make_split(
        items,
        args.benchmark_ratio,
        args.min_per_label,
        args.split_by,
        args.min_speakers,
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)

    train_speakers = sorted({speaker_id(item) for item in train if speaker_id(item) is not None})
    benchmark_speakers = sorted({speaker_id(item) for item in benchmark if speaker_id(item) is not None})
    base_split_info = {
        "method": split_method,
        "benchmark_ratio": args.benchmark_ratio,
        "train_speakers": train_speakers,
        "benchmark_speakers": benchmark_speakers,
    }

    with train_path.open("w", encoding="utf-8") as f:
        split_info = dict(base_split_info)
        split_info["role"] = "train"
        json.dump(with_samples(payload, train, split_info), f, ensure_ascii=False, indent=2)

    with benchmark_path.open("w", encoding="utf-8") as f:
        split_info = dict(base_split_info)
        split_info["role"] = "benchmark"
        json.dump(with_samples(payload, benchmark, split_info), f, ensure_ascii=False, indent=2)

    print(f"Input samples: {len(items)}")
    print(f"Split method: {split_method}")
    if split_method == "speaker":
        print(f"Train speakers: {sorted({speaker_id(item) for item in train})}")
        print(f"Benchmark speakers: {sorted({speaker_id(item) for item in benchmark})}")
    print(f"Train samples: {len(train)} -> {train_path}")
    print(f"Benchmark samples: {len(benchmark)} -> {benchmark_path}")


if __name__ == "__main__":
    main()

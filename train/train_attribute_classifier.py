from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from attribute_classifier import (
    AttributeSoftmaxClassifier,
    build_examples,
    labels_to_indices,
    load_json,
    matrix_from_rows,
    predict_payload,
    softmax,
    write_json,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def split_indices(length: int, val_ratio: float, seed: int):
    indices = list(range(length))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if length < 2:
        return indices, []
    val_size = max(1, int(round(length * val_ratio)))
    val_size = min(val_size, length - 1)
    return indices[val_size:], indices[:val_size]


def split_indices_by_speaker(rows: list[dict], val_ratio: float, seed: int):
    by_speaker: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        speaker_id = row.get("speaker_id") or row.get("sample", {}).get("speaker_id") or "unknown_speaker"
        by_speaker.setdefault(str(speaker_id), []).append(index)

    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        print("Only one speaker found in classifier rows; falling back to segment-level validation split.")
        return split_indices(len(rows), val_ratio, seed)

    rng = random.Random(seed)
    rng.shuffle(speakers)
    val_speaker_count = max(1, int(round(len(speakers) * val_ratio)))
    if val_speaker_count >= len(speakers):
        val_speaker_count = len(speakers) - 1

    val_speakers = set(speakers[:val_speaker_count])
    train_speakers = [speaker for speaker in speakers if speaker not in val_speakers]
    train_indices = [
        index
        for speaker in train_speakers
        for index in by_speaker[speaker]
    ]
    val_indices = [
        index
        for speaker in speakers
        if speaker in val_speakers
        for index in by_speaker[speaker]
    ]

    if not train_indices or not val_indices:
        print("Speaker validation split produced an empty side; falling back to segment-level validation split.")
        return split_indices(len(rows), val_ratio, seed)

    print(f"Classifier train speakers: {', '.join(train_speakers)}")
    print(f"Classifier val speakers: {', '.join(sorted(val_speakers))}")
    return train_indices, val_indices


def weighted_loss_and_grad(logits, y, class_weights):
    probs = softmax(logits)
    row_ids = np.arange(len(y))
    sample_weights = class_weights[y]
    losses = -np.log(np.maximum(probs[row_ids, y], 1e-8)) * sample_weights
    grad = probs
    grad[row_ids, y] -= 1.0
    grad *= (sample_weights / max(1, len(y)))[:, None]
    return float(losses.mean()), grad


def accuracy(model, x, y):
    if len(y) == 0:
        return None
    probs = model.predict_proba(x)
    pred = np.argmax(probs, axis=1)
    return float((pred == y).mean())


def evaluate(model, x, y, class_weights):
    if len(y) == 0:
        return None, None
    logits = model.logits(x)
    loss, _ = weighted_loss_and_grad(logits, y, class_weights)
    return loss, accuracy(model, x, y)


def make_class_weights(rows, label_names):
    counts = Counter(row["label"] for row in rows)
    weights = []
    for label in label_names:
        count = max(1, counts.get(label, 0))
        weights.append(1.0 / (count ** 0.5))
    weights = np.array(weights, dtype=np.float32)
    return weights / max(float(weights.mean()), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an attribute baseline classifier for pronunciation errors.")
    parser.add_argument("--dataset", default="data/final/train_dataset.json")
    parser.add_argument("--output", default="train/attribute_classifier.npz")
    parser.add_argument("--predictions", default="data/final/classifier_predictions.json")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target",
        choices=("label", "category", "severity", "primary_evidence"),
        default="label",
        help="Prediction target to train: fine label, coarse category, severity, or primary evidence.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("full", "requested", "recommended"),
        default="full",
        help="Use full legacy features or only the requested acoustic/visual/WavLM summary feature set.",
    )
    parser.add_argument("--no-rule-features", action="store_true")
    parser.add_argument(
        "--no-wavlm-features",
        action="store_true",
        help="Ablation: train without WavLM embedding/audio_attributes features.",
    )
    parser.add_argument(
        "--no-phoneme-identity-features",
        action="store_true",
        help="Ablation: do not use expected/observed phoneme IDs, deletion flag, or same-as-standard flag.",
    )
    parser.add_argument(
        "--no-phonetic-features",
        action="store_true",
        help="Ablation: do not use expected_features, observed_features, predicted_features, or feature_errors.",
    )
    parser.add_argument(
        "--allow-non-speaker-split",
        action="store_true",
        help="Debug only: allow training from a dataset that was not split by speaker.",
    )
    parser.add_argument(
        "--train-on-all",
        action="store_true",
        help="Final-fit mode: train on every labeled row and skip the internal validation split.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run tools/08_build_dataset.py, then tools/09_make_stability_benchmark.py first."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_json = load_json(dataset_path)
    split_method = dataset_json.get("split", {}).get("method") if isinstance(dataset_json, dict) else None
    if not args.allow_non_speaker_split and not args.train_on_all and split_method != "speaker":
        raise ValueError(
            f"{dataset_path} is not a speaker split dataset (split.method={split_method!r}). "
            "Add speaker_id metadata, run tools/09_make_stability_benchmark.py, then train again. "
            "For smoke tests only, pass --allow-non-speaker-split. "
            "For the final model after validation, pass --train-on-all."
        )

    rows, feature_names, label_names = build_examples(
        dataset_json,
        include_rule=not args.no_rule_features,
        include_wavlm=not args.no_wavlm_features,
        include_phoneme_identity=not args.no_phoneme_identity_features,
        include_phonetic_features=not args.no_phonetic_features,
        target=args.target,
        feature_set=args.feature_set,
    )
    if not rows:
        raise ValueError("No trainable segments found in dataset.")

    x_all = matrix_from_rows(rows, feature_names)
    y_all = labels_to_indices(rows, label_names)
    if args.train_on_all:
        train_indices = list(range(len(rows)))
        val_indices = []
        print("Final-fit mode: training on all labeled rows; validation is skipped.")
    else:
        train_indices, val_indices = split_indices_by_speaker(rows, args.val_ratio, args.seed)
    x_train = x_all[train_indices]
    y_train = y_all[train_indices]
    x_val = x_all[val_indices] if val_indices else np.zeros((0, x_all.shape[1]), dtype=np.float32)
    y_val = y_all[val_indices] if val_indices else np.zeros((0,), dtype=np.int64)

    model = AttributeSoftmaxClassifier(
        input_dim=len(feature_names),
        num_labels=len(label_names),
        seed=args.seed,
    )
    model.fit_normalizer(x_train)
    class_weights = make_class_weights([rows[index] for index in train_indices], label_names)

    best_state = None
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        order = np.arange(len(train_indices))
        np.random.shuffle(order)
        total_loss = 0.0
        total_seen = 0

        for start in range(0, len(order), args.batch_size):
            batch_positions = order[start : start + args.batch_size]
            xb = x_train[batch_positions]
            yb = y_train[batch_positions]

            xb_norm = model.normalize(xb)
            logits = xb_norm @ model.weights + model.bias
            loss, grad_logits = weighted_loss_and_grad(logits, yb, class_weights)

            grad_w = xb_norm.T @ grad_logits + args.weight_decay * model.weights
            grad_b = grad_logits.sum(axis=0)
            model.weights -= args.lr * grad_w.astype(np.float32)
            model.bias -= args.lr * grad_b.astype(np.float32)

            total_loss += loss * len(yb)
            total_seen += len(yb)

        train_loss = total_loss / max(total_seen, 1)
        train_acc = accuracy(model, x_train, y_train)
        val_loss, val_acc = evaluate(model, x_val, y_val, class_weights)
        score = val_loss if val_loss is not None else train_loss
        if score <= best_score:
            best_score = score
            best_state = (model.weights.copy(), model.bias.copy(), model.mean.copy(), model.std.copy())

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_text = "n/a" if val_loss is None else f"{val_loss:.4f} acc={val_acc:.3f}"
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} val={val_text}"
            )

    if best_state is not None:
        model.weights, model.bias, model.mean, model.std = best_state

    metadata = {
        "schema_version": "attribute_classifier_checkpoint_v1",
        "model": "attribute_softmax_classifier",
        "feature_names": feature_names,
        "label_names": label_names,
        "include_rule_features": False if args.feature_set in {"requested", "recommended"} else not args.no_rule_features,
        "include_wavlm_features": not args.no_wavlm_features,
        "include_phoneme_identity_features": False if args.feature_set in {"requested", "recommended"} else not args.no_phoneme_identity_features,
        "include_phonetic_features": False if args.feature_set in {"requested", "recommended"} else not args.no_phonetic_features,
        "feature_set": args.feature_set,
        "target": args.target,
        "source_dataset": str(dataset_path).replace("\\", "/"),
        "validation_split_method": "all_labeled_no_validation" if args.train_on_all else "speaker",
    }
    output_path = Path(args.output)
    model.save(output_path, metadata)

    predictions = predict_payload(dataset_json, model, metadata)
    write_json(Path(args.predictions), predictions)

    print(f"Segments: {len(rows)}")
    print(f"Features: {len(feature_names)}")
    print(f"Labels: {label_names}")
    print(f"Saved model: {output_path}")
    print(f"Wrote predictions: {args.predictions}")


if __name__ == "__main__":
    main()

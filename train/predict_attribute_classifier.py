from __future__ import annotations

import argparse
import sys
from pathlib import Path

from attribute_classifier import AttributeSoftmaxClassifier, load_json, predict_payload, write_json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained attribute classifier and write fusion-ready predictions.")
    parser.add_argument("--dataset", default="data/final/dataset.json")
    parser.add_argument("--checkpoint", default="train/attribute_classifier.npz")
    parser.add_argument("--output", default="data/final/classifier_predictions.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    checkpoint_path = Path(args.checkpoint)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run tools/08_build_dataset.py first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train/train_attribute_classifier.py first.")

    dataset = load_json(dataset_path)
    model, metadata = AttributeSoftmaxClassifier.load(checkpoint_path)
    predictions = predict_payload(dataset, model, metadata)
    write_json(Path(args.output), predictions)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

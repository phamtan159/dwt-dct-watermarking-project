"""
Evaluation script: compute metrics on a test dataset.

Metrics:
    - Per-class Precision, Recall, F1
    - Overall Accuracy
    - Confusion matrix summary
    - Error detection rate (binary: OK vs any error)

Usage:
    cd models
    python evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from transformers import Wav2Vec2Processor

from wav2vec2_crf import Wav2Vec2_BiLSTM_CRF
from audio_dataset import AudioDataset, collate_fn
from utils import build_frame_labels, build_frame_mask, label_vocab, LabelVocab


def evaluate(model, loader, vocab, device):
    """Run evaluation and compute all metrics."""
    model.eval()

    all_preds = []
    all_golds = []

    with torch.no_grad():
        for input_values, mask, phonemes, labels, audio_lens in loader:
            input_values = input_values.to(device)

            outputs = model.wav2vec2(input_values)
            hidden = outputs.last_hidden_state
            num_frames = hidden.shape[1]
            bs = hidden.shape[0]

            lstm_out, _ = model.lstm(hidden)
            lstm_out = model.dropout(lstm_out)
            emissions = model.fc(lstm_out)

            # Build labels
            batch_labels = []
            for b in range(bs):
                fl = build_frame_labels(
                    phonemes[b], labels[b], num_frames,
                    audio_lens[b], vocab
                )
                batch_labels.append(fl)

            batch_labels = nn.utils.rnn.pad_sequence(
                batch_labels, batch_first=True,
                padding_value=vocab.stoi["OK"]
            ).to(device)

            if batch_labels.shape[1] < num_frames:
                pad = torch.full(
                    (bs, num_frames - batch_labels.shape[1]),
                    vocab.stoi["OK"], dtype=torch.long, device=device
                )
                batch_labels = torch.cat([batch_labels, pad], dim=1)
            elif batch_labels.shape[1] > num_frames:
                batch_labels = batch_labels[:, :num_frames]

            frame_mask = build_frame_mask(audio_lens, num_frames, bs, device)

            # Decode predictions
            preds = model.crf.decode(emissions, mask=frame_mask)

            # Collect per-frame results (only for real frames)
            for b in range(bs):
                real_len = frame_mask[b].sum().item()
                all_preds.extend(preds[b][:real_len])
                all_golds.extend(batch_labels[b][:real_len].cpu().tolist())

    return all_preds, all_golds


def compute_metrics(preds, golds, vocab):
    """Compute precision, recall, F1 per class."""
    # Per-class counts
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for p, g in zip(preds, golds):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    # Compute metrics
    results = {}
    for idx in range(len(vocab)):
        label = vocab.itos[idx]
        t = tp.get(idx, 0)
        f_p = fp.get(idx, 0)
        f_n = fn.get(idx, 0)

        precision = t / (t + f_p) if (t + f_p) > 0 else 0.0
        recall = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": t + f_n
        }

    # Overall accuracy
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    accuracy = correct / len(preds) if preds else 0.0

    # Error detection (binary: OK vs error)
    ok_idx = vocab.stoi["OK"]
    binary_correct = sum(
        1 for p, g in zip(preds, golds)
        if (p == ok_idx) == (g == ok_idx)
    )
    error_detection_acc = binary_correct / len(preds) if preds else 0.0

    return results, accuracy, error_detection_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--data", default="../data/final/dataset.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocab = LabelVocab(ckpt["label_vocab"]) if "label_vocab" in ckpt else label_vocab

    model = Wav2Vec2_BiLSTM_CRF(num_labels=len(vocab)).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    # Load data
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    dataset = AudioDataset(args.data, processor, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # Evaluate
    print("🔍 Running evaluation...")
    preds, golds = evaluate(model, loader, vocab, device)
    results, accuracy, err_det_acc = compute_metrics(preds, golds, vocab)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"📊 EVALUATION RESULTS ({len(preds)} frames)")
    print(f"{'=' * 60}")
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"🔍 Error Detection Accuracy: {err_det_acc:.4f} ({err_det_acc*100:.1f}%)")

    print(f"\n{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for label, m in sorted(results.items(), key=lambda x: -x[1]["support"]):
        if m["support"] > 0:
            print(f"{label:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f} {m['support']:>10}")

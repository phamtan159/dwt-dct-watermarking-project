"""
Inference script for pronunciation error detection.

Usage:
    python predict.py --audio path/to/audio.wav
    python predict.py --audio path/to/audio.wav --phonemes phonemes.json
"""

import argparse
import json
import os
import torch
import torchaudio
from collections import Counter
from transformers import Wav2Vec2Processor
from wav2vec2_crf import Wav2Vec2_BiLSTM_CRF
from utils import label_vocab, LabelVocab


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pt")


def load_model(checkpoint_path, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = LabelVocab(ckpt["label_vocab"]) if "label_vocab" in ckpt else label_vocab
    model = Wav2Vec2_BiLSTM_CRF(num_labels=len(vocab)).to(device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model, vocab, device


def predict_frames(audio_path, model, processor, device):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.squeeze(0)
    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None
    with torch.no_grad():
        predictions = model(input_values, attention_mask=attention_mask)
    return predictions[0], len(waveform)


def frames_to_phonemes(frame_labels, phonemes, audio_len_samples):
    num_frames = len(frame_labels)
    spf = (audio_len_samples / 16000.0) / num_frames if num_frames > 0 else 1.0
    results = []
    for p in phonemes:
        sf = max(0, min(int(p["s"] / spf), num_frames))
        ef = max(0, min(int(p["e"] / spf), num_frames))
        pf = frame_labels[sf:ef]
        pred = Counter(pf).most_common(1)[0][0] if pf else "OK"
        results.append({"phone": p["phone"], "start": p["s"], "end": p["e"], "predicted": pred})
    return results


def summarize_errors(pred_labels):
    errors = []
    cur, start = None, 0
    for i, label in enumerate(pred_labels):
        if label != "OK":
            if cur != label:
                if cur and cur != "OK":
                    errors.append({"type": cur, "start_frame": start, "end_frame": i})
                cur, start = label, i
        else:
            if cur and cur != "OK":
                errors.append({"type": cur, "start_frame": start, "end_frame": i})
            cur, start = "OK", i
    if cur and cur != "OK":
        errors.append({"type": cur, "start_frame": start, "end_frame": len(pred_labels)})
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pronunciation errors")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--phonemes", default=None, help="JSON with phoneme alignments")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    print(f"🔍 Analyzing: {args.audio}")
    model, vocab, device = load_model(args.checkpoint, args.device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    pred_indices, audio_len = predict_frames(args.audio, model, processor, device)
    pred_labels = [vocab.itos[idx] for idx in pred_indices]

    errors = summarize_errors(pred_labels)
    if not errors:
        print("\n✅ No pronunciation errors detected!")
    else:
        print(f"\n⚠️ Found {len(errors)} error region(s):")
        for e in errors:
            print(f"  🔴 {e['type']} (frames {e['start_frame']}–{e['end_frame']})")

    if args.phonemes:
        with open(args.phonemes, "r", encoding="utf-8") as f:
            phonemes = json.load(f)
        results = frames_to_phonemes(pred_labels, phonemes, audio_len)
        print(f"\n📋 Phoneme-level results:")
        for r in results:
            s = "✅" if r["predicted"] == "OK" else "🔴"
            print(f"  {s} [{r['phone']}] {r['start']:.2f}s–{r['end']:.2f}s → {r['predicted']}")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {args.output}")

    counts = Counter(pred_labels)
    print(f"\n📊 Label distribution ({len(pred_labels)} frames):")
    for label, count in counts.most_common():
        print(f"  {label}: {count} ({100*count/len(pred_labels):.1f}%)")

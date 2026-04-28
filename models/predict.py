"""
Inference script: load trained model and predict pronunciation errors from audio.

Usage:
    cd models
    python predict.py --audio path/to/audio.wav --checkpoint checkpoints/model.pt
"""

import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Processor

from wav2vec2_crf import Wav2Vec2_BiLSTM_CRF
from utils import label_vocab


def predict(audio_path, checkpoint_path, device="cuda"):
    """
    Run inference on a single audio file.

    Returns:
        list of predicted frame-level labels
    """
    device = device if torch.cuda.is_available() else "cpu"

    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )

    # Load model
    model = Wav2Vec2_BiLSTM_CRF(num_labels=len(label_vocab)).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load and process audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.squeeze(0)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    input_values = inputs.input_values.to(device)

    # Predict
    with torch.no_grad():
        predictions = model(input_values)

    # Decode labels
    pred_labels = [label_vocab.itos[idx] for idx in predictions[0]]

    return pred_labels


def summarize_errors(pred_labels):
    """Summarize which error types were detected and their positions."""
    errors = []
    current_error = None
    start_frame = 0

    for i, label in enumerate(pred_labels):
        if label != "OK":
            if current_error is None or current_error != label:
                if current_error is not None and current_error != "OK":
                    errors.append({
                        "type": current_error,
                        "start_frame": start_frame,
                        "end_frame": i
                    })
                current_error = label
                start_frame = i
        else:
            if current_error is not None and current_error != "OK":
                errors.append({
                    "type": current_error,
                    "start_frame": start_frame,
                    "end_frame": i
                })
            current_error = "OK"
            start_frame = i

    # Handle last segment
    if current_error is not None and current_error != "OK":
        errors.append({
            "type": current_error,
            "start_frame": start_frame,
            "end_frame": len(pred_labels)
        })

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict pronunciation errors from audio"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (cuda or cpu)"
    )
    args = parser.parse_args()

    print(f"🔍 Analyzing: {args.audio}")

    pred_labels = predict(args.audio, args.checkpoint, args.device)

    errors = summarize_errors(pred_labels)

    if not errors:
        print("✅ No pronunciation errors detected!")
    else:
        print(f"\n⚠️ Found {len(errors)} error(s):")
        for err in errors:
            print(
                f"  🔴 {err['type']} "
                f"(frames {err['start_frame']}–{err['end_frame']})"
            )

    # Also show raw distribution
    from collections import Counter
    counts = Counter(pred_labels)
    print(f"\n📊 Label distribution ({len(pred_labels)} frames):")
    for label, count in counts.most_common():
        pct = 100 * count / len(pred_labels)
        print(f"  {label}: {count} ({pct:.1f}%)")

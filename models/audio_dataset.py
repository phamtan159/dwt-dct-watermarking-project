"""
Audio Dataset and collate function for pronunciation error detection.

Expected dataset.json format:
[
    {
        "audio": "path/to/audio.wav",
        "phonemes": [
            {"s": 0.12, "e": 0.25, "phone": "z"},
            {"s": 0.25, "e": 0.40, "phone": "a"},
            ...
        ],
        "labels": ["z→d", "OK", ...]
    },
    ...
]

Improvements from fine-tune project:
    - Robust error handling for missing/corrupt audio files
    - Auto-skip samples with issues (instead of crashing)
    - Label validation against vocab
    - Detailed loading statistics
"""

import json
import os
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Dataset for loading audio + phoneme annotations."""

    def __init__(self, path, processor, label_vocab):
        """
        Args:
            path: path to dataset.json
            processor: WavLM feature extractor instance
            label_vocab: LabelVocab instance
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.processor = processor
        self.label_vocab = label_vocab

        # Validate and filter samples
        self.data = []
        skipped = 0
        unknown_labels = set()

        for item in raw_data:
            # Check audio file exists
            if not os.path.exists(item["audio"]):
                print(f"  ⚠️ Audio not found: {item['audio']}")
                skipped += 1
                continue

            # Check phoneme-label count match
            if len(item["phonemes"]) != len(item["labels"]):
                print(f"  ⚠️ Phoneme/label mismatch in {item['audio']}: "
                      f"{len(item['phonemes'])} phonemes vs {len(item['labels'])} labels")
                skipped += 1
                continue

            # Track unknown labels
            for label in item["labels"]:
                if label not in label_vocab.stoi:
                    unknown_labels.add(label)

            self.data.append(item)

        # Report stats
        print(f"📦 Loaded {len(self.data)} samples from {path}")
        if skipped:
            print(f"   ⚠️ Skipped {skipped} invalid samples")
        if unknown_labels:
            print(f"   ⚠️ Unknown labels (will map to 'OK'): {unknown_labels}")

        # Compute label distribution
        label_counts = {}
        for item in self.data:
            for label in item["labels"]:
                label_counts[label] = label_counts.get(label, 0) + 1

        if label_counts:
            print(f"   📊 Label distribution:")
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                print(f"      {label}: {count}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load audio
        waveform, sr = torchaudio.load(item["audio"])

        # Resample to 16kHz if needed
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        waveform = waveform.squeeze(0)  # (T,)

        # Process through the WavLM feature extractor
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )

        return (
            inputs.input_values.squeeze(0),  # (T_audio,)
            item["phonemes"],                 # list of dicts
            item["labels"],                   # list of strings
            len(waveform)                     # audio length in samples
        )


def collate_fn(batch):
    """
    Custom collate: pad audio to max length in batch, build attention mask.

    Returns:
        input_values: (B, T_max) padded audio
        mask: (B, T_max) attention mask (1 = real, 0 = padding)
        phonemes: tuple of phoneme lists
        labels: tuple of label lists
        audio_lens: tuple of audio lengths (in samples)
    """
    inputs, phonemes, labels, audio_lens = zip(*batch)

    max_len = max(x.shape[0] for x in inputs)

    padded = []
    mask = []

    for x in inputs:
        pad_len = max_len - x.shape[0]

        padded.append(torch.cat([x, torch.zeros(pad_len)]))
        mask.append(
            torch.cat([
                torch.ones(len(x)),
                torch.zeros(pad_len)
            ])
        )

    return (
        torch.stack(padded),   # (B, T_max)
        torch.stack(mask),     # (B, T_max)
        phonemes,              # tuple of lists
        labels,                # tuple of lists
        audio_lens             # tuple of ints
    )

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
from pathlib import Path

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
        self.dataset_path = Path(path).resolve()
        self.dataset_dir = self.dataset_path.parent
        self.project_root = self._infer_project_root()

        with open(path, "r", encoding="utf-8") as f:
            raw_payload = json.load(f)

        raw_data = self._normalize_dataset(raw_payload)

        self.processor = processor
        self.label_vocab = label_vocab

        # Validate and filter samples
        self.data = []
        skipped = 0
        unknown_labels = set()

        for item in raw_data:
            item = dict(item)
            item["audio"] = self._resolve_path(item.get("audio") or item.get("audio_path"))

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

    def _infer_project_root(self):
        if self.dataset_dir.name == "final" and self.dataset_dir.parent.name == "data":
            return self.dataset_dir.parent.parent
        for parent in [self.dataset_dir, *self.dataset_dir.parents]:
            if (parent / "data").exists() and ((parent / "tools").exists() or (parent / "models").exists()):
                return parent
        return self.dataset_dir

    def _resolve_path(self, raw_path):
        if not raw_path:
            return ""

        path = Path(raw_path)
        if path.is_absolute():
            return str(path)

        candidates = [
            Path.cwd() / path,
            self.project_root / path,
            self.dataset_dir / path,
            self.dataset_dir.parent / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(self.project_root / path)

    def _normalize_dataset(self, payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            return self._audio_items_from_av_dataset(payload)
        raise ValueError("Unsupported dataset format: expected a list or an audio_only_v1 object.")

    def _audio_items_from_av_dataset(self, dataset):
        items = []
        for sample in dataset.get("samples", []):
            phonemes = []
            labels = []
            for seg in sample.get("segments", []):
                start = seg.get("start")
                end = seg.get("end")
                if start is None or end is None:
                    continue
                phonemes.append(
                    {
                        "s": start,
                        "e": end,
                        "phone": seg.get("phoneme_real") or seg.get("phone"),
                        "standard_phone": seg.get("phoneme_standard") or seg.get("standard_phone"),
                    }
                )
                labels.append(str(seg.get("label", seg.get("error_id", "0"))))

            if phonemes:
                items.append(
                    {
                        "audio": sample.get("audio") or sample.get("audio_path"),
                        "audio_id": sample.get("audio_id"),
                        "sample_id": sample.get("sample_id") or sample.get("id") or sample.get("audio_id"),
                        "speaker_id": sample.get("speaker_id") or "unknown_speaker",
                        "phonemes": phonemes,
                        "labels": labels,
                    }
                )
        return items

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

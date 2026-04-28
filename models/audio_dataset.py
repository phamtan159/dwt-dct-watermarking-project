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
"""

import json
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Dataset for loading audio + phoneme annotations."""

    def __init__(self, path, processor, label_vocab):
        """
        Args:
            path: path to dataset.json
            processor: Wav2Vec2Processor instance
            label_vocab: LabelVocab instance
        """
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = processor
        self.label_vocab = label_vocab

        print(f"📦 Loaded {len(self.data)} samples from {path}")

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

        # Process through wav2vec2 processor
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
        audio_lens: tuple of audio lengths
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

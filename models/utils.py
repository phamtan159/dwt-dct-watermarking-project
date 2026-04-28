"""
Utility functions for pronunciation error detection.
- LabelVocab: maps error labels to indices
- build_frame_labels: aligns phoneme-level labels to wav2vec2 frame indices
"""

import torch


class LabelVocab:
    """Simple vocabulary for error labels."""

    def __init__(self, labels):
        """
        Args:
            labels: list of label strings, e.g. ["OK", "z→d", "s→x", ...]
        """
        self.labels = labels
        self.stoi = {label: i for i, label in enumerate(labels)}
        self.itos = {i: label for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"LabelVocab({self.labels})"


# ================================================
# Default label set for Vietnamese pronunciation errors
# Customize this list based on your dataset
# ================================================
DEFAULT_LABELS = [
    "OK",       # correct pronunciation
    "z→d",      # z misread as d
    "s→x",      # s misread as x
    "r→g",      # r misread as g
    "l→n",      # l misread as n
    "tr→ch",    # tr misread as ch
    "n→l",      # n misread as l
    "OTHER",    # other error types
]

label_vocab = LabelVocab(DEFAULT_LABELS)


def build_frame_labels(phonemes, labels, num_frames, audio_len, vocab):
    """
    Convert phoneme-level labels to frame-level labels for CRF.

    Each wav2vec2 output frame corresponds to a time window.
    This function maps phoneme boundaries → frame indices → label IDs.

    Args:
        phonemes: list of dicts with {"s": start_sec, "e": end_sec, "phone": "..."}
        labels:   list of label strings matching each phoneme (e.g. "OK", "z→d")
        num_frames: number of output frames from wav2vec2 (T dimension)
        audio_len: total number of audio samples (at 16kHz)
        vocab: LabelVocab instance

    Returns:
        torch.LongTensor of shape (num_frames,)
    """
    # Time duration per frame
    frame_time = audio_len / num_frames

    # Default: all frames are "OK"
    frame_labels = [vocab.stoi["OK"]] * num_frames

    for p, label in zip(phonemes, labels):
        start_frame = int(p["s"] / frame_time)
        end_frame = int(p["e"] / frame_time)

        label_id = vocab.stoi.get(label, vocab.stoi["OK"])

        for i in range(start_frame, min(end_frame, num_frames)):
            frame_labels[i] = label_id

    return torch.tensor(frame_labels, dtype=torch.long)

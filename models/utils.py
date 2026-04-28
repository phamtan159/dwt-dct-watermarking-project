"""
Utility functions for pronunciation error detection.
- LabelVocab: maps error labels to indices
- build_frame_labels: aligns phoneme-level labels to wav2vec2 frame indices
- build_frame_mask: marks valid wav2vec2 frames for CRF training
"""

import math
import torch


WAV2VEC2_CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
WAV2VEC2_CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)


class LabelVocab:
    """Simple vocabulary for error labels."""

    def __init__(self, labels):
        """
        Args:
            labels: list of label strings, e.g. ["OK", "z→d", "s→x", ...]
        """
        self.labels = list(labels)
        self.stoi = {label: i for i, label in enumerate(labels)}
        self.itos = {i: label for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.labels)

    def add_label(self, label):
        """Add a new label at runtime if it does not already exist."""
        if label in self.stoi:
            return self.stoi[label]

        idx = len(self.labels)
        self.labels.append(label)
        self.stoi[label] = idx
        self.itos[idx] = label
        return idx

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


def estimate_wav2vec2_output_length(audio_len_samples):
    """Estimate the number of feature frames after wav2vec2's conv frontend."""
    length = max(int(audio_len_samples), 0)

    for kernel, stride in zip(WAV2VEC2_CONV_KERNELS, WAV2VEC2_CONV_STRIDES):
        if length <= 0:
            return 0
        length = (length - kernel) // stride + 1

    return max(length, 0)


def build_frame_labels(phonemes, labels, num_frames, audio_len_samples, vocab, sample_rate=16000):
    """
    Convert phoneme-level labels to frame-level labels for CRF.

    Each wav2vec2 output frame corresponds to a time window.
    This function maps phoneme boundaries → frame indices → label IDs.

    Args:
        phonemes: list of dicts with {"s": start_sec, "e": end_sec, "phone": "..."}
        labels:   list of label strings matching each phoneme (e.g. "OK", "z→d")
        num_frames: number of output frames from wav2vec2 (T dimension)
        audio_len_samples: total number of audio samples
        vocab: LabelVocab instance

    Returns:
        torch.LongTensor of shape (num_frames,)
    """
    ok_id = vocab.stoi["OK"]
    frame_labels = [ok_id] * num_frames

    if num_frames <= 0 or audio_len_samples <= 0:
        return torch.tensor(frame_labels, dtype=torch.long)

    audio_duration_sec = audio_len_samples / float(sample_rate)
    sec_per_frame = audio_duration_sec / num_frames if audio_duration_sec > 0 else 0.0

    if sec_per_frame <= 0:
        return torch.tensor(frame_labels, dtype=torch.long)

    for p, label in zip(phonemes, labels):
        start_sec = max(0.0, float(p.get("s", 0.0)))
        end_sec = max(start_sec, float(p.get("e", start_sec)))

        start_frame = min(num_frames, max(0, int(start_sec / sec_per_frame)))
        end_frame = min(num_frames, max(start_frame + 1, math.ceil(end_sec / sec_per_frame)))

        label_id = vocab.stoi.get(label, ok_id)

        for i in range(start_frame, end_frame):
            frame_labels[i] = label_id

    return torch.tensor(frame_labels, dtype=torch.long)


def build_frame_mask(audio_lens_samples, num_frames, batch_size, device):
    """
    Build a mask that marks only real wav2vec2 frames as valid for the CRF.

    Args:
        audio_lens_samples: iterable of original waveform lengths in samples
        num_frames: padded frame length in the current batch
        batch_size: batch size
        device: torch device

    Returns:
        torch.BoolTensor of shape (batch_size, num_frames)
    """
    mask = torch.zeros((batch_size, num_frames), dtype=torch.bool, device=device)

    for b, audio_len in enumerate(audio_lens_samples):
        if b >= batch_size:
            break

        real_frames = estimate_wav2vec2_output_length(audio_len)
        real_frames = min(real_frames, num_frames)

        if audio_len > 0 and num_frames > 0 and real_frames == 0:
            real_frames = 1

        mask[b, :real_frames] = True

    return mask

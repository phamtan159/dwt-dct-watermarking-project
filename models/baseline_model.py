"""
Baseline model: Simple BiLSTM classifier WITHOUT pretrained wav2vec2.

This serves as a comparison point to prove that fine-tuning wav2vec2
actually improves pronunciation error detection. Learned from fine-tune
project which had separate baseline + advanced models.

Architecture:
    audio features (MFCC) → BiLSTM → FC → labels

Usage:
    cd models
    python train_baseline.py
"""

import torch
import torch.nn as nn
import torchaudio


class BaselineModel(nn.Module):
    """Simple MFCC + BiLSTM baseline (no pretrained features)."""

    def __init__(self, num_labels, n_mfcc=40, hidden_dim=128):
        super().__init__()
        self.num_labels = num_labels

        # MFCC feature extractor (no pretrained weights needed)
        self.n_mfcc = n_mfcc

        self.lstm = nn.LSTM(
            n_mfcc,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def extract_mfcc(self, waveform, sr=16000):
        """Extract MFCC features from raw audio."""
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 320, "n_mels": 80}
        ).to(waveform.device)

        # waveform: (B, T_audio)
        mfcc = mfcc_transform(waveform)  # (B, n_mfcc, T_frames)
        return mfcc.permute(0, 2, 1)     # (B, T_frames, n_mfcc)

    def forward(self, input_values, labels=None):
        """
        Args:
            input_values: (B, T_audio) raw waveform
            labels: (B, T_frames) frame labels, or None

        Returns:
            If labels: CrossEntropy loss
            If no labels: (B, T_frames, num_labels) logits
        """
        features = self.extract_mfcc(input_values)  # (B, T, n_mfcc)

        x, _ = self.lstm(features)
        x = self.dropout(x)
        logits = self.fc(x)  # (B, T, num_labels)

        if labels is not None:
            # Adjust label length to match feature length
            T = logits.shape[1]
            if labels.shape[1] > T:
                labels = labels[:, :T]
            elif labels.shape[1] < T:
                pad = torch.full(
                    (labels.shape[0], T - labels.shape[1]),
                    0, dtype=torch.long, device=labels.device
                )
                labels = torch.cat([labels, pad], dim=1)

            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            return loss
        else:
            return logits.argmax(dim=-1)  # (B, T)

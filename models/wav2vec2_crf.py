"""
Wav2Vec2 + BiLSTM + CRF model for pronunciation error detection.

Architecture:
    audio → wav2vec2 (feature extractor) → BiLSTM (context) → CRF (sequence constraint) → labels

Includes:
    - forward(): full pipeline (wav2vec2 → LSTM → CRF)
    - forward_from_features(): LSTM → CRF only (avoids redundant wav2vec2 forward)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from torchcrf import CRF


class Wav2Vec2_BiLSTM_CRF(nn.Module):
    def __init__(self, num_labels, hidden_dim=128, dropout=0.1):
        """
        Args:
            num_labels: number of error label classes
            hidden_dim: LSTM hidden dimension (output will be hidden_dim * 2 due to bidirectional)
            dropout: dropout rate between LSTM and FC
        """
        super().__init__()

        # ==============================
        # 🔊 Wav2Vec2 Feature Extractor
        # ==============================
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        input_dim = self.wav2vec2.config.hidden_size  # 768

        # ==============================
        # 🧠 BiLSTM Context Encoder
        # ==============================
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # ==============================
        # 🎯 Emission Layer
        # ==============================
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

        # ==============================
        # 🔗 CRF Sequence Constraint
        # ==============================
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_values, attention_mask=None, labels=None):
        """
        Full forward pass: wav2vec2 → LSTM → CRF.

        Args:
            input_values: (B, T_audio) raw waveform
            attention_mask: (B, T_audio) mask for padded positions
            labels: (B, T_frames) frame-level label IDs, or None for inference

        Returns:
            If labels is not None: CRF negative log-likelihood loss (scalar)
            If labels is None: list of predicted label sequences
        """
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )

        features = outputs.last_hidden_state  # (B, T_frames, H)

        return self.forward_from_features(features, labels=labels)

    def forward_from_features(self, features, frame_mask=None, labels=None):
        """
        Forward pass from pre-extracted wav2vec2 features.
        Avoids redundant wav2vec2 computation during training.

        Args:
            features: (B, T_frames, H) wav2vec2 hidden states
            frame_mask: (B, T_frames) bool mask for CRF
            labels: (B, T_frames) frame-level label IDs, or None for inference

        Returns:
            If labels is not None: CRF negative log-likelihood loss (scalar)
            If labels is None: list of predicted label sequences
        """
        x, _ = self.lstm(features)
        x = self.dropout(x)
        emissions = self.fc(x)  # (B, T_frames, num_labels)

        if frame_mask is None:
            frame_mask = torch.ones(
                emissions.shape[:2],
                dtype=torch.bool,
                device=emissions.device
            )

        if labels is not None:
            # Training: return CRF loss
            loss = -self.crf(emissions, labels, mask=frame_mask)
            return loss
        else:
            # Inference: return decoded sequences
            pred = self.crf.decode(emissions, mask=frame_mask)
            return pred

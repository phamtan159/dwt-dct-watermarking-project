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
        features = self.extract_features(
            input_values,
            attention_mask=attention_mask
        )

        frame_mask = None
        if attention_mask is not None:
            frame_mask = self.wav2vec2._get_feature_vector_attention_mask(
                features.shape[1],
                attention_mask.long()
            )

        return self.forward_from_features(features, frame_mask=frame_mask, labels=labels)

    def extract_features(self, input_values, attention_mask=None):
        """Run the wav2vec2 encoder once and return frame-level features."""
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state  # (B, T_frames, H)

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

    def freeze_wav2vec2(self, freeze_layers_below=8):
        """
        Freeze the wav2vec2 frontend and lower encoder layers.

        Layers with index >= freeze_layers_below remain trainable.
        """
        frozen = 0
        trainable = 0

        for name, param in self.wav2vec2.named_parameters():
            should_train = False

            if name.startswith("encoder.layers."):
                layer_idx = int(name.split("encoder.layers.")[1].split(".")[0])
                should_train = layer_idx >= freeze_layers_below

            param.requires_grad = should_train

            if should_train:
                trainable += param.numel()
            else:
                frozen += param.numel()

        return {"frozen": frozen, "trainable": trainable}

    def unfreeze_all(self):
        """Unfreeze every parameter in the model."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_stats(self):
        """Report how many parameters are trainable vs frozen."""
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "pct_trainable": (100.0 * trainable / total) if total else 0.0,
        }

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
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torchcrf import CRF


DEFAULT_PRETRAINED_MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "pretrained"
    / "facebook-wav2vec2-lv-60-espeak-cv-ft"
)


def get_pretrained_model_path(model_path=None):
    """Return the local wav2vec2 checkpoint used for fine-tuning."""
    path = Path(model_path) if model_path else DEFAULT_PRETRAINED_MODEL_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pretrained wav2vec2 model at {path}. "
            "Download or copy facebook/wav2vec2-lv-60-espeak-cv-ft into pretrained/ first."
        )
    return str(path)


def load_wav2vec2_processor(model_path=None):
    """Load the local processor without requiring the eSpeak runtime."""
    return Wav2Vec2Processor.from_pretrained(
        get_pretrained_model_path(model_path),
        local_files_only=True,
        do_phonemize=False,
    )


class Wav2Vec2_BiLSTM_CRF(nn.Module):
    def __init__(self, num_labels, hidden_dim=128, dropout=0.1, pretrained_model_path=None):
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
            get_pretrained_model_path(pretrained_model_path),
            local_files_only=True,
        )

        input_dim = self.wav2vec2.config.hidden_size

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

    def freeze_wav2vec2_all(self):
        """Freeze the whole wav2vec2 encoder and train only the task head."""
        frozen_count = 0

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            frozen_count += 1

        return {"frozen": frozen_count, "trainable": 0}

    def freeze_wav2vec2(self, freeze_layers_below=8):
        """
        Freeze early layers of wav2vec2.
        
        Args:
            freeze_layers_below: Layers with index < this will be frozen.
                                 Base has 12 layers (0-11).
        """
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.wav2vec2.named_parameters():
            if "encoder.layers" in name:
                layer_id = int(name.split("encoder.layers.")[1].split(".")[0])
                if layer_id < freeze_layers_below:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
                    trainable_count += 1
            else:
                # Freeze feature extractor (CNNs) by default
                param.requires_grad = False
                frozen_count += 1
                
        return {"frozen": frozen_count, "trainable": trainable_count}

    def unfreeze_top_wav2vec2_layers(self, num_trainable_layers=2):
        """
        Freeze wav2vec2 except for the last encoder layers.

        Args:
            num_trainable_layers: Number of top transformer layers to fine-tune.
                                  Base wav2vec2 has 12 layers (0-11).
        """
        total_layers = self.wav2vec2.config.num_hidden_layers
        first_trainable_layer = max(0, total_layers - num_trainable_layers)
        frozen_count = 0
        trainable_count = 0

        for name, param in self.wav2vec2.named_parameters():
            if "encoder.layers" in name:
                layer_id = int(name.split("encoder.layers.")[1].split(".")[0])
                param.requires_grad = layer_id >= first_trainable_layer
            else:
                # Keep the convolutional feature extractor and projections stable.
                param.requires_grad = False

            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1

        return {"frozen": frozen_count, "trainable": trainable_count}

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_stats(self):
        """Get statistics about trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "pct_trainable": 100 * trainable / total
        }

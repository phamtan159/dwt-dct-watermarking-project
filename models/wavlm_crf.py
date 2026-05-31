"""
WavLM + BiLSTM + CRF model for pronunciation error detection.

Architecture:
    audio -> WavLM feature extractor -> BiLSTM context encoder -> CRF labels

The module keeps the old public names as aliases so existing checkpoints and
scripts can be migrated without a broad file rename.
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel

try:
    from torchcrf import CRF
except ImportError:  # Some Windows installs expose pytorch-crf as `crf`.
    from crf import CRF as _SeqFirstCRF

    class CRF(_SeqFirstCRF):
        def __init__(self, num_tags: int, batch_first: bool = False):
            super().__init__(num_tags)
            self.batch_first = batch_first

        def forward(self, emissions, tags, mask=None):
            if self.batch_first:
                emissions = emissions.transpose(0, 1)
                tags = tags.transpose(0, 1)
                if mask is not None:
                    mask = mask.transpose(0, 1)
            return super().forward(emissions, tags, mask=mask)

        def decode(self, emissions, mask=None):
            if self.batch_first:
                emissions = emissions.transpose(0, 1)
                if mask is not None:
                    mask = mask.transpose(0, 1)
            return super().decode(emissions, mask=mask)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "microsoft-wavlm-large"
DEFAULT_PRETRAINED_MODEL_DIRS = [
    PROJECT_ROOT / "pretrained" / MODEL_NAME,
]


def get_pretrained_model_path(model_path=None):
    """Return the local WavLM checkpoint used for fine-tuning."""
    candidates = [Path(model_path)] if model_path else DEFAULT_PRETRAINED_MODEL_DIRS
    for path in candidates:
        if path.exists():
            return str(path)
    if candidates:
        path = candidates[0]
        raise FileNotFoundError(
            f"Missing pretrained WavLM model at {path}. "
            f"Copy the full model into pretrained/{MODEL_NAME} first."
        )
    raise FileNotFoundError("No WavLM pretrained model path was configured.")


def load_wavlm_processor(model_path=None):
    """Load the local WavLM feature extractor."""
    return AutoFeatureExtractor.from_pretrained(
        get_pretrained_model_path(model_path),
        local_files_only=True,
    )


class WavLM_BiLSTM_CRF(nn.Module):
    def __init__(self, num_labels, hidden_dim=128, dropout=0.1, pretrained_model_path=None):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained(
            get_pretrained_model_path(pretrained_model_path),
            local_files_only=True,
        )

        input_dim = self.wavlm.config.hidden_size

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def extract_features(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(self, input_values, attention_mask=None, labels=None):
        features = self.extract_features(input_values, attention_mask=attention_mask)
        return self.forward_from_features(features, labels=labels)

    def forward_from_features(self, features, frame_mask=None, labels=None):
        x, _ = self.lstm(features)
        x = self.dropout(x)
        emissions = self.fc(x)

        if frame_mask is None:
            frame_mask = torch.ones(
                emissions.shape[:2],
                dtype=torch.bool,
                device=emissions.device,
            )

        if labels is not None:
            return -self.crf(emissions, labels, mask=frame_mask)

        return self.crf.decode(emissions, mask=frame_mask)

    def freeze_wavlm_all(self):
        """Freeze the whole WavLM encoder and train only the task head."""
        frozen_count = 0
        for param in self.wavlm.parameters():
            param.requires_grad = False
            frozen_count += 1
        return {"frozen": frozen_count, "trainable": 0}

    def freeze_wavlm(self, freeze_layers_below=20):
        """
        Freeze early WavLM layers.

        WavLM large has 24 encoder layers. By default this leaves only the last
        four trainable if this method is used directly.
        """
        frozen_count = 0
        trainable_count = 0

        for name, param in self.wavlm.named_parameters():
            if "encoder.layers" in name:
                layer_id = int(name.split("encoder.layers.")[1].split(".")[0])
                param.requires_grad = layer_id >= freeze_layers_below
            else:
                param.requires_grad = False

            if param.requires_grad:
                trainable_count += 1
            else:
                frozen_count += 1

        return {"frozen": frozen_count, "trainable": trainable_count}

    def unfreeze_top_wavlm_layers(self, num_trainable_layers=2):
        """Freeze WavLM except for the last encoder layers."""
        total_layers = self.wavlm.config.num_hidden_layers
        first_trainable_layer = max(0, total_layers - num_trainable_layers)
        frozen_count = 0
        trainable_count = 0

        for name, param in self.wavlm.named_parameters():
            if "encoder.layers" in name:
                layer_id = int(name.split("encoder.layers.")[1].split(".")[0])
                param.requires_grad = layer_id >= first_trainable_layer
            else:
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
            "pct_trainable": 100 * trainable / total,
        }

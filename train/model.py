import os
import sys

import torch
import torch.nn as nn


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "auto_avsr"))

try:
    from datamodule.transforms import TextTransform
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
except ImportError as exc:
    raise ImportError(
        "Could not import auto_avsr modules. Run from the project root and make "
        "sure the auto_avsr folder and its dependencies are installed. "
        f"Original error: {exc}"
    ) from exc


def get_visual_frontend(checkpoint_path, device="cuda"):
    """Load auto_avsr and return only its pretrained visual frontend."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    text_transform = TextTransform()
    model = E2E(len(text_transform.token_list), "video", ctc_weight=0.1)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)

    frontend = model.frontend
    frontend.eval()
    return frontend.to(device)


class LipModelAdvanced(nn.Module):
    def __init__(self, num_classes, encoder_checkpoint):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = get_visual_frontend(encoder_checkpoint, self.device)
        self.encoder_trainable = False

        # auto_avsr base visual frontend returns 512-dim frame features.
        self.feature_dim = 512

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(256, num_classes)
        self.freeze_encoder()

    def freeze_encoder(self):
        self.encoder_trainable = False
        self.visual_encoder.eval()
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder_tail(self, trainable_tensors=8):
        """Unfreeze only the last few visual-encoder tensors for cautious tuning."""
        self.freeze_encoder()
        params = list(self.visual_encoder.named_parameters())
        selected = params[-trainable_tensors:]
        for _, param in selected:
            param.requires_grad = True

        self.encoder_trainable = bool(selected)
        self.visual_encoder.train(self.encoder_trainable)
        return [name for name, _ in selected]

    def forward(self, x):
        # x: (B, T, C, H, W); auto_avsr frontend expects this shape.
        if self.encoder_trainable:
            feat_seq = self.visual_encoder(x)
        else:
            self.visual_encoder.eval()
            with torch.no_grad():
                feat_seq = self.visual_encoder(x)

        lstm_out, _ = self.lstm(feat_seq)
        out = lstm_out[:, -1, :]
        return self.fc(out)

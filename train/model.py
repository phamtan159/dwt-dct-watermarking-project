import torch
import torch.nn as nn
import sys
import os

# Thêm path tới auto_avsr để import espnet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "auto_avsr"))

try:
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
    from datamodule.transforms import TextTransform
except ImportError:
    print("Error: Could not import espnet or datamodule. Make sure you are running from the project root and auto_avsr is present.")

def get_visual_frontend(checkpoint_path, device="cuda"):
    """
    Khởi tạo model E2E và lấy phần frontend (Visual Encoder) từ checkpoint của auto_avsr.
    """
    text_transform = TextTransform()
    token_list = text_transform.token_list
    
    # Khởi tạo model với modality là video
    model = E2E(len(token_list), "video", ctc_weight=0.1)
    
    # Load weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại {checkpoint_path}. Hãy tải về trước.")
        
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt)
    
    # Chỉ lấy phần frontend (ResNet + 3D Conv)
    frontend = model.frontend
    frontend.eval()
    return frontend.to(device)

class LipModelAdvanced(nn.Module):
    def __init__(self, num_classes, encoder_checkpoint):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = get_visual_frontend(encoder_checkpoint, self.device)
        
        # Frontend của auto_avsr base trả về feature 512-dim
        self.feature_dim = 512 

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x input: (B, T, C, H, W)
        # auto_avsr frontend nhận (B, T, C, H, W)
        with torch.no_grad(): # Thường freeze encoder ở phase đầu
            # Nếu encoder đã unfreeze thì bỏ torch.no_grad()
            # Ở đây ta mặc định xử lý feature extraction
            feat_seq = self.visual_encoder(x) # Output: (B, T, D)
        
        lstm_out, _ = self.lstm(feat_seq)
        # Lấy output của frame cuối cùng
        out = lstm_out[:, -1, :] 
        return self.fc(out)

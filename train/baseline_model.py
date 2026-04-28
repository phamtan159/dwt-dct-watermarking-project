import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Mạng CNN 3D đơn giản để trích xuất đặc trưng không gian (H, W) và thời gian (T)
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Ảnh gốc 88x88 qua 3 lớp MaxPooling 2x2: 88 -> 44 -> 22 -> 11
        # Số features sau CNN: 128 channels * 11 height * 11 width = 15488
        self.feature_dim = 128 * 11 * 11
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x input từ dataset: (B, T, C, H, W)
        # Conv3D của PyTorch yêu cầu input: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Đi qua khối Conv3D
        feat = self.conv3d(x) 
        # Output feat: (B, 128, T, 11, 11)
        
        # Chuẩn bị input cho LSTM: yêu cầu (B, T, Features)
        B, C, T, H, W = feat.shape
        feat = feat.permute(0, 2, 1, 3, 4).contiguous() # (B, T, 128, 11, 11)
        feat = feat.view(B, T, -1) # Ép phẳng thành (B, T, 15488)
        
        # Đi qua khối LSTM
        lstm_out, _ = self.lstm(feat)
        
        # Chỉ lấy output của timestep (frame) cuối cùng để phân loại
        out = lstm_out[:, -1, :] 
        return self.fc(out)

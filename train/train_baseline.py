import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from baseline_model import BaselineModel
from dataset import LipDataset   

dataset = LipDataset("data")   

if len(dataset) > 0:
    NUM_CLASSES = len(dataset.label_map)
else:
    NUM_CLASSES = 2

BATCH_SIZE = 4    
EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(dataset) == 0:
    print("⚠️ Dataset trống! Hãy đảm bảo bạn đã chạy các bước tools 01-06 và gán nhãn manual.")
else:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Khởi tạo Baseline Model
    model = BaselineModel(NUM_CLASSES).to(device)

    print("\n>>> Bắt đầu Training Baseline Model (CNN 3D + LSTM tự code)")
    print(f"Tổng số lớp phân loại (classes): {NUM_CLASSES}")
    
    # Train tất cả các tham số từ đầu (Train from scratch)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    # Lưu trọng số của Baseline model
    torch.save(model.state_dict(), "train/baseline_model_weights.pth")
    print("\n✅ Đã lưu model baseline tại: train/baseline_model_weights.pth")

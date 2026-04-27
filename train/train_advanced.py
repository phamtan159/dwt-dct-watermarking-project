import torch
from torch.utils.data import DataLoader
import sys
import os

# Đảm bảo import được dataset và model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import LipModelAdvanced
from dataset import LipDataset   

import json

# Cấu hình
ENCODER_CKPT = "pretrained/vsr_trlrs3_base.pth" 

dataset = LipDataset("data")   

if len(dataset) > 0:
    NUM_CLASSES = len(dataset.label_map)
else:
    NUM_CLASSES = 2

BATCH_SIZE = 4    
EPOCHS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(ENCODER_CKPT):
    print(f"⚠️ CẢNH BÁO: Không tìm thấy file checkpoint tại {ENCODER_CKPT}")
    print("Hãy chạy lệnh tải model: curl -L http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs3_base.pth -o pretrained/vsr_trlrs3_base.pth")

if len(dataset) == 0:
    print("⚠️ Dataset trống! Hãy đảm bảo bạn đã chạy các bước tools 01-06 và gán nhãn manual.")
else:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LipModelAdvanced(NUM_CLASSES, ENCODER_CKPT).to(device)

    # 1. Phase 1: Chỉ train Classifier (LSTM + FC), đóng băng Visual Encoder
    print("\n>>> Phase 1: Training classifier head (Visual Encoder is frozen)")
    for name, param in model.named_parameters():
        if "visual_encoder" in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
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
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(loader):.4f}")

    # 2. Phase 2: Unfreeze một phần Visual Encoder để fine-tune (tùy chọn)
    print("\n>>> Phase 2: Fine-tuning the whole model with low LR")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for epoch in range(5, EPOCHS):
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
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

    # Lưu model thành phẩm
    torch.save(model.state_dict(), "train/final_model.pth")
    print("\n✅ Đã lưu model thành phẩm tại: train/final_model.pth")

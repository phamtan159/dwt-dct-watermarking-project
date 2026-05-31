import os
import sys

import torch
from torch.utils.data import DataLoader

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
    print("Dataset trong. Hay chay pipeline tao clips va dataset truoc.")
else:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = BaselineModel(NUM_CLASSES).to(device)

    print("\n>>> Training visual-only baseline (CNN3D + LSTM)")
    print(f"Classes: {NUM_CLASSES}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "train/visual_baseline_model_weights.pth")
    print("\nSaved visual baseline: train/visual_baseline_model_weights.pth")

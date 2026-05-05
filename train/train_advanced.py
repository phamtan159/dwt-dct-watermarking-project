import copy
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import LipDataset


ENCODER_CKPT = "pretrained/vsr_trlrs3_base.pth"
BATCH_SIZE = 4
EPOCHS = 15
PHASE1_EPOCHS = 5
VAL_RATIO = 0.2
VAL_MANIFEST = "data/validation_clips.txt"

# Phase 2 choices:
# - frozen: safest default, keep pretrained encoder fixed and continue training head.
# - partial: unfreeze only the last few encoder tensors.
# - distill: partial unfreeze + teacher/student KL loss to reduce forgetting.
PHASE2_MODE = os.environ.get("PHASE2_MODE", "frozen").lower()
PHASE2_LR = 1e-5
PHASE2_TRAINABLE_ENCODER_TENSORS = 8
DISTILL_LAMBDA = 1.0
DISTILL_TEMPERATURE = 0.5


def make_loaders(dataset):
    if len(dataset) < 2:
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True), None

    if os.path.exists(VAL_MANIFEST):
        with open(VAL_MANIFEST, "r", encoding="utf-8") as f:
            val_ids = {
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            }

        val_indices = [
            idx
            for idx, sample in enumerate(dataset.samples)
            if sample.get("sample_id") in val_ids
        ]
        train_indices = [
            idx
            for idx, sample in enumerate(dataset.samples)
            if sample.get("sample_id") not in val_ids
        ]

        if val_indices and train_indices:
            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            val_loader = DataLoader(
                Subset(dataset, val_indices),
                batch_size=BATCH_SIZE,
                shuffle=False,
            )
            print(f"Using validation manifest: {VAL_MANIFEST} ({len(val_indices)} clips)")
            return train_loader, val_loader

        print(
            f"WARNING: {VAL_MANIFEST} exists but did not produce a usable split; "
            "falling back to seeded random split."
        )

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = len(dataset) - val_size
    if train_size == 0:
        train_size, val_size = 1, len(dataset) - 1

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]


def configure_phase2(model):
    if PHASE2_MODE == "frozen":
        model.freeze_encoder()
        print("\n>>> Phase 2: encoder stays frozen; training only LSTM + classifier")
    elif PHASE2_MODE in {"partial", "distill"}:
        names = model.unfreeze_encoder_tail(PHASE2_TRAINABLE_ENCODER_TENSORS)
        print(f"\n>>> Phase 2: partial encoder tuning ({len(names)} tensors)")
        for name in names:
            print(f"    unfreezed: visual_encoder.{name}")
    else:
        raise ValueError("PHASE2_MODE must be one of: frozen, partial, distill")


def distillation_loss(student_logits, teacher_logits):
    temperature = DISTILL_TEMPERATURE
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def run_epoch(model, loader, criterion, optimizer, device, teacher=None):
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            loss = loss + DISTILL_LAMBDA * distillation_loss(logits, teacher_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()

    return total_loss / max(1, len(loader)), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()

    return total_loss / max(1, len(loader)), correct / max(1, total)


def print_metrics(epoch, total_epochs, train_loss, train_acc, val_metrics):
    msg = f"Epoch {epoch:02d}/{total_epochs} | train loss={train_loss:.4f} acc={train_acc:.2%}"
    if val_metrics is not None:
        val_loss, val_acc = val_metrics
        msg += f" | val loss={val_loss:.4f} acc={val_acc:.2%}"
    print(msg)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(ENCODER_CKPT):
        print(f"WARNING: checkpoint not found: {ENCODER_CKPT}")
        print(
            "Download it first, for example: "
            "curl -L http://www.doc.ic.ac.uk/~pm4115/autoAVSR/vsr_trlrs3_base.pth "
            "-o pretrained/vsr_trlrs3_base.pth"
        )
        return

    dataset = LipDataset("data")
    if len(dataset) == 0:
        print("WARNING: Dataset is empty. Run tools 01-06 and add manual labels first.")
        return

    from model import LipModelAdvanced

    num_classes = len(dataset.label_map)
    train_loader, val_loader = make_loaders(dataset)
    model = LipModelAdvanced(num_classes, ENCODER_CKPT).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n>>> Phase 1: encoder frozen; training LSTM + classifier")
    model.freeze_encoder()
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=1e-3)
    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print_metrics(epoch, PHASE1_EPOCHS, train_loss, train_acc, val_metrics)

    teacher = None
    if PHASE2_MODE == "distill":
        teacher = copy.deepcopy(model).to(device)
        teacher.freeze_encoder()
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        print("\n>>> Saved Phase 1 snapshot as distillation teacher")

    configure_phase2(model)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=PHASE2_LR)

    for epoch in range(PHASE1_EPOCHS + 1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            teacher=teacher,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        print_metrics(epoch, EPOCHS, train_loss, train_acc, val_metrics)

    torch.save(model.state_dict(), "train/final_model.pth")
    print("\nSaved model to train/final_model.pth")


if __name__ == "__main__":
    main()

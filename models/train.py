"""
Production training script for WavLM + BiLSTM + CRF pronunciation error detector.

TWO-PHASE TRAINING (learned from fine-tune project):
    Phase 1: Freeze WavLM, train classifier head (LSTM + FC + CRF)
    Phase 2: Unfreeze all layers, fine-tune end-to-end with lower LR

Features:
    ✅ WavLM forward runs ONCE per batch (not twice)
    ✅ Batch CRF — no per-sample loop for forward/loss
    ✅ Mixed Precision (AMP) → ~40% faster
    ✅ Smart freeze: layers 0-7 frozen, 8-11 trainable (Phase 1)
    ✅ Two-phase training (freeze → unfreeze)
    ✅ Gradient accumulation → simulate larger batch
    ✅ Linear warmup scheduler
    ✅ Checkpoint save/resume with full state
    ✅ Proper frame-level mask for CRF (padding-aware!)
    ✅ Clear logging every N steps
    ✅ Validation loss tracking
    ✅ Early stopping support

Usage:
    cd models
    python train.py
"""

import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader, Subset, random_split
from transformers import get_linear_schedule_with_warmup

from wavlm_crf import WavLM_BiLSTM_CRF, load_wavlm_processor
from audio_dataset import AudioDataset, collate_fn
from utils import build_frame_labels, build_frame_mask, label_vocab

# =========================
# ⚙️ CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4          # increase if GPU allows
EPOCHS_PHASE1 = 5       # freeze WavLM -> train classifier only
EPOCHS_PHASE2 = 10      # unfreeze → fine-tune everything
LR_PHASE1 = 1e-3        # higher LR for classifier head
LR_PHASE2 = 1e-5        # lower LR for full fine-tuning
EPOCHS_HEAD_ONLY = 8
EPOCHS_ENCODER_TOP = 5
EPOCHS_FULL_FINETUNE = 0
TOP_WAVLM_LAYERS = 2
LR_HEAD_ONLY = 1e-3
LR_ENCODER_TOP = 3e-5
LR_FULL_FINETUNE = 1e-5

EPOCHS_PHASE1 = EPOCHS_HEAD_ONLY
EPOCHS_PHASE2 = EPOCHS_ENCODER_TOP
LR_PHASE1 = LR_HEAD_ONLY
LR_PHASE2 = LR_ENCODER_TOP

USE_SELF_DISTILLATION = True
DISTILL_LAMBDA = 0.3
DISTILL_TEMPERATURE = 1.0

ACCUM_STEPS = 2          # effective batch = BATCH_SIZE * ACCUM_STEPS
LOG_EVERY = 10           # log every N steps
VAL_SPLIT = 0.1          # 10% for validation
RANDOM_SEED = 42
SAVE_DIR = "checkpoints"
SAVE_PATH = os.path.join(SAVE_DIR, "model.pt")
BEST_PATH = os.path.join(SAVE_DIR, "best_model.pt")

USE_AMP = torch.cuda.is_available()  # only use AMP on GPU

FULL_DATA_PATH = "../data/final/dataset.json"
TRAIN_SPLIT_PATH = "../data/final/train_dataset.json"
DATA_PATH = TRAIN_SPLIT_PATH if os.path.exists(TRAIN_SPLIT_PATH) else FULL_DATA_PATH
STABILITY_BENCHMARK_PATH = "../data/final/stability_benchmark.json"


def fallback_sample_split(dataset, val_split=VAL_SPLIT):
    """Fallback for tiny or single-speaker datasets."""
    if len(dataset) == 1:
        print("Only one sample available; using it for both train and validation.")
        only = Subset(dataset, [0])
        return only, only

    val_size = max(1, int(len(dataset) * val_split))
    if val_size >= len(dataset):
        val_size = len(dataset) - 1
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    return random_split(dataset, [train_size, val_size], generator=generator)


def split_dataset_by_speaker(dataset, val_split=VAL_SPLIT):
    """Split train/validation by speaker so one speaker cannot leak across both sets."""
    by_speaker = defaultdict(list)
    for index, item in enumerate(dataset.data):
        speaker_id = item.get("speaker_id") or "unknown_speaker"
        by_speaker[str(speaker_id)].append(index)

    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        print("Only one speaker found; falling back to sample-level split.")
        return fallback_sample_split(dataset, val_split)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(speakers)

    val_speaker_count = max(1, int(round(len(speakers) * val_split)))
    if val_speaker_count >= len(speakers):
        val_speaker_count = len(speakers) - 1

    val_speakers = set(speakers[:val_speaker_count])
    train_speakers = [speaker for speaker in speakers if speaker not in val_speakers]

    train_indices = [
        index
        for speaker in train_speakers
        for index in by_speaker[speaker]
    ]
    val_indices = [
        index
        for speaker in speakers
        if speaker in val_speakers
        for index in by_speaker[speaker]
    ]

    if not train_indices or not val_indices:
        print("Speaker split produced an empty side; falling back to sample-level split.")
        return fallback_sample_split(dataset, val_split)

    print(f"Speaker split train speakers: {', '.join(train_speakers)}")
    print(f"Speaker split val speakers: {', '.join(sorted(val_speakers))}")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# =========================
# 📦 LOAD
# =========================
print("=" * 60)
print("🚀 Pronunciation Error Detection - Training")
print("=" * 60)

print(f"\n🔹 Device: {DEVICE}")
print(f"🔹 Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUM_STEPS})")
print(f"🔹 Phase 1: {EPOCHS_PHASE1} epochs (classifier only, LR={LR_PHASE1})")
print(f"🔹 Phase 2: {EPOCHS_PHASE2} epochs (full fine-tune, LR={LR_PHASE2})")
print(f"🔹 AMP: {USE_AMP}")

print("\n🔹 Loading processor...")
processor = load_wavlm_processor()

print("🔹 Loading dataset...")
full_dataset = AudioDataset(DATA_PATH, processor, label_vocab)

if len(full_dataset) == 0:
    print("❌ Dataset is empty! Please prepare data first.")
    print("   See: python tools/03_textgrid_to_dataset.py")
    sys.exit(1)

# Train/Val split by speaker
train_dataset, val_dataset = split_dataset_by_speaker(full_dataset, VAL_SPLIT)
train_size = len(train_dataset)
val_size = len(val_dataset)

print(f"🔹 Train: {train_size} | Val: {val_size}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=True
)

benchmark_loader = None
if os.path.exists(STABILITY_BENCHMARK_PATH):
    print(f"Loading stability benchmark: {STABILITY_BENCHMARK_PATH}")
    benchmark_dataset = AudioDataset(STABILITY_BENCHMARK_PATH, processor, label_vocab)
    benchmark_loader = DataLoader(
        benchmark_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    print(f"Stability benchmark: {len(benchmark_dataset)} samples")
else:
    print(f"Stability benchmark not found: {STABILITY_BENCHMARK_PATH}")
    print("Create it from older/known-good phonemes, speakers, and common errors to track forgetting.")

print(f"🔹 Labels: {label_vocab.labels}")
print(f"🔹 Num labels: {len(label_vocab)}")

print("\n🔹 Building model...")
model = WavLM_BiLSTM_CRF(num_labels=len(label_vocab)).to(DEVICE)

# =========================
# 💾 LOAD CHECKPOINT (if exists)
# =========================
start_epoch = 0
current_phase = 1
best_val_loss = float("inf")
teacher_state = None
os.makedirs(SAVE_DIR, exist_ok=True)

if os.path.exists(SAVE_PATH):
    print(f"\n🔁 Resuming from checkpoint: {SAVE_PATH}")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    current_phase = ckpt.get("phase", 1)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    teacher_state = ckpt.get("teacher_model")
    print(f"   Resuming from epoch {start_epoch}, phase {current_phase}")


def clone_state_dict_to_cpu(model):
    """Create a CPU snapshot that can be stored in checkpoints."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def build_teacher_model(state_dict):
    """Build the frozen teacher used for self-distillation."""
    if state_dict is None:
        return None

    teacher = WavLM_BiLSTM_CRF(num_labels=len(label_vocab)).to(DEVICE)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def distillation_loss(student_emissions, teacher_emissions, frame_mask):
    """KL over CRF emissions, restricted to real audio frames."""
    temperature = DISTILL_TEMPERATURE
    student_log_probs = F.log_softmax(student_emissions / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_emissions / temperature, dim=-1)
    kl_per_frame = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none"
    ).sum(dim=-1)

    mask = frame_mask.to(dtype=kl_per_frame.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (kl_per_frame * mask).sum() / denom * (temperature ** 2)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, phase, teacher_model=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    step = 0

    for i, (input_values, mask, phonemes, labels, audio_lens) in enumerate(loader):
        input_values = input_values.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            # =============================================
            # WavLM forward (ONCE per batch!)
            # =============================================
            hidden = model.extract_features(input_values)

            num_frames = hidden.shape[1]
            batch_size_actual = hidden.shape[0]

            # =============================================
            # 🔥 LSTM + FC → emissions
            # =============================================
            lstm_out, _ = model.lstm(hidden)
            lstm_out = model.dropout(lstm_out)
            emissions = model.fc(lstm_out)  # (B, T_frames, num_labels)

            # =============================================
            # 🧠 BUILD BATCH LABELS (frame-level)
            # =============================================
            batch_labels = []

            for b in range(batch_size_actual):
                frame_labels = build_frame_labels(
                    phonemes[b],
                    labels[b],
                    num_frames,
                    audio_lens[b],
                    label_vocab
                )
                batch_labels.append(frame_labels)

            batch_labels = nn.utils.rnn.pad_sequence(
                batch_labels,
                batch_first=True,
                padding_value=label_vocab.stoi["OK"]
            ).to(DEVICE)

            # Ensure labels match emissions shape
            if batch_labels.shape[1] < num_frames:
                pad = torch.full(
                    (batch_size_actual, num_frames - batch_labels.shape[1]),
                    label_vocab.stoi["OK"],
                    dtype=torch.long,
                    device=DEVICE
                )
                batch_labels = torch.cat([batch_labels, pad], dim=1)
            elif batch_labels.shape[1] > num_frames:
                batch_labels = batch_labels[:, :num_frames]

            # =============================================
            # 🔥 FIX: frame-level mask (padding-aware!)
            # CRF needs mask aligned with emissions shape
            # =============================================
            frame_mask = build_frame_mask(
                audio_lens, num_frames, batch_size_actual, DEVICE
            )

            # =============================================
            # 🔥 CRF loss (BATCH — no per-sample loop!)
            # =============================================
            loss = -model.crf(
                emissions,
                batch_labels,
                mask=frame_mask
            )

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_hidden = teacher_model.extract_features(input_values)
                    teacher_lstm_out, _ = teacher_model.lstm(teacher_hidden)
                    teacher_lstm_out = teacher_model.dropout(teacher_lstm_out)
                    teacher_emissions = teacher_model.fc(teacher_lstm_out)

                distill = distillation_loss(
                    emissions,
                    teacher_emissions,
                    frame_mask
                )
                loss = loss + DISTILL_LAMBDA * distill

            loss = loss / ACCUM_STEPS

        # Backward with AMP scaler
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * ACCUM_STEPS
        step += 1

        # Periodic logging
        if step % LOG_EVERY == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  [P{phase}] Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss {loss.item() * ACCUM_STEPS:.4f} | "
                f"LR {current_lr:.2e}"
            )

    avg_loss = total_loss / max(step, 1)
    return avg_loss


@torch.no_grad()
def validate(model, loader):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    step = 0

    for input_values, mask, phonemes, labels, audio_lens in loader:
        input_values = input_values.to(DEVICE)

        hidden = model.extract_features(input_values)
        num_frames = hidden.shape[1]
        batch_size_actual = hidden.shape[0]

        lstm_out, _ = model.lstm(hidden)
        lstm_out = model.dropout(lstm_out)
        emissions = model.fc(lstm_out)

        batch_labels = []
        for b in range(batch_size_actual):
            frame_labels = build_frame_labels(
                phonemes[b], labels[b], num_frames,
                audio_lens[b], label_vocab
            )
            batch_labels.append(frame_labels)

        batch_labels = nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True,
            padding_value=label_vocab.stoi["OK"]
        ).to(DEVICE)

        if batch_labels.shape[1] < num_frames:
            pad = torch.full(
                (batch_size_actual, num_frames - batch_labels.shape[1]),
                label_vocab.stoi["OK"], dtype=torch.long, device=DEVICE
            )
            batch_labels = torch.cat([batch_labels, pad], dim=1)
        elif batch_labels.shape[1] > num_frames:
            batch_labels = batch_labels[:, :num_frames]

        frame_mask = build_frame_mask(
            audio_lens, num_frames, batch_size_actual, DEVICE
        )

        loss = -model.crf(emissions, batch_labels, mask=frame_mask)
        total_loss += loss.item()
        step += 1

    return total_loss / max(step, 1)


@torch.no_grad()
def evaluate_benchmark(model, loader):
    """Evaluate a fixed stability benchmark to catch forgetting across epochs."""
    if loader is None:
        return None

    model.eval()
    all_preds = []
    all_golds = []

    for input_values, mask, phonemes, labels, audio_lens in loader:
        input_values = input_values.to(DEVICE)

        hidden = model.extract_features(input_values)
        num_frames = hidden.shape[1]
        batch_size_actual = hidden.shape[0]

        lstm_out, _ = model.lstm(hidden)
        lstm_out = model.dropout(lstm_out)
        emissions = model.fc(lstm_out)

        batch_labels = []
        for b in range(batch_size_actual):
            frame_labels = build_frame_labels(
                phonemes[b], labels[b], num_frames,
                audio_lens[b], label_vocab
            )
            batch_labels.append(frame_labels)

        batch_labels = nn.utils.rnn.pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=label_vocab.stoi["OK"]
        ).to(DEVICE)

        if batch_labels.shape[1] < num_frames:
            pad = torch.full(
                (batch_size_actual, num_frames - batch_labels.shape[1]),
                label_vocab.stoi["OK"],
                dtype=torch.long,
                device=DEVICE
            )
            batch_labels = torch.cat([batch_labels, pad], dim=1)
        elif batch_labels.shape[1] > num_frames:
            batch_labels = batch_labels[:, :num_frames]

        frame_mask = build_frame_mask(
            audio_lens, num_frames, batch_size_actual, DEVICE
        )
        preds = model.crf.decode(emissions, mask=frame_mask)

        for b in range(batch_size_actual):
            real_len = frame_mask[b].sum().item()
            all_preds.extend(preds[b][:real_len])
            all_golds.extend(batch_labels[b][:real_len].cpu().tolist())

    if not all_golds:
        return None

    ok_idx = label_vocab.stoi["OK"]
    correct = sum(1 for pred, gold in zip(all_preds, all_golds) if pred == gold)
    binary_correct = sum(
        1 for pred, gold in zip(all_preds, all_golds)
        if (pred == ok_idx) == (gold == ok_idx)
    )

    per_label = {}
    for idx, label in label_vocab.itos.items():
        tp = sum(1 for pred, gold in zip(all_preds, all_golds) if pred == idx and gold == idx)
        fp = sum(1 for pred, gold in zip(all_preds, all_golds) if pred == idx and gold != idx)
        fn = sum(1 for pred, gold in zip(all_preds, all_golds) if pred != idx and gold == idx)
        support = sum(1 for gold in all_golds if gold == idx)
        if support == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {"f1": f1, "support": support}

    return {
        "accuracy": correct / len(all_golds),
        "error_detection_acc": binary_correct / len(all_golds),
        "per_label": per_label,
        "frames": len(all_golds),
    }


def print_benchmark_metrics(metrics, phase, epoch):
    """Print compact fixed-benchmark metrics after each epoch."""
    if metrics is None:
        return

    print(
        f"Stability benchmark [P{phase} E{epoch}] | "
        f"Frames: {metrics['frames']} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Error detection: {metrics['error_detection_acc']:.4f}"
    )

    label_summary = ", ".join(
        f"{label}:F1={stats['f1']:.3f}/n={stats['support']}"
        for label, stats in sorted(
            metrics["per_label"].items(),
            key=lambda item: item[1]["support"],
            reverse=True
        )
    )
    if label_summary:
        print(f"Stability per-label: {label_summary}")


def save_checkpoint(model, optimizer, scheduler, epoch, phase, val_loss, teacher_state=None):
    """Save training checkpoint with all state."""
    payload = {
        "epoch": epoch,
        "phase": phase,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "label_vocab": label_vocab.labels,
        "best_val_loss": best_val_loss,
    }
    if teacher_state is not None:
        payload["teacher_model"] = teacher_state

    torch.save(payload, SAVE_PATH)


# =========================
# 🏋️ PHASE 1: Train classifier only
# =========================
total_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2

if current_phase == 1 and start_epoch < EPOCHS_PHASE1:
    print("\n" + "=" * 60)
    print("PHASE 1: Training classifier (WavLM frozen)")
    print("=" * 60)

    # Freeze WavLM
    freeze_info = model.freeze_wavlm_all()
    stats = model.get_param_stats()
    print(f"Frozen WavLM params: {freeze_info['frozen']}")
    print(f"Trainable WavLM params: {freeze_info['trainable']}")
    print(f"📊 Total trainable: {stats['trainable']:,} / {stats['total']:,} ({stats['pct_trainable']:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE1,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * EPOCHS_PHASE1 // ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    for epoch in range(start_epoch, EPOCHS_PHASE1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, phase=1)
        val_loss = validate(model, val_loader)
        benchmark_metrics = evaluate_benchmark(model, benchmark_loader)
        elapsed = time.time() - t0

        print(f"\n✅ [P1] Epoch {epoch} DONE | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {elapsed:.1f}s")

        print_benchmark_metrics(benchmark_metrics, phase=1, epoch=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "label_vocab": label_vocab.labels}, BEST_PATH)
            print(f"   🏆 New best model! Val Loss: {val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, 1, val_loss)
        print(f"💾 Saved checkpoint → {SAVE_PATH}\n")

    start_epoch = EPOCHS_PHASE1
    current_phase = 2

if current_phase == 1 and start_epoch >= EPOCHS_PHASE1:
    current_phase = 2

if current_phase == 2 and USE_SELF_DISTILLATION and teacher_state is None:
    teacher_state = clone_state_dict_to_cpu(model)
    print("Created self-distillation teacher snapshot from the head-only model.")

# =========================
# 🏋️ PHASE 2: Conservative encoder fine-tuning
# =========================
if current_phase == 2:
    teacher_model = build_teacher_model(teacher_state) if USE_SELF_DISTILLATION else None
    if teacher_model is not None:
        print("Self-distillation teacher is active for phase 2.")

    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning top WavLM layers")
    print("=" * 60)

    # Unfreeze only the top WavLM encoder layers.
    model.unfreeze_top_wavlm_layers(num_trainable_layers=TOP_WAVLM_LAYERS)
    stats = model.get_param_stats()
    print(f"Top WavLM layers unfrozen: {TOP_WAVLM_LAYERS}")
    print(f"🔥 Trainable parameters: {stats['trainable']:,} / {stats['total']:,} ({stats['pct_trainable']:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_PHASE2,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * EPOCHS_PHASE2 // ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    phase2_start = max(start_epoch, EPOCHS_PHASE1)

    for epoch in range(phase2_start, EPOCHS_PHASE1 + EPOCHS_PHASE2):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            epoch,
            phase=2,
            teacher_model=teacher_model
        )
        val_loss = validate(model, val_loader)
        benchmark_metrics = evaluate_benchmark(model, benchmark_loader)
        elapsed = time.time() - t0

        print(f"\n✅ [P2] Epoch {epoch} DONE | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | {elapsed:.1f}s")

        print_benchmark_metrics(benchmark_metrics, phase=2, epoch=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "label_vocab": label_vocab.labels}, BEST_PATH)
            print(f"   🏆 New best model! Val Loss: {val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, 2, val_loss, teacher_state=teacher_state)
        print(f"💾 Saved checkpoint → {SAVE_PATH}\n")

print("=" * 60)
print("🎉 Training complete!")
print(f"   Best model saved at: {BEST_PATH}")
print(f"   Last checkpoint at: {SAVE_PATH}")
print("=" * 60)

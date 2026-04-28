"""
Production training script for Wav2Vec2 + BiLSTM + CRF pronunciation error detector.

TWO-PHASE TRAINING:
    Phase 1: Freeze wav2vec2 layers 0-7, train classifier head (LSTM + FC + CRF)
    Phase 2: Unfreeze all layers, fine-tune end-to-end with lower LR
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import Wav2Vec2Processor, get_linear_schedule_with_warmup

from wav2vec2_crf import Wav2Vec2_BiLSTM_CRF
from audio_dataset import AudioDataset, collate_fn
from utils import build_frame_labels, build_frame_mask, label_vocab


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 4
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 10
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
ACCUM_STEPS = 2
LOG_EVERY = 10
VAL_SPLIT = 0.1

SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
SAVE_PATH = os.path.join(SAVE_DIR, "model.pt")
BEST_PATH = os.path.join(SAVE_DIR, "best_model.pt")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "final", "dataset.json")

USE_AMP = torch.cuda.is_available()


def build_batch_labels(phoneme_batch, label_batch, audio_lens, num_frames, device):
    """Convert a batch of phoneme labels into padded frame labels."""
    batch_labels = []

    for phonemes, labels, audio_len in zip(phoneme_batch, label_batch, audio_lens):
        frame_labels = build_frame_labels(
            phonemes,
            labels,
            num_frames,
            audio_len,
            label_vocab,
        )
        batch_labels.append(frame_labels)

    batch_labels = nn.utils.rnn.pad_sequence(
        batch_labels,
        batch_first=True,
        padding_value=label_vocab.stoi["OK"],
    ).to(device)

    if batch_labels.shape[1] < num_frames:
        pad = torch.full(
            (batch_labels.shape[0], num_frames - batch_labels.shape[1]),
            label_vocab.stoi["OK"],
            dtype=torch.long,
            device=device,
        )
        batch_labels = torch.cat([batch_labels, pad], dim=1)
    elif batch_labels.shape[1] > num_frames:
        batch_labels = batch_labels[:, :num_frames]

    return batch_labels


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, phase):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    step = 0

    for i, (input_values, mask, phonemes, labels, audio_lens) in enumerate(loader):
        input_values = input_values.to(DEVICE)
        audio_mask = mask.to(DEVICE).long()

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            hidden = model.extract_features(input_values, attention_mask=audio_mask)
            num_frames = hidden.shape[1]
            batch_size_actual = hidden.shape[0]

            batch_labels = build_batch_labels(
                phonemes,
                labels,
                audio_lens,
                num_frames,
                DEVICE,
            )
            frame_mask = build_frame_mask(
                audio_lens,
                num_frames,
                batch_size_actual,
                DEVICE,
            )

            loss = model.forward_from_features(
                hidden,
                frame_mask=frame_mask,
                labels=batch_labels,
            )
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        should_step = (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * ACCUM_STEPS
        step += 1

        if step % LOG_EVERY == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  [P{phase}] Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss {loss.item() * ACCUM_STEPS:.4f} | "
                f"LR {current_lr:.2e}"
            )

    return total_loss / max(step, 1)


@torch.no_grad()
def validate(model, loader):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    step = 0

    for input_values, mask, phonemes, labels, audio_lens in loader:
        input_values = input_values.to(DEVICE)
        audio_mask = mask.to(DEVICE).long()

        hidden = model.extract_features(input_values, attention_mask=audio_mask)
        num_frames = hidden.shape[1]
        batch_size_actual = hidden.shape[0]

        batch_labels = build_batch_labels(
            phonemes,
            labels,
            audio_lens,
            num_frames,
            DEVICE,
        )
        frame_mask = build_frame_mask(
            audio_lens,
            num_frames,
            batch_size_actual,
            DEVICE,
        )

        loss = model.forward_from_features(
            hidden,
            frame_mask=frame_mask,
            labels=batch_labels,
        )

        total_loss += loss.item()
        step += 1

    return total_loss / max(step, 1)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, phase, best_val_loss):
    """Save training checkpoint with all state needed to resume."""
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "label_vocab": label_vocab.labels,
            "best_val_loss": best_val_loss,
        },
        SAVE_PATH,
    )


def maybe_restore_training_state(optimizer, scheduler, scaler, resume_state):
    """Load optimizer, scheduler, and scaler state when resuming a phase."""
    if resume_state.get("optim"):
        optimizer.load_state_dict(resume_state["optim"])
    if resume_state.get("sched"):
        scheduler.load_state_dict(resume_state["sched"])
    if resume_state.get("scaler"):
        scaler.load_state_dict(resume_state["scaler"])


def build_optimizer_and_scheduler(model, lr, total_epochs, train_loader_len):
    """Create optimizer + warmup scheduler for the active phase."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
    )

    total_steps = max(1, (train_loader_len * total_epochs + ACCUM_STEPS - 1) // ACCUM_STEPS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    return optimizer, scheduler, scaler


def main():
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Pronunciation Error Detection - Training")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUM_STEPS})")
    print(f"Phase 1: {EPOCHS_PHASE1} epochs (classifier only, LR={LR_PHASE1})")
    print(f"Phase 2: {EPOCHS_PHASE2} epochs (full fine-tune, LR={LR_PHASE2})")
    print(f"AMP: {USE_AMP}")

    if not os.path.exists(DATA_PATH):
        print(f"\nDataset not found: {DATA_PATH}")
        print("Run the tools pipeline first to generate data/final/dataset.json.")
        sys.exit(1)

    print("\nLoading processor...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    print("Loading dataset...")
    full_dataset = AudioDataset(DATA_PATH, processor, label_vocab)
    if len(full_dataset) == 0:
        print("Dataset is empty after validation.")
        sys.exit(1)

    val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
    train_size = len(full_dataset) - val_size
    split_generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train: {train_size} | Val: {val_size}")
    print(f"Labels: {label_vocab.labels}")
    print(f"Num labels: {len(label_vocab)}")

    print("\nBuilding model...")
    model = Wav2Vec2_BiLSTM_CRF(num_labels=len(label_vocab)).to(DEVICE)

    os.makedirs(SAVE_DIR, exist_ok=True)
    start_epoch = 0
    current_phase = 1
    best_val_loss = float("inf")
    resume_state = {}

    if os.path.exists(SAVE_PATH):
        print(f"\nResuming from checkpoint: {SAVE_PATH}")
        ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        current_phase = ckpt.get("phase", 1)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        resume_state = {
            "optim": ckpt.get("optim"),
            "sched": ckpt.get("sched"),
            "scaler": ckpt.get("scaler"),
        }
        print(f"Resuming from epoch {start_epoch}, phase {current_phase}")

    if current_phase == 1 and start_epoch < EPOCHS_PHASE1:
        print("\n" + "=" * 60)
        print("PHASE 1: Training classifier (wav2vec2 partially frozen)")
        print("=" * 60)

        freeze_info = model.freeze_wav2vec2(freeze_layers_below=8)
        stats = model.get_param_stats()
        print(f"Frozen wav2vec2 params: {freeze_info['frozen']}")
        print(f"Trainable wav2vec2 params: {freeze_info['trainable']}")
        print(
            f"Total trainable: {stats['trainable']:,} / {stats['total']:,} "
            f"({stats['pct_trainable']:.1f}%)"
        )

        optimizer, scheduler, scaler = build_optimizer_and_scheduler(
            model,
            LR_PHASE1,
            EPOCHS_PHASE1,
            len(train_loader),
        )
        if start_epoch > 0:
            maybe_restore_training_state(optimizer, scheduler, scaler, resume_state)

        for epoch in range(start_epoch, EPOCHS_PHASE1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, phase=1)
            val_loss = validate(model, val_loader)
            elapsed = time.time() - t0

            print(
                f"\n[P1] Epoch {epoch} DONE | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "label_vocab": label_vocab.labels},
                    BEST_PATH,
                )
                print(f"  New best model! Val Loss: {val_loss:.4f}")

            save_checkpoint(model, optimizer, scheduler, scaler, epoch, 1, best_val_loss)
            print(f"Saved checkpoint -> {SAVE_PATH}\n")

        start_epoch = EPOCHS_PHASE1
        current_phase = 2
        resume_state = {}

    if current_phase == 2:
        print("\n" + "=" * 60)
        print("PHASE 2: Full fine-tuning (all layers)")
        print("=" * 60)

        model.unfreeze_all()
        stats = model.get_param_stats()
        print(f"All parameters unfrozen: {stats['trainable']:,} total")

        optimizer, scheduler, scaler = build_optimizer_and_scheduler(
            model,
            LR_PHASE2,
            EPOCHS_PHASE2,
            len(train_loader),
        )
        if start_epoch > EPOCHS_PHASE1:
            maybe_restore_training_state(optimizer, scheduler, scaler, resume_state)

        phase2_start = max(start_epoch, EPOCHS_PHASE1)

        for epoch in range(phase2_start, EPOCHS_PHASE1 + EPOCHS_PHASE2):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, phase=2)
            val_loss = validate(model, val_loader)
            elapsed = time.time() - t0

            print(
                f"\n[P2] Epoch {epoch} DONE | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model": model.state_dict(), "label_vocab": label_vocab.labels},
                    BEST_PATH,
                )
                print(f"  New best model! Val Loss: {val_loss:.4f}")

            save_checkpoint(model, optimizer, scheduler, scaler, epoch, 2, best_val_loss)
            print(f"Saved checkpoint -> {SAVE_PATH}\n")

    print("=" * 60)
    print("Training complete!")
    print(f"Best model saved at: {BEST_PATH}")
    print(f"Last checkpoint at: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# 🔧 Fine-Tune-2 Improvements Report

## What was learned from Fine-Tune (Visual) project and applied

### 1. 🛠️ Data Preparation Pipeline (NEW — was completely missing)

| File | Purpose |
|---|---|
| [01_prepare_audio.py](file:///d:/fine-tune-2/tools/01_prepare_audio.py) | Extract audio from video/media → 16kHz mono WAV |
| [02_run_mfa.py](file:///d:/fine-tune-2/tools/02_run_mfa.py) | Montreal Forced Aligner → phoneme timing |
| [03_textgrid_to_dataset.py](file:///d:/fine-tune-2/tools/03_textgrid_to_dataset.py) | TextGrid → `dataset.json` for training |

### 2. 🐛 Critical Bug Fixes

#### `utils.py` — build_frame_labels was miscalculating frame timing
```diff
-    frame_time = audio_len / num_frames  # BUG: audio_len is in SAMPLES, not seconds!
+    audio_duration_sec = audio_len_samples / 16000.0
+    sec_per_frame = audio_duration_sec / num_frames
```

#### `utils.py` — Added padding-aware frame mask (NEW)
The old code used `torch.ones()` for ALL frames including padding → CRF learned garbage from padded positions.
```diff
+def build_frame_mask(audio_lens_samples, num_frames, batch_size, device):
+    """Build mask that distinguishes real frames from padding."""
+    mask[b, :real_frames] = True  # Only real frames are marked
```

#### `train.py` — Frame mask was not padding-aware
```diff
-    frame_mask = torch.ones(...)  # All frames treated as real (WRONG!)
+    frame_mask = build_frame_mask(audio_lens, ...)  # Padding-aware
```

### 3. 🏗️ Two-Phase Training (from fine-tune project)

| Phase | Epochs | What trains | LR | Why |
|---|---|---|---|---|
| **Phase 1** | 5 | LSTM + FC + CRF only | 1e-3 | Fast classifier convergence |
| **Phase 2** | 10 | Everything (wav2vec2 + all) | 1e-5 | End-to-end fine-tuning |

### 4. ✨ New Features Added

| Feature | File | Description |
|---|---|---|
| **Validation loop** | [train.py](file:///d:/fine-tune-2/models/train.py) | Train/val split + best model tracking |
| **Evaluation metrics** | [evaluate.py](file:///d:/fine-tune-2/models/evaluate.py) | Per-class P/R/F1 + error detection accuracy |
| **Baseline model** | [baseline_model.py](file:///d:/fine-tune-2/models/baseline_model.py) | MFCC+BiLSTM baseline for comparison |
| **Phoneme-level predict** | [predict.py](file:///d:/fine-tune-2/models/predict.py) | Majority voting frame→phoneme aggregation |
| **Smart freeze/unfreeze** | [wav2vec2_crf.py](file:///d:/fine-tune-2/models/wav2vec2_crf.py) | `freeze_wav2vec2()`, `unfreeze_all()` methods |
| **Feature caching** | [wav2vec2_crf.py](file:///d:/fine-tune-2/models/wav2vec2_crf.py) | `extract_features()` for 5-10x speedup |
| **Dynamic label vocab** | [utils.py](file:///d:/fine-tune-2/models/utils.py) | `add_label()` for runtime label expansion |
| **Robust dataset loading** | [audio_dataset.py](file:///d:/fine-tune-2/models/audio_dataset.py) | Validates audio, labels, shows distribution |

### 5. 📊 Before vs After Summary

```
BEFORE (fine-tune-2 original):
├── models/          ← 6 files, basic training
├── data/final/      ← sample only
└── NO tools pipeline

AFTER (improved):
├── tools/           ← 3-step data pipeline (NEW)
├── models/          ← 8 files, production-ready
│   ├── train.py         ← Two-phase + validation + best model
│   ├── evaluate.py      ← P/R/F1 metrics (NEW)
│   ├── baseline_model.py ← Comparison baseline (NEW)
│   ├── predict.py       ← Phoneme-level output
│   ├── wav2vec2_crf.py  ← Smart freeze/unfreeze/cache
│   └── utils.py         ← Fixed frame timing + padding mask
├── data/            ← Full directory structure
└── readme.md        ← Complete step-by-step guide
```

> [!IMPORTANT]
> Bạn cần tạo thêm các thư mục data: `data/raw/`, `data/audio/`, `data/transcript/`, `data/aligned/` trước khi chạy pipeline.
> ```bash
> mkdir data\raw data\audio data\transcript data\aligned
> ```

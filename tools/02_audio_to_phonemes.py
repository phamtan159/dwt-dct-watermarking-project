"""
02_audio_to_phonemes.py
Nhận dạng âm vị trực tiếp từ audio bằng wav2vec2-phoneme.
Model: Speech31/wav2vec2-large-english-TIMIT-phoneme_v3

Input:  data/audio/*.wav (16kHz mono)
Output: data/annotations/auto/*.txt  (chuỗi phoneme thực tế người nói)
        data/annotations/auto/*.json (segments với timestamp)
"""

import os, json, torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf

# ── Load model ──────────────────────────────────────────────
MODEL_ID = "Speech31/wav2vec2-large-english-TIMIT-phoneme_v3"
print(f"Loading model: {MODEL_ID} ...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, local_files_only=True)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, local_files_only=True)
model.eval()
print("Model loaded.")

os.makedirs("data/annotations/auto", exist_ok=True)

# ── Process each audio ──────────────────────────────────────
for file in os.listdir("data/audio"):
    if not file.endswith(".wav"):
        continue

    name = file.replace(".wav", "")
    audio_path = f"data/audio/{file}"

    print(f"\nProcessing: {audio_path}")

    # Read audio
    speech, sr = sf.read(audio_path)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"

    # Run inference
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    # Decode phonemes
    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes_str = processor.batch_decode(predicted_ids)[0]

    # Clean up: remove special tokens and extra spaces
    phonemes_str = phonemes_str.replace("<pad>", "").replace("<s>", "").replace("</s>", "")
    phonemes_str = " ".join(phonemes_str.split())  # normalize whitespace

    # ── Save .txt (plain IPA string) ────────────────────────
    txt_path = f"data/annotations/auto/{name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(phonemes_str.strip())
    safe_str = phonemes_str.strip().encode('ascii', errors='replace').decode('ascii')
    print(f"  -> {txt_path}: {safe_str}")

    # ── Build segments with timestamps ──────────────────────
    pred_ids_seq = predicted_ids[0].tolist()

    # wav2vec2 frame rate: ~49.95 frames/sec (320 samples per frame at 16kHz)
    frame_duration = 320 / 16000  # 0.02 seconds per frame

    segments = []
    blank_id = processor.tokenizer.pad_token_id

    # Group consecutive identical predictions (CTC decoding with timestamps)
    current_id = None
    start_frame = 0

    for i, pid in enumerate(pred_ids_seq):
        if pid != current_id:
            # Save previous segment if it was a real phoneme (not blank/pad)
            if current_id is not None and current_id != blank_id:
                token = processor.tokenizer.convert_ids_to_tokens(current_id)
                if token and token.strip():
                    seg_id = f"{len(segments):03d}_{token}"
                    segments.append({
                        "id": seg_id,
                        "phoneme": token,
                        "start": round(start_frame * frame_duration, 3),
                        "end": round(i * frame_duration, 3),
                        "error": None
                    })
            current_id = pid
            start_frame = i

    # Handle last segment
    if current_id is not None and current_id != blank_id:
        token = processor.tokenizer.convert_ids_to_tokens(current_id)
        if token and token.strip():
            seg_id = f"{len(segments):03d}_{token}"
            segments.append({
                "id": seg_id,
                "phoneme": token,
                "start": round(start_frame * frame_duration, 3),
                "end": round(len(pred_ids_seq) * frame_duration, 3),
                "error": None
            })

    # ── Save .json (segments with timestamps) ───────────────
    json_path = f"data/annotations/auto/{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, indent=2, ensure_ascii=False)
    print(f"  -> {json_path}: {len(segments)} segments")

print("\nDone!")

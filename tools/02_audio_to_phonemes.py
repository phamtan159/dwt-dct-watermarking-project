"""
Run wav2vec2 phoneme recognition on normalized audio.

Input:
    data/audio/*.wav

Output:
    data/annotations/wav2vec2_raw/*.txt
    data/annotations/wav2vec2_raw/*.json
"""

import json
import os

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_ID = "Speech31/wav2vec2-large-english-TIMIT-phoneme_v3"
OUTPUT_DIR = "data/annotations/wav2vec2_raw"

print(f"Loading model: {MODEL_ID} ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, local_files_only=False)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, local_files_only=False)
model.eval()
print("Model loaded.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir("data/audio"):
    if not file.endswith(".wav"):
        continue

    name = file.replace(".wav", "")
    audio_path = f"data/audio/{file}"
    print(f"\nProcessing: {audio_path}")

    speech, sr = sf.read(audio_path)
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes_str = processor.batch_decode(predicted_ids)[0]
    phonemes_str = phonemes_str.replace("<pad>", "").replace("<s>", "").replace("</s>", "")
    phonemes_str = " ".join(phonemes_str.split())

    txt_path = f"{OUTPUT_DIR}/{name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(phonemes_str.strip())
    safe_str = phonemes_str.strip().encode("ascii", errors="replace").decode("ascii")
    print(f"  -> {txt_path}: {safe_str}")

    pred_ids_seq = predicted_ids[0].tolist()
    frame_duration = 320 / 16000
    blank_id = processor.tokenizer.pad_token_id

    segments = []
    current_id = None
    start_frame = 0

    for i, pid in enumerate(pred_ids_seq):
        if pid != current_id:
            if current_id is not None and current_id != blank_id:
                token = processor.tokenizer.convert_ids_to_tokens(current_id)
                if token and token.strip():
                    seg_id = f"{len(segments):03d}_{token}"
                    segments.append(
                        {
                            "id": seg_id,
                            "phoneme": token,
                            "phoneme_raw": token,
                            "start": round(start_frame * frame_duration, 3),
                            "end": round(i * frame_duration, 3),
                            "error": None,
                        }
                    )
            current_id = pid
            start_frame = i

    if current_id is not None and current_id != blank_id:
        token = processor.tokenizer.convert_ids_to_tokens(current_id)
        if token and token.strip():
            seg_id = f"{len(segments):03d}_{token}"
            segments.append(
                {
                    "id": seg_id,
                    "phoneme": token,
                    "phoneme_raw": token,
                    "start": round(start_frame * frame_duration, 3),
                    "end": round(len(pred_ids_seq) * frame_duration, 3),
                    "error": None,
                }
            )

    json_path = f"{OUTPUT_DIR}/{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments}, f, indent=2, ensure_ascii=False)
    print(f"  -> {json_path}: {len(segments)} segments")

print("\nDone!")

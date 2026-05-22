"""
Run Wav2Vec2 phoneme recognition on normalized audio.

Input:
    data/audio/*.wav

Output:
    data/annotations/wav2vec2_raw/*.txt
    data/annotations/wav2vec2_raw/*.json
"""

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
LOCAL_MODEL_DIR = Path(__file__).resolve().parents[1] / "pretrained" / "facebook-wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_OUTPUT_DIR = Path("data/annotations/wav2vec2_raw")
DEFAULT_AUDIO_DIR = Path("data/audio")


def find_local_snapshot(model_id: str) -> Path | None:
    required_files = ["config.json", "preprocessor_config.json", "pytorch_model.bin", "vocab.json"]
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_cache = cache_home / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if model_cache.exists():
        snapshots = sorted(
            (path for path in model_cache.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if all((snapshot / filename).exists() for filename in required_files):
                return snapshot

    if LOCAL_MODEL_DIR.exists() and all((LOCAL_MODEL_DIR / filename).exists() for filename in required_files):
        return LOCAL_MODEL_DIR

    return None


def load_model(model_id: str, model_path: str | None = None):
    resolved_model = Path(model_path) if model_path else find_local_snapshot(model_id)
    local_only = resolved_model is not None
    source = str(resolved_model) if resolved_model else model_id

    print(f"Loading Wav2Vec2 phoneme model: {source} ...")
    processor = Wav2Vec2Processor.from_pretrained(
        source,
        local_files_only=local_only,
        do_phonemize=False,
    )
    model = Wav2Vec2ForCTC.from_pretrained(source, local_files_only=local_only)
    model.eval()
    print("Model loaded.")
    return processor, model


def write_prediction(audio_path: Path, output_dir: Path, processor, model) -> None:
    name = audio_path.stem
    print(f"\nProcessing: {audio_path}")

    speech, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError(f"{audio_path}: expected 16kHz, got {sr}Hz")

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes_str = processor.batch_decode(predicted_ids)[0]
    phonemes_str = phonemes_str.replace("<pad>", "").replace("<s>", "").replace("</s>", "")
    phonemes_str = " ".join(phonemes_str.split())

    txt_path = output_dir / f"{name}.txt"
    txt_path.write_text(phonemes_str.strip(), encoding="utf-8")
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
                            "model": "wav2vec2_ctc_phoneme",
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
                    "model": "wav2vec2_ctc_phoneme",
                }
            )

    json_path = output_dir / f"{name}.json"
    payload = {
        "audio_id": name,
        "model": MODEL_ID,
        "decoder": "wav2vec2_ctc_phoneme",
        "segments": segments,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {json_path}: {len(segments)} segments")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wav2Vec2 phoneme recognition on normalized audio.")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-path", default=None, help="Optional local Hugging Face model directory.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"No wav files found in {audio_dir}")
        return

    processor, model = load_model(args.model_id, args.model_path)
    for audio_path in audio_files:
        write_prediction(audio_path, output_dir, processor, model)

    print("\nDone!")


if __name__ == "__main__":
    main()

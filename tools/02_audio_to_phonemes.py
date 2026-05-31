"""
Run Wav2Vec2 phoneme recognition on normalized audio.

Input:
    data/audio/<speaker>/*.wav

Output:
    data/annotations/wav2vec2_raw/<speaker>/*.txt
    data/annotations/wav2vec2_raw/<speaker>/*.json
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from speaker_paths import iter_speaker_files, relative_id, relative_output_path, sample_id_from_relative, speaker_id_from_relative, warn_root_level_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "facebook-wav2vec2-lv-60-espeak-cv-ft"
LOCAL_MODEL_DIR = PROJECT_ROOT / "pretrained" / MODEL_NAME
DEFAULT_OUTPUT_DIR = Path("data/annotations/wav2vec2_raw")
DEFAULT_AUDIO_DIR = Path("data/audio")


def resolve_local_model(model_path: str | None = None) -> Path:
    required_files = ["config.json", "preprocessor_config.json", "pytorch_model.bin", "vocab.json"]
    path = Path(model_path) if model_path else LOCAL_MODEL_DIR
    missing = [filename for filename in required_files if not (path / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing local Wav2Vec2 model files in {path}: {', '.join(missing)}. "
            f"Copy the full model into pretrained/{MODEL_NAME} first."
        )
    return path


def load_model(model_path: str | None = None):
    source = resolve_local_model(model_path)

    print(f"Loading Wav2Vec2 phoneme model: {source} ...")
    processor = Wav2Vec2Processor.from_pretrained(
        str(source),
        local_files_only=True,
        do_phonemize=False,
    )
    model = Wav2Vec2ForCTC.from_pretrained(str(source), local_files_only=True)
    model.eval()
    print("Model loaded.")
    return processor, model, source


def write_prediction(audio_path: Path, audio_dir: Path, output_dir: Path, processor, model, model_source: Path) -> None:
    rel_id = relative_id(audio_path, audio_dir)
    name = sample_id_from_relative(rel_id)
    speaker_id = speaker_id_from_relative(rel_id)
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

    txt_path = relative_output_path(output_dir, rel_id, ".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
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

    json_path = relative_output_path(output_dir, rel_id, ".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": rel_id,
        "speaker_id": speaker_id,
        "sample_id": name,
        "audio_id": name,
        "model": str(model_source).replace("\\", "/"),
        "decoder": "wav2vec2_ctc_phoneme",
        "segments": segments,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {json_path}: {len(segments)} segments")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wav2Vec2 phoneme recognition on normalized audio.")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=str(LOCAL_MODEL_DIR), help="Local Hugging Face model directory.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warn_root_level_files(audio_dir, {".wav"}, "audio")
    audio_files = iter_speaker_files(audio_dir, {".wav"})
    if not audio_files:
        print(f"No speaker wav files found in {audio_dir}")
        return

    processor, model, model_source = load_model(args.model_path)
    for audio_path in audio_files:
        write_prediction(audio_path, audio_dir, output_dir, processor, model, model_source)

    print("\nDone!")


if __name__ == "__main__":
    main()

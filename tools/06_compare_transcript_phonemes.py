"""
Compare MFA-aligned canonical phones against wav2vec2 raw phoneme predictions.

Inputs:
    data/transcript/<video_name>.txt
    data/annotations/auto/<video_name>.json
    data/annotations/wav2vec2_raw/<video_name>.json

Output:
    data/annotations/compare/<video_name>.json
"""

import argparse
import json
from pathlib import Path

from phoneme_utils import (
    collapse_compare_phone_tokens,
    is_silence,
    normalize_phone_sequence_to_compare,
    normalize_token_to_compare,
    read_transcript,
)


KNOWN_ERROR_BY_PAIR = {
    ("θ", "t"): "th_to_t",
    ("θ", "th"): "th_to_t",
    ("ð", "d"): "dh_to_d",
    ("s", ""): "s_final_omitted",
    ("ʃ", "s"): "sh_to_s",
    ("z", "s"): "z_to_s",
    ("r", "g"): "r_to_g_y",
    ("r", "j"): "r_to_g_y",
    ("v", "d"): "v_to_d_dz",
    ("v", "z"): "v_to_d_dz",
    ("dʒ", "d"): "dj_to_d_y",
    ("dʒ", "j"): "dj_to_d_y",
}


def load_label_map(path):
    if not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    id_to_code = {}
    code_to_id = {}
    if isinstance(data, dict):
        for code, item in data.items():
            if isinstance(item, dict) and item.get("id"):
                id_to_code[item["id"]] = str(code)
                code_to_id[str(code)] = item["id"]
    return id_to_code, code_to_id


def classify_error(std_phone, real_phone, id_to_code):
    if std_phone == real_phone:
        return "no_error", "0"
    error_id = KNOWN_ERROR_BY_PAIR.get((std_phone, real_phone), "OTHER")
    return error_id, id_to_code.get(error_id)


def find_matching_file(directory, stem, suffix):
    exact = directory / f"{stem}{suffix}"
    if exact.exists():
        return exact
    lower_stem = stem.lower()
    for path in directory.glob(f"*{suffix}"):
        if path.stem.lower() == lower_stem:
            return path
    return None


def segment_overlap(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def infer_real_phone(aligned_segment, wav_segments):
    start = float(aligned_segment["start"])
    end = float(aligned_segment["end"])
    duration = max(end - start, 1e-6)

    overlaps = []
    for wav_segment in wav_segments:
        overlap = segment_overlap(
            start,
            end,
            float(wav_segment["start"]),
            float(wav_segment["end"]),
        )
        if overlap > 0:
            overlaps.append((wav_segment, overlap))

    if not overlaps:
        return "", 0.0

    overlaps.sort(key=lambda item: (float(item[0]["start"]), float(item[0]["end"])))

    raw_tokens = []
    overlap_by_token = {}
    for wav_segment, overlap in overlaps:
        tokens = normalize_phone_sequence_to_compare(
            [wav_segment.get("phoneme_raw", wav_segment.get("phoneme"))]
        )
        for token in tokens:
            raw_tokens.append(token)
            overlap_by_token[token] = overlap_by_token.get(token, 0.0) + overlap

    collapsed_tokens = collapse_compare_phone_tokens(raw_tokens)
    if not collapsed_tokens:
        return "", 0.0

    if len(collapsed_tokens) == 1:
        real_phone = collapsed_tokens[0]
    else:
        real_phone = max(collapsed_tokens, key=lambda token: overlap_by_token.get(token, 0.0))

    support = round(overlap_by_token.get(real_phone, 0.0) / duration, 3)
    return real_phone, support


def compare_file(auto_path, wav2vec2_dir, transcript_dir, mfa_transcript_dir, output_dir, id_to_code, error_field):
    stem = auto_path.stem
    transcript_path = find_matching_file(transcript_dir, stem, ".txt")
    if transcript_path is None:
        print(f"WARNING: missing transcript for {stem}")
        return None

    mfa_transcript_path = find_matching_file(mfa_transcript_dir, stem, ".txt")
    if mfa_transcript_path and mfa_transcript_path.stat().st_mtime > auto_path.stat().st_mtime:
        print(
            f"WARNING: skip {stem} because {auto_path.name} is older than {mfa_transcript_path.name}. "
            "Run MFA align and tools/04_textgrid_to_json.py again."
        )
        return None

    wav2vec2_path = find_matching_file(wav2vec2_dir, stem, ".json")
    if wav2vec2_path is None:
        print(f"WARNING: missing wav2vec2 raw file for {stem}")
        return None

    data = json.loads(auto_path.read_text(encoding="utf-8"))
    wav_data = json.loads(wav2vec2_path.read_text(encoding="utf-8"))

    aligned_segments = [
        seg
        for seg in data.get("segments", [])
        if not is_silence(seg.get("phoneme_standard", seg.get("phoneme")))
    ]
    wav_segments = [
        seg
        for seg in wav_data.get("segments", [])
        if not is_silence(seg.get("phoneme_raw", seg.get("phoneme")))
    ]
    words = read_transcript(transcript_path)

    output_segments = []
    standard_phonemes = []
    real_phonemes = []
    mismatch_count = 0

    for out_idx, source in enumerate(aligned_segments):
        std_phone = normalize_token_to_compare(source.get("phoneme_standard", source.get("phoneme")))
        if is_silence(std_phone):
            continue

        real_phone, acoustic_support = infer_real_phone(source, wav_segments)
        error_id, error_code = classify_error(std_phone, real_phone, id_to_code)
        error_value = error_code if error_field == "code" and error_code is not None else error_id
        if error_id != "no_error":
            mismatch_count += 1

        standard_phonemes.append(std_phone)
        real_phonemes.append(real_phone)
        source_id = source.get("id", f"{out_idx:03d}_{std_phone}")

        output_segments.append(
            {
                "id": source_id,
                "phoneme_standard": std_phone,
                "phoneme_real": real_phone,
                "phoneme": real_phone,
                "start": source["start"],
                "end": source["end"],
                "error": error_value,
                "error_id": error_id,
                "error_code": error_code,
                "acoustic_support": acoustic_support,
            }
        )

    result = {
        "video_id": stem,
        "transcript": " ".join(words),
        "standard_phonemes": standard_phonemes,
        "real_phonemes": real_phonemes,
        "alignment_distance": mismatch_count,
        "mismatch_count": mismatch_count,
        "unknown_words": [],
        "segments": output_segments,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, len(output_segments), mismatch_count


def main():
    parser = argparse.ArgumentParser(description="Compare MFA-aligned phones with wav2vec2 raw phones.")
    parser.add_argument("--annotation_dir", default="data/annotations/auto")
    parser.add_argument("--wav2vec2_dir", default="data/annotations/wav2vec2_raw")
    parser.add_argument("--transcript_dir", default="data/transcript")
    parser.add_argument("--mfa_transcript_dir", default="data/audio")
    parser.add_argument("--out_dir", default="data/annotations/compare")
    parser.add_argument("--label_map", default="data/label_map.json")
    parser.add_argument("--error_field", choices=["id", "code"], default="id")
    args = parser.parse_args()

    id_to_code, _ = load_label_map(Path(args.label_map))

    annotation_dir = Path(args.annotation_dir)
    wav2vec2_dir = Path(args.wav2vec2_dir)
    transcript_dir = Path(args.transcript_dir)
    mfa_transcript_dir = Path(args.mfa_transcript_dir)
    output_dir = Path(args.out_dir)

    outputs = []
    for auto_path in sorted(annotation_dir.glob("*.json")):
        result = compare_file(
            auto_path,
            wav2vec2_dir,
            transcript_dir,
            mfa_transcript_dir,
            output_dir,
            id_to_code,
            args.error_field,
        )
        if result:
            outputs.append(result)

    for path, segment_count, mismatch_count in outputs:
        print(f"Wrote {path} ({segment_count} segments, {mismatch_count} mismatches)")
    print(f"Done: {len(outputs)} comparison files")


if __name__ == "__main__":
    main()

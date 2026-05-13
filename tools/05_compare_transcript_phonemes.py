"""
Compare MFA-aligned canonical phones against wav2vec2 raw phoneme predictions.

Inputs:
    data/transcript/<audio_name>.txt
    data/annotations/auto/<audio_name>.json
    data/annotations/wav2vec2_raw/<audio_name>.json

Output:
    data/annotations/compare/<audio_name>.json
"""

import argparse
import json
from pathlib import Path

from phoneme_utils import (
    COMBINED_TOKENS,
    is_silence,
    load_dictionary,
    normalize_phone_sequence_to_compare,
    normalize_token_to_compare,
    read_transcript,
    word_to_compare_phones,
)

DELETE_COST = 0.95
INSERT_COST = 0.75
TIME_COST_WEIGHT = 0.35


KNOWN_ERROR_BY_PAIR = {
    ("u", ""): "Thiếu độ tròn môi",
    ("u", "ə"): "Thiếu độ tròn môi",
    ("u", "ʊ"): "Thiếu độ tròn môi",
    ("ʊ", ""): "Thiếu độ tròn môi",
    ("ʊ", "ə"): "Thiếu độ tròn môi",
    ("ɔ", ""): "Thiếu độ tròn môi",
    ("ɔ", "ə"): "Thiếu độ tròn môi",
    ("ɑ", ""): "Hàm không mở đủ sâu",
    ("ɑ", "ə"): "Hàm không mở đủ sâu",
    ("æ", ""): "Hàm không mở đủ sâu",
    ("æ", "ə"): "Hàm không mở đủ sâu",
    ("v", "j"): "Nhầm lẫn V và Y",
    ("v", "i"): "Nhầm lẫn V và Y",
    ("v", "d"): "Nhầm lẫn V và Y",
    ("v", "z"): "Nhầm lẫn V và Y",
    ("p", ""): "Đóng môi cuối từ",
    ("b", ""): "Đóng môi cuối từ",
}

TECHNICAL_ERROR_BY_PAIR = {
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
    fallback = {
        "OK": {"id": "0", "name": "OK", "code": "OK"},
        "OTHER": {"id": "OTHER", "name": "OTHER", "code": "OTHER"},
    }
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = {}
    if isinstance(data, dict):
        for key, item in data.items():
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index", item.get("id", key))
            name = item.get("loai_loi", key)
            if key == "no_error" or str(raw_index) == "0":
                labels["OK"] = {"id": str(raw_index), "name": name, "code": "OK"}
            else:
                labels[name] = {
                    "id": str(raw_index),
                    "name": name,
                    "code": item.get("code", key),
                }
                labels[key] = labels[name]
    labels.setdefault("OK", fallback["OK"])
    labels.setdefault("OTHER", fallback["OTHER"])
    return labels


def classify_error(std_phone, real_phone, labels):
    if std_phone == real_phone:
        item = labels["OK"]
        return item["name"], item["id"], item["code"]

    error_name = KNOWN_ERROR_BY_PAIR.get((std_phone, real_phone), "OTHER")
    item = labels.get(error_name, labels["OTHER"])
    error_code = TECHNICAL_ERROR_BY_PAIR.get((std_phone, real_phone), item.get("code", error_name))
    return item["name"], item["id"], error_code


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


def midpoint(item):
    return (float(item["start"]) + float(item["end"])) / 2


def overlap_support(standard_item, raw_item):
    if raw_item is None:
        return 0.0
    start = float(standard_item["start"])
    end = float(standard_item["end"])
    duration = max(end - start, 1e-6)
    overlap = segment_overlap(start, end, float(raw_item["start"]), float(raw_item["end"]))
    return round(overlap / duration, 3)


def time_cost(standard_item, raw_item):
    std_start = float(standard_item["start"])
    std_end = float(standard_item["end"])
    raw_start = float(raw_item["start"])
    raw_end = float(raw_item["end"])
    std_mid = (std_start + std_end) / 2
    raw_mid = (raw_start + raw_end) / 2
    std_dur = max(std_end - std_start, 1e-6)
    raw_dur = max(raw_end - raw_start, 1e-6)
    outside = max(0.0, abs(std_mid - raw_mid) - ((std_dur + raw_dur) / 2))
    return min(0.8, outside / 1.0) * TIME_COST_WEIGHT


def phone_substitution_cost(std_phone, raw_phone, standard_item, raw_item):
    if std_phone == raw_phone:
        base = 0.0
    elif (std_phone, raw_phone) in KNOWN_ERROR_BY_PAIR or (std_phone, raw_phone) in TECHNICAL_ERROR_BY_PAIR:
        base = 0.65
    else:
        base = 1.0
    return base + time_cost(standard_item, raw_item)


def raw_segment_to_items(wav_segments):
    raw_items = []
    for seg_idx, segment in enumerate(wav_segments):
        tokens = normalize_phone_sequence_to_compare(
            [segment.get("phoneme_raw", segment.get("phoneme"))]
        )
        for token in tokens:
            raw_items.append(
                {
                    "phone": token,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "raw_index": seg_idx,
                    "raw_segment_ids": [segment.get("id", str(seg_idx))],
                }
            )

    deduped = []
    for item in raw_items:
        if deduped and deduped[-1]["phone"] == item["phone"]:
            deduped[-1]["end"] = item["end"]
            deduped[-1]["raw_segment_ids"].extend(item["raw_segment_ids"])
            continue
        deduped.append(item)

    collapsed = []
    i = 0
    while i < len(deduped):
        pair = tuple(item["phone"] for item in deduped[i : i + 2])
        if len(pair) == 2 and pair in COMBINED_TOKENS:
            first = deduped[i]
            second = deduped[i + 1]
            collapsed.append(
                {
                    "phone": COMBINED_TOKENS[pair],
                    "start": first["start"],
                    "end": second["end"],
                    "raw_index": first["raw_index"],
                    "raw_segment_ids": first["raw_segment_ids"] + second["raw_segment_ids"],
                }
            )
            i += 2
            continue
        collapsed.append(deduped[i])
        i += 1

    return collapsed


def build_standard_items(aligned_segments, words, dictionary):
    standard_items = []
    cursor = 0

    for word_index, word in enumerate(words):
        word_phones, _ = word_to_compare_phones(word, dictionary)
        for phoneme_index in range(len(word_phones)):
            if cursor >= len(aligned_segments):
                break
            segment = aligned_segments[cursor]
            std_phone = normalize_token_to_compare(
                segment.get("phoneme_standard", segment.get("phoneme"))
            )
            if not is_silence(std_phone):
                standard_items.append(
                    {
                        "source": segment,
                        "phone": std_phone,
                        "start": float(segment["start"]),
                        "end": float(segment["end"]),
                        "word": word,
                        "word_index": word_index,
                        "phoneme_index_in_word": phoneme_index,
                    }
                )
            cursor += 1

    while cursor < len(aligned_segments):
        segment = aligned_segments[cursor]
        std_phone = normalize_token_to_compare(segment.get("phoneme_standard", segment.get("phoneme")))
        if not is_silence(std_phone):
            standard_items.append(
                {
                    "source": segment,
                    "phone": std_phone,
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "word": None,
                    "word_index": None,
                    "phoneme_index_in_word": None,
                }
            )
        cursor += 1

    return standard_items


def align_phone_sequences(standard_items, raw_items):
    n = len(standard_items)
    m = len(raw_items)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + DELETE_COST
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + INSERT_COST
        back[0][j] = "insert"

    for i in range(1, n + 1):
        standard_item = standard_items[i - 1]
        for j in range(1, m + 1):
            raw_item = raw_items[j - 1]
            substitute = dp[i - 1][j - 1] + phone_substitution_cost(
                standard_item["phone"],
                raw_item["phone"],
                standard_item,
                raw_item,
            )
            delete = dp[i - 1][j] + DELETE_COST
            insert = dp[i][j - 1] + INSERT_COST

            best_cost = substitute
            best_op = "match" if standard_item["phone"] == raw_item["phone"] else "substitute"
            if delete < best_cost:
                best_cost = delete
                best_op = "delete"
            if insert < best_cost:
                best_cost = insert
                best_op = "insert"

            dp[i][j] = best_cost
            back[i][j] = best_op

    assignments = [None] * n
    insertions = []
    i = n
    j = m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"match", "substitute"}:
            assignments[i - 1] = {
                "raw": raw_items[j - 1],
                "op": op,
                "cost": round(
                    phone_substitution_cost(
                        standard_items[i - 1]["phone"],
                        raw_items[j - 1]["phone"],
                        standard_items[i - 1],
                        raw_items[j - 1],
                    ),
                    3,
                ),
            }
            i -= 1
            j -= 1
        elif op == "delete":
            assignments[i - 1] = {"raw": None, "op": "delete", "cost": DELETE_COST}
            i -= 1
        elif op == "insert":
            insertions.append(raw_items[j - 1])
            j -= 1
        else:
            break

    insertions.reverse()
    return assignments, insertions, round(dp[n][m], 3)


def align_phone_sequences_by_word(standard_items, raw_items):
    groups = []
    current_key = object()
    for index, item in enumerate(standard_items):
        key = item.get("word_index")
        if key is None:
            return align_phone_sequences(standard_items, raw_items)
        if key != current_key:
            groups.append(
                {
                    "word_index": key,
                    "word": item.get("word"),
                    "indices": [],
                    "start": item["start"],
                    "end": item["end"],
                }
            )
            current_key = key
        groups[-1]["indices"].append(index)
        groups[-1]["start"] = min(groups[-1]["start"], item["start"])
        groups[-1]["end"] = max(groups[-1]["end"], item["end"])

    if not groups:
        return align_phone_sequences(standard_items, raw_items)

    boundaries = []
    for idx in range(len(groups) - 1):
        boundaries.append((groups[idx]["end"] + groups[idx + 1]["start"]) / 2)

    raw_groups = [[] for _ in groups]
    group_idx = 0
    for raw_item in raw_items:
        raw_mid = midpoint(raw_item)
        while group_idx < len(boundaries) and raw_mid > boundaries[group_idx]:
            group_idx += 1
        raw_groups[group_idx].append(raw_item)

    assignments = [None] * len(standard_items)
    insertions = []
    total_score = 0.0
    for idx, group in enumerate(groups):
        group_items = [standard_items[item_index] for item_index in group["indices"]]
        group_assignments, group_insertions, group_score = align_phone_sequences(
            group_items,
            raw_groups[idx],
        )
        total_score += group_score
        for item_index, assignment in zip(group["indices"], group_assignments):
            assignments[item_index] = assignment
        for insertion in group_insertions:
            insertion = dict(insertion)
            insertion["word"] = group["word"]
            insertion["word_index"] = group["word_index"]
            insertions.append(insertion)

    return assignments, insertions, round(total_score, 3)


def compare_file(auto_path, wav2vec2_dir, transcript_dir, mfa_transcript_dir, output_dir, labels, dictionary):
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

    standard_items = build_standard_items(aligned_segments, words, dictionary)
    raw_items = raw_segment_to_items(wav_segments)
    assignments, insertions, alignment_score = align_phone_sequences_by_word(standard_items, raw_items)

    output_segments = []
    standard_phonemes = []
    real_phonemes = []
    mismatch_count = 0

    for out_idx, (standard_item, assignment) in enumerate(zip(standard_items, assignments)):
        source = standard_item["source"]
        std_phone = standard_item["phone"]
        raw_item = assignment["raw"] if assignment else None
        real_phone = raw_item["phone"] if raw_item else ""
        acoustic_support = overlap_support(standard_item, raw_item)
        error_name, error_id, error_code = classify_error(std_phone, real_phone, labels)
        if error_id != labels["OK"]["id"]:
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
                "error": error_name,
                "error_id": error_id,
                "error_code": error_code,
                "acoustic_support": acoustic_support,
                "alignment_op": assignment["op"] if assignment else "delete",
                "alignment_cost": assignment["cost"] if assignment else DELETE_COST,
                "word": standard_item["word"],
                "word_index": standard_item["word_index"],
                "phoneme_index_in_word": standard_item["phoneme_index_in_word"],
                "raw_start": raw_item["start"] if raw_item else None,
                "raw_end": raw_item["end"] if raw_item else None,
                "raw_segment_ids": raw_item["raw_segment_ids"] if raw_item else [],
            }
        )

    result = {
        "audio_id": stem,
        "transcript": " ".join(words),
        "standard_phonemes": standard_phonemes,
        "real_phonemes": real_phonemes,
        "alignment_distance": mismatch_count,
        "alignment_method": "word_sequence_dp",
        "alignment_score": alignment_score,
        "raw_phonemes": [item["phone"] for item in raw_items],
        "raw_insertions": [
            {
                "phoneme": item["phone"],
                "start": item["start"],
                "end": item["end"],
                "raw_segment_ids": item["raw_segment_ids"],
                "word": item.get("word"),
                "word_index": item.get("word_index"),
            }
            for item in insertions
        ],
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
    parser.add_argument("--auto-dir", default="data/annotations/auto")
    parser.add_argument("--wav2vec2-dir", default="data/annotations/wav2vec2_raw")
    parser.add_argument("--transcript-dir", default="data/transcript")
    parser.add_argument("--mfa-transcript-dir", default="data/audio")
    parser.add_argument("--output-dir", default="data/annotations/compare")
    parser.add_argument("--label-map", default="data/label_map.json")
    parser.add_argument("--dictionary", default="custom_mfa.dict")
    args = parser.parse_args()

    labels = load_label_map(Path(args.label_map))
    dictionary = load_dictionary(Path(args.dictionary))

    auto_dir = Path(args.auto_dir)
    wav2vec2_dir = Path(args.wav2vec2_dir)
    transcript_dir = Path(args.transcript_dir)
    mfa_transcript_dir = Path(args.mfa_transcript_dir)
    output_dir = Path(args.output_dir)

    outputs = []
    for auto_path in sorted(auto_dir.glob("*.json")):
        result = compare_file(
            auto_path,
            wav2vec2_dir,
            transcript_dir,
            mfa_transcript_dir,
            output_dir,
            labels,
            dictionary,
        )
        if result:
            outputs.append(result)

    for path, segment_count, mismatch_count in outputs:
        print(f"Wrote {path} ({segment_count} segments, {mismatch_count} mismatches)")
    print(f"Done: {len(outputs)} comparison files")


if __name__ == "__main__":
    main()

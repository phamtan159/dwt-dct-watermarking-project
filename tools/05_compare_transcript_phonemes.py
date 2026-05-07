"""
Compare expected transcript phonemes against real aligned phonemes.

Inputs:
    data/transcript/<audio_name>.txt          transcript text
    data/annotations/auto/<audio_name>.json  real phonemes + timing

Output:
    data/annotations/compare/<audio_name>.json

Each output segment keeps the real timing and adds:
    phoneme_standard: expected phoneme from transcript
    phoneme_real: phoneme detected/aligned from the audio
    error: trainable error label ("OK", "th_to_t", "z_to_s", "OTHER", ...)
    error_code: numeric key from data/label_map.json when available

For best accuracy, add a word->phoneme dictionary file and pass --dictionary.
Without one, this tool uses a small built-in English lexicon plus simple fallback
letter rules so the pipeline can run without external services.
"""

import argparse
import json
import re
from pathlib import Path


BUILTIN_LEXICON = {
    "a": ["ə"],
    "bus": ["b", "ə", "s"],
    "cat": ["k", "æ", "t"],
    "check": ["tʃ", "ɛ", "k"],
    "for": ["f", "ə", "r"],
    "her": ["h", "ə", "r"],
    "is": ["ɪ", "z"],
    "it": ["ɪ", "t"],
    "job": ["dʒ", "ɑ", "b"],
    "likes": ["l", "a", "ɪ", "k", "s"],
    "mother": ["m", "ə", "ð", "ə", "r"],
    "my": ["m", "a", "ɪ"],
    "new": ["n", "u"],
    "on": ["ɑ", "n"],
    "out": ["a", "ʊ", "t"],
    "pen": ["p", "ɛ", "n"],
    "perfect": ["p", "ə", "r", "f", "ɛ", "k", "t"],
    "runs": ["r", "ə", "n", "z"],
    "she": ["ʃ", "i"],
    "sit": ["s", "ɪ", "t"],
    "so": ["s", "o"],
    "the": ["ð", "ə"],
    "thin": ["θ", "ɪ", "n"],
    "this": ["ð", "ɪ", "s"],
    "to": ["t", "ə"],
    "today": ["t", "ə", "d", "e", "ɪ"],
    "vest": ["v", "ɛ", "s", "t"],
    "with": ["w", "ɪ", "θ"],
}

ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "a ʊ",
    "AY": "a ɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ə r", "EY": "e ɪ", "F": "f", "G": "g",
    "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "o",
    "OY": "ɔ ɪ", "P": "p", "R": "r", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

LETTER_FALLBACK = {
    "a": "ə", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f",
    "g": "g", "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l",
    "m": "m", "n": "n", "o": "ɑ", "p": "p", "q": "k", "r": "r",
    "s": "s", "t": "t", "u": "ʊ", "v": "v", "w": "w", "x": "k s",
    "y": "j", "z": "z",
}

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

SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>", "<sil>", "SIL"}


def clean_word(word):
    return re.sub(r"[^a-z']", "", word.lower())


def read_transcript(path):
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\u200b", " ")
    return [clean_word(w) for w in re.findall(r"[A-Za-z']+", text) if clean_word(w)]


def normalize_phone(phone):
    return str(phone or "").strip()


def is_silence(phone):
    return normalize_phone(phone) in SILENCE_PHONES


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


def load_dictionary(path):
    lexicon = {}
    if not path or not path.exists():
        return lexicon

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        word = clean_word(parts[0].split("(")[0])
        phones = []
        for token in parts[1:]:
            token = re.sub(r"\d", "", token)
            if token in ARPABET_TO_IPA:
                phones.extend(ARPABET_TO_IPA[token].split())
            else:
                phones.append(token)
        if word and phones:
            lexicon.setdefault(word, phones)
    return lexicon


def fallback_g2p(word):
    phones = []
    i = 0
    while i < len(word):
        chunk = word[i:i + 2]
        if chunk == "th":
            phones.append("θ")
            i += 2
        elif chunk == "sh":
            phones.append("ʃ")
            i += 2
        elif chunk == "ch":
            phones.append("tʃ")
            i += 2
        elif chunk in {"ng", "ck"}:
            phones.append("ŋ" if chunk == "ng" else "k")
            i += 2
        else:
            mapped = LETTER_FALLBACK.get(word[i])
            if mapped:
                phones.extend(mapped.split())
            i += 1
    return phones


def transcript_to_phonemes(words, dictionary):
    phones = []
    unknown_words = []
    for word in words:
        if word in dictionary:
            phones.extend(dictionary[word])
        elif word in BUILTIN_LEXICON:
            phones.extend(BUILTIN_LEXICON[word])
        else:
            unknown_words.append(word)
            phones.extend(fallback_g2p(word))
    return phones, unknown_words


def align_sequences(standard, real):
    n, m = len(standard), len(real)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if standard[i - 1] == real[j - 1] else 1
            choices = [
                (dp[i - 1][j - 1] + sub_cost, "match"),
                (dp[i - 1][j] + 1, "del"),
                (dp[i][j - 1] + 1, "ins"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "match":
            aligned.append((standard[i - 1], real[j - 1], j - 1))
            i -= 1
            j -= 1
        elif move == "del":
            aligned.append((standard[i - 1], "", None))
            i -= 1
        else:
            aligned.append(("", real[j - 1], j - 1))
            j -= 1

    return list(reversed(aligned)), dp[n][m]


def classify_error(std_phone, real_phone, id_to_code):
    if std_phone == real_phone:
        return "OK", "0"
    error_id = KNOWN_ERROR_BY_PAIR.get((std_phone, real_phone), "OTHER")
    return error_id, id_to_code.get(error_id)


def estimate_deleted_timing(real_segments, prev_real_index, next_real_index):
    prev_seg = real_segments[prev_real_index] if prev_real_index is not None and prev_real_index >= 0 else None
    next_seg = real_segments[next_real_index] if next_real_index is not None and next_real_index < len(real_segments) else None
    if prev_seg and next_seg:
        start = float(prev_seg["end"])
        end = float(next_seg["start"])
    elif prev_seg:
        start = end = float(prev_seg["end"])
    elif next_seg:
        start = end = float(next_seg["start"])
    else:
        start = end = 0.0
    return start, end


def find_matching_file(directory, stem):
    exact = directory / f"{stem}.txt"
    if exact.exists():
        return exact
    lower_stem = stem.lower()
    for path in directory.glob("*.txt"):
        if path.stem.lower() == lower_stem:
            return path
    return None


def compare_file(auto_path, transcript_dir, output_dir, dictionary, id_to_code, error_field):
    stem = auto_path.stem
    transcript_path = find_matching_file(transcript_dir, stem)
    if transcript_path is None:
        print(f"WARNING: missing transcript for {stem}")
        return None

    data = json.loads(auto_path.read_text(encoding="utf-8"))
    real_segments = [
        seg for seg in data.get("segments", [])
        if not is_silence(seg.get("phoneme_real", seg.get("phoneme")))
    ]
    real_phones = [normalize_phone(seg.get("phoneme_real", seg.get("phoneme"))) for seg in real_segments]
    words = read_transcript(transcript_path)
    standard_phones, unknown_words = transcript_to_phonemes(words, dictionary)
    alignment, distance = align_sequences(standard_phones, real_phones)

    output_segments = []
    for out_idx, (std_phone, real_phone, real_index) in enumerate(alignment):
        if is_silence(std_phone):
            # Real-only insertions are usually pauses, fillers, or alignment noise.
            # They do not teach a useful standard -> real pronunciation relation.
            continue

        if real_index is None:
            prev_real_index = next(
                (item[2] for item in reversed(alignment[:out_idx]) if item[2] is not None),
                None
            )
            next_real_index = next(
                (item[2] for item in alignment[out_idx + 1:] if item[2] is not None),
                None
            )
            start, end = estimate_deleted_timing(real_segments, prev_real_index, next_real_index)
            source_id = f"{out_idx:03d}_{std_phone}_missing"
        else:
            source = real_segments[real_index]
            start = source["start"]
            end = source["end"]
            source_id = source.get("id", f"{out_idx:03d}_{real_phone}")

        error_id, error_code = classify_error(std_phone, real_phone, id_to_code)
        error_value = error_code if error_field == "code" and error_code is not None else error_id

        output_segments.append({
            "id": source_id,
            "phoneme_standard": std_phone,
            "phoneme_real": real_phone,
            "phoneme": real_phone,
            "start": start,
            "end": end,
            "error": error_value,
            "error_id": error_id,
            "error_code": error_code,
        })

    result = {
        "audio_id": stem,
        "transcript": " ".join(words),
        "standard_phonemes": standard_phones,
        "real_phonemes": real_phones,
        "alignment_distance": distance,
        "unknown_words": sorted(set(unknown_words)),
        "segments": output_segments,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, len(output_segments), len(unknown_words)


def main():
    parser = argparse.ArgumentParser(description="Compare transcript-standard phonemes with real aligned phonemes.")
    parser.add_argument("--auto-dir", default="data/annotations/auto")
    parser.add_argument("--transcript-dir", default="data/transcript")
    parser.add_argument("--output-dir", default="data/annotations/compare")
    parser.add_argument("--dictionary", default=None, help="Optional word-to-phoneme dictionary.")
    parser.add_argument("--label-map", default="data/label_map.json")
    parser.add_argument("--error-field", choices=["id", "code"], default="id")
    args = parser.parse_args()

    dictionary = load_dictionary(Path(args.dictionary)) if args.dictionary else {}
    id_to_code, _ = load_label_map(Path(args.label_map))

    auto_dir = Path(args.auto_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir = Path(args.output_dir)

    outputs = []
    for auto_path in sorted(auto_dir.glob("*.json")):
        result = compare_file(
            auto_path,
            transcript_dir,
            output_dir,
            dictionary,
            id_to_code,
            args.error_field,
        )
        if result:
            outputs.append(result)

    for path, segment_count, unknown_count in outputs:
        print(f"Wrote {path} ({segment_count} segments, {unknown_count} unknown words)")
    print(f"Done: {len(outputs)} comparison files")


if __name__ == "__main__":
    main()

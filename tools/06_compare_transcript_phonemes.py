"""
07_compare_transcript_phonemes.py

Compare actual phonemes in data/annotations/auto/*.json with the standard
transcript in data/transcript/<audio_name>.txt, then write aligned training
annotations to data/annotations/compare/*.json.

Optional dictionary format:
  WORD PH1 PH2 PH3
  word<TAB>PH1 PH2 PH3
"""

import argparse
import json
import os
import re
from pathlib import Path


DEFAULT_ANNOTATION_DIR = "data/annotations/auto"
DEFAULT_TRANSCRIPT_DIR = "data/transcript"
DEFAULT_OUTPUT_DIR = "data/annotations/compare"
SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>", "<sil>"}

BUILTIN_LEXICON = {
    "a": ["ə"],
    "and": ["æ", "n", "d"],
    "at": ["æ", "t"],
    "blue": ["b", "l", "u"],
    "bob": ["b", "ɑ", "b"],
    "cat": ["k", "æ", "t"],
    "club": ["k", "l", "ʌ", "b"],
    "door": ["d", "ɔ", "r"],
    "fan": ["f", "æ", "n"],
    "fast": ["f", "æ", "s", "t"],
    "father": ["f", "ɑ", "ð", "ə", "r"],
    "find": ["f", "aɪ", "n", "d"],
    "fly": ["f", "l", "aɪ"],
    "food": ["f", "u", "d"],
    "had": ["h", "æ", "d"],
    "he": ["h", "i"],
    "help": ["h", "ɛ", "l", "p"],
    "job": ["dʒ", "ɑ", "b"],
    "like": ["l", "aɪ", "k"],
    "looked": ["l", "ʊ", "k", "t"],
    "map": ["m", "æ", "p"],
    "my": ["m", "aɪ"],
    "near": ["n", "ɪ", "r"],
    "please": ["p", "l", "i", "z"],
    "star": ["s", "t", "ɑ", "r"],
    "stop": ["s", "t", "ɑ", "p"],
    "the": ["ð", "ə"],
    "to": ["t", "u"],
    "took": ["t", "ʊ", "k"],
    "very": ["v", "ɛ", "r", "i"],
    "voice": ["v", "ɔɪ", "s"],
}

PHONE_ALIASES = {
    "ɝ": "ə",
    "ɚ": "ə",
    "ʧ": "tʃ",
    "ʤ": "dʒ",
    "oʊ": "ɔ",
    "ɔː": "ɔ",
    "ɑː": "ɑ",
    "uː": "u",
}


def repair_mojibake(text):
    if not isinstance(text, str):
        return ""
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    return repaired if repaired else text


def normalize_phone(phone):
    phone = repair_mojibake(phone).strip()
    phone = re.sub(r"\d+", "", phone)
    return PHONE_ALIASES.get(phone, phone)


def is_silence_phone(phone):
    if phone is None:
        return True
    return normalize_phone(phone).strip().lower() in SILENCE_PHONES


def tokenize_transcript(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def load_dictionary(path):
    lexicon = {word: phones[:] for word, phones in BUILTIN_LEXICON.items()}
    if not path:
        return lexicon

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                word, phones = line.split("\t", 1)
                parts = phones.split()
            else:
                fields = line.split()
                word, parts = fields[0], fields[1:]
            if parts:
                lexicon[word.lower()] = [normalize_phone(phone) for phone in parts]
    return lexicon


def transcript_to_phones(text, lexicon):
    phones = []
    missing = []
    for word in tokenize_transcript(text):
        word_phones = lexicon.get(word)
        if not word_phones:
            missing.append(word)
            continue
        phones.extend(word_phones)
    return phones, missing


def load_actual_segments(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg in data.get("segments", []):
        real = normalize_phone(seg.get("phoneme_real") or seg.get("phoneme"))
        if is_silence_phone(real):
            continue
        segments.append(
            {
                "id": seg.get("id"),
                "phoneme_real": real,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
            }
        )
    return segments


def substitution_cost(std_phone, real_phone):
    return 0 if std_phone == real_phone else 1


def align_phones(standard_phones, actual_phones):
    n = len(standard_phones)
    m = len(actual_phones)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete_standard"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert_actual"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i - 1][j - 1] + substitution_cost(standard_phones[i - 1], actual_phones[j - 1]), "match_or_sub"),
                (dp[i - 1][j] + 1, "delete_standard"),
                (dp[i][j - 1] + 1, "insert_actual"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda item: item[0])

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "match_or_sub":
            pairs.append((standard_phones[i - 1], actual_phones[j - 1]))
            i -= 1
            j -= 1
        elif move == "delete_standard":
            pairs.append((standard_phones[i - 1], ""))
            i -= 1
        elif move == "insert_actual":
            pairs.append((None, actual_phones[j - 1]))
            j -= 1
        else:
            break

    pairs.reverse()
    return pairs


def build_compare_segments(standard_phones, actual_segments):
    actual_phones = [seg["phoneme_real"] for seg in actual_segments]
    pairs = align_phones(standard_phones, actual_phones)

    output = []
    actual_index = 0
    for standard_phone, real_phone in pairs:
        standard_is_silence = is_silence_phone(standard_phone)
        real_is_silence = is_silence_phone(real_phone)

        # Extra actual phones aligned against no standard phone are usually
        # pauses, fillers, or alignment noise, so keep them out of compare data.
        if standard_is_silence:
            if not real_is_silence:
                actual_index += 1
            continue

        src = None
        if not real_is_silence:
            if actual_index >= len(actual_segments):
                break
            src = actual_segments[actual_index]
            actual_index += 1

        if real_is_silence:
            error_id = f"{standard_phone}_omitted"
            seg_id = f"{len(output):03d}_{standard_phone}_omitted"
        else:
            error_id = "no_error" if standard_phone == real_phone else "OTHER"
            seg_id = f"{len(output):03d}_{real_phone}"
        output.append(
            {
                "id": seg_id,
                "phoneme_standard": standard_phone,
                "phoneme_real": "" if real_is_silence else real_phone,
                "phoneme": standard_phone if real_is_silence else real_phone,
                "start": round(src["start"], 3) if src else None,
                "end": round(src["end"], 3) if src else None,
                "error": error_id,
                "error_id": error_id,
                "error_code": None,
            }
        )

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_dir", default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--transcript_dir", default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dictionary", default=None)
    args = parser.parse_args()

    lexicon = load_dictionary(args.dictionary)
    os.makedirs(args.out_dir, exist_ok=True)

    written = 0
    for annotation_path in sorted(Path(args.annotation_dir).glob("*.json")):
        name = annotation_path.stem
        transcript_path = Path(args.transcript_dir) / f"{name}.txt"
        if not transcript_path.exists():
            print(f"WARNING: missing transcript for {name}: {transcript_path}")
            continue

        transcript = transcript_path.read_text(encoding="utf-8")
        standard_phones, missing_words = transcript_to_phones(transcript, lexicon)
        if missing_words:
            print(f"WARNING: {name}: missing dictionary words: {', '.join(sorted(set(missing_words)))}")
        if not standard_phones:
            print(f"WARNING: {name}: no standard phones; skipped")
            continue

        actual_segments = load_actual_segments(annotation_path)
        if not actual_segments:
            print(f"WARNING: {name}: no actual phoneme segments; skipped")
            continue

        compared = build_compare_segments(standard_phones, actual_segments)
        out_path = Path(args.out_dir) / f"{name}.json"
        out_path.write_text(json.dumps({"segments": compared}, indent=2, ensure_ascii=False), encoding="utf-8")

        errors = sum(1 for seg in compared if seg["error_id"] != "no_error")
        print(f"{name}: wrote {len(compared)} segments to {out_path} (errors={errors})")
        written += 1

    print(f"\nDone. Wrote {written} compare files to {args.out_dir}")


if __name__ == "__main__":
    main()


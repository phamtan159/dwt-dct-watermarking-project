"""
Shared phoneme utilities for MFA preparation and comparison.
"""

from __future__ import annotations

import re
from pathlib import Path


SILENCE_PHONES = {"", " ", "|", "sil", "sp", "spn", "<eps>", "<sil>", "SIL"}

# Canonical comparison phone set. MFA-specific phones are converted into this
# space before being compared against acoustic-model predictions.
BUILTIN_LEXICON = {
    "a": ["ə"],
    "apple": ["æ", "p", "ə", "l"],
    "around": ["ə", "r", "aw", "n", "d"],
    "back": ["b", "æ", "k"],
    "bags": ["b", "æ", "g", "z"],
    "banana": ["b", "ə", "n", "æ", "n", "ə"],
    "bit": ["b", "ɪ", "t"],
    "brother": ["b", "r", "ə", "ð", "ə", "r"],
    "bus": ["b", "ə", "s"],
    "cat": ["k", "æ", "t"],
    "check": ["tʃ", "ɛ", "k"],
    "class": ["k", "l", "æ", "s"],
    "easy": ["i", "z", "i"],
    "eat": ["i", "t"],
    "face": ["f", "ej", "s"],
    "for": ["f", "ə", "r"],
    "george": ["dʒ", "ɔ", "r", "dʒ"],
    "get": ["g", "ɛ", "t"],
    "glass": ["g", "l", "æ", "s"],
    "her": ["h", "ə", "r"],
    "is": ["ɪ", "z"],
    "it": ["ɪ", "t"],
    "job": ["dʒ", "ɑ", "b"],
    "juice": ["dʒ", "u", "s"],
    "june": ["dʒ", "u", "n"],
    "late": ["l", "ej", "t"],
    "like": ["l", "aj", "k"],
    "likes": ["l", "aj", "k", "s"],
    "map": ["m", "æ", "p"],
    "mother": ["m", "ə", "ð", "ə", "r"],
    "my": ["m", "aj"],
    "new": ["n", "u"],
    "nice": ["n", "aj", "s"],
    "off": ["ɔ", "f"],
    "on": ["ɑ", "n"],
    "out": ["aw", "t"],
    "paper": ["p", "ej", "p", "ə", "r"],
    "pen": ["p", "ɛ", "n"],
    "perfect": ["p", "ə", "r", "f", "ɛ", "k", "t"],
    "pick": ["p", "ɪ", "k"],
    "pig": ["p", "ɪ", "g"],
    "please": ["p", "l", "i", "z"],
    "present": ["p", "r", "ɛ", "z", "ə", "n", "t"],
    "pressure": ["p", "r", "ɛ", "ʃ", "ə", "r"],
    "put": ["p", "ʊ", "t"],
    "rain": ["r", "ej", "n"],
    "red": ["r", "ɛ", "d"],
    "rice": ["r", "aj", "s"],
    "right": ["r", "aj", "t"],
    "river": ["r", "ɪ", "v", "ə", "r"],
    "rose": ["r", "ow", "z"],
    "run": ["r", "ə", "n"],
    "runs": ["r", "ə", "n", "z"],
    "seven": ["s", "ɛ", "v", "ə", "n"],
    "seat": ["s", "i", "t"],
    "she": ["ʃ", "i"],
    "sheep": ["ʃ", "i", "p"],
    "shirt": ["ʃ", "ə", "r", "t"],
    "shop": ["ʃ", "ɑ", "p"],
    "sit": ["s", "ɪ", "t"],
    "so": ["s", "ow"],
    "speak": ["s", "p", "i", "k"],
    "sugar": ["ʃ", "ʊ", "g", "ə", "r"],
    "thank": ["θ", "æ", "ŋ", "k"],
    "that": ["ð", "æ", "t"],
    "the": ["ð", "ə"],
    "thin": ["θ", "ɪ", "n"],
    "think": ["θ", "ɪ", "ŋ", "k"],
    "this": ["ð", "ɪ", "s"],
    "those": ["ð", "ow", "z"],
    "through": ["θ", "r", "u"],
    "three": ["θ", "r", "i"],
    "thumb": ["θ", "ə", "m"],
    "to": ["t", "ə"],
    "today": ["t", "ə", "d", "ej"],
    "turn": ["t", "ə", "r", "n"],
    "up": ["ə", "p"],
    "very": ["v", "ɛ", "r", "i"],
    "vest": ["v", "ɛ", "s", "t"],
    "view": ["v", "j", "u"],
    "visit": ["v", "ɪ", "z", "ɪ", "t"],
    "voice": ["v", "ɔj", "s"],
    "vote": ["v", "ow", "t"],
    "white": ["w", "aj", "t"],
    "with": ["w", "ɪ", "θ"],
}

ARPABET_TO_COMPARE = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɔ",
    "AW": "aw",
    "AY": "aj",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ə r",
    "EY": "ej",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "ow",
    "OY": "ɔj",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

LETTER_FALLBACK = {
    "a": "ə",
    "b": "b",
    "c": "k",
    "d": "d",
    "e": "ɛ",
    "f": "f",
    "g": "g",
    "h": "h",
    "i": "ɪ",
    "j": "dʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "ɔ",
    "p": "p",
    "q": "k",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "ʊ",
    "v": "v",
    "w": "w",
    "x": "k s",
    "y": "j",
    "z": "z",
}

COMPARE_TO_MFA = {
    "g": "ɡ",
    "r": "ɹ",
}

RAW_TO_COMPARE = {
    "ɡ": "g",
    "ɹ": "r",
    "ʧ": "tʃ",
    "ʤ": "dʒ",
    "t͡ʃ": "tʃ",
    "d͡ʒ": "dʒ",
    "o": "ow",
    "e": "ej",
    "eɪ": "ej",
    "oʊ": "ow",
    "ɔɪ": "ɔj",
    "aɪ": "aj",
    "aʊ": "aw",
    "ɑː": "ɑ",
    "ɔː": "ɔ",
    "eɪ": "ej",
    "iː": "i",
    "oʊ": "ow",
    "th": "θ",
    "uː": "u",
    "ɚ": "ə r",
    "ᵻ": "ɪ",
}

COMBINED_TOKENS = {
    ("a", "ɪ"): "aj",
    ("a", "j"): "aj",
    ("a", "ʊ"): "aw",
    ("e", "ɪ"): "ej",
    ("o", "ʊ"): "ow",
    ("ɔ", "ɪ"): "ɔj",
    ("d", "ʒ"): "dʒ",
    ("t", "ʃ"): "tʃ",
}


def clean_word(word: str) -> str:
    return re.sub(r"[^a-z']", "", str(word or "").lower())


def read_transcript(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\u200b", " ")
    return [clean_word(w) for w in re.findall(r"[A-Za-z']+", text) if clean_word(w)]


def normalize_phone(phone: str | None) -> str:
    return str(phone or "").strip()


def is_silence(phone: str | None) -> bool:
    return normalize_phone(phone) in SILENCE_PHONES


def normalize_token_to_compare(token: str | None) -> str:
    token = normalize_phone(token)
    token = re.sub(r"\d+", "", token)
    if not token or token in SILENCE_PHONES:
        return ""
    return RAW_TO_COMPARE.get(token, token)


def load_dictionary(path: Path | None) -> dict[str, list[str]]:
    lexicon: dict[str, list[str]] = {}
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
        phones: list[str] = []
        for token in parts[1:]:
            token = re.sub(r"\d", "", token)
            mapped = ARPABET_TO_COMPARE.get(token)
            if mapped:
                phones.extend(mapped.split())
            else:
                normalized = normalize_token_to_compare(token)
                if normalized:
                    phones.append(normalized)

        if word and phones:
            lexicon.setdefault(word, phones)
    return lexicon


def fallback_g2p(word: str) -> list[str]:
    phones: list[str] = []
    i = 0
    while i < len(word):
        chunk = word[i : i + 2]
        if chunk == "th":
            phones.append("θ")
            i += 2
        elif chunk == "sh":
            phones.append("ʃ")
            i += 2
        elif chunk == "ch":
            phones.append("tʃ")
            i += 2
        elif chunk == "ph":
            phones.append("f")
            i += 2
        elif chunk == "ng":
            phones.append("ŋ")
            i += 2
        elif chunk == "ck":
            phones.append("k")
            i += 2
        else:
            mapped = LETTER_FALLBACK.get(word[i])
            if mapped:
                phones.extend(mapped.split())
            i += 1
    return phones


def word_to_compare_phones(word: str, dictionary: dict[str, list[str]]) -> tuple[list[str], bool]:
    if word in dictionary:
        return list(dictionary[word]), False
    if word in BUILTIN_LEXICON:
        return list(BUILTIN_LEXICON[word]), False
    return fallback_g2p(word), True


def transcript_to_compare_phones(
    words: list[str], dictionary: dict[str, list[str]]
) -> tuple[list[str], list[str]]:
    phones: list[str] = []
    unknown_words: list[str] = []
    for word in words:
        word_phones, used_fallback = word_to_compare_phones(word, dictionary)
        phones.extend(word_phones)
        if used_fallback:
            unknown_words.append(word)
    return phones, unknown_words


def compare_phone_to_mfa(phone: str) -> str:
    return COMPARE_TO_MFA.get(phone, phone)


def compare_phones_to_mfa(phones: list[str]) -> list[str]:
    return [compare_phone_to_mfa(phone) for phone in phones]


def normalize_phone_sequence_to_compare(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    for token in tokens:
        mapped = normalize_token_to_compare(token)
        if mapped:
            normalized.extend(mapped.split())
    return normalized


def collapse_compare_phone_tokens(tokens: list[str]) -> list[str]:
    deduped: list[str] = []
    for token in tokens:
        if token and (not deduped or deduped[-1] != token):
            deduped.append(token)

    collapsed: list[str] = []
    i = 0
    while i < len(deduped):
        pair = tuple(deduped[i : i + 2])
        if len(pair) == 2 and pair in COMBINED_TOKENS:
            collapsed.append(COMBINED_TOKENS[pair])
            i += 2
            continue
        collapsed.append(deduped[i])
        i += 1
    return collapsed

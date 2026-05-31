"""
Prepare canonical transcript files and a custom lexicon for MFA.

Input:
    data/transcript/<speaker>/*.txt
    data/audio/<speaker>/*.wav

Output:
    data/audio/<speaker>/*.txt
    custom_mfa.dict
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phoneme_utils import (
    compare_phones_to_mfa,
    load_dictionary,
    read_transcript,
    word_to_compare_phones,
)
from speaker_paths import find_matching_file, iter_speaker_files, relative_id, warn_root_level_files


def find_transcript(transcript_dir: Path, rel_id: str) -> Path | None:
    exact = transcript_dir / Path(rel_id).with_suffix(".txt")
    if exact.exists():
        return exact

    flat = transcript_dir / f"{Path(rel_id).name}.txt"
    if flat.exists():
        return flat

    return find_matching_file(transcript_dir, rel_id, ".txt")


def write_text_preserving_requested_case(path: Path, text: str) -> None:
    for existing in path.parent.glob("*.txt"):
        if existing.name.lower() == path.name.lower() and existing.name != path.name:
            temp_path = existing.with_name(f"{existing.stem}.__casefix__.tmp")
            existing.rename(temp_path)
            temp_path.rename(path)
            break
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare transcript text and lexicon for MFA.")
    parser.add_argument("--transcript-dir", default="data/transcript")
    parser.add_argument("--audio-dir", default="data/audio")
    parser.add_argument("--dictionary", default=None, help="Optional word-to-phone lexicon.")
    parser.add_argument("--dict-out", default="custom_mfa.dict")
    args = parser.parse_args()

    transcript_dir = Path(args.transcript_dir)
    audio_dir = Path(args.audio_dir)
    dict_out = Path(args.dict_out)
    dictionary = load_dictionary(Path(args.dictionary)) if args.dictionary else {}
    audio_dir.mkdir(parents=True, exist_ok=True)

    warn_root_level_files(audio_dir, {".wav"}, "audio")
    audio_files = iter_speaker_files(audio_dir, {".wav"})
    if not audio_files:
        print(f"ERROR: no speaker audio files found in {audio_dir}")
        return

    lexicon: dict[str, list[str]] = {}
    fallback_words: set[str] = set()
    prepared_files = 0

    for audio_path in audio_files:
        rel_id = relative_id(audio_path, audio_dir)
        transcript_path = find_transcript(transcript_dir, rel_id)
        if transcript_path is None:
            print(f"Skip {audio_path}: missing transcript {transcript_dir / Path(rel_id).with_suffix('.txt')}")
            continue

        words = read_transcript(transcript_path)
        if not words:
            print(f"Skip {transcript_path}: empty transcript after normalization")
            continue

        output_path = audio_path.with_suffix(".txt")
        write_text_preserving_requested_case(output_path, " ".join(words))

        for word in words:
            phones, used_fallback = word_to_compare_phones(word, dictionary)
            lexicon[word] = compare_phones_to_mfa(phones)
            if used_fallback:
                fallback_words.add(word)

        prepared_files += 1
        print(f"Wrote MFA transcript: {output_path}")

    with dict_out.open("w", encoding="utf-8") as f:
        for word in sorted(lexicon):
            f.write(f"{word}\t{' '.join(lexicon[word])}\n")

    print(f"\nPrepared {prepared_files} transcript files for MFA.")
    print(f"Wrote MFA dictionary: {dict_out}")
    if fallback_words:
        words_str = ", ".join(sorted(fallback_words))
        print(f"INFO: used fallback G2P for {len(fallback_words)} word(s): {words_str}")
    print("\nNext:")
    print(f"mfa align --clean {audio_dir} {dict_out} english_mfa data/aligned")


if __name__ == "__main__":
    main()

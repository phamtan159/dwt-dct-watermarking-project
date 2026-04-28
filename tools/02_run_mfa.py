"""
Prepare an MFA corpus and run Montreal Forced Aligner.

Expected inputs:
    data/audio/<name>.wav
    data/transcript/<name>.txt
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcript"
CORPUS_DIR = PROJECT_ROOT / "data" / "_mfa_corpus"
ALIGNED_DIR = PROJECT_ROOT / "data" / "aligned"


def ensure_directories(corpus_dir, aligned_dir):
    corpus_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)


def reset_corpus_dir(corpus_dir):
    if corpus_dir.exists():
        for path in corpus_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def link_or_copy(src_path, dst_path):
    try:
        if dst_path.exists():
            dst_path.unlink()
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def build_corpus(audio_dir, transcript_dir, corpus_dir):
    ensure_directories(corpus_dir, ALIGNED_DIR)
    reset_corpus_dir(corpus_dir)

    paired = 0
    for audio_path in sorted(audio_dir.glob("*.wav")):
        transcript_path = transcript_dir / f"{audio_path.stem}.txt"
        if not transcript_path.exists():
            print(f"  Skipping {audio_path.name}: missing transcript {transcript_path.name}")
            continue

        corpus_audio = corpus_dir / audio_path.name
        corpus_lab = corpus_dir / f"{audio_path.stem}.lab"

        link_or_copy(audio_path, corpus_audio)
        corpus_lab.write_text(transcript_path.read_text(encoding="utf-8").strip(), encoding="utf-8")
        paired += 1

    return paired


def run_mfa(mfa_binary, corpus_dir, dictionary, acoustic_model, output_dir, clean):
    command = [
        mfa_binary,
        "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
    ]
    if clean:
        command.append("--clean")

    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run Montreal Forced Aligner on prepared audio")
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR))
    parser.add_argument("--transcript-dir", default=str(TRANSCRIPT_DIR))
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--output-dir", default=str(ALIGNED_DIR))
    parser.add_argument("--dictionary", required=True, help="MFA pronunciation dictionary name or path")
    parser.add_argument("--acoustic-model", required=True, help="MFA acoustic model name or path")
    parser.add_argument("--mfa-bin", default="mfa", help="Path or command name for the MFA executable")
    parser.add_argument("--clean", action="store_true", help="Ask MFA to clean temporary state before aligning")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    transcript_dir = Path(args.transcript_dir).resolve()
    corpus_dir = Path(args.corpus_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    ensure_directories(corpus_dir, output_dir)
    paired = build_corpus(audio_dir, transcript_dir, corpus_dir)
    if paired == 0:
        print("No audio/transcript pairs were found. Nothing to align.")
        return

    print(f"Built MFA corpus with {paired} file(s) in {corpus_dir}")
    run_mfa(args.mfa_bin, corpus_dir, args.dictionary, args.acoustic_model, output_dir, args.clean)
    print(f"Alignment complete. TextGrid files are in {output_dir}")


if __name__ == "__main__":
    main()

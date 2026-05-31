"""
Move or copy existing flat data files into the speaker-based layout.

Example:
  python tools/00_migrate_flat_data_to_speaker.py --speaker S01
  python tools/00_migrate_flat_data_to_speaker.py --speaker S01 --execute

The first command is a dry run. Add --execute only after checking the printed plan.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MEDIA_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".avi", ".webm"}
TEXTGRID_EXTENSIONS = {".textgrid"}
ANNOTATION_EXTENSIONS = {".json", ".txt", ".csv"}
SKIP_NAMES = {".gitkeep"}


def eligible_file(path: Path, suffixes: set[str] | None = None) -> bool:
    if not path.is_file() or path.name in SKIP_NAMES:
        return False
    if suffixes is None:
        return True
    return path.suffix.lower() in suffixes


def plan_file_moves(root: Path, speaker_id: str, suffixes: set[str] | None = None) -> list[tuple[Path, Path]]:
    moves = []
    if not root.exists():
        return moves
    for path in sorted(root.iterdir()):
        if eligible_file(path, suffixes):
            moves.append((path, root / speaker_id / path.name))
    return moves


def plan_processed_moves(root: Path, speaker_id: str) -> list[tuple[Path, Path]]:
    moves = []
    if not root.exists():
        return moves

    for path in sorted(root.iterdir()):
        if path.name in SKIP_NAMES or path.name == speaker_id:
            continue
        if path.is_file() or path.is_dir():
            moves.append((path, root / speaker_id / path.name))
    return moves


def collect_moves(data_dir: Path, speaker_id: str) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    moves.extend(plan_file_moves(data_dir / "raw", speaker_id, MEDIA_EXTENSIONS))
    moves.extend(plan_file_moves(data_dir / "transcript", speaker_id, {".txt"}))
    moves.extend(plan_file_moves(data_dir / "audio", speaker_id, MEDIA_EXTENSIONS | {".txt"}))
    moves.extend(plan_file_moves(data_dir / "aligned", speaker_id, TEXTGRID_EXTENSIONS | {".csv"}))

    annotation_root = data_dir / "annotations"
    if annotation_root.exists():
        for annotation_dir in sorted(path for path in annotation_root.iterdir() if path.is_dir()):
            moves.extend(plan_file_moves(annotation_dir, speaker_id, ANNOTATION_EXTENSIONS))

    processed_root = data_dir / "processed"
    if processed_root.exists():
        for processed_dir in sorted(path for path in processed_root.iterdir() if path.is_dir()):
            moves.extend(plan_processed_moves(processed_dir, speaker_id))

    return moves


def apply_move(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"Skip existing destination: {dst}")
        return
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate flat data files into data/<kind>/<speaker>/...")
    parser.add_argument("--speaker", required=True, help="Speaker folder name, for example S01 or nguyen_van_a")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--execute", action="store_true", help="Actually move/copy files. Without this, only print a plan.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them when --execute is set.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    moves = collect_moves(data_dir, args.speaker)
    if not moves:
        print(f"No flat files found under {data_dir}")
        return

    action = "Copy" if args.copy else "Move"
    for src, dst in moves:
        print(f"{action}: {src} -> {dst}")

    if not args.execute:
        print("Dry run only. Add --execute to apply this migration.")
        return

    for src, dst in moves:
        apply_move(src, dst, args.copy)
    print(f"Done migrating {len(moves)} item(s) to speaker {args.speaker}.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


def relative_id(path: Path, root: Path, suffix: str | None = None) -> str:
    rel = path.relative_to(root)
    if suffix and rel.name.endswith(suffix):
        rel = rel.with_name(rel.name[: -len(suffix)])
    else:
        rel = rel.with_suffix("")
    return rel.as_posix()


def speaker_id_from_relative(rel_id: str) -> str:
    parts = Path(rel_id).parts
    return parts[0] if len(parts) > 1 else "unknown_speaker"


def sample_id_from_relative(rel_id: str) -> str:
    return Path(rel_id).name


def relative_output_path(output_root: Path, rel_id: str, suffix: str) -> Path:
    return output_root / Path(rel_id).with_suffix(suffix)


def iter_files(root: Path, suffixes: set[str] | tuple[str, ...] | list[str]):
    suffixes = {suffix.lower() for suffix in suffixes}
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def iter_speaker_files(root: Path, suffixes: set[str] | tuple[str, ...] | list[str]):
    return [
        path
        for path in iter_files(root, suffixes)
        if path.parent != root
    ]


def warn_root_level_files(root: Path, suffixes: set[str] | tuple[str, ...] | list[str], label: str) -> None:
    suffixes = {suffix.lower() for suffix in suffixes}
    root_level = [
        path
        for path in sorted(root.glob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    for path in root_level:
        print(
            f"Skip {path}: {label} must be inside a speaker folder, "
            f"for example {root / 'S01' / path.name}"
        )


def find_matching_file(root: Path, rel_id: str, suffix: str) -> Path | None:
    exact = root / Path(rel_id).with_suffix(suffix)
    if exact.exists():
        return exact

    stem = Path(rel_id).name.lower()
    candidates = [
        path
        for path in root.rglob(f"*{suffix}")
        if path.stem.lower() == stem
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None

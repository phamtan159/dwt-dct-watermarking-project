import json
import os
import re
from pathlib import Path

import textgrid


ALIGNED_DIR = Path("data/aligned")
MFA_TRANSCRIPT_DIR = Path("data/audio")
AUTO_DIR = Path("data/annotations/auto")

AUTO_DIR.mkdir(parents=True, exist_ok=True)


def find_matching_path(directory: Path, stem: str, suffix: str) -> Path | None:
    exact = directory / f"{stem}{suffix}"
    if exact.exists():
        return exact

    lower_stem = stem.lower()
    for path in directory.glob(f"*{suffix}"):
        if path.stem.lower() == lower_stem:
            return path
    return None


for file in os.listdir(ALIGNED_DIR):
    if not file.endswith(".TextGrid"):
        continue

    textgrid_path = ALIGNED_DIR / file
    transcript_path = find_matching_path(MFA_TRANSCRIPT_DIR, textgrid_path.stem, ".txt")
    if transcript_path and transcript_path.stat().st_mtime > textgrid_path.stat().st_mtime:
        print(
            f"Skip {textgrid_path.name}: TextGrid is older than {transcript_path.name}. "
            "Run MFA align again before exporting JSON."
        )
        continue

    tg = textgrid.TextGrid.fromFile(str(textgrid_path))

    segments = []
    full_phonemes = []

    try:
        tier = tg.getFirst("phones")
    except ValueError:
        tier = tg[1] if len(tg) > 1 else tg[0]

    for interval in tier:
        phoneme = interval.mark.strip()
        if phoneme and phoneme not in ["", "spn", "sil"]:
            base_phoneme = re.sub(r"\d+", "", phoneme)
            full_phonemes.append(base_phoneme)

            seg_id = f"{len(segments):03d}_{base_phoneme}"
            segments.append(
                {
                    "id": seg_id,
                    "phoneme": base_phoneme,
                    "phoneme_standard": base_phoneme,
                    "start": interval.minTime,
                    "end": interval.maxTime,
                    "error": None,
                }
            )

    full_phonemes_str = " ".join(full_phonemes).strip()

    txt_path = AUTO_DIR / file.replace(".TextGrid", ".txt")
    json_path = AUTO_DIR / file.replace(".TextGrid", ".json")

    txt_path.write_text(full_phonemes_str, encoding="utf-8")
    json_path.write_text(json.dumps({"segments": segments}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_path}")

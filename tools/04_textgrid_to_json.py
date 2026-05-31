import json
import os
import re
from pathlib import Path

import textgrid

from speaker_paths import (
    find_matching_file,
    iter_speaker_files,
    relative_id,
    relative_output_path,
    sample_id_from_relative,
    speaker_id_from_relative,
    warn_root_level_files,
)


ALIGNED_DIR = Path("data/aligned")
MFA_TRANSCRIPT_DIR = Path("data/audio")
AUTO_DIR = Path("data/annotations/auto")

AUTO_DIR.mkdir(parents=True, exist_ok=True)


warn_root_level_files(ALIGNED_DIR, {".TextGrid"}, "TextGrid")

for textgrid_path in iter_speaker_files(ALIGNED_DIR, {".TextGrid"}):
    rel_id = relative_id(textgrid_path, ALIGNED_DIR, ".TextGrid")
    speaker_id = speaker_id_from_relative(rel_id)
    sample_id = sample_id_from_relative(rel_id)
    transcript_path = find_matching_file(MFA_TRANSCRIPT_DIR, rel_id, ".txt")
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

    txt_path = relative_output_path(AUTO_DIR, rel_id, ".txt")
    json_path = relative_output_path(AUTO_DIR, rel_id, ".json")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    txt_path.write_text(full_phonemes_str, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "id": rel_id,
                "speaker_id": speaker_id,
                "sample_id": sample_id,
                "audio_id": sample_id,
                "segments": segments,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")

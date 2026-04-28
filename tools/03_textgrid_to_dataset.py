"""
Convert MFA TextGrid output into data/final/dataset.json.
"""

import argparse
import json
from pathlib import Path

from textgrid import IntervalTier, TextGrid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
ALIGNED_DIR = PROJECT_ROOT / "data" / "aligned"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
DEFAULT_OUTPUT = FINAL_DIR / "dataset.json"

PHONE_TIER_NAMES = {"phones", "phonemes", "phone"}
SKIP_PHONES = {"", "sil", "sp", "spn"}


def select_phone_tier(textgrid_obj, preferred_name=None):
    if preferred_name:
        for tier in textgrid_obj.tiers:
            if tier.name.lower() == preferred_name.lower():
                return tier

    for tier in textgrid_obj.tiers:
        if isinstance(tier, IntervalTier) and tier.name.lower() in PHONE_TIER_NAMES:
            return tier

    for tier in textgrid_obj.tiers:
        if isinstance(tier, IntervalTier):
            return tier

    raise ValueError("No interval tier found in TextGrid")


def textgrid_to_item(textgrid_path, audio_dir, final_dir, preferred_tier=None):
    tg = TextGrid.fromFile(str(textgrid_path))
    tier = select_phone_tier(tg, preferred_name=preferred_tier)

    audio_path = audio_dir / f"{textgrid_path.stem}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio for {textgrid_path.name}: {audio_path}")

    phonemes = []
    labels = []
    for interval in tier.intervals:
        phone = interval.mark.strip()
        if phone.lower() in SKIP_PHONES:
            continue

        phonemes.append(
            {
                "s": float(interval.minTime),
                "e": float(interval.maxTime),
                "phone": phone,
            }
        )
        labels.append("OK")

    return {
        "audio": str(audio_path.relative_to(final_dir).as_posix()),
        "phonemes": phonemes,
        "labels": labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert TextGrid alignments into dataset.json")
    parser.add_argument("--aligned-dir", default=str(ALIGNED_DIR))
    parser.add_argument("--audio-dir", default=str(AUDIO_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--phone-tier", default=None, help="Force a specific TextGrid tier name")
    args = parser.parse_args()

    aligned_dir = Path(args.aligned_dir).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    textgrids = sorted(aligned_dir.glob("*.TextGrid"))
    if not textgrids:
        print(f"No TextGrid files found in {aligned_dir}")
        return

    dataset = []
    skipped = 0

    for textgrid_path in textgrids:
        try:
            item = textgrid_to_item(
                textgrid_path,
                audio_dir,
                output_path.parent,
                preferred_tier=args.phone_tier,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"  Skipping {textgrid_path.name}: {exc}")
            skipped += 1
            continue

        dataset.append(item)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(dataset)} sample(s) to {output_path}")
    if skipped:
        print(f"Skipped {skipped} file(s) due to missing audio or invalid tiers")


if __name__ == "__main__":
    main()

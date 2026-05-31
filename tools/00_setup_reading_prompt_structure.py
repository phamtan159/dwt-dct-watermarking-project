from __future__ import annotations

import csv
import shutil
from pathlib import Path


SPEAKER_COUNT = 10
SPEAKER_PREFIX = "speaker"
GROUP_REPEATS = 3
CONTEXT_REPEATS = 3
THETA = "\u03b8"
DH = "\u00f0"

GROUPS = [
    {
        "group_id": 1,
        "target_group": "final_d",
        "phone_folder": "d",
        "target_phoneme": "d",
        "target_position": "final",
        "default_error_code": "final_d_weak_or_omitted",
        "items": [
            ("bed", "I want to go to bed."),
            ("good", "Have a good day!"),
            ("played", "The boys played football."),
        ],
    },
    {
        "group_id": 2,
        "target_group": "final_t",
        "phone_folder": "t",
        "target_phoneme": "t",
        "target_position": "final",
        "default_error_code": "final_t_weak_or_omitted",
        "items": [
            ("cat", "I see a cute black cat."),
            ("light", "Please turn on the light."),
            ("boat", "Look at the small boat."),
        ],
    },
    {
        "group_id": 3,
        "target_group": "final_p",
        "phone_folder": "p",
        "target_phoneme": "p",
        "target_position": "final",
        "default_error_code": "final_p_weak_or_omitted",
        "items": [
            ("map", "I need a map to find the way."),
            ("cup", "She has a cup of tea."),
            ("help", "Can you help me, please?"),
        ],
    },
    {
        "group_id": 4,
        "target_group": "dh_medial_er_final",
        "phone_folder": f"{DH}_er",
        "target_phoneme": "dh+er",
        "target_position": "medial+final",
        "default_error_code": "dh_to_d",
        "items": [
            ("mother", "I love my mother very much."),
            ("brother", "My brother is very tall."),
            ("weather", "The weather is very nice today."),
        ],
    },
    {
        "group_id": 5,
        "target_group": "theta_medial",
        "phone_folder": f"{THETA}_medial",
        "target_phoneme": "theta",
        "target_position": "medial",
        "default_error_code": "th_to_t",
        "items": [
            ("birthday", "Today is my birthday."),
            ("healthy", "Apples are healthy food."),
            ("nothing", "There is nothing in the box."),
        ],
    },
    {
        "group_id": 6,
        "target_group": "dh_initial",
        "phone_folder": DH,
        "target_phoneme": "dh",
        "target_position": "initial",
        "default_error_code": "dh_to_d",
        "items": [
            ("this", "This is my new book."),
            ("that", "I like that red car."),
            ("these", "These are my favorite shoes."),
            ("those", "Look at those birds in the sky."),
        ],
    },
    {
        "group_id": 7,
        "target_group": "theta_initial",
        "phone_folder": THETA,
        "target_phoneme": "theta",
        "target_position": "initial",
        "default_error_code": "th_to_t",
        "items": [
            ("think", "I think it will rain today."),
            ("thin", "The paper is very thin."),
            ("three", "I have three red apples."),
            ("thumb", "My thumb hurts a lot."),
        ],
    },
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def touch_gitkeep(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".gitkeep"
    if not marker.exists():
        marker.write_text("", encoding="utf-8")


def clear_generated_transcripts(transcript_root: Path, speakers: list[str]) -> None:
    for speaker in speakers:
        path = transcript_root / speaker
        if path.exists():
            shutil.rmtree(path)


def group_words(group: dict) -> list[str]:
    return [word for word, _ in group["items"]]


def group_sentences(group: dict) -> list[str]:
    return [sentence for _, sentence in group["items"]]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw"
    transcript_root = project_root / "data" / "transcript"
    metadata_path = project_root / "data" / "sample_metadata.csv"
    prompt_path = project_root / "data" / "reading_prompts.csv"

    speakers = [f"{SPEAKER_PREFIX}-{idx:02d}" for idx in range(1, SPEAKER_COUNT + 1)]
    clear_generated_transcripts(transcript_root, speakers)

    metadata_rows: list[dict[str, str]] = []
    prompt_rows: list[dict[str, str]] = []

    for group in GROUPS:
        for word, sentence in group["items"]:
            prompt_rows.append(
                {
                    "group_id": str(group["group_id"]),
                    "target_group": group["target_group"],
                    "phone_folder": group["phone_folder"],
                    "target_word": word,
                    "target_phoneme": group["target_phoneme"],
                    "target_position": group["target_position"],
                    "sentence": sentence,
                    "default_error_code": group["default_error_code"],
                }
            )

    for speaker in speakers:
        for group in GROUPS:
            folder_base = Path(speaker) / group["phone_folder"]
            words = group_words(group)
            target_words = "|".join(words)
            word_transcript = " ".join(words)

            for mode, expected_label, read_style in [
                ("T", "OK", "group_words_correct"),
                ("F", group["default_error_code"], "group_words_error"),
            ]:
                folder = folder_base / mode
                touch_gitkeep(raw_root / folder)
                for repeat in range(1, GROUP_REPEATS + 1):
                    stem = f"{mode}_{repeat:02d}"
                    rel_id = (folder / stem).as_posix()
                    write_text(transcript_root / folder / f"{stem}.txt", word_transcript)
                    metadata_rows.append(
                        {
                            "sample_id": rel_id,
                            "speaker_id": speaker,
                            "phone_folder": group["phone_folder"],
                            "mode": mode,
                            "take_id": f"{mode}{repeat:02d}",
                            "read_style": read_style,
                            "target_group": group["target_group"],
                            "target_word": target_words,
                            "target_words": target_words,
                            "target_phoneme": group["target_phoneme"],
                            "target_position": group["target_position"],
                            "expected_label": expected_label,
                            "needs_expert_label": "false" if mode == "T" else "true",
                            "transcript": word_transcript,
                        }
                    )

            context_folder = folder_base / "C"
            touch_gitkeep(raw_root / context_folder)
            for word, sentence in group["items"]:
                for repeat in range(1, CONTEXT_REPEATS + 1):
                    stem = f"{word}_C_{repeat:02d}"
                    rel_id = (context_folder / stem).as_posix()
                    write_text(transcript_root / context_folder / f"{stem}.txt", sentence)
                    metadata_rows.append(
                        {
                            "sample_id": rel_id,
                            "speaker_id": speaker,
                            "phone_folder": group["phone_folder"],
                            "mode": "C",
                            "take_id": f"C{repeat:02d}",
                            "read_style": "sentence_error",
                            "target_group": group["target_group"],
                            "target_word": word,
                            "target_words": word,
                            "target_phoneme": group["target_phoneme"],
                            "target_position": group["target_position"],
                            "expected_label": group["default_error_code"],
                            "needs_expert_label": "true",
                            "transcript": sentence,
                        }
                    )

    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    with prompt_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "target_group",
                "phone_folder",
                "target_word",
                "target_phoneme",
                "target_position",
                "sentence",
                "default_error_code",
            ],
        )
        writer.writeheader()
        writer.writerows(prompt_rows)

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Wrote {prompt_path.relative_to(project_root)}")
    print(f"Wrote {metadata_path.relative_to(project_root)} with {len(metadata_rows)} planned files")
    print(f"Created transcript tree under {transcript_root.relative_to(project_root)}")
    print(f"Created raw folder tree under {raw_root.relative_to(project_root)}")
    print("Naming: T_01/F_01 are group-word files; <word>_C_01 are sentence-context files.")


if __name__ == "__main__":
    main()

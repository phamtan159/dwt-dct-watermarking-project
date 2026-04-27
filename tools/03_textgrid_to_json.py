import textgrid, json, os

os.makedirs("data/annotations/auto", exist_ok=True)

# Danh sách các âm vị dễ sai của người Miền Tây (bạn có thể tùy chỉnh thêm dựa theo từ điển MFA)
TARGET_PHONEMES = ["v", "r", "tr", "s", "gi", "d"]

for file in os.listdir("data/aligned"):
    if not file.endswith(".TextGrid"):
        continue

    tg = textgrid.TextGrid.fromFile(f"data/aligned/{file}")

    segments = []

    try:
        tier = tg.getFirst("phones")
    except ValueError:
        tier = tg[1] if len(tg) > 1 else tg[0]

    for interval in tier:
        phoneme = interval.mark.strip()
        # Chỉ lấy những âm vị nằm trong danh sách cần chú ý
        if phoneme and phoneme in TARGET_PHONEMES:
            seg_id = f"{len(segments):03d}_{phoneme}"

            segments.append({
                "id": seg_id,
                "phoneme": phoneme,
                "start": interval.minTime,
                "end": interval.maxTime,
                "error": None
            })

    json.dump(
        {"segments": segments},
        open(f"data/annotations/auto/{file.replace('.TextGrid','.json')}", "w"),
        indent=2
    )

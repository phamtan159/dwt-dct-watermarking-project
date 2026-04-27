import textgrid, json, os

os.makedirs("data/annotations/auto", exist_ok=True)

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
        if interval.mark.strip():
            seg_id = f"{len(segments):03d}_{interval.mark}"

            segments.append({
                "id": seg_id,
                "phoneme": interval.mark,
                "start": interval.minTime,
                "end": interval.maxTime,
                "error": None
            })

    json.dump(
        {"segments": segments},
        open(f"data/annotations/auto/{file.replace('.TextGrid','.json')}", "w"),
        indent=2
    )

import json, os, shutil

for file in os.listdir("data/annotations/auto"):
    if not file.endswith(".json"):
        continue

    name = file.replace(".json", "")

    frames_dir = f"data/processed/mouth/{name}"
    if not os.path.exists(frames_dir):
        continue

    frames = sorted(os.listdir(frames_dir))

    meta = json.load(open(f"data/meta/{name}.json", encoding="utf-8"))
    ann = json.load(open(f"data/annotations/auto/{file}", encoding="utf-8"))

    fps = meta["fps"]

    out_dir = f"data/processed/clips/{name}"
    os.makedirs(out_dir, exist_ok=True)

    for seg in ann["segments"]:
        PAD = 3
        start_f = max(0, round(seg["start"] * fps) - PAD)
        end_f = round(seg["end"] * fps) + PAD

        clip_dir = f"{out_dir}/{seg['id']}"
        os.makedirs(clip_dir, exist_ok=True)

        for i in range(start_f, min(end_f, len(frames))):
            shutil.copy(
                f"{frames_dir}/{frames[i]}",
                f"{clip_dir}/{frames[i]}"
            )

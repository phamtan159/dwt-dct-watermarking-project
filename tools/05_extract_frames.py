import cv2, os, json

for f in os.listdir("data/raw"):
    if not f.endswith(".mp4"):
        continue

    name = f.replace(".mp4", "")
    cap = cv2.VideoCapture(f"data/raw/{f}")

    out_dir = f"data/processed/frames/{name}"
    os.makedirs(out_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imwrite(f"{out_dir}/{i:04d}.jpg", frame)
        i += 1

    os.makedirs("data/meta", exist_ok=True)
    json.dump({"fps": fps}, open(f"data/meta/{name}.json", "w"))

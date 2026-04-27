import os

os.makedirs("data/audio", exist_ok=True)

for f in os.listdir("data/raw"):
    if f.endswith(".mp4"):
        os.system(
            f"ffmpeg -y -i data/raw/{f} -ar 16000 -ac 1 data/audio/{f.replace('.mp4','.wav')}"
        )

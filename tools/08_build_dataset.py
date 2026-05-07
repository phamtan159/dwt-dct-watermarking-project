import os
import json

# Run from project root, so paths are relative to root.
# But audio paths stored in dataset.json need to be relative to models/ directory, 
# because train.py is executed from inside models/.
ANNOTATION_DIR = os.environ.get("ANNOTATION_DIR", "data/annotations/manual")
AUDIO_DIR = "../data/audio"
OUTPUT_PATH = "data/final/dataset.json"
LABEL_MAP_PATH = "data/label_map.json"
SILENCE_PHONES = {"", " ", "sil", "sp", "spn", "<eps>", "<sil>", "SIL"}


def is_silence(phone):
    return str(phone or "").strip() in SILENCE_PHONES

def build_dataset():
    if not os.path.exists(ANNOTATION_DIR):
        print(f"ERROR: Annotation directory not found: {ANNOTATION_DIR}")
        print("Create annotations/manual, or run tools/07_compare_transcript_phonemes.py and set ANNOTATION_DIR=data/annotations/compare.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    dataset = []
    
    for filename in os.listdir(ANNOTATION_DIR):
        if not filename.endswith(".json"):
            continue
            
        name = filename.replace(".json", "")
        json_path = os.path.join(ANNOTATION_DIR, filename)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        phonemes = []
        labels = []
        
        for seg in data.get("segments", []):
            standard_phone = seg.get("phoneme_standard")
            real_phone = seg.get("phoneme_real", seg.get("phoneme", ""))

            if is_silence(standard_phone):
                continue
            if is_silence(real_phone) and str(seg.get("error_id", seg.get("error", ""))).strip().upper() == "OK":
                continue

            phonemes.append({
                "s": seg["start"],
                "e": seg["end"],
                "phone": real_phone,
                "standard_phone": standard_phone
            })
            
            # If error is null or empty, it's "OK"
            error = seg.get("error_id", seg.get("error"))
            if error is None or str(error).strip() == "" or str(error).strip().upper() == "OK":
                labels.append("OK")
            else:
                labels.append(str(error).strip())
                
        # Check if the audio file exists (relative to the tools directory script)
        audio_check_path = f"data/audio/{name}.wav"
        if not os.path.exists(audio_check_path):
            print(f"WARNING: Audio file not found, skipping: {audio_check_path}")
            continue
            
        dataset.append({
            "audio": f"{AUDIO_DIR}/{name}.wav",
            "phonemes": phonemes,
            "labels": labels
        })
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"OK: Built dataset with {len(dataset)} samples at: {OUTPUT_PATH}")
    print("Next: cd models && python train.py")

if __name__ == "__main__":
    build_dataset()

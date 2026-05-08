import os
import json

# Run from project root, so paths are relative to root.
# But audio paths stored in dataset.json need to be relative to models/ directory, 
# because train.py is executed from inside models/.
DEFAULT_ANNOTATION_DIR = "data/annotations/manual"
FALLBACK_ANNOTATION_DIR = "data/annotations/compare"
ANNOTATION_DIR = os.environ.get("ANNOTATION_DIR", DEFAULT_ANNOTATION_DIR)
AUDIO_DIR = "../data/audio"
OUTPUT_PATH = "data/final/dataset.json"
LABEL_MAP_PATH = "data/label_map.json"
SILENCE_PHONES = {"", " ", "sil", "sp", "spn", "<eps>", "<sil>", "SIL"}


def is_silence(phone):
    return str(phone or "").strip() in SILENCE_PHONES


def load_label_map(path):
    labels_by_name = {"OK": "0", "no_error": "0"}
    valid_ids = {"0"}
    if not os.path.exists(path):
        valid_ids.add("OTHER")
        return labels_by_name, valid_ids

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key, item in data.items():
            if not isinstance(item, dict):
                continue
            label_id = str(item.get("index", item.get("id", key)))
            label_name = item.get("loai_loi", key)
            valid_ids.add(label_id)
            labels_by_name[key] = label_id
            labels_by_name[label_name] = label_id
            if key == "no_error" or label_id == "0":
                labels_by_name["OK"] = label_id

    if "OTHER" not in labels_by_name:
        valid_ids.add("OTHER")
        labels_by_name["OTHER"] = "OTHER"
    return labels_by_name, valid_ids


def resolve_label_id(seg, labels_by_name, valid_ids):
    raw_id = seg.get("error_id")
    raw_error = seg.get("error")

    if raw_id is not None and str(raw_id).strip() != "":
        label_id = str(raw_id).strip()
        if label_id.upper() == "OK":
            return labels_by_name["OK"]
        if label_id in valid_ids:
            return label_id

    if raw_error is None or str(raw_error).strip() == "":
        return labels_by_name["OK"]

    label_name = str(raw_error).strip()
    if label_name.upper() == "OK":
        return labels_by_name["OK"]

    return labels_by_name.get(label_name, "OTHER")


def resolve_annotation_dir():
    if os.environ.get("ANNOTATION_DIR"):
        return ANNOTATION_DIR

    if os.path.exists(DEFAULT_ANNOTATION_DIR):
        return DEFAULT_ANNOTATION_DIR

    if os.path.exists(FALLBACK_ANNOTATION_DIR):
        print(
            f"INFO: {DEFAULT_ANNOTATION_DIR} not found, using fallback annotation directory: "
            f"{FALLBACK_ANNOTATION_DIR}"
        )
        return FALLBACK_ANNOTATION_DIR

    return ANNOTATION_DIR

def build_dataset():
    annotation_dir = resolve_annotation_dir()
    labels_by_name, valid_ids = load_label_map(LABEL_MAP_PATH)

    if not os.path.exists(annotation_dir):
        print(f"ERROR: Annotation directory not found: {annotation_dir}")
        print(
            "Create data/annotations/manual, or run "
            "tools/05_compare_transcript_phonemes.py, or set "
            "ANNOTATION_DIR=data/annotations/compare."
        )
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    dataset = []
    
    for filename in os.listdir(annotation_dir):
        if not filename.endswith(".json"):
            continue
            
        name = filename.replace(".json", "")
        json_path = os.path.join(annotation_dir, filename)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        phonemes = []
        labels = []
        
        for seg in data.get("segments", []):
            standard_phone = seg.get("phoneme_standard")
            real_phone = seg.get("phoneme_real", seg.get("phoneme", ""))

            if is_silence(standard_phone):
                continue
            if is_silence(real_phone) and resolve_label_id(seg, labels_by_name, valid_ids) == labels_by_name["OK"]:
                continue

            phonemes.append({
                "s": seg["start"],
                "e": seg["end"],
                "phone": real_phone,
                "standard_phone": standard_phone
            })
            
            labels.append(resolve_label_id(seg, labels_by_name, valid_ids))
                
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
    print(f"Annotation source: {annotation_dir}")
    print("Next: cd models && python train.py")

if __name__ == "__main__":
    build_dataset()

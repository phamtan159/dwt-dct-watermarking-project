import os
import json

# Run from project root, so paths are relative to root.
# But audio paths stored in dataset.json need to be relative to models/ directory, 
# because train.py is executed from inside models/.
MANUAL_DIR = "data/annotations/manual"
AUDIO_DIR = "../data/audio"
OUTPUT_PATH = "data/final/dataset.json"
LABEL_MAP_PATH = "data/label_map.json"

def build_dataset():
    if not os.path.exists(MANUAL_DIR):
        print(f"❌ Lỗi: Thư mục {MANUAL_DIR} không tồn tại.")
        print("Hãy tạo thư mục này, copy các file JSON từ annotations/auto sang và bắt đầu gán nhãn trường 'error'.")
        return

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    dataset = []
    
    for filename in os.listdir(MANUAL_DIR):
        if not filename.endswith(".json"):
            continue
            
        name = filename.replace(".json", "")
        json_path = os.path.join(MANUAL_DIR, filename)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        phonemes = []
        labels = []
        
        for seg in data.get("segments", []):
            phonemes.append({
                "s": seg["start"],
                "e": seg["end"],
                "phone": seg["phoneme"]
            })
            
            # If error is null or empty, it's "OK"
            error = seg.get("error")
            if error is None or str(error).strip() == "" or str(error).strip().upper() == "OK":
                labels.append("OK")
            else:
                labels.append(str(error).strip())
                
        # Check if the audio file exists (relative to the tools directory script)
        audio_check_path = f"data/audio/{name}.wav"
        if not os.path.exists(audio_check_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy file audio {audio_check_path}, đang bỏ qua file này.")
            continue
            
        dataset.append({
            "audio": f"{AUDIO_DIR}/{name}.wav",
            "phonemes": phonemes,
            "labels": labels
        })
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Đã tạo thành công dataset tổng hợp gồm {len(dataset)} mẫu tại: {OUTPUT_PATH}")
    print("Bây giờ bạn đã có thể chạy huấn luyện bằng lệnh: cd models && python train.py")

if __name__ == "__main__":
    build_dataset()

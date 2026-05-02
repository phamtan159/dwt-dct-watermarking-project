import cv2, os, json
import mediapipe as mp
import numpy as np

# Danh sách các ID quan trọng vùng môi
LIPS_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

os.makedirs("data/annotations/visual_check", exist_ok=True)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

for file in os.listdir("data/annotations/auto"):
    if not file.endswith(".json"):
        continue

    name = file.replace(".json", "")
    frames_dir = f"data/processed/frames/{name}"
    
    if not os.path.exists(frames_dir):
        continue

    frames = sorted(os.listdir(frames_dir))
    meta = json.load(open(f"data/meta/{name}.json"))
    ann = json.load(open(f"data/annotations/auto/{file}", encoding="utf-8"))
    fps = meta["fps"]

    # Đọc trước toàn bộ thông số khẩu hình của video để chạy nhanh hơn
    print(f"Đang phân tích khẩu hình video: {name}...")
    frame_features = []
    
    for f_name in frames:
        img = cv2.imread(f"{frames_dir}/{f_name}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(img_rgb)
        
        if res.multi_face_landmarks:
            pts = res.multi_face_landmarks[0].landmark
            
            # Tính toán các chỉ số
            # 1. Độ mở dọc (Height): Điểm 0 (môi trên) và 17 (môi dưới)
            height = calculate_distance(pts[0], pts[17])
            
            # 2. Độ bè ngang (Width): Điểm 61 (khóe trái) và 291 (khóe phải)
            width = calculate_distance(pts[61], pts[291])
            
            # 3. MAR (Mouth Aspect Ratio)
            mar = height / width if width > 0 else 0
            
            frame_features.append({
                "frame": f_name,
                "height": round(height, 4),
                "width": round(width, 4),
                "mar": round(mar, 4)
            })
        else:
            frame_features.append(None)

    # Đánh giá từng âm vị
    for seg in ann["segments"]:
        PAD = 1 # Chỉ lấy +-1 frame để tránh lấn sang âm vị khác
        start_f = max(0, round(seg["start"] * fps) - PAD)
        end_f = min(len(frame_features)-1, round(seg["end"] * fps) + PAD)
        
        valid_features = [frame_features[i] for i in range(start_f, end_f+1) if frame_features[i] is not None]
        
        if not valid_features:
            seg["visual_check"] = "Không nhận diện được khuôn mặt"
            continue
            
        avg_mar = sum(f["mar"] for f in valid_features) / len(valid_features)
        min_height = min(f["height"] for f in valid_features)
        avg_width = sum(f["width"] for f in valid_features) / len(valid_features)
        
        phoneme = seg["phoneme"]
        check_msg = "Khẩu hình chuẩn"
        
        # Áp dụng Rule (Bước 1: Visual)
        if phoneme in ['p', 'b', 'm']:
            if min_height > 0.02: # Môi chưa chạm nhau
                check_msg = "LỖI VISUAL: Chưa mím chặt hai môi."
        elif phoneme in ['u', 'uː', 'w', 'ʃ', 'tʃ', 'dʒ']:
            if avg_width > 0.08: # Môi chưa chu / còn quá rộng
                check_msg = "LỖI VISUAL: Môi chưa đủ chu ra phía trước."
        elif phoneme in ['i', 'iː', 'ɪ', 'e']:
            if avg_width < 0.06: # Khóe miệng chưa mở rộng
                check_msg = "LỖI VISUAL: Khóe miệng chưa giãn đều sang hai bên."
        elif phoneme in ['f', 'v']:
            if avg_mar > 0.15: # Miệng mở quá to, không cắn môi dưới
                check_msg = "LỖI VISUAL: Răng trên không chạm môi dưới."
                
        seg["visual_features"] = {
            "avg_mar": round(avg_mar, 4),
            "min_height": round(min_height, 4),
            "avg_width": round(avg_width, 4)
        }
        seg["visual_check"] = check_msg

    out_file = f"data/annotations/visual_check/{file}"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)
        
    print(f"Đã lưu kết quả phân tích khẩu hình tại: {out_file}")

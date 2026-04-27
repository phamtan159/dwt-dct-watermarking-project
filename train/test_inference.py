import torch
import cv2
import os
import numpy as np
import json
from model import LipModelAdvanced

def load_clip(clip_dir, max_len=12):
    """Load và tiền xử lý một folder chứa các ảnh miệng"""
    frames = sorted(os.listdir(clip_dir))
    imgs = []
    for f in frames[:max_len]:
        p = os.path.join(clip_dir, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (88, 88))
        img = img[..., None] / 255.0 # (H, W, 1)
        imgs.append(img)
    
    # Padding nếu thiếu frame
    while len(imgs) < max_len:
        imgs.append(imgs[-1] if imgs else np.zeros((88, 88, 1)))
        
    # Chuyển thành tensor (T, C, H, W) -> (1, T, C, H, W) cho batch
    imgs = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2)
    return imgs.unsqueeze(0)

def inference(clip_path, model_path, encoder_ckpt, label_map_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    
    # Khởi tạo model
    num_classes = len(label_map)
    model = LipModelAdvanced(num_classes, encoder_ckpt).to(device)
    
    # Load weights đã train
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded trained model from {model_path}")
    else:
        print(f"⚠️ Không tìm thấy file model đã train tại {model_path}. Đang dùng weights khởi tạo.")
    
    model.eval()
    
    # Load data
    input_tensor = load_clip(clip_path).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(output, dim=1).item()
        
    result = inv_label_map.get(pred_idx, "Unknown")
    confidence = prob[0][pred_idx].item()
    
    print(f"\n--- KẾT QUẢ DỰ ĐOÁN ---")
    print(f"Clip: {clip_path}")
    print(f"Dự đoán lỗi: {result}")
    print(f"Độ tin cậy: {confidence:.2%}")
    print(f"----------------------")

if __name__ == "__main__":
    # Ví dụ chạy test
    # inference(
    #     clip_path="data/processed/clips/sample/000_v", 
    #     model_path="train/final_model.pth",
    #     encoder_ckpt="pretrained/vsr_trlrs3_base.pth",
    #     label_map_path="data/label_map.json"
    # )
    print("Hướng dẫn sử dụng: Sửa hàm main trong file này, trỏ tới clip bạn muốn test và chạy python train/test_inference.py")

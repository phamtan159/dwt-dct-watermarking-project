import os, json, cv2
import torch
import numpy as np

MAX_LEN = 12

class LipDataset:
    def __init__(self, base):
        self.samples = []
        self.base = base
        
        # Build or load label_map
        label_map_path = os.path.join(base, "label_map.json")
        labels_set = set()
        
        # Đọc tất cả annotations để gom nhãn
        manual_dir = os.path.join(base, "annotations", "manual")
        if os.path.exists(manual_dir):
            for file in os.listdir(manual_dir):
                if not file.endswith(".json"): continue
                ann = json.load(open(os.path.join(manual_dir, file)))
                for seg in ann["segments"]:
                    labels_set.add(seg.get("error") or "no_error")
                    
        # Nếu đã có label_map thì đọc, không thì tạo mới
        if os.path.exists(label_map_path):
            self.label_map = json.load(open(label_map_path))
            # Cập nhật thêm nhãn mới nếu có
            new_labels = False
            for label in labels_set:
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                    new_labels = True
            if new_labels:
                json.dump(self.label_map, open(label_map_path, "w"), indent=2)
        else:
            self.label_map = {lbl: i for i, lbl in enumerate(sorted(list(labels_set)))}
            if self.label_map:
                json.dump(self.label_map, open(label_map_path, "w"), indent=2)

        if not os.path.exists(manual_dir):
            return

        for file in os.listdir(manual_dir):
            if not file.endswith(".json"): continue
            
            name = file.replace(".json", "")
            ann = json.load(open(os.path.join(manual_dir, file)))

            for seg in ann["segments"]:
                label = seg.get("error") or "no_error"

                clip_dir = os.path.join(base, "processed", "clips", name, seg['id'])
                if not os.path.exists(clip_dir):
                    continue

                frames = sorted(os.listdir(clip_dir))
                paths = [os.path.join(clip_dir, f) for f in frames]

                self.samples.append({
                    "paths": paths,
                    "label": self.label_map[label]
                })

    def __getitem__(self, i):
        sample = self.samples[i]

        imgs = []
        for p in sample["paths"][:MAX_LEN]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (88, 88))
            img = img[..., None] / 255.0  # (H, W, 1)
            imgs.append(img)

        # Fix 6: Xử lý mảng rỗng
        if len(imgs) == 0:
            imgs = [np.zeros((88, 88, 1)) for _ in range(MAX_LEN)]
            
        while len(imgs) < MAX_LEN:
            imgs.append(imgs[-1])

        # Fix 7: Tensor conversion
        imgs_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2) # (T, C, H, W)
        
        # Fix 9: Normalize frame
        imgs_tensor = (imgs_tensor - 0.5) / 0.5
        
        return imgs_tensor, sample["label"]

    def __len__(self):
        return len(self.samples)

import json
import os

import cv2
import numpy as np
import torch


MAX_LEN = 12
SILENCE_PHONES = {"", "sil", "sp", "spn", "<eps>", "<sil>"}


def is_silence_phone(phone):
    if phone is None:
        return True
    return str(phone).strip().lower() in SILENCE_PHONES


def should_skip_segment(seg):
    if "phoneme_standard" in seg:
        return is_silence_phone(seg.get("phoneme_standard"))
    if "standard_phone" in seg:
        return is_silence_phone(seg.get("standard_phone"))
    phone = seg.get("phoneme") or seg.get("phone") or seg.get("phoneme_real")
    return is_silence_phone(phone)


class LipDataset:
    def __init__(self, base):
        self.samples = []
        self.base = base
        self.project_root = os.path.dirname(os.path.abspath(base)) or "."
        self.label_map = {}

        final_dataset_path = os.path.join(base, "final", "dataset.json")
        compare_dir = os.path.join(base, "annotations", "compare")
        manual_dir = os.path.join(base, "annotations", "manual")

        if os.path.exists(final_dataset_path):
            self._load_final_dataset(final_dataset_path)
        elif os.path.exists(compare_dir) and any(f.endswith(".json") for f in os.listdir(compare_dir)):
            self._load_compare_annotations(compare_dir)
        else:
            self._load_manual_annotations(manual_dir)

    def _label_index(self, label):
        label = label or "no_error"
        value = self.label_map.get(label)
        if isinstance(value, dict):
            return value["index"]
        if isinstance(value, int):
            return value

        index = len(self.label_map)
        self.label_map[label] = {"index": index, "loai_loi": label}
        return index

    def _normalize_existing_label_map(self, raw_map):
        normalized = {}
        for label, value in raw_map.items():
            if isinstance(value, dict):
                normalized[label] = value
            else:
                normalized[label] = {"index": int(value), "loai_loi": label}
        self.label_map = normalized

    def _load_label_map_file(self, path):
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            self._normalize_existing_label_map(json.load(f))
        return True

    def _resolve_path(self, path):
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.project_root, path))

    def _frame_paths(self, clip_dir):
        if not clip_dir or not os.path.exists(clip_dir):
            return []
        return [
            os.path.join(clip_dir, f)
            for f in sorted(os.listdir(clip_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def _add_sample(self, video, segment_id, clip_dir, label, phoneme=None, phoneme_standard=None, phoneme_real=None):
        paths = self._frame_paths(clip_dir)
        if not paths:
            return

        self.samples.append(
            {
                "sample_id": f"{video}/{segment_id}",
                "video": video,
                "segment_id": segment_id,
                "phoneme": phoneme or phoneme_real,
                "phoneme_standard": phoneme_standard,
                "phoneme_real": phoneme_real,
                "error_id": label or "no_error",
                "paths": paths,
                "label": self._label_index(label),
            }
        )

    def _load_final_dataset(self, final_dataset_path):
        with open(final_dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        self._normalize_existing_label_map(dataset.get("label_map", {}))
        fallback_clip_root = dataset.get("clip_root") or os.path.join(self.base, "processed", "clips")

        for sample in dataset.get("samples", []):
            video = sample.get("video_id") or sample.get("id")
            for seg in sample.get("segments", []):
                if should_skip_segment(seg):
                    continue
                segment_id = seg.get("id")
                clip_dir = self._resolve_path(seg.get("clip_dir"))
                if not clip_dir and video and segment_id:
                    clip_dir = self._resolve_path(os.path.join(fallback_clip_root, video, segment_id))

                self._add_sample(
                    video=video,
                    segment_id=segment_id,
                    clip_dir=clip_dir,
                    label=seg.get("error_id") or seg.get("label") or "no_error",
                    phoneme=seg.get("phone") or seg.get("phoneme"),
                    phoneme_standard=seg.get("standard_phone") or seg.get("phoneme_standard"),
                    phoneme_real=seg.get("phoneme_real") if "phoneme_real" in seg else (seg.get("phone") or seg.get("phoneme")),
                )

    def _load_compare_annotations(self, compare_dir):
        labels = set()
        files = [f for f in sorted(os.listdir(compare_dir)) if f.endswith(".json")]
        for file in files:
            with open(os.path.join(compare_dir, file), "r", encoding="utf-8") as f:
                ann = json.load(f)
            for seg in ann.get("segments", []):
                if should_skip_segment(seg):
                    continue
                labels.add(seg.get("error_id") or seg.get("error") or "no_error")

        self.label_map = {
            label: {"index": i, "loai_loi": label}
            for i, label in enumerate(sorted(labels or {"no_error"}))
        }

        for file in files:
            video = file.replace(".json", "")
            with open(os.path.join(compare_dir, file), "r", encoding="utf-8") as f:
                ann = json.load(f)

            for seg in ann.get("segments", []):
                if should_skip_segment(seg):
                    continue
                segment_id = seg.get("id")
                clip_dir = os.path.join(self.base, "processed", "clips", video, segment_id)
                self._add_sample(
                    video=video,
                    segment_id=segment_id,
                    clip_dir=clip_dir,
                    label=seg.get("error_id") or seg.get("error") or "no_error",
                    phoneme=seg.get("phoneme"),
                    phoneme_standard=seg.get("phoneme_standard"),
                    phoneme_real=seg.get("phoneme_real") if "phoneme_real" in seg else seg.get("phoneme"),
                )

    def _load_manual_annotations(self, manual_dir):
        label_map_path = os.path.join(self.base, "label_map.json")
        labels = set()

        if os.path.exists(manual_dir):
            for file in os.listdir(manual_dir):
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(manual_dir, file), "r", encoding="utf-8") as f:
                    ann = json.load(f)
                for seg in ann.get("segments", []):
                    labels.add(seg.get("error") or "no_error")

        if not self._load_label_map_file(label_map_path):
            self.label_map = {
                label: {"index": i, "loai_loi": label}
                for i, label in enumerate(sorted(labels or {"no_error"}))
            }
            if self.label_map:
                with open(label_map_path, "w", encoding="utf-8") as f:
                    json.dump(self.label_map, f, indent=2, ensure_ascii=False)

        if not os.path.exists(manual_dir):
            return

        for file in os.listdir(manual_dir):
            if not file.endswith(".json"):
                continue

            video = file.replace(".json", "")
            with open(os.path.join(manual_dir, file), "r", encoding="utf-8") as f:
                ann = json.load(f)

            for seg in ann.get("segments", []):
                if should_skip_segment(seg):
                    continue
                segment_id = seg.get("id")
                clip_dir = os.path.join(self.base, "processed", "clips", video, segment_id)
                self._add_sample(
                    video=video,
                    segment_id=segment_id,
                    clip_dir=clip_dir,
                    label=seg.get("error") or "no_error",
                    phoneme=seg.get("phoneme"),
                )

    def __getitem__(self, i):
        sample = self.samples[i]

        imgs = []
        for p in sample["paths"][:MAX_LEN]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (88, 88))
            img = img[..., None] / 255.0
            imgs.append(img)

        if len(imgs) == 0:
            imgs = [np.zeros((88, 88, 1)) for _ in range(MAX_LEN)]

        while len(imgs) < MAX_LEN:
            imgs.append(imgs[-1])

        imgs_tensor = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(0, 3, 1, 2)
        imgs_tensor = (imgs_tensor - 0.5) / 0.5

        return imgs_tensor, sample["label"]

    def __len__(self):
        return len(self.samples)


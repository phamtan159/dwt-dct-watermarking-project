"""
Utility functions for pronunciation error detection.
- LabelVocab: maps error labels to indices
- build_frame_labels: aligns phoneme-level labels to WavLM frame indices
"""

import json
import os
import torch


class LabelVocab:
    """Simple vocabulary for error label IDs."""

    def __init__(self, labels):
        """
        Args:
            labels: list of label IDs, e.g. ["0", "1", "2", ...]
        """
        self.labels = [str(label) for label in labels]
        self.stoi = {label: i for i, label in enumerate(self.labels)}
        self.itos = {i: label for i, label in enumerate(self.labels)}
        if "0" in self.stoi:
            self.stoi.setdefault("OK", self.stoi["0"])
            self.ok_label = "0"
        else:
            self.stoi.setdefault("OK", 0)
            self.ok_label = self.labels[0] if self.labels else "OK"

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"LabelVocab({self.labels})"


# ================================================
# Load label set dynamically from label_map.json
# ================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FINAL_LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "data/final/label_map.json")
LEGACY_LABEL_MAP_PATH = os.path.join(PROJECT_ROOT, "data/label_map.json")
LABEL_MAP_PATH = FINAL_LABEL_MAP_PATH if os.path.exists(FINAL_LABEL_MAP_PATH) else LEGACY_LABEL_MAP_PATH


def normalize_label(value):
    if value is None or str(value).strip() == "":
        return "OK"
    label = str(value).strip()
    if label.upper() == "OK" or label in {"0", "no_error"}:
        return "OK"
    return label

def load_labels_from_map(path):
    if not os.path.exists(path):
        return ["OK", "OTHER"]
    
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except:
            return ["0", "OTHER"]
            
    labels = []
    has_other_label = False
    
    # Support list format
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], str):
            # List of strings
            for item in data:
                label = normalize_label(item)
                if label not in labels:
                    labels.append(label)
        elif isinstance(data[0], dict):
            # List of dicts (prefer numeric "index", fall back to "id")
            for item in data:
                label = None
                if "index" in item:
                    label = normalize_label(item["index"])
                elif "id" in item:
                    label = normalize_label(item["id"])
                    
                if label and label not in labels:
                    labels.append(label)
                    
    # Support dict format used by data/label_map.json
    elif isinstance(data, dict):
        indexed_labels = []
        fallback_labels = []

        def add_label(raw_index, raw_label):
            label = normalize_label(raw_label)
            try:
                indexed_labels.append((int(raw_index), label))
            except (TypeError, ValueError):
                fallback_labels.append(label)

        if isinstance(data.get("no_error"), dict):
            item = data["no_error"]
            add_label(item.get("index", item.get("id", 0)), item.get("code", "OK"))
        if isinstance(data.get("other"), dict):
            item = data["other"]
            has_other_label = True
            add_label(item.get("index", item.get("id", "OTHER")), item.get("code", "OTHER"))
        for phoneme_group in (data.get("phonemes") or {}).values():
            if not isinstance(phoneme_group, dict):
                continue
            for category_key, category in (phoneme_group.get("categories") or {}).items():
                if not isinstance(category, dict):
                    continue
                add_label(category.get("index", category.get("id")), category.get("code", category_key))
                for fine_key, fine_label in (category.get("labels") or {}).items():
                    if isinstance(fine_label, dict):
                        add_label(fine_label.get("index", fine_label.get("id")), fine_label.get("code", fine_key))

        for key, item in data.items():
            if key in {"schema_version", "description", "phonemes", "no_error", "other"}:
                continue
            if str(key).upper() == "OTHER":
                has_other_label = True
            if isinstance(item, int):
                indexed_labels.append((item, normalize_label(key)))
                continue
            if isinstance(item, dict):
                raw_index = item.get("index", item.get("id"))
                if key == "no_error" or str(raw_index) == "0":
                    label = "OK"
                else:
                    label = normalize_label(item.get("code", key))
                try:
                    indexed_labels.append((int(raw_index), label))
                except (TypeError, ValueError):
                    fallback_labels.append(label)
                continue
            if str(key).isdigit():
                indexed_labels.append((int(key), normalize_label(key)))
            else:
                fallback_labels.append(normalize_label(key))

        for _, label in sorted(indexed_labels):
            if label not in labels:
                labels.append(label)
        for label in fallback_labels:
            if label not in labels:
                labels.append(label)

    if "0" in labels and "OK" not in labels:
        labels[labels.index("0")] = "OK"
    if "OK" not in labels:
        labels.insert(0, "OK")
    if not has_other_label and "OTHER" not in labels:
        labels.append("OTHER")
        
    return labels

DEFAULT_LABELS = load_labels_from_map(LABEL_MAP_PATH)
label_vocab = LabelVocab(DEFAULT_LABELS)


def build_frame_labels(phonemes, labels, num_frames, audio_len, vocab):
    """
    Convert phoneme-level labels to frame-level labels for CRF.

    Each WavLM output frame corresponds to a time window.
    This function maps phoneme boundaries to frame indices to label IDs.

    Args:
        phonemes: list of dicts with {"s": start_sec, "e": end_sec, "phone": "..."}
        labels:   list of label ID strings matching each phoneme (e.g. "0", "1")
        num_frames: number of output frames from WavLM (T dimension)
        audio_len: total number of audio samples (at 16kHz)
        vocab: LabelVocab instance

    Returns:
        torch.LongTensor of shape (num_frames,)
    """
    # Time duration per WavLM frame in seconds.
    frame_time = (audio_len / 16000.0) / max(num_frames, 1)

    # Default: all frames are "OK"
    frame_labels = [vocab.stoi["OK"]] * num_frames

    for p, label in zip(phonemes, labels):
        start_frame = int(p["s"] / frame_time)
        end_frame = int(p["e"] / frame_time)

        label_id = vocab.stoi.get(str(label), vocab.stoi["OK"])

        for i in range(start_frame, min(end_frame, num_frames)):
            frame_labels[i] = label_id

    return torch.tensor(frame_labels, dtype=torch.long)


def build_frame_mask(audio_lens, num_frames, batch_size, device):
    """
    Build a padding-aware mask for the CRF.
    
    Args:
        audio_lens: List of original audio lengths (in samples)
        num_frames: Total frames output by WavLM for this batch
        batch_size: Number of samples in batch
        device: 'cpu' or 'cuda'
        
    Returns:
        torch.BoolTensor of shape (B, T_frames)
    """
    mask = torch.zeros((batch_size, num_frames), dtype=torch.bool, device=device)
    
    for b, length in enumerate(audio_lens):
        # Calculate how many real frames this audio sample has
        # WavLM downsamples audio by about 320x.
        # We can also infer it from the emissions shape in the model
        # Here we use the proportion of audio length
        real_frames = int((length / max(audio_lens)) * num_frames)
        mask[b, :min(real_frames, num_frames)] = True
        
    return mask


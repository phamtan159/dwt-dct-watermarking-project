"""
Utility functions for pronunciation error detection.
- LabelVocab: maps error labels to indices
- build_frame_labels: aligns phoneme-level labels to wav2vec2 frame indices
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
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "../data/label_map.json")

def load_labels_from_map(path):
    if not os.path.exists(path):
        return ["0", "OTHER"]
    
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
                label = str(item)
                if label not in labels:
                    labels.append(label)
        elif isinstance(data[0], dict):
            # List of dicts (prefer numeric "index", fall back to "id")
            for item in data:
                label = None
                if "index" in item:
                    label = str(item["index"])
                elif "id" in item:
                    label = str(item["id"])
                    
                if label and label not in labels:
                    labels.append(label)
                    
    # Support dict format used by data/label_map.json
    elif isinstance(data, dict):
        indexed_labels = []
        fallback_labels = []
        for key, item in data.items():
            if not isinstance(item, dict):
                continue
            if str(key).upper() == "OTHER":
                has_other_label = True
            if "index" in item:
                indexed_labels.append((int(item["index"]), str(item["index"])))
            elif "id" in item:
                fallback_labels.append(str(item["id"]))
            elif str(key).isdigit():
                indexed_labels.append((int(key), str(key)))

        for _, label in sorted(indexed_labels):
            if label not in labels:
                labels.append(label)
        for label in fallback_labels:
            if label not in labels:
                labels.append(label)

    if "0" not in labels:
        labels.insert(0, "0")
    if not has_other_label and "OTHER" not in labels:
        labels.append("OTHER")
        
    return labels

DEFAULT_LABELS = load_labels_from_map(LABEL_MAP_PATH)
label_vocab = LabelVocab(DEFAULT_LABELS)


def build_frame_labels(phonemes, labels, num_frames, audio_len, vocab):
    """
    Convert phoneme-level labels to frame-level labels for CRF.

    Each wav2vec2 output frame corresponds to a time window.
    This function maps phoneme boundaries → frame indices → label IDs.

    Args:
        phonemes: list of dicts with {"s": start_sec, "e": end_sec, "phone": "..."}
        labels:   list of label ID strings matching each phoneme (e.g. "0", "1")
        num_frames: number of output frames from wav2vec2 (T dimension)
        audio_len: total number of audio samples (at 16kHz)
        vocab: LabelVocab instance

    Returns:
        torch.LongTensor of shape (num_frames,)
    """
    # Time duration per frame
    frame_time = audio_len / num_frames

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
        num_frames: Total frames output by wav2vec2 for this batch
        batch_size: Number of samples in batch
        device: 'cpu' or 'cuda'
        
    Returns:
        torch.BoolTensor of shape (B, T_frames)
    """
    mask = torch.zeros((batch_size, num_frames), dtype=torch.bool, device=device)
    
    for b, length in enumerate(audio_lens):
        # Calculate how many real frames this audio sample has
        # Wav2Vec2 downsamples by ~320x (20ms hops at 16kHz)
        # We can also infer it from the emissions shape in the model
        # Here we use the proportion of audio length
        real_frames = int((length / max(audio_lens)) * num_frames)
        mask[b, :min(real_frames, num_frames)] = True
        
    return mask


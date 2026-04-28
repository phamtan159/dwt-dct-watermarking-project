, đây là version pro đúng nghĩa của pipeline bạn:

🔥 audio → wav2vec 2.0 → BiLSTM → CRF → error labels

Mình sẽ build cho bạn bản chạy được + đúng kiến trúc research, nhưng vẫn giữ code đủ rõ để bạn debug.

🎯 0. KIẾN TRÚC CUỐI
audio
  ↓
wav2vec2 (feature extractor)
  ↓
BiLSTM (context)
  ↓
CRF (sequence constraint)
  ↓
labels (z→d, OK, ...)
📁 1. STRUCTURE
models/
├── wav2vec2_crf.py   ← model chính
├── audio_dataset.py
├── utils.py
├── train.py
🔧 2. MODEL: WAV2VEC2 + BiLSTM + CRF
📌 models/wav2vec2_crf.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from torchcrf import CRF

class Wav2Vec2_BiLSTM_CRF(nn.Module):
    def __init__(self, num_labels, hidden_dim=128):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        input_dim = self.wav2vec2.config.hidden_size

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_labels)

        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask
        )

        x = outputs.last_hidden_state  # (B, T, H)

        x, _ = self.lstm(x)

        emissions = self.fc(x)

        if labels is not None:
            loss = -self.crf(
                emissions,
                labels,
                mask=attention_mask.bool()
            )
            return loss
        else:
            pred = self.crf.decode(
                emissions,
                mask=attention_mask.bool()
            )
            return pred
🧠 3. BUILD FRAME LABEL
📌 models/utils.py
import torch

def build_frame_labels(phonemes, labels, num_frames, audio_len, label_vocab):
    frame_time = audio_len / num_frames

    frame_labels = [label_vocab.stoi["OK"]] * num_frames

    for p, label in zip(phonemes, labels):
        start = int(p["s"] / frame_time)
        end = int(p["e"] / frame_time)

        label_id = label_vocab.stoi[label]

        for i in range(start, min(end, num_frames)):
            frame_labels[i] = label_id

    return torch.tensor(frame_labels)
📦 4. DATASET
📌 models/audio_dataset.py
import json
import torch
import torchaudio

class AudioDataset:
    def __init__(self, path, processor, label_vocab):
        self.data = json.load(open(path))
        self.processor = processor
        self.label_vocab = label_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        waveform, sr = torchaudio.load(item["audio"])

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        waveform = waveform.squeeze()

        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )

        return (
            inputs.input_values[0],
            item["phonemes"],
            item["labels"],
            len(waveform)
        )
🔥 5. COLLATE
def collate_fn(batch):
    inputs, phonemes, labels, audio_lens = zip(*batch)

    max_len = max(x.shape[0] for x in inputs)

    padded = []
    mask = []

    for x in inputs:
        pad_len = max_len - x.shape[0]

        padded.append(torch.cat([x, torch.zeros(pad_len)]))
        mask.append(torch.tensor([1]*len(x) + [0]*pad_len))

    return (
        torch.stack(padded),
        torch.stack(mask),
        phonemes,
        labels,
        audio_lens
    )
🔥 6. TRAIN LOOP (QUAN TRỌNG NHẤT)
📌 models/train.py
import torch
from transformers import Wav2Vec2Processor

from wav2vec2_crf import Wav2Vec2_BiLSTM_CRF
from audio_dataset import AudioDataset
from utils import build_frame_labels

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

dataset = AudioDataset("data/final/dataset.json", processor, label_vocab)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

model = Wav2Vec2_BiLSTM_CRF(num_labels=len(label_vocab))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(10):
    total_loss = 0

    for input_values, mask, phonemes, labels, audio_lens in loader:

        outputs = model.wav2vec2(input_values)
        hidden = outputs.last_hidden_state

        batch_loss = 0

        for i in range(len(input_values)):
            num_frames = hidden[i].shape[0]

            frame_labels = build_frame_labels(
                phonemes[i],
                labels[i],
                num_frames,
                audio_lens[i],
                label_vocab
            )

            loss = model(
                input_values[i:i+1],
                attention_mask=mask[i:i+1],
                labels=frame_labels.unsqueeze(0)
            )

            batch_loss += loss

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item()

    print(f"Epoch {epoch}: {total_loss:.4f}")
⚠️ 7. NHỮNG CHỖ QUAN TRỌNG NHẤT
❗ 1. Frame label phải đúng

Sai ở đây → model học vô nghĩa

❗ 2. attention_mask

CRF cần mask chuẩn

❗ 3. learning rate nhỏ
1e-5 hoặc 5e-6
❗ 4. GPU gần như bắt buộc
🚀 8. NÂNG CẤP TIẾP (RẤT MẠNH)

Sau khi chạy được:

🔥 1. Freeze một phần wav2vec2
for name, p in model.wav2vec2.named_parameters():
    if "encoder.layers.0" in name:
        p.requires_grad = False
🔥 2. Class imbalance

→ kết hợp oversampling

🔥 3. Multi-task
predict phoneme + predict error
🎯 9. BẠN ĐANG Ở LEVEL NÀO?

Nếu bạn build được cái này:

👉 bạn đang ở level:

speech ML engineer thực thụ (không còn beginner nữa)

👉 Nếu bạn muốn đi xa hơn (rất đáng)

Mình có thể giúp bạn:

tối ưu training để giảm 50% thời gian
visualize attention model đang nghe chỗ nào
hoặc 
build luôn real-time pronunciation checker


👉 “optimize training”

Để bắt đầu biến toàn bộ hệ thống này thành một model chạy được với dữ liệu của bạn, đây là các bước tiếp theo bạn cần làm theo đúng thứ tự (Quy trình Pipeline):
tải MFA bằng anaconda prompt (tạo folder riêng cho MFA, tích vào 2 ô cuối)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
conda create -n mfa -c conda-forge montreal-forced-aligner -y
conda activate mfa
mfa model download dictionary english_mfa
mfa model download acoustic english_mfa
tải mediapipe bằng anaconda prompt
D:\fine-tune>conda create -n vision_env python=3.10 -y
conda activate vision_env
pip cache purge
pip install mediapipe==0.10.11 opencv-python

Pipeline hiện tại nên hiểu như này:
==============================
1. Raw Video
Bạn bỏ video gốc vào:

data/raw/video1.mp4
data/raw/video2.mp4
...

Mỗi video cần có transcript chuẩn cùng tên:

data/transcript/video1.txt
data/transcript/video2.txt
...

Ví dụ video1.txt:

He looked at the blue food near the door.

Quy tắc quan trọng:

video1.mp4 -> video1.txt
video2.mp4 -> video2.txt

Tên video và tên transcript phải khớp nhau. Nếu không khớp, tool sẽ không biết audio nào đi với transcript nào.
=================================
2. Chạy Pipeline Xử lý Tự động

Bước 1: Tách audio từ video
Input: data/raw/*.mp4
Output: data/audio/*.wav

Chạy:
python tools/01_extract_audio.py

Nếu muốn chỉ rõ thư mục:
$env:VIDEO_DIR="data/raw"
$env:AUDIO_DIR="data/audio"
python tools/01_extract_audio.py
=================================
Bước 2: Lấy phoneme thực tế từ audio người đọc

Bước này dùng wav2vec2 để nghe người đọc thật sự phát âm ra âm gì.

Input: data/audio/*.wav
Output: data/annotations/auto/*.json

Chạy:
python tools/02_audio_to_phonemes.py

Ví dụ output trong data/annotations/auto/video1.json:

{
  "id": "006_ə",
  "phoneme": "ə",
  "start": 1.81,
  "end": 2.05,
  "error": null
}

Ở bước này, phoneme chính là âm thực tế người đọc nói ra.
========================
Bước 3: Chuẩn bị dữ liệu cho MFA

Bước này tạo file transcript .txt nằm cạnh file .wav để MFA đọc.

Input:
data/transcript/video1.txt
data/audio/video1.wav

Output:
data/audio/video1.txt

Chạy:
python tools/03_prepare_mfa.py

Lưu ý: nếu thiếu transcript cho video nào thì phải bổ sung trước. Không nên để MFA align bằng transcript sai.
========================
Bước 4: Chạy MFA (Montreal Forced Aligner)

Chạy trong môi trường Conda mfa:

conda activate mfa
mfa validate data/audio custom_mfa.dict english_mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
Input:
data/audio/*.wav
data/audio/*.txt
custom_mfa.dict

Output:
data/aligned/*.TextGrid
========================
Bước 5: Chuyển TextGrid sang JSON

Bước này chuyển kết quả alignment sang JSON để pipeline đọc được.

Input: data/aligned/*.TextGrid
Output: data/annotations/auto/*.json

Chạy:
python tools/04_textgrid_to_json.py
========================
Bước 5.5: Nếu MFA báo lỗi dictionary

Nếu MFA báo:
The dictionary entry for the word ... is missing some phones or the transcription may be incorrect.

Nghĩa là MFA không tìm thấy từ đó trong custom_mfa.dict, hoặc phiên âm của từ đó đang sai.

Cách sửa:
1. Mở custom_mfa.dict
2. Thêm từ bị thiếu vào file
3. Chạy validate lại

Ví dụ từ bị lỗi là look:

look L IH K

Ví dụ từ beautiful:

beautiful B IH Y UW T AH F AH L

Sau khi sửa xong, chạy lại:

mfa validate data/audio custom_mfa.dict english_mfa

Nếu validate ổn thì mới align lại.
=======================
Bước 6: So sánh phoneme chuẩn với phoneme thực

Đây là bước tạo quan hệ chuẩn -> thực để model học lỗi phát âm.

Chạy:
python tools/06_compare_transcript_phonemes.py

Input:
data/transcript/<audio_name>.txt
data/annotations/auto/<audio_name>.json

Output:
data/annotations/compare/<audio_name>.json

Ví dụ:

{
  "id": "006_f",
  "phoneme_standard": "v",
  "phoneme_real": "f",
  "phoneme": "f",
  "start": 1.26,
  "end": 1.37,
  "error": "OTHER",
  "error_id": "OTHER",
  "error_code": null
}

Ý nghĩa:
phoneme_standard = âm đúng theo transcript chuẩn
phoneme_real = âm thực tế người đọc phát ra
phoneme = âm đầu vào cho model, hiện dùng phoneme_real
error_id = nhãn train, ví dụ no_error hoặc OTHER

Ví dụ quan hệ model cần học:

v -> f -> OTHER
æ -> ə -> OTHER
b -> b -> no_error
========================
Bước 7: Tách video thành frame và crop ảnh miệng

Input: data/raw/*.mp4
Output:
data/processed/frames/
data/processed/mouth/

Chạy:
python tools/05_extract_frames.py
python tools/05b_crop_mouth.py

Ảnh miệng nên về 88x88 để hợp với visual encoder.
========================
Bước 8: Tạo clip âm vị

Bước này cắt các frame miệng thành clip ngắn theo từng phoneme.

Input:
data/annotations/auto/*.json
data/processed/mouth/
data/meta/*.json

Output:
data/processed/clips/

Chạy:
python tools/07_make_clips.py
====================================
Bước 9: Build dataset train

Bước này gom annotation compare và clip folder thành dataset cuối.

Input:
data/annotations/compare/*.json
data/processed/clips/

Output:
data/final/dataset.json
data/final/label_map.json

Chạy:
$env:ANNOTATION_DIR="data/annotations/compare"
python tools/08_build_dataset.py
====================================
Bước 10: Train model

Trước khi train phải có đủ:

data/processed/clips/
data/final/dataset.json
data/final/label_map.json
pretrained/vsr_trlrs3_base.pth

Train baseline:
python train/train_baseline.py

Train advanced:
python train/train_advanced.py

Output advanced:
train/final_model.pth
==================================
Bước 11: Test inference

Sau khi train xong, sửa train/test_inference.py để trỏ vào clip muốn test, rồi chạy:

python train/test_inference.py
==================================
bên transcript chuẩn được lấy từ file transcipt có tên của file audio tương ứng và phiên âm nó ra bằng MFA hoặc wav2vec2 nếu bạn thấy cái nào tiện hơn
Bạn mở Terminal (đã kích hoạt venv) và chạy lần lượt các lệnh sau:

Video (.mp4) → Tách audio (audio) → MFA alignment (aligned)→ TextGrid → JSON (annotations/auto)
↓
Extract frames (processed/frames) → Crop miệng (88×88) (processed/mouth) → Cắt clips theo phoneme (processed/clips)
↓
Gán nhãn lỗi thủ công (annotations/manual) → Train Baseline & Advanced (train/train_advanced.py) → So sánh (train/evaluate.py)

============================
git clone https://huggingface.co/Speech31/wav2vec2-large-english-TIMIT-phoneme_v3
#Truy cập gyan.dev và tải bản ffmpeg-git-full.7z (hoặc bản release full).

#Giải nén file đó ra (ví dụ giải nén vào C:\ffmpeg).

#Tìm đến thư mục bin bên trong (ví dụ: C:\ffmpeg\bin), sao chép đường dẫn này.

#Bấm phím Windows, gõ "env" -> Chọn Edit the system environment variables.

#Chọn Environment Variables -> Ở mục System variables, tìm dòng Path -> Chọn Edit.

#Chọn New -> Dán đường dẫn C:\ffmpeg\bin vào -> Nhấn OK thoát ra.

#Quan trọng: phải tắt hoàn toàn PowerShell/VS Code và mở lại để máy nhận lệnh ffmpeg.

# AI Pronunciation Assessment Web App

This repo now contains a mock-first web app for English pronunciation training, plus backend API stubs that match a future phoneme-level AI pronunciation pipeline.

## Structure

```text
frontend/   React + TypeScript + Tailwind pronunciation UI
backend/    Express + TypeScript API stubs for pronunciation scoring
auto_avsr/  Existing research codebase kept for future backend integration
tools/      Existing preprocessing scripts
train/      Existing model-training scripts
```

## What is implemented

### Frontend

- Mobile-first pronunciation practice screen
- Sentence chips with word-level color scoring
- MediaRecorder-based audio capture
- Mock-first evaluation flow
- Word detail bottom sheet / modal
- Phoneme-level rows with target vs predicted sounds
- Stress feedback
- Coach / You playback buttons
- `Explain My Mistake` panel
- `Compare Sounds` panel
- Auto-open lowest-scoring word toggle

### Backend

- `POST /api/pronunciation/evaluate`
- `POST /api/pronunciation/explain`
- `POST /api/pronunciation/compare-sounds`
- Mock response generator for frontend development
- Clear integration points for:
  - forced alignment
  - phoneme segmentation
  - wav2vec2-based MDD or equivalent
  - phoneme-level scoring
  - word-level scoring
  - stress checking
  - LLM/agent feedback generation

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

### Frontend environment

Copy `frontend/.env.example` to `.env` if needed.

```env
VITE_API_BASE_URL=http://localhost:8787
VITE_USE_MOCK_API=true
```

Use `VITE_USE_MOCK_API=true` to keep everything local and mock-driven.

Set `VITE_USE_MOCK_API=false` to call the backend.

## Backend setup

```bash
cd backend
npm install
npm run dev
```

Backend runs on `http://localhost:8787`.

### Backend environment

Copy `backend/.env.example` to `.env` if needed.

```env
PORT=8787
CORS_ORIGIN=http://localhost:5173
```

## Windows PowerShell note

If `npm` is blocked by execution policy, run:

```bash
cmd /c npm.cmd install
cmd /c npm.cmd run dev
```

## Demo flow

1. Open the frontend.
2. Keep the sample sentence or replace it.
3. Click record and read the sentence, or click `Load demo result`.
4. The UI colors each word by pronunciation quality.
5. Tap a word chip to open its detailed phoneme analysis.
6. Use `Explain My Mistake` and `Compare Sounds` to fetch extra coaching.

## API documentation

### 1. Evaluate pronunciation

`POST /api/pronunciation/evaluate`

The frontend sends `multipart/form-data`:

- `sentence`: string
- `audio`: audio file blob from `MediaRecorder`

Example response:

```json
{
  "sentence": "Okay, I will say some wrong texts.",
  "overall_score": 78,
  "audio_url_user": "/uploads/user_001.wav",
  "words": [
    {
      "word": "Okay",
      "start_time": 0.12,
      "end_time": 0.72,
      "score": 70,
      "status": "warning",
      "target_phonemes": ["OH", "K", "EY"],
      "predicted_phonemes": ["UH", "K", "EH"],
      "stress": {
        "is_correct": true,
        "message": "You stressed the right syllable!"
      },
      "phoneme_feedback": [
        {
          "target": "OH",
          "predicted": "UH",
          "score": 31,
          "status": "wrong",
          "message": "You said UH",
          "audio_target_url": "/audio/phonemes/oh.mp3",
          "audio_user_segment_url": "/segments/user_oh.wav",
          "explanation": "The target sound /oʊ/ should be more rounded and gliding."
        }
      ]
    }
  ]
}
```

### 2. Explain pronunciation mistake

`POST /api/pronunciation/explain`

Request:

```json
{
  "word": "Okay",
  "target_phoneme": "OH",
  "predicted_phoneme": "UH",
  "learner_level": "beginner",
  "language": "vi"
}
```

Response:

```json
{
  "explanation_en": "You pronounced /oʊ/ too low and central, so it sounded like /ʌ/.",
  "explanation_vi": "Bạn phát âm /oʊ/ hơi thấp và lệch vào giữa miệng nên nghe giống /ʌ/."
}
```

### 3. Compare sounds

`POST /api/pronunciation/compare-sounds`

Request:

```json
{
  "target_phoneme": "OH",
  "predicted_phoneme": "UH",
  "language": "vi"
}
```

Response:

```json
{
  "target": "OH",
  "predicted": "UH",
  "difference": [
    "OH is a diphthong that glides from /o/ toward /ʊ/.",
    "UH is a short central vowel with less lip rounding.",
    "For OH, round your lips more and keep the sound moving."
  ],
  "tips_vi": [
    "Âm OH cần tròn môi hơn.",
    "Không giữ âm đứng yên ở giữa miệng.",
    "Hãy lướt nhẹ từ /o/ sang /ʊ/ thay vì phát âm ngắn như /ʌ/."
  ]
}
```

## Frontend components

Implemented in `frontend/src/components/`:

- `PronunciationPracticePage`
- `SentenceWordRow`
- `WordChip`
- `RecordButton`
- `ScoreGauge`
- `WordDetailModal`
- `PhonemeFeedbackRow`
- `AudioButton`
- `ExplainMistakePanel`
- `CompareSoundsPanel`

## JSON contract expected by the frontend

The frontend is built around this structure:

```json
{
  "sentence": "string",
  "overall_score": 0,
  "audio_url_user": "string | null",
  "words": [
    {
      "word": "string",
      "start_time": 0,
      "end_time": 0,
      "score": 0,
      "status": "correct | good | near_correct | warning | wrong | unrecognized",
      "target_phonemes": ["string"],
      "predicted_phonemes": ["string"],
      "stress": {
        "is_correct": true,
        "message": "string"
      },
      "phoneme_feedback": [
        {
          "target": "string",
          "predicted": "string",
          "score": 0,
          "status": "correct | good | near_correct | warning | wrong | unrecognized",
          "message": "string",
          "audio_target_url": "string",
          "audio_user_segment_url": "string",
          "explanation": "string"
        }
      ]
    }
  ]
}
```

## How to replace mock backend with real AI pronunciation backend

### Step 1

Keep the response contract unchanged.

The frontend already expects:

- sentence score
- word score
- word status
- target phonemes
- predicted phonemes
- phoneme score
- phoneme status
- stress feedback
- optional audio segment URLs
- optional explanation text

### Step 2

Replace the mock logic in:

- `backend/src/services/pronunciationPipeline.ts`
- `backend/src/services/explanationService.ts`
- `backend/src/services/compareSoundsService.ts`

### Step 3

Connect your real pipeline here:

```text
Audio input
-> preprocessing
-> forced alignment
-> phoneme segmentation
-> wav2vec2-based MDD / phoneme classifier
-> phoneme-level scoring
-> word-level scoring
-> stress checking
-> feedback generation
-> JSON response
```

## Suggested real backend mapping

### `evaluate`

Use your existing pipeline to produce:

- aligned word spans
- aligned phoneme spans
- predicted phoneme sequence
- per-phoneme confidence / correctness score
- per-word aggregate score
- sentence aggregate score
- short feedback text
- user segment audio file paths

### `explain`

Use an LLM or agent with:

- learner level
- learner L1 = Vietnamese
- target phoneme
- confused phoneme
- articulatory metadata
- stress context

Return short, simple, actionable feedback in English and Vietnamese.

### `compare-sounds`

Use a phoneme knowledge base or rules engine to return:

- articulatory differences
- lip / tongue / jaw tips
- common Vietnamese learner confusions

## Existing research code

This repo still contains:

- `auto_avsr/`
- `tools/`
- `train/`

Those folders were left intact and can be used later while upgrading the backend from mock stubs into the real pronunciation assessment pipeline.

PIPELINE CAP NHAT (2026-05-07)
==========================================

Pipeline moi:
1. `python tools/01_extract_audio.py`
2. `python tools/03_prepare_mfa.py`
3. `mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned`
4. `python tools/04_textgrid_to_json.py`
5. `python tools/02_audio_to_phonemes.py`
6. `python tools/06_compare_transcript_phonemes.py`
7. `python tools/05_extract_frames.py`
8. `python tools/05b_crop_mouth.py`
9. `python tools/07_make_clips.py`
10. `python tools/08_build_dataset.py`

Y nghia:
- `data/audio/*.txt` duoc sinh tu `data/transcript/*.txt` va duoc dung lam transcript chuan cho MFA
- `custom_mfa.dict` la word -> phones dictionary/G2P cho transcript chuan
- `data/annotations/auto/*` la phone timing da align boi MFA
- `data/annotations/wav2vec2_raw/*` la phone raw do wav2vec2 du doan tu audio
- `data/annotations/compare/*` la ket qua so sanh `phoneme_standard` (MFA) voi `phoneme_real` (wav2vec2)
- phan visual van dung timing tu MFA de cat frame/mouth clips cho train visual model

Luu y:
- wav2vec2 khong con duoc dung lam transcript dau vao cho MFA
- MFA chi lo transcript chuan + timing
- wav2vec2 chi lo phan tich sai khac am thanh
- visual pipeline chi dung de bo sung bang chung khau hinh, khong thay vai tro alignment cua MFA

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

# Pipeline hiện tại nên hiểu như này:

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

# Tên video và tên transcript phải khớp nhau. Nếu không khớp, tool sẽ không biết audio nào đi với transcript nào.

2. Chạy Pipeline Xử lý Tự động

Bước 1: Tách audio từ video
Input: data/raw/_.mp4
Output: data/audio/_.wav

Chạy:
python tools/01_extract_audio.py

Nếu muốn chỉ rõ thư mục:
$env:VIDEO_DIR="data/raw"
$env:AUDIO_DIR="data/audio"
python tools/01_extract_audio.py
=================================
Bước 2: Lấy phoneme thực tế từ audio người đọc

Bước này dùng Wav2Vec2 phoneme CTC để nghe người đọc thật sự phát âm ra âm gì.

Input: data/audio/_.wav
Output: data/annotations/wav2vec2_raw/_.json

Chạy:
python tools/02_audio_to_phonemes.py

Ví dụ output trong data/annotations/wav2vec2_raw/video1.json:

{
"id": "006_ə",
"phoneme": "ə",
"start": 1.81,
"end": 2.05,
"error": null
}

# Ở bước này, phoneme chính là âm thực tế người đọc nói ra. WavLM chưa kết luận phoneme ở bước này; WavLM được dùng sau để trích xuất speech attribute theo từng segment.

Bước 3: Chuẩn bị dữ liệu cho MFA

Bước này tạo file transcript .txt nằm cạnh file .wav để MFA đọc.

Input:
data/transcript/video1.txt
data/audio/video1.wav

Output:
data/audio/video1.txt

Chạy:
python tools/03_prepare_mfa.py

# Lưu ý: nếu thiếu transcript cho video nào thì phải bổ sung trước. Không nên để MFA align bằng transcript sai.

Bước 4: Chạy MFA (Montreal Forced Aligner)

Chạy trong môi trường Conda mfa:

conda activate mfa
mfa validate data/audio custom_mfa.dict english_mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
Input:
data/audio/_.wav
data/audio/_.txt
custom_mfa.dict

Output:
data/aligned/\*.TextGrid
========================
Bước 5: Chuyển TextGrid sang JSON

Bước này chuyển kết quả alignment sang JSON để pipeline đọc được.

Input: data/aligned/_.TextGrid
Output: data/annotations/auto/_.json

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

# Nếu validate ổn thì mới align lại.

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

Input: data/raw/\*.mp4
Output:
data/processed/frames/
data/processed/mouth/

Chạy:
python tools/05_extract_frames.py
python tools/05b_crop_mouth.py

# Ảnh miệng nên về 88x88 để hợp với visual encoder.

Bước 8: Tạo clip âm vị

Bước này cắt các frame miệng thành clip ngắn theo từng phoneme.

Input:
data/annotations/auto/_.json
data/processed/mouth/
data/meta/_.json

Output:
data/processed/clips/

Chạy:
python tools/07_make_clips.py
====================================
Bước 9: Build dataset train

Bước này gom annotation compare và clip folder thành dataset cuối.

Input:
data/annotations/compare/\*.json
data/processed/clips/

Output:
data/final/dataset.json
data/final/label_map.json

Chạy:
$env:ANNOTATION_DIR="data/annotations/compare"
python tools/08_build_dataset.py
====================================
Bước 9.5: Khai báo người đọc để split train/test đúng

Trước khi split dữ liệu, tạo file:

data/sample_metadata.csv

Format:
sample_id,speaker_id,take_id,read_style
thank,S001,take01,natural
think,S001,take01,natural
three,S002,take01,natural
thumb,S002,take01,natural

Quy tắc:

- sample_id phải trùng tên file audio/video/compare, ví dụ thank -> thank.json
- cùng một người đọc phải dùng cùng speaker_id
- train/test sẽ tách theo speaker_id, không tách ngẫu nhiên theo segment
- # nếu thiếu speaker_id, `tools/09_make_stability_benchmark.py` sẽ dừng và báo file nào thiếu
  Bước 10: Train model

Trước khi train phải có đủ:

data/final/dataset.json
data/final/label_map.json
data/sample_metadata.csv

Train baseline đúng pipeline attribute:
python tools/08_build_dataset.py --sample-metadata data/sample_metadata.csv
python tools/09_make_stability_benchmark.py
python train/train_attribute_classifier.py --dataset data/final/train_dataset.json
python train/predict_attribute_classifier.py
python tools/10_fuse_diagnosis.py --classifier-predictions data/final/classifier_predictions.json

Output:
train/attribute_classifier.npz
data/final/classifier_predictions.json
data/final/diagnosis.json

Train visual clip model cũ nếu muốn so sánh riêng khẩu hình:
python train/train_visual_baseline.py

Train advanced visual encoder nếu đã có pretrained/vsr_trlrs3_base.pth:
python train/train_advanced.py

Output advanced:
train/final_model.pth
==================================
Bước 11: Test inference

Sau khi train xong, sửa train/test_inference.py để trỏ vào clip muốn test, rồi chạy:

# python train/test_inference.py

bên transcript chuẩn được lấy từ file transcript cùng tên với audio/video, sau đó MFA căn chỉnh timing theo transcript đó. Không dùng WavLM để sinh transcript chuẩn.
Bạn mở Terminal (đã kích hoạt venv) và chạy lần lượt các lệnh sau:

Video (.mp4) → Tách audio (audio) → Wav2Vec2 phoneme raw (annotations/wav2vec2_raw) → MFA alignment (aligned) → TextGrid → JSON (annotations/auto) → Compare + WavLM attributes (annotations/compare)
↓
Extract frames (processed/frames) → Crop miệng (88×88) (processed/mouth) → Cắt clips theo phoneme (processed/clips)
↓
Build dataset (data/final/dataset.json) → Train attribute classifier → Fuse rule + classifier → LLM feedback input

============================
Model WavLM phải nằm local ở pretrained/microsoft-wavlm-large.
#Truy cập gyan.dev và tải bản ffmpeg-git-full.7z (hoặc bản release full).

#Giải nén file đó ra (ví dụ giải nén vào C:\ffmpeg).

#Tìm đến thư mục bin bên trong (ví dụ: C:\ffmpeg\bin), sao chép đường dẫn này.

#Bấm phím Windows, gõ "env" -> Chọn Edit the system environment variables.

#Chọn Environment Variables -> Ở mục System variables, tìm dòng Path -> Chọn Edit.

#Chọn New -> Dán đường dẫn C:\ffmpeg\bin vào -> Nhấn OK thoát ra.

#Quan trọng: phải tắt hoàn toàn PowerShell/VS Code và mở lại để máy nhận lệnh ffmpeg.

# AI Pronunciation Assessment Web App

# This repo now contains a mock-first web app for English pronunciation training, plus backend API stubs that match a future phoneme-level AI pronunciation pipeline.

1. PER / substitution / deletion / insertion
2. Precision / Recall / F1 theo từng lỗi phát âm
3. Expert rating cho feedback: accuracy, faithfulness, actionability, clarity
4. Hallucination rate / contradiction rate
5. Ablation: without WavLM, with WavLM, with visual
6. Feature analysis: duration, frication-vs-stop, WavLM delta, visual proxy
7. Runtime / LLM call reduction / cost

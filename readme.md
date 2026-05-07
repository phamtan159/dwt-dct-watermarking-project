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

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
==========================================
Pipeline Chính

Bạn bỏ audio/video gốc vào data/raw/, và transcript chuẩn theo cùng tên vào data/transcript/.

Ví dụ:
data/raw/audio1.wav
data/transcript/audio1.txt

Transcript là câu đúng chuẩn, ví dụ:
"This thin vest is for my mother."

==========================================
Bước 1: Tách audio sang WAV

Chạy:
python tools/01_extract_audio.py

Output:
data/audio/*.wav

Tất cả audio được chuẩn hóa thành WAV mono 16kHz để MFA và model đọc ổn định.

==========================================
Bước 2: Chuẩn bị dữ liệu cho MFA

Chạy:
python tools/02_audio_to_phonemes.py
python tools/03_prepare_mfa.py

Output chính:
data/audio/<tên_file>.txt
data/annotations/auto/<tên_file>.txt

Bước này chuẩn bị text/phoneme để MFA căn thời gian với audio.

==========================================
Bước 3: Chạy MFA align

Mở Anaconda Prompt hoặc PowerShell đã init conda, rồi activate MFA:
conda activate mfa

Sau đó đứng ở thư mục project và chạy MFA:
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned

Output:
data/aligned/*.TextGrid

MFA tạo file TextGrid, trong đó có từng âm vị và mốc thời gian bắt đầu/kết thúc.

==========================================
Bước 4: Chuyển TextGrid sang JSON

Chạy:
python tools/04_textgrid_to_json.py

Output:
data/annotations/auto/*.json

Ví dụ mỗi segment:
{
  "id": "000_s",
  "phoneme": "s",
  "start": 1.23,
  "end": 1.35,
  "error": null
}

Ở đây phoneme là âm thực/aligned từ audio người đọc.

==========================================
Bước 5: So sánh âm chuẩn và âm thực

Chạy:
python tools/05_compare_transcript_phonemes.py

Tool này đọc:
data/transcript/<tên_file>.txt
data/annotations/auto/<tên_file>.json

Output:
data/annotations/compare/<tên_file>.json

Mỗi segment sẽ có cả âm chuẩn và âm thực:
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

Lưu ý:
- phoneme_standard lấy từ transcript chuẩn
- phoneme_real lấy từ audio/MFA
- không tính khoảng trống/silence như phoneme thật
- nếu standard rỗng mà real có âm thì bỏ, xem như âm thừa hoặc align noise
- nếu standard có âm mà real rỗng thì giữ, vì có thể là lỗi nuốt âm

==========================================
Bước 6: Cắt audio thành clip phoneme

Chạy:
python tools/06_make_audio_clips.py

Input:
data/audio/<tên_file>.wav
data/annotations/auto/<tên_file>.json

Output:
data/processed/clips/<tên_file>/*.wav

Bước này không trực tiếp train model, nhưng rất quan trọng để nghe lại từng segment và kiểm tra MFA cắt đúng chưa.

==========================================
Bước 7: Build dataset train

Sau khi đã có data/annotations/compare/*.json, gom toàn bộ quan hệ âm chuẩn -> âm thực thành dataset cho model học.

Chạy:
$env:ANNOTATION_DIR="data/annotations/compare"
python tools/08_build_dataset.py

Output:
data/final/dataset.json

Trong file này, mỗi audio sẽ có:
- audio: đường dẫn tới file wav
- phonemes: từng đoạn âm, lấy phoneme_real làm âm người đọc phát ra
- standard_phone: âm chuẩn từ transcript
- labels: nhãn lỗi, lấy từ error_id trong file compare

Ví dụ model học:
/v/ chuẩn nhưng người đọc ra /f/ -> label lỗi tương ứng
/θ/ chuẩn nhưng người đọc ra /t/ -> label lỗi tương ứng
OK thì giữ là OK

==========================================
Bước 8: Tạo benchmark chống quên

Bước này không bắt buộc, nhưng nên làm trước khi train thật.
Nó tách dataset thành 2 phần:
- train_dataset.json: dữ liệu chính để train
- stability_benchmark.json: tập cố định để kiểm tra model có học cái mới rồi quên cái cũ không

Chạy:
python tools/09_make_stability_benchmark.py

Output:
data/final/train_dataset.json
data/final/stability_benchmark.json

Khi train, models/train.py sẽ ưu tiên dùng train_dataset.json nếu file này tồn tại.
Nếu có stability_benchmark.json, sau mỗi epoch nó sẽ in thêm accuracy/F1 trên tập benchmark này.

==========================================
Bước 9: Train model audio

Chạy từ thư mục models:
cd models
python train.py

Model đang train theo hướng an toàn cho dataset nhỏ:
- Phase 1: freeze toàn bộ Wav2Vec2, chỉ train phần head BiLSTM + FC + CRF
- Phase 2: chỉ mở vài layer cuối của Wav2Vec2 để fine-tune nhẹ
- Có self-distillation để model mới không drift quá xa model teacher sau phase 1
- Có benchmark chống quên nếu đã tạo ở bước trên

Output:
models/checkpoints/model.pt
models/checkpoints/best_model.pt

==========================================
Bước 10: Đánh giá model

Sau khi train xong, chạy:
cd models
python evaluate.py --checkpoint checkpoints/best_model.pt

Nếu muốn đánh giá bằng benchmark:
python evaluate.py --checkpoint checkpoints/best_model.pt --data ../data/final/stability_benchmark.json
===============================


  3. Gán nhãn Lỗi (Thủ công)
  C:\Users\Windows 10\.cache\huggingface\hub\models--Speech31--wav2vec2-large-english-TIMIT-phoneme_v3\snapshots. copy file trong đây đưa vào pretrained và đổi tên thành wav2vec2_phoneme_v3. 
Đây là lúc bạn dạy cho model biết đâu là lỗi.

Vào thư mục data/annotations/auto/, copy toàn bộ các file .json sang thư mục data/annotations/manual/.
Mở các file ở mục manual lên, tìm đến các âm vị người đọc phát âm sai và sửa trường "error": null thành tên lỗi. Ví dụ: "error": "v_to_d". 4. Huấn luyện Model (Training)
Khi đã gán nhãn xong vài mẫu dữ liệu, bạn chỉ cần chạy: python train/train_advanced.py

(Hệ thống sẽ tự động tổng hợp nhãn, load file pretrained của AV-HuBERT và bắt đầu huấn luyện. Khi xong, bạn sẽ thu được file final_model.pth) data/annotations/manual/\*.json + data/processed/clips + label_map.json + pretrained/vsr_trlrs3_base.pth

Media có âm thanh (.mp4/.mp3/.wav/.m4a/...) → Tách audio WAV 16k mono (audio) → MFA alignment (aligned) → TextGrid → JSON (annotations/auto) → Audio clips theo file gốc (processed/clips/<tên_file>/)
↓
 Cắt clips theo phoneme (processed/clips)
↓
Gán nhãn lỗi thủ công (annotations/manual) → Train Baseline & Advanced (train/train_advanced.py) → So sánh (train/evaluate.py)

Bước 1 (Visual): Dùng MediaPipe kiểm tra khung xương ngoài (môi, độ mở hàm). Nếu môi chưa chu (âm /ʃ/), báo lỗi ngay lập tức về tư thế cơ mặt.
Bước 2 (Audio): Nếu khẩu hình đã chuẩn, dùng mô hình AI Speech (như Wav2Vec2 hoặc HuBERT) để phân tích sóng âm.
Bước 3 (Tổng hợp):Nếu Visual ĐÚNG + Audio SAI $\rightarrow$ Lỗi do luồng hơi hoặc vị trí lưỡi (hướng dẫn người dùng về cách đặt lưỡi). Nếu Visual SAI + Audio SAI $\rightarrow$ Lỗi do khẩu hình (hướng dẫn chu môi/mở miệng).
============================
git clone https://huggingface.co/Speech31/wav2vec2-large-english-TIMIT-phoneme_v3
#Truy cập gyan.dev và tải bản ffmpeg-git-full.7z (hoặc bản release full).

#Giải nén file đó ra (ví dụ giải nén vào C:\ffmpeg).

#Tìm đến thư mục bin bên trong (ví dụ: C:\ffmpeg\bin), sao chép đường dẫn này.

#Bấm phím Windows, gõ "env" -> Chọn Edit the system environment variables.

#Chọn Environment Variables -> Ở mục System variables, tìm dòng Path -> Chọn Edit.

#Chọn New -> Dán đường dẫn C:\ffmpeg\bin vào -> Nhấn OK thoát ra.

#Quan trọng: phải tắt hoàn toàn PowerShell/VS Code và mở lại để máy nhận lệnh ffmpeg.
============================================
📂 phoneme-recognition-english-tutor (fine-tune-2)
├── 📂 data/                    # Nơi lưu trữ toàn bộ dữ liệu
│   ├── 📂 raw/                 # [1] Video/Audio gốc (.mp4, .mp3...)
│   ├── 📂 audio/               # [2] Audio đã chuẩn hóa 16kHz (.wav)
│   ├── 📂 transcript/          # [3] Kết quả nhận diện âm vị thô
│   ├── 📂 aligned/             # [4] Kết quả căn chỉnh thời gian (.TextGrid) từ MFA
│   ├── 📂 annotations/         # [5] Nhãn dữ liệu
│   │   ├── 📂 auto/            # JSON tự động tạo từ TextGrid
│   │   └── 📂 manual/          # JSON bạn đã gán nhãn lỗi thủ công
│   ├── 📂 processed/           # [6] Audio đã cắt nhỏ theo từng âm vị
│   │   └── 📂 clips/           
│   └── 📂 final/               # [7] Dataset cuối cùng để train
│       └── 📄 dataset.json     # File tổng hợp từ thư mục manual
├── 📂 tools/                   # Các công cụ tiền xử lý (Pipeline)
│   ├── 📄 01_extract_audio.py
│   ├── 📄 02_audio_to_phonemes.py
│   ├── 📄 03_prepare_mfa.py
│   ├── 📄 04_textgrid_to_json.py
│   └── 📄 05_make_audio_clips.py
├── 📂 models/                  # Mã nguồn của mô hình AI
│   ├── 📄 wav2vec2_crf.py      # Kiến trúc mô hình Advanced (Wav2Vec2 + CRF)
│   ├── 📄 baseline_model.py    # Kiến trúc mô hình Baseline (so sánh)
│   ├── 📄 train.py             # Script chạy huấn luyện chính
│   ├── 📄 evaluate.py          # Script đánh giá độ chính xác
│   ├── 📄 predict.py           # Script chạy dự đoán thực tế
│   ├── 📄 audio_dataset.py     # Bộ nạp dữ liệu cho PyTorch
│   └── 📄 utils.py             # Các hàm bổ trợ (xử lý nhãn, vocab)
├── 📂 pretrained/              # Lưu các mô hình tải về (Wav2Vec2 phoneme...)
├── 📂 venv/                    # Môi trường ảo Python
├── 📄 custom_mfa.dict          # Từ điển âm vị cho MFA
├── 📄 requirements.txt         # Các thư viện cần cài đặt
└── 📄 readme.md                # Hướng dẫn sử dụng
PIPELINE CAP NHAT (2026-05-07)
==========================================

Pipeline moi:
1. `python tools/01_extract_audio.py`
2. `python tools/03_prepare_mfa.py`
3. `mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned`
4. `python tools/04_textgrid_to_json.py`
5. `python tools/02_audio_to_phonemes.py`
6. `python tools/05_compare_transcript_phonemes.py`
7. `python tools/06_make_audio_clips.py`
8. `python tools/08_build_dataset.py`
9. `python tools/09_make_stability_benchmark.py`

Y nghia:
- `data/audio/*.txt` duoc sinh tu `data/transcript/*.txt` va duoc dung lam transcript chuan cho MFA
- `custom_mfa.dict` la word -> phones dictionary/G2P cho transcript chuan
- `data/annotations/auto/*` la phone timing da align boi MFA
- `data/annotations/wav2vec2_raw/*` la phone raw do wav2vec2 du doan tu audio
- `data/annotations/compare/*` la ket qua so sanh `phoneme_standard` (MFA) voi `phoneme_real` (wav2vec2)

Luu y:
- wav2vec2 khong con duoc dung lam transcript dau vao cho MFA
- MFA chi lo transcript chuan + timing
- wav2vec2 chi lo phan tich sai khac am thanh

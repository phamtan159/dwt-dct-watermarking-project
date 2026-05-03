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

1. Chuẩn bị Dữ liệu thô (Raw Data)
   Bạn hãy copy các đoạn video có hình người đang đọc/phát âm (.mp4) vào thư mục data/raw/.
   Tạo các file text chứa nội dung họ đọc (transcript) và lưu vào data/transcript/ (Tên file phải trùng với video, ví dụ: video1.mp4 thì transcript là video1.txt).
2. Chạy Pipeline Xử lý Tự động (Bước 1 - 6)

Bạn mở Terminal (đã kích hoạt venv) và chạy lần lượt các lệnh sau:

Tách Audio:  D:\fine-tune\data\raw
    python tools/01_extract_audio.py
Phiên âm vào: data\annotations\auto
    python tools/02_audio_to_phonemes.py
Sau đó Dựa vào file từ điển Python (dictionary) để ánh xạ (map) 1-1 từ kết quả TIMIT sang chuẩn IPA của MFA. rồi xuất ra file txt trong (data/audio) để MFA đọc
    python tools/03_prepare_mfa.py
Chạy MFA (Montreal Forced Aligner): Lưu ý: Bạn cần có dict (từ điển) và acoustic model tiếng Việt hoặc ngôn ngữ tương ứng. Lệnh chạy sẽ tương tự:  D:\fine-tune\data\aligned
    mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
Chuyển TextGrid sang JSON: D:\fine-tune\data\annotations\auto
python tools/03_textgrid_to_json.py
Cắt Video thành các Frames: D:\fine-tune\data\processed\frames
python tools/04_extract_frames.py
Dùng AI dò mặt và Cắt riêng vùng Miệng (88x88): D:\fine-tune\data\processed\mouth
python tools/05_crop_mouth.py
python tools/05b_visual_check.py
Ghép Frames thành các Clip ngắn (chứa từng âm vị): D:\fine-tune\data\processed\clips
python tools/06_make_clips.py 3. Gán nhãn Lỗi (Thủ công)
Đây là lúc bạn dạy cho model biết đâu là lỗi.

Vào thư mục data/annotations/auto/, copy toàn bộ các file .json sang thư mục data/annotations/manual/.
Mở các file ở mục manual lên, tìm đến các âm vị người đọc phát âm sai và sửa trường "error": null thành tên lỗi. Ví dụ: "error": "v_to_d". 4. Huấn luyện Model (Training)
Khi đã gán nhãn xong vài mẫu dữ liệu, bạn chỉ cần chạy: python train/train_advanced.py

(Hệ thống sẽ tự động tổng hợp nhãn, load file pretrained của AV-HuBERT và bắt đầu huấn luyện. Khi xong, bạn sẽ thu được file final_model.pth)

Video (.mp4) → Tách audio (audio) → MFA alignment (aligned)→ TextGrid → JSON (annotations/auto)
↓
Extract frames (processed/frames) → Crop miệng (88×88) (processed/mouth) → Cắt clips theo phoneme (processed/clips)
↓
Gán nhãn lỗi thủ công (annotations/manual) → Train Baseline & Advanced (train/train_advanced.py) → So sánh (train/evaluate.py)

Bước 1 (Visual): Dùng MediaPipe kiểm tra khung xương ngoài (môi, độ mở hàm). Nếu môi chưa chu (âm /ʃ/), báo lỗi ngay lập tức về tư thế cơ mặt.
Bước 2 (Audio): Nếu khẩu hình đã chuẩn, dùng mô hình AI Speech (như Wav2Vec2 hoặc HuBERT) để phân tích sóng âm.
Bước 3 (Tổng hợp):Nếu Visual ĐÚNG + Audio SAI $\rightarrow$ Lỗi do luồng hơi hoặc vị trí lưỡi (hướng dẫn người dùng về cách đặt lưỡi). Nếu Visual SAI + Audio SAI $\rightarrow$ Lỗi do khẩu hình (hướng dẫn chu môi/mở miệng).

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

Pipeline Chính

data/raw
Bạn bỏ file gốc vào đây. Có thể là video .mp4 hoặc các file có âm thanh như .mp3, .wav, .m4a, .flac.

Tách audio sang WAV

python tools/01_extract_audio.py
Kết quả nằm ở data/audio/*.wav. Tất cả được chuẩn hóa thành WAV mono 16kHz để các bước sau đọc ổn định.

Nhận diện/chuẩn bị âm vị
python tools/02_audio_to_phonemes.py
python tools/03_prepare_mfa.py
Bước này tạo dữ liệu âm vị/txt để MFA có thể căn thời gian.
Chạy MFA (Montreal Forced Aligner): Lưu ý: Bạn cần có dict (từ điển) và acoustic model tiếng Việt hoặc ngôn ngữ tương ứng. Lệnh chạy sẽ tương tự: D:\fine-tune\data\aligned
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
MFA tạo file .TextGrid, trong đó có từng âm vị và mốc thời gian bắt đầu/kết thúc.

Chuyển TextGrid sang JSON
python tools/04_textgrid_to_json.py
Kết quả nằm ở data/annotations/auto/*.json, ví dụ mỗi segment có:

{
  "id": "000_s",
  "phoneme": "s",
  "start": 1.23,
  "end": 1.35,
  "error": null
}
Cắt audio thành clip nhỏ theo âm vị
python tools/05_make_audio_clips.py
Mỗi file gốc có thư mục riêng:

data/processed/clips/audio1/
  000_s.wav
  001_ə.wav
  ...
  3. Gán nhãn Lỗi (Thủ công)
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

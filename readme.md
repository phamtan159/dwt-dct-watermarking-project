Để bắt đầu biến toàn bộ hệ thống này thành một model chạy được với dữ liệu của bạn, đây là các bước tiếp theo bạn cần làm theo đúng thứ tự (Quy trình Pipeline):

1. Chuẩn bị Dữ liệu thô (Raw Data)
Bạn hãy copy các đoạn video có hình người đang đọc/phát âm (.mp4) vào thư mục data/raw/.
Tạo các file text chứa nội dung họ đọc (transcript) và lưu vào data/transcript/ (Tên file phải trùng với video, ví dụ: video1.mp4 thì transcript là video1.txt).
2. Chạy Pipeline Xử lý Tự động (Bước 1 - 6)
Bạn mở Terminal (đã kích hoạt venv) và chạy lần lượt các lệnh sau:

Tách Audio: python tools/01_extract_audio.py
Chạy MFA (Montreal Forced Aligner): Lưu ý: Bạn cần có dict (từ điển) và acoustic model tiếng Việt hoặc ngôn ngữ tương ứng. Lệnh chạy sẽ tương tự: mfa align data/audio data/dict.txt model_name data/aligned
Chuyển TextGrid sang JSON: python tools/03_textgrid_to_json.py
Cắt Video thành các Frames: python tools/04_extract_frames.py
Dùng AI dò mặt và Cắt riêng vùng Miệng (88x88): python tools/05_crop_mouth.py
Ghép Frames thành các Clip ngắn (chứa từng âm vị): python tools/06_make_clips.py
3. Gán nhãn Lỗi (Thủ công)
Đây là lúc bạn dạy cho model biết đâu là lỗi.

Vào thư mục data/annotations/auto/, copy toàn bộ các file .json sang thư mục data/annotations/manual/.
Mở các file ở mục manual lên, tìm đến các âm vị người đọc phát âm sai và sửa trường "error": null thành tên lỗi. Ví dụ: "error": "v_to_d".
4. Huấn luyện Model (Training)
Khi đã gán nhãn xong vài mẫu dữ liệu, bạn chỉ cần chạy: python train/train_advanced.py

(Hệ thống sẽ tự động tổng hợp nhãn, load file pretrained của AV-HuBERT và bắt đầu huấn luyện. Khi xong, bạn sẽ thu được file final_model.pth)
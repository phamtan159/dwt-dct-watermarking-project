import torch
import librosa
import os
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 1. Cấu hình file
audio_input = "audio.m4a"
audio_output = "audio.wav"

# 2. Chuyển đổi định dạng bằng pydub
print("--- Đang xử lý file âm thanh ---")
try:
    # Load file m4a
    track = AudioSegment.from_file(audio_input, format="m4a")
    
    # Ép về mono (1 kênh), sample rate 16000Hz theo yêu cầu của model Wav2Vec2
    track = track.set_frame_rate(16000).set_channels(1)
    
    # Xuất ra file wav
    track.export(audio_output, format="wav")
    print(f"--- Đã chuyển đổi thành công sang {audio_output} ---")
except Exception as e:
    print(f"Lỗi khi xử lý file: {e}")
    print("Mẹo: Đảm bảo bạn đã cài FFmpeg và thêm vào PATH hệ thống.")
    exit()

# 3. Load model (Sử dụng cache để lần sau chạy nhanh hơn)
model_name = "./wav2vec2-large-english-TIMIT-phoneme_v3"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# 4. Xử lý Audio bằng librosa
audio, sr = librosa.load(audio_output, sr=16000)
input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

# 5. Dự đoán
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print("\n" + "="*30)
print("PHONEME RESULT:", transcription[0])
print("="*30)
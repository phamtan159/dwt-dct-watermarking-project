import torch
import librosa
import os
import asyncio
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# 1. Cấu hình file
audio_input = "audio.m4a"
audio_output = "audio.wav"

def get_transcription():
    # 0. Kiểm tra Device (GPU vs CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Đang chạy trên thiết bị: {device.upper()} ---")

    # 2. Chuyển đổi định dạng bằng pydub
    print("--- Đang xử lý file âm thanh ---")
    try:
        if os.path.exists(audio_input):
            track = AudioSegment.from_file(audio_input, format="m4a")
            track = track.set_frame_rate(16000).set_channels(1)
            track.export(audio_output, format="wav")
            print(f"--- Đã chuyển đổi thành công sang {audio_output} ---")
        else:
            print(f"File {audio_input} không tồn tại.")
            return None
    except Exception as e:
        print(f"Lỗi khi xử lý file: {e}")
        return None

    # 3. Load model
    model_name = "./wav2vec2-large-english-TIMIT-phoneme_v3"
    print(f"Loading model {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

    # 4. Xử lý Audio
    audio, sr = librosa.load(audio_output, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    input_values = input_values.to(device)

    # 5. Dự đoán
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    result = transcription[0]
    print("\nPHONEME RESULT:", result)
    return result

async def main():
    # Step 1: Transcription
    phonemes = get_transcription()
    
    if not phonemes:
        print("Transcription failed.")
        return

    # Step 2: Prepare Prompt for Ollama
    # We want Ollama to provide feedback based on phonemes
    prompt = f"""
I am practicing English pronunciation. Here is the sequence of phonemes recognized from my speech:
"{phonemes}"

Can you:
1. Guess what word or sentence I was trying to say?
2. Analyze the pronunciation and tell me if it's correct?
3. Provide feedback in Vietnamese on how to improve?
"""

    print("\nStarting Ollama Analysis...")
    # Step 3: Send to Ollama and get response
    from ollama_service import get_ollama_response_async
    response = await get_ollama_response_async(prompt)
    
    if response:
        print("\nWorkflow completed successfully.")
    else:
        print("\nWorkflow failed at Ollama stage.")

if __name__ == "__main__":
    asyncio.run(main())

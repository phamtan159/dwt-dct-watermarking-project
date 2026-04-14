import os
import shutil
import tempfile
import asyncio
import torch
import librosa
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from ollama_service import get_ollama_response_async
from gemini_app import get_gemini_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./wav2vec2-large-english-TIMIT-phoneme_v3"
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(DEVICE)

@app.post("/predict")
async def predict_and_analyze(
    file: UploadFile = File(...), 
    target_text: str = Form(...),
    ai_type: str = Form("ollama") # "gemini" hoặc "ollama"
):
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, file.filename)
        wav_path = os.path.join(temp_dir, "temp.wav")
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            track = AudioSegment.from_file(input_path)
            track = track.set_frame_rate(16000).set_channels(1)
            track.export(wav_path, format="wav")
            
            audio, sr = librosa.load(wav_path, sr=16000)
            input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(DEVICE)
            
            with torch.no_grad():
                logits = model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            print(f"\n[PHONEME RECOGNITION]: {transcription}")

            # TẠO PROMPT SO SÁNH TRỰC TIẾP
            full_prompt = f"""
I am practicing English. Respond DIRECTLY and CONCISELY.
Intended Sentence/Word: "{target_text}"
Recognized Phonemes from AI: "{transcription}"

Task:
1. Compare and point out specific mistakes (wrong sounds/vowels).
2. Give short feedback in Vietnamese on how to fix.
No introduction, no yapping, just the facts.
"""
            if ai_type == "gemini":
                print("--- Sending comparison prompt to Gemini ---")
                answer = await get_gemini_response(full_prompt)
            else:
                print("--- Sending comparison prompt to Ollama ---")
                answer = await get_ollama_response_async(full_prompt)
            
            return {
                "status": "success",
                "target_text": target_text,
                "transcription": transcription,
                "ai_answer": answer,
                "ai_type": ai_type
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

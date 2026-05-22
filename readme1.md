venv\Scripts\python.exe tools/01_extract_audio.py
venv\Scripts\python.exe tools/02_audio_to_phonemes.py
venv\Scripts\python.exe tools/03_prepare_mfa.py
D:
cd "D:\A Project YTB\fine-tune-audio"
conda activate mfa
mfa validate data/audio custom_mfa.dict english_mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
venv\Scripts\python.exe tools/04_textgrid_to_json.py
venv\Scripts\python.exe tools/05_compare_transcript_phonemes.py
venv\Scripts\python.exe tools/06_make_audio_clips.py
$env:ANNOTATION_DIR="data/annotations/compare"
venv\Scripts\python.exe tools/08_build_dataset.py
venv\Scripts\python.exe tools/09_make_stability_benchmark.py
cd models
python train.py
python evaluate.py --checkpoint checkpoints/best_model.pt
python evaluate.py --checkpoint checkpoints/best_model.pt --data ../data/final/
python clear.py

python tools/01_extract_audio.py
python tools/02_audio_to_phonemes.py
python tools/03_prepare_mfa.py
==========
conda activate mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
python tools/04_textgrid_to_json.py
python tools/05_compare_transcript_phonemes.py
python tools/06_make_audio_clips.py
python tools/08_build_dataset.py
python tools/09_make_stability_benchmark.py
================
cd models
python train.py
python evaluate.py --checkpoint checkpoints/best_model.pt

python evaluate.py --checkpoint checkpoints/best_model.pt --data ../data/final/stability_benchmark.json
===============================
python clear_data.py

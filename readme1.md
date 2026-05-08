python tools/01_extract_audio.py
python tools/02_audio_to_phonemes.py
python tools/03_prepare_mfa.py
conda activate mfa
mfa validate data/audio custom_mfa.dict english_mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
python tools/04_textgrid_to_json.py
python tools/05_extract_frames.py
python tools/05b_crop_mouth.py
python tools/06_compare_transcript_phonemes.py
python tools/07_make_clips.py
python tools/08_build_dataset.py
=============
Train baseline:
python train/train_baseline.py

Train advanced:
python train/train_advanced.py
==============
python train/test_inference.py
===============
python clear_data.py 
cd "D:\A Project YTB\fine-tune-visual"
.\venv\Scripts\activate
$env:PYTHONIOENCODING='utf-8'
python clear.py
python tools/00_setup_reading_prompt_structure.py
python tools/01_extract_audio.py
python tools/02_audio_to_phonemes.py
python tools/03_prepare_mfa.py
D:
cd "D:\A Project YTB\fine-tune-visual"
conda activate mfa
mfa validate data/audio custom_mfa.dict english_mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
conda deactivate
.\venv\Scripts\activate
$env:PYTHONIOENCODING='utf-8'
python tools/04_textgrid_to_json.py
python tools/05_extract_frames.py
python tools/05b_crop_mouth.py
python tools/06_compare_transcript_phonemes.py
python tools/07_make_clips.py
python tools/08_build_dataset.py
python tools/11_extract_segment_attributes.py
python tools/12_export_label_review_files.py --overwrite
python tools/09_make_stability_benchmark.py --input data/final/segment_attributes.json --train-output data/final/train_dataset.json --benchmark-output data/final/stability_benchmark.json

## Train AI co WavLM

python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --output train/attribute_classifier_with_wavlm.npz --predictions data/final/classifier_predictions_with_wavlm.json
python train/predict_attribute_classifier.py --dataset data/final/segment_attributes.json --checkpoint train/attribute_classifier_with_wavlm.npz --output data/final/classifier_predictions_with_wavlm.json
python tools/10_fuse_diagnosis.py --dataset data/final/segment_attributes.json --classifier-predictions data/final/classifier_predictions_with_wavlm.json --output data/final/diagnosis_with_wavlm.json

## Train AI khong WavLM

python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --no-wavlm-features --output train/attribute_classifier_no_wavlm.npz --predictions data/final/classifier_predictions_no_wavlm.json
python train/predict_attribute_classifier.py --dataset data/final/segment_attributes.json --checkpoint train/attribute_classifier_no_wavlm.npz --output data/final/classifier_predictions_no_wavlm.json
python tools/10_fuse_diagnosis.py --dataset data/final/segment_attributes.json --classifier-predictions data/final/classifier_predictions_no_wavlm.json --output data/final/diagnosis_no_wavlm.json

```

```

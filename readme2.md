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
===================
python tools/12_export_label_review_files.py --overwrite
python tools/14_llm_suggest_labels.py
python tools/13_merge_human_labels.py

python tools/09_make_stability_benchmark.py --input data/final/segment_attributes.json --train-output data/final/train_dataset.json --benchmark-output data/final/stability_benchmark.json

lệnh này để làm gì

## Train AI co WavLM

python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --output train/attribute_classifier_with_wavlm.npz --predictions data/final/classifier_predictions_with_wavlm.json
python train/predict_attribute_classifier.py --dataset data/final/segment_attributes.json --checkpoint train/attribute_classifier_with_wavlm.npz --output data/final/classifier_predictions_with_wavlm.json
python tools/10_fuse_diagnosis.py --dataset data/final/segment_attributes.json --classifier-predictions data/final/classifier_predictions_with_wavlm.json --output data/final/diagnosis_with_wavlm.json

## Train AI khong WavLM

python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --no-wavlm-features --output train/attribute_classifier_no_wavlm.npz --predictions data/final/classifier_predictions_no_wavlm.json
python train/predict_attribute_classifier.py --dataset data/final/segment_attributes.json --checkpoint train/attribute_classifier_no_wavlm.npz --output data/final/classifier_predictions_no_wavlm.json
python tools/10_fuse_diagnosis.py --dataset data/final/segment_attributes.json --classifier-predictions data/final/classifier_predictions_no_wavlm.json --output data/final/diagnosis_no_wavlm.json

```
# baseline đầy đủ
python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --target label --output train/attribute_classifier_full.npz --predictions data/final/pred_full.json

# kiểm tra có học vẹt expected/observed không
python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --target label --no-phoneme-identity-features --output train/attribute_classifier_no_phoneme_identity.npz --predictions data/final/pred_no_phoneme_identity.json

# kiểm tra WavLM có giúp không
python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --target label --no-wavlm-features --output train/attribute_classifier_no_wavlm.npz --predictions data/final/pred_no_wavlm.json

# train severity riêng
python train/train_attribute_classifier.py --dataset data/final/train_dataset.json --target severity --output train/attribute_classifier_severity.npz --predictions data/final/pred_severity.json
```

python tools/15_predict_recommended_human_label.py --dataset data/final/segment_attributes.json --category-checkpoint train/final_pronunciation_category_recommended_no_wavlm.npz --label-checkpoint train/final_pronunciation_label_recommended_no_wavlm.npz --severity-checkpoint train/final_pronunciation_severity_recommended_no_wavlm.npz --output data/final/recommended_human_label_predictions_no_wavlm.json
"AI đoán những thứ cần đoán, đưa cho LLM"

python tools/16_export_direct_llm_feedback.py --input data/final/segment_attributes.json --output data/final/direct_llm_feedback_inputs.json
"gom tất cả cho LLM"
$env:LLM_API_KEY='YOUR_API_KEY'
$env:LLM_MODEL='YOUR_MODEL_NAME'

# python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.json

python tools/18_evaluate_phoneme_alignment.py --dataset data/final/segment_attributes.json --output data/final/eval_phoneme_alignment.json --per-phoneme-csv data/final/eval_phoneme_alignment_per_phoneme.csv

python tools/19_evaluate_error_detection.py --ground-truth data/final/segment_attributes_labeled.json --predictions data/final/recommended_human_label_predictions_no_wavlm.json --target label --output data/final/eval_error_detection_label_no_wavlm.json --per-class-csv data/final/eval_error_detection_label_no_wavlm_per_class.csv

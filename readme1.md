# Pipeline hien tai

Huong moi la audio-only:

```text
audio/video + transcript
-> extract audio
-> MFA + Wav2Vec2 + WavLM + Speech Attributes
-> Stage 1: phat hien dung/sai
-> Stage 2: du doan observed phoneme
-> LLM sinh feedback
```

Neu input la video, he thong chi dung video de trich audio.

## Pipeline thuc chien

Chay cac buoc nen:

```powershell
cd "D:\A Project YTB\fine-tune-visual"
.\venv\Scripts\activate
$env:PYTHONIOENCODING='utf-8'

python clear.py
python tools/00_setup_reading_prompt_structure.py
python tools/01_extract_audio.py
python tools/02_audio_to_phonemes.py
python tools/03_prepare_mfa.py
```

Chay MFA:

```powershell
conda activate mfa
mfa validate data/audio custom_mfa.dict english_mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
conda deactivate
```

Hoac:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_mfa.ps1
```

Sau MFA:

```powershell
.\venv\Scripts\activate
$env:PYTHONIOENCODING='utf-8'

python tools/04_textgrid_to_json.py
python tools/06_compare_transcript_phonemes.py
python tools/06_make_audio_clips.py
python tools/08_build_dataset.py
python tools/11_extract_segment_attributes.py
```

Stage 1:

```powershell
python tools/18_predict_mdd_classifier.py `
  --input data/final/segment_attributes.json `
  --checkpoint train/current/stage1_meta_mdd_classifier.pt `
  --output data/final/mdd_predictions.json `
  --require-posterior
```

Stage 2:

```powershell
python tools/19_predict_stage2_observed_phone.py `
  --input data/final/segment_attributes.json `
  --checkpoint train/current/stage2_observed_phone_classifier.pt `
  --mdd-predictions data/final/mdd_predictions.json `
  --output data/final/stage2_observed_phone_predictions.json `
  --require-posterior
```

Xuat input cho LLM:

```powershell
python tools/16_export_direct_llm_feedback.py `
  --input data/final/segment_attributes.json `
  --mdd-predictions data/final/mdd_predictions.json `
  --stage2-predictions data/final/stage2_observed_phone_predictions.json `
  --output data/final/direct_llm_feedback_inputs.json
```

Goi LLM:

```powershell
$env:LLM_API_KEY='YOUR_API_KEY'
$env:LLM_MODEL='YOUR_MODEL_NAME'

python tools/17_generate_direct_llm_feedback.py `
  --input data/final/direct_llm_feedback_inputs.json `
  --output data/final/direct_llm_feedback_outputs.json
```

Output chinh:

```text
data/final/segment_attributes.json
data/final/mdd_predictions.json
data/final/stage2_observed_phone_predictions.json
data/final/direct_llm_feedback_inputs.json
data/final/direct_llm_feedback_outputs.json
```

## Pipeline nghien cuu

Chi khac pipeline thuc chien o phan train/evaluate model tren L2-ARCTIC.

```powershell
python tools/30_regenerate_l2arctic_feature_cache.py `
  --corpus-root "D:\A Project YTB\L2artic" `
  --output-dir "D:\A Project YTB\L2artic\regenerated_cache_v1" `
  --device cuda `
  --resume `
  --save-every 500
```

Train Stage 1:

```powershell
python tools/31_train_l2arctic_frozen_protocol.py `
  --protocol data/protocols/l2arctic_frozen_source_vietnamese_raw_cache_v1.json `
  --summary "D:\A Project YTB\L2artic\regenerated_cache_v1\source_only_summary.csv" `
  --epochs 80 `
  --seed 13
```

Train Stage 2:

```powershell
python tools/26_train_stage2_observed_phone_classifier.py `
  --details-csv "D:\A Project YTB\L2artic\regenerated_cache_v1_full\regenerated_cache_v1\eval_l2arctic_raw_regenerated_details.csv" `
  --feature-cache "D:\A Project YTB\L2artic\regenerated_cache_v1_full\regenerated_cache_v1\meta_classifier_feature_cache.raw_v1.json" `
  --stage1-models-root "D:\A Project YTB\L2artic\regenerated_cache_v1_full\regenerated_cache_v1\models" `
  --output-dir "D:\A Project YTB\L2artic\regenerated_cache_v1_full\regenerated_cache_v1\stage2_observed_phone" `
  --epochs 120 `
  --seed 17
```

Output nghien cuu chinh:

```text
source_only_summary.csv
summary_tables.md
summary_tables_extended_metrics.md
stage2_observed_phone/stage2_summary_aggregate_vietnamese.csv
stage2_observed_phone/stage2_summary_by_run.csv
stage2_observed_phone/stage2_paper_tables.md
models/**/meta_mdd_classifier.pt
stage2_observed_phone/**/stage2_observed_phone_classifier.pt
```

## File con dung

```text
tools/01_extract_audio.py
tools/02_audio_to_phonemes.py
tools/03_prepare_mfa.py
tools/04_textgrid_to_json.py
tools/06_compare_transcript_phonemes.py
tools/06_make_audio_clips.py
tools/08_build_dataset.py
tools/11_extract_segment_attributes.py
tools/18_predict_mdd_classifier.py
tools/19_predict_stage2_observed_phone.py
tools/16_export_direct_llm_feedback.py
tools/17_generate_direct_llm_feedback.py
tools/25_train_meta_mdd_classifier.py
tools/26_train_stage2_observed_phone_classifier.py
tools/30_regenerate_l2arctic_feature_cache.py
tools/31_train_l2arctic_frozen_protocol.py
```

## File legacy khong goi trong pipeline moi

```text
tools/09_make_stability_benchmark.py
tools/10_fuse_diagnosis.py
tools/12_export_label_review_files.py
tools/13_merge_human_labels.py
tools/14_llm_suggest_labels.py
tools/15_predict_recommended_human_label.py
tools/19_evaluate_error_detection.py
train/train_attribute_classifier.py
train/predict_attribute_classifier.py
```

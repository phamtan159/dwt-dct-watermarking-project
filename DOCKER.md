# Docker Usage

Docker chi dung cho pipeline audio-only hien tai. Thu muc lam viec trong container:

```text
/workspace/fine-tune-visual
```

Build tu repository root:

```powershell
cd "D:\A Project YTB\fine-tune-visual"
docker compose build pronunciation
```

Chay pipeline:

```powershell
docker compose run --rm pronunciation python clear.py
docker compose run --rm pronunciation python tools/00_setup_reading_prompt_structure.py
docker compose run --rm pronunciation python tools/01_extract_audio.py
docker compose run --rm pronunciation python tools/02_audio_to_phonemes.py
docker compose run --rm pronunciation python tools/03_prepare_mfa.py
```

Chay MFA:

```powershell
docker compose run --rm pronunciation mfa validate data/audio custom_mfa.dict english_mfa
docker compose run --rm pronunciation mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
```

Sau MFA:

```powershell
docker compose run --rm pronunciation python tools/04_textgrid_to_json.py
docker compose run --rm pronunciation python tools/06_compare_transcript_phonemes.py
docker compose run --rm pronunciation python tools/06_make_audio_clips.py
docker compose run --rm pronunciation python tools/08_build_dataset.py
docker compose run --rm pronunciation python tools/11_extract_segment_attributes.py
docker compose run --rm pronunciation python tools/18_predict_mdd_classifier.py --input data/final/segment_attributes.json --checkpoint train/current/stage1_meta_mdd_classifier.pt --output data/final/mdd_predictions.json --require-posterior
docker compose run --rm pronunciation python tools/19_predict_stage2_observed_phone.py --input data/final/segment_attributes.json --checkpoint train/current/stage2_observed_phone_classifier.pt --mdd-predictions data/final/mdd_predictions.json --output data/final/stage2_observed_phone_predictions.json --require-posterior
docker compose run --rm pronunciation python tools/16_export_direct_llm_feedback.py --input data/final/segment_attributes.json --mdd-predictions data/final/mdd_predictions.json --stage2-predictions data/final/stage2_observed_phone_predictions.json --output data/final/direct_llm_feedback_inputs.json
```

Goi LLM:

```powershell
$env:LLM_API_KEY='YOUR_API_KEY'
$env:LLM_MODEL='YOUR_MODEL_NAME'
docker compose run --rm pronunciation python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.json
```

Output chinh:

```text
data/final/segment_attributes.json
data/final/mdd_predictions.json
data/final/stage2_observed_phone_predictions.json
data/final/direct_llm_feedback_inputs.json
data/final/direct_llm_feedback_outputs.json
```

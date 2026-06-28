# Reproducibility Guide

This guide describes how to reproduce the audio-only pronunciation pipeline.
The recommended route is Docker because it installs Python, ffmpeg, and
Montreal Forced Aligner in one controlled environment.

## 1. Recommended: Docker

Run from the repository root, the folder that contains this file and
`docker-compose.yml`:

```powershell
cd "D:\A Project YTB\fine-tune-visual"
docker compose build pronunciation
```

Download MFA models once:

```powershell
docker compose run --rm pronunciation mfa model download dictionary english_mfa
docker compose run --rm pronunciation mfa model download acoustic english_mfa
```

Prepare project artifacts before running:

```text
fine-tune-visual/pretrained/facebook-wav2vec2-lv-60-espeak-cv-ft/
fine-tune-visual/pretrained/microsoft-wavlm-large/
fine-tune-visual/pretrained/mostafaashahin-SA_US_Adult/
fine-tune-visual/train/current/stage1_meta_mdd_classifier.pt
fine-tune-visual/train/current/stage2_observed_phone_classifier.pt
```

Put input files in:

```text
fine-tune-visual/data/raw/
fine-tune-visual/data/transcript/
```

If you cloned this repository directly, these paths are relative to the project
root:

```text
pretrained/facebook-wav2vec2-lv-60-espeak-cv-ft/
pretrained/microsoft-wavlm-large/
pretrained/mostafaashahin-SA_US_Adult/
train/current/stage1_meta_mdd_classifier.pt
train/current/stage2_observed_phone_classifier.pt
data/raw/
data/transcript/
```

Run the pipeline:

```powershell
docker compose run --rm pronunciation python clear.py
docker compose run --rm pronunciation python tools/00_setup_reading_prompt_structure.py
docker compose run --rm pronunciation python tools/01_extract_audio.py
docker compose run --rm pronunciation python tools/02_audio_to_phonemes.py
docker compose run --rm pronunciation python tools/03_prepare_mfa.py

docker compose run --rm pronunciation mfa validate data/audio custom_mfa.dict english_mfa
docker compose run --rm pronunciation mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned

docker compose run --rm pronunciation python tools/04_textgrid_to_json.py
docker compose run --rm pronunciation python tools/06_compare_transcript_phonemes.py
docker compose run --rm pronunciation python tools/06_make_audio_clips.py
docker compose run --rm pronunciation python tools/08_build_dataset.py
docker compose run --rm pronunciation python tools/11_extract_segment_attributes.py
```

Run Stage 1 and Stage 2 classifiers:

```powershell
docker compose run --rm pronunciation python tools/18_predict_mdd_classifier.py --input data/final/segment_attributes.json --checkpoint train/current/stage1_meta_mdd_classifier.pt --output data/final/mdd_predictions.json --require-posterior

docker compose run --rm pronunciation python tools/19_predict_stage2_observed_phone.py --input data/final/segment_attributes.json --checkpoint train/current/stage2_observed_phone_classifier.pt --mdd-predictions data/final/mdd_predictions.json --output data/final/stage2_observed_phone_predictions.json --require-posterior
```

Export LLM input:

```powershell
docker compose run --rm pronunciation python tools/16_export_direct_llm_feedback.py --input data/final/segment_attributes.json --mdd-predictions data/final/mdd_predictions.json --stage2-predictions data/final/stage2_observed_phone_predictions.json --output data/final/direct_llm_feedback_inputs.json
```

Generate LLM feedback if an OpenAI-compatible API is available:

```powershell
$env:LLM_API_KEY='YOUR_API_KEY'
$env:LLM_MODEL='YOUR_MODEL_NAME'

docker compose run --rm pronunciation python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.json
```

Main outputs:

```text
data/final/segment_attributes.json
data/final/mdd_predictions.json
data/final/stage2_observed_phone_predictions.json
data/final/direct_llm_feedback_inputs.json
data/final/direct_llm_feedback_outputs.json
```

## 2. Native Windows Setup

Install Miniconda, then create the MFA environment:

```powershell
conda create -n mfa -c conda-forge montreal-forced-aligner ffmpeg -y
conda activate mfa
mfa model download dictionary english_mfa
mfa model download acoustic english_mfa
conda deactivate
```

Create the project Python environment:

```powershell
cd "D:\A Project YTB\fine-tune-visual"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Then run the same pipeline commands from `readme1.md`.

## 3. What Is Not Stored In Git

The repository intentionally does not store large or generated files:

```text
venv/
data/raw/
data/audio/
data/aligned/
data/processed/
data/final/
pretrained/
```

The small runtime classifiers are stored in:

```text
train/current/
```

Pretrained acoustic models should be downloaded separately and placed under
`pretrained/`.

## 4. Notes For Paper Reproduction

- Docker is the preferred environment for reproducibility.
- MFA is used only for forced alignment, not for final error decisions.
- Wav2Vec2 posterior, Speech Attributes, duration/energy, and WavLM-derived
  features are exported into `segment_attributes.json`.
- Stage 1 predicts correct/incorrect.
- Stage 2 predicts the observed phoneme for segments flagged by Stage 1.
- The LLM receives compact evidence and generates learner-facing feedback.

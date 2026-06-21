# Docker Usage

This Docker setup is for the current `fine-tune-visual` pipeline. The main
working directory inside the container is:

```text
/workspace/fine-tune-visual
```

The image does not copy `data/`, `pretrained/`, or `venv/`. Those stay on the host
and are mounted into the container by `docker-compose.yml`.

The pipeline uses local model folders under:

```text
pretrained/
```

## Build

Run from the workspace root:

```powershell
cd "D:\A Project YTB"
docker compose build visual
```

## First-Time MFA Model Setup

The container includes Montreal Forced Aligner, but the MFA acoustic/dictionary
models may need to be downloaded once into the `mfa-cache` Docker volume:

```powershell
docker compose run --rm visual mfa model download dictionary english_mfa
docker compose run --rm visual mfa model download acoustic english_mfa
```

If you run fully offline, download the MFA models on a machine with network and
copy them into the mounted MFA cache.

## Run The Pipeline

From the workspace root:

```powershell
docker compose run --rm visual python clear.py
docker compose run --rm visual python tools/00_setup_reading_prompt_structure.py
docker compose run --rm visual python tools/01_extract_audio.py
docker compose run --rm visual python tools/02_audio_to_phonemes.py
docker compose run --rm visual python tools/03_prepare_mfa.py
```

Run MFA inside the same container:

```powershell
docker compose run --rm visual mfa validate data/audio custom_mfa.dict english_mfa
docker compose run --rm visual mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
```

Continue after MFA:

```powershell
docker compose run --rm visual python tools/04_textgrid_to_json.py
docker compose run --rm visual python tools/05_extract_frames.py
docker compose run --rm visual python tools/05b_crop_mouth.py
docker compose run --rm visual python tools/06_compare_transcript_phonemes.py
docker compose run --rm visual python tools/07_make_clips.py
docker compose run --rm visual python tools/08_build_dataset.py
docker compose run --rm visual python tools/11_extract_segment_attributes.py
```

Important output:

```text
data/final/segment_attributes.json
```

## Classifier Route

Create review files, merge labels, train/evaluate, then predict human-label shaped
output:

```powershell
docker compose run --rm visual python tools/12_export_label_review_files.py --overwrite
docker compose run --rm visual python tools/13_merge_human_labels.py

docker compose run --rm visual python train/train_attribute_classifier.py --dataset data/final/segment_attributes_labeled.json --target category --feature-set recommended --train-on-all --epochs 120 --output train/final_pronunciation_category_recommended_wavlm.npz --predictions data/final/final_predictions_category_recommended_wavlm.json
docker compose run --rm visual python train/train_attribute_classifier.py --dataset data/final/segment_attributes_labeled.json --target label --feature-set recommended --train-on-all --epochs 120 --output train/final_pronunciation_label_recommended_wavlm.npz --predictions data/final/final_predictions_label_recommended_wavlm.json
docker compose run --rm visual python train/train_attribute_classifier.py --dataset data/final/segment_attributes_labeled.json --target severity --feature-set recommended --train-on-all --epochs 120 --output train/final_pronunciation_severity_recommended_wavlm.npz --predictions data/final/final_predictions_severity_recommended_wavlm.json

docker compose run --rm visual python tools/15_predict_recommended_human_label.py --dataset data/final/segment_attributes.json --output data/final/recommended_human_label_predictions.json
```

For the no-WavLM ablation, add `--no-wavlm-features` to each train command and
pass the no-WavLM checkpoints into `tools/15_predict_recommended_human_label.py`.

## Direct LLM Route

Export direct LLM input:

```powershell
docker compose run --rm visual python tools/16_export_direct_llm_feedback.py --input data/final/segment_attributes.json --output data/final/direct_llm_feedback_inputs.json --all-segments
```

Dry run:

```powershell
docker compose run --rm visual python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.dry_run.json --dry-run
```

Real API call:

```powershell
$env:LLM_API_KEY='YOUR_API_KEY'
$env:LLM_MODEL='YOUR_MODEL_NAME'
docker compose run --rm -e LLM_API_KEY -e LLM_MODEL visual python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.json
```

## Notes

- Local models must already exist under `fine-tune-visual/pretrained`.
- Do not mount Windows Conda folders into Docker; Docker installs its own Linux environment.
- Docker runs Linux paths internally, but all project files remain on your Windows filesystem through bind mounts.
- CPU works, but WavLM/SA extraction can be slow. For GPU, install NVIDIA Container Toolkit and run with Docker's GPU options or add a GPU compose override.

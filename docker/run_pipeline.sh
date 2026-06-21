#!/usr/bin/env bash
set -euo pipefail

export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ "${SKIP_CLEAR:-0}" != "1" ]]; then
  python clear.py
fi

python tools/00_setup_reading_prompt_structure.py
python tools/01_extract_audio.py
python tools/02_audio_to_phonemes.py
python tools/03_prepare_mfa.py

if [[ "${SKIP_MFA_VALIDATE:-0}" != "1" ]]; then
  mfa validate data/audio custom_mfa.dict english_mfa
fi

clean_flag=(--clean)
if [[ "${MFA_NO_CLEAN:-0}" == "1" ]]; then
  clean_flag=()
fi
mfa align "${clean_flag[@]}" data/audio custom_mfa.dict english_mfa data/aligned

python tools/04_textgrid_to_json.py
python tools/05_extract_frames.py
python tools/05b_crop_mouth.py
python tools/06_compare_transcript_phonemes.py
python tools/07_make_clips.py
python tools/08_build_dataset.py
python tools/11_extract_segment_attributes.py

if [[ "${RUN_DIRECT_LLM_INPUT:-0}" == "1" ]]; then
  python tools/16_export_direct_llm_feedback.py \
    --input data/final/segment_attributes.json \
    --output data/final/direct_llm_feedback_inputs.json \
    --all-segments
fi

echo "Pipeline done."
echo "Main output: data/final/segment_attributes.json"

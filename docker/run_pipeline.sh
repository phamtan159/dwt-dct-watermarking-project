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
python tools/06_compare_transcript_phonemes.py
python tools/06_make_audio_clips.py
python tools/08_build_dataset.py
python tools/11_extract_segment_attributes.py

if [[ -f train/current/stage1_meta_mdd_classifier.pt ]]; then
  python tools/18_predict_mdd_classifier.py \
    --input data/final/segment_attributes.json \
    --checkpoint train/current/stage1_meta_mdd_classifier.pt \
    --output data/final/mdd_predictions.json \
    --require-posterior
fi

if [[ -f train/current/stage2_observed_phone_classifier.pt && -f data/final/mdd_predictions.json ]]; then
  python tools/19_predict_stage2_observed_phone.py \
    --input data/final/segment_attributes.json \
    --checkpoint train/current/stage2_observed_phone_classifier.pt \
    --mdd-predictions data/final/mdd_predictions.json \
    --output data/final/stage2_observed_phone_predictions.json \
    --require-posterior
fi

if [[ "${RUN_DIRECT_LLM_INPUT:-1}" == "1" ]]; then
  python tools/16_export_direct_llm_feedback.py \
    --input data/final/segment_attributes.json \
    --mdd-predictions data/final/mdd_predictions.json \
    --stage2-predictions data/final/stage2_observed_phone_predictions.json \
    --output data/final/direct_llm_feedback_inputs.json \
    ${DIRECT_LLM_ALL_SEGMENTS:+--all-segments}
fi

echo "Pipeline done."
echo "Main outputs:"
echo "- data/final/segment_attributes.json"
echo "- data/final/mdd_predictions.json"
echo "- data/final/stage2_observed_phone_predictions.json"
echo "- data/final/direct_llm_feedback_inputs.json"

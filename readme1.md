# Pipeline hien tai: video + transcript -> feedback LLM

Huong hien tai tam thoi bo qua classifier. Pipeline van tao phoneme alignment, visual attributes, speech attributes va WavLM summary, sau do dua tung segment vao LLM de sinh:

```json
{
  "diagnosis": "...",
  "correction_steps": ["...", "..."]
}
```

## 1. Chay pipeline den segment attributes

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

Sau do chay MFA:

```powershell
D:
cd "D:\A Project YTB\fine-tune-visual"
conda activate mfa
mfa validate data/audio custom_mfa.dict english_mfa
mfa align --clean data/audio custom_mfa.dict english_mfa data/aligned
conda deactivate
```

Quay lai venv cua du an:

```powershell
.\venv\Scripts\activate
$env:PYTHONIOENCODING='utf-8'

python tools/04_textgrid_to_json.py
python tools/05_extract_frames.py
python tools/05b_crop_mouth.py
python tools/06_compare_transcript_phonemes.py
python tools/07_make_clips.py
python tools/08_build_dataset.py
python tools/11_extract_segment_attributes.py
```

Ket qua quan trong nhat cua phan nay:

```text
data/final/segment_attributes.json
```

## 2. Tao input cho LLM

```powershell
python tools/16_export_direct_llm_feedback.py --input data/final/segment_attributes.json --output data/final/direct_llm_feedback_inputs.json --all-segments
```

Tool nay dung `pronunciation_error.md` lam danh sach am vi can tap trung.

Policy:

- `focus_articulatory`: target phoneme nam trong `pronunciation_error.md` -> LLM duoc nhin sau vao duration, speech_attribute_prediction, frication_vs_stop, vowel_quality, WavLM summary/delta va visual summary.
- `nonfocus_ok`: target phoneme khong nam trong nhom focus va Wav2Vec2/MFA nghe dung ky vong -> khong can goi LLM.
- `nonfocus_substitution`: target phoneme khong nam trong nhom focus nhung bi thay the thanh am khac -> van cho LLM phan tich loi nhu thuong.
- `nonfocus_insertion_or_deletion`: target phoneme khong nam trong nhom focus va bi them/thieu -> chi feedback co ban, khong phan tich dac trung cau am.

## 3. Goi LLM de tao feedback

Dry run de kiem tra input/policy:

```powershell
python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.dry_run.json --dry-run
```

Chay that voi API:

```powershell
$env:LLM_API_KEY='YOUR_API_KEY'
$env:LLM_MODEL='YOUR_MODEL_NAME'
python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.json
```

Co the loc rieng mot speaker/file:

```powershell
python tools/17_generate_direct_llm_feedback.py --input data/final/direct_llm_feedback_inputs.json --output data/final/direct_llm_feedback_outputs.json --speaker speaker-01 --sample-id thin_C_01
```

target_phoneme
observed_phoneme
alignment_op
duration
speech_attribute_prediction
WavLM summary
visual summary
primary_evidence_policy
feedback_policy
llm_prompt

## 4. Nhung buoc khong bat buoc trong huong direct-LLM

Nhung lenh sau chi can neu quay lai huong gan nhan/train classifier:

```powershell
python tools/12_export_label_review_files.py --overwrite
python tools/14_llm_suggest_labels.py
python tools/13_merge_human_labels.py
```

Hien tai, voi huong direct-LLM, LLM khong nhin `human_label`, khong nhin classifier label, va khong dung rule label de suy luan. No chi nhin evidence da duoc router cho phep theo policy o tren.

# Pipeline fine-tune-visual sau khi doi ten file

## Cach dat ten hien tai

Trong moi folder am vi, `T` va `F` la file doc ca nhom tu. `C` la file doc tung cau theo target word.

Vi du nhom `/theta/` dau tu:

```text
data/raw/speaker-01/θ/T/T_01.mp4
data/raw/speaker-01/θ/T/T_02.mp4
data/raw/speaker-01/θ/T/T_03.mp4

data/raw/speaker-01/θ/F/F_01.mp4
data/raw/speaker-01/θ/F/F_02.mp4
data/raw/speaker-01/θ/F/F_03.mp4

data/raw/speaker-01/θ/C/think_C_01.mp4
data/raw/speaker-01/θ/C/think_C_02.mp4
data/raw/speaker-01/θ/C/think_C_03.mp4
data/raw/speaker-01/θ/C/thin_C_01.mp4
data/raw/speaker-01/θ/C/thin_C_02.mp4
data/raw/speaker-01/θ/C/thin_C_03.mp4
data/raw/speaker-01/θ/C/three_C_01.mp4
data/raw/speaker-01/θ/C/three_C_02.mp4
data/raw/speaker-01/θ/C/three_C_03.mp4
data/raw/speaker-01/θ/C/thumb_C_01.mp4
data/raw/speaker-01/θ/C/thumb_C_02.mp4
data/raw/speaker-01/θ/C/thumb_C_03.mp4
```

## Transcript tuong ung

`T/F` doc ca nhom tu, nen transcript la danh sach tu:

```text
data/transcript/speaker-01/θ/T/T_01.txt
think thin three thumb

data/transcript/speaker-01/θ/F/F_01.txt
think thin three thumb
```

`C` doc trong cau, nen transcript la ca cau:

```text
data/transcript/speaker-01/θ/C/thin_C_01.txt
The paper is very thin.
```

Nghia la:

```text
T = doc dung ca nhom tu
F = doc sai ca nhom tu
C = doc sai target word trong cau
```

## So luong setup

Hien script setup sinh:

```text
7 nhom am vi
10 speaker
1110 planned files
```

Moi speaker, moi nhom:

```text
T = 3 file group-level
F = 3 file group-level
C = 3 file cho moi target word trong nhom
```

Vi du nhom `/theta/` co 4 target word:

```text
T: 3 file
F: 3 file
C: 4 word x 3 = 12 file
Tong: 18 file / speaker cho nhom nay
```

## File setup quan trong

```text
tools/00_setup_reading_prompt_structure.py
data/reading_prompts.csv
data/sample_metadata.csv
data/transcript/speaker-01..speaker-10/...
data/raw/speaker-01..speaker-10/...
```

Da setup lai theo convention moi. Chi can chay lai lenh setup khi muon sinh lai transcript/metadata:

```powershell
python tools/00_setup_reading_prompt_structure.py
```

## Chay lai toan bo pipeline

Vi ban da doi ten file raw, nen nen xoa output cu truoc khi chay lai:

```powershell
python clear.py
```

Lenh nay xoa output sinh ra tu pipeline, bao gom:

```text
data/audio
data/aligned
data/annotations/auto
data/annotations/compare
data/annotations/wav2vec2_raw
data/final
data/processed/frames
data/processed/mouth
data/processed/clips
```

Lenh nay khong xoa:

```text
data/raw
data/transcript
data/sample_metadata.csv
data/reading_prompts.csv
```

Sau do chay:

```powershell
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
.\venv\Scripts\activate
```

Chay tiep:

```powershell
python tools/04_textgrid_to_json.py
python tools/05_extract_frames.py
python tools/05b_crop_mouth.py
python tools/06_compare_transcript_phonemes.py
python tools/07_make_clips.py
python tools/08_build_dataset.py
python tools/11_extract_segment_attributes.py
```

## Neu 06 hoac 07 bao loi

`06_compare_transcript_phonemes.py` can:

```text
data/annotations/auto
data/annotations/wav2vec2_raw
```

`07_make_clips.py` can:

```text
data/annotations/auto
```

Neu `data/annotations/auto` chua co file JSON, tuc la MFA va `04_textgrid_to_json.py` chua chay thanh cong.

## Output chinh

```text
data/final/segment_attributes.json
```

File nay la dau vao chinh cho AI/rule/LLM.

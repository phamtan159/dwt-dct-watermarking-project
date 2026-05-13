# AGENTS.md

Huong dan nay danh cho AI/coder khi sua du an `fine-tune-audio`.
Muc tieu la giu pipeline phat am on dinh, de kiem chung, va khong sua lan man.

## Nguyen tac chung

- Noi ro gia dinh truoc khi sua neu yeu cau con mo ho.
- Uu tien cach don gian nhat giai quyet dung van de hien tai.
- Chi sua file lien quan truc tiep den yeu cau.
- Khong refactor, doi format, doi ten bien, hay xoa code khong lien quan.
- Neu thay code chet hoac diem co the cai tien ngoai pham vi, hay ghi chu lai thay vi tu sua.
- Moi thay doi nen co cach verify ro rang: chay script, test, hoac kiem tra output.

## Pipeline cua du an

Du an xu ly theo huong:

```text
data/raw
-> tools/01_extract_audio.py
-> data/audio
-> tools/03_prepare_mfa.py
-> MFA / aligner tao TextGrid
-> tools/04_textgrid_to_json.py
-> data/annotations/auto
-> tools/05_compare_transcript_phonemes.py
-> data/annotations/compare
-> tools/08_build_dataset.py
-> data/final/dataset.json
-> models/train.py
```

Khi sua pipeline:

- Khong doi schema JSON neu khong that su can.
- Giu cac truong quan trong: `segments`, `id`, `phoneme`, `phoneme_standard`, `phoneme_real`, `start`, `end`, `error`, `error_id`.
- Khong tron lan giua `phoneme_standard` va `phoneme_real`.
- TextGrid sinh ra tu MFA hoac aligner phu chi la moc thoi gian can am, khong mac dinh la bang chung phat am dung/sai.
- Neu them aligner moi nhu Wav2TextGrid, hay dat output vao folder rieng truoc, vi du `data/aligned_w2tg`, roi so sanh voi MFA.

## Khi sua code

- Doc script hien co truoc khi viet script moi.
- Tai su dung helper trong `tools/phoneme_utils.py` neu phu hop.
- Giu input/output path mac dinh theo cau truc `data/...`.
- Neu them tham so CLI, phai giu default tuong thich voi cach chay hien tai.
- Khong them dependency nang neu co the dung thu vien da co trong `requirements.txt`.

## Khi verify

Voi thay doi tien xu ly, uu tien chay dung buoc bi anh huong, vi du:

```bash
python tools/03_prepare_mfa.py
python tools/04_textgrid_to_json.py
python tools/05_compare_transcript_phonemes.py
python tools/08_build_dataset.py
```

Voi thay doi model:

```bash
cd models
python evaluate.py --checkpoint checkpoints/best_model.pt
```

Neu khong chay duoc vi thieu model, thieu data, hoac thieu moi truong, ghi ro ly do va neu cach kiem tra thay the.

## Khong nen lam

- Khong thay MFA bang tool khac ma khong tao output rieng de doi chieu.
- Khong sua toan bo README chi vi mot thay doi nho.
- Khong xoa file trong `data/`, `models/checkpoints/`, hoac cache model neu khong duoc yeu cau.
- Khong bien mot script tien xu ly nho thanh framework lon.


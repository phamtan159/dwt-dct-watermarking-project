import textgrid, json, os, re

os.makedirs("data/annotations/auto", exist_ok=True)

for file in os.listdir("data/aligned"):
    if not file.endswith(".TextGrid"):
        continue

    tg = textgrid.TextGrid.fromFile(f"data/aligned/{file}")

    segments = []
    full_phonemes = []

    try:
        tier = tg.getFirst("phones")
    except ValueError:
        tier = tg[1] if len(tg) > 1 else tg[0]

    for interval in tier:
        phoneme = interval.mark.strip()
        
        # Lọc bỏ khoảng lặng (sil), lỗi căn chỉnh (spn) hoặc khoảng trống
        if phoneme and phoneme not in ["", "spn", "sil"]:
            # Tiếng Anh dùng từ điển ARPA thường có số đi kèm (ví dụ: AA1, IY0). 
            # Ta cần bỏ các số này để quy về âm vị gốc (AA, IY)
            base_phoneme = re.sub(r'\d+', '', phoneme)
            
            full_phonemes.append(base_phoneme)
            
            seg_id = f"{len(segments):03d}_{base_phoneme}"

            segments.append({
                "id": seg_id,
                "phoneme": base_phoneme,
                "start": interval.minTime,
                "end": interval.maxTime,
                "error": None
            })
        else:
            if full_phonemes and full_phonemes[-1] != " ":
                full_phonemes.append(" ")

    full_phonemes_str = "".join(full_phonemes).strip()
    
    with open(f"data/annotations/auto/{file.replace('.TextGrid','.txt')}", "w", encoding="utf-8") as f_txt:
        f_txt.write(full_phonemes_str)

    with open(f"data/annotations/auto/{file.replace('.TextGrid','.json')}", "w", encoding="utf-8") as f:
        json.dump(
            {"segments": segments},
            f,
            indent=2,
            ensure_ascii=False
        )

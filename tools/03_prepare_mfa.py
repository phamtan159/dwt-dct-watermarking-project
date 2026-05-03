import os
import glob

# Mô hình Wav2Vec2 TIMIT-phoneme_v3 thực chất xuất ra bảng mã IPA (ví dụ: ɑ, æ, ʃ, ʧ, ʤ, ŋ...)
# chứ không phải Arpabet (aa, ae, sh, ch...).
# Do đó, bảng mapping này dùng để ánh xạ IPA của Wav2Vec2 sang chuẩn IPA của MFA english_mfa (nếu có sự khác biệt).
# Các ký tự cơ bản (a, b, d, f, k...) được giữ nguyên.
ipa_to_mfa_ipa = {
    "ʧ": "tʃ",   # Wav2Vec2 dùng ký tự đơn ʧ, MFA dùng tʃ
    "ʤ": "dʒ",   # Wav2Vec2 dùng ký tự đơn ʤ, MFA dùng dʒ
    "oʊ": "oʊ",
    "aɪ": "aɪ",
    # Thêm các quy tắc ánh xạ khác nếu MFA báo lỗi "missing phones"
}

def map_phoneme(p):
    return ipa_to_mfa_ipa.get(p, p)

def main():
    auto_dir = "data/annotations/auto"
    audio_dir = "data/audio"
    dict_path = "custom_mfa.dict"
    
    txt_files = glob.glob(os.path.join(auto_dir, "*.txt"))
    
    if not txt_files:
        print(f"Không tìm thấy file .txt nào trong {auto_dir}")
        return

    unique_phonemes = set()

    for txt_file in txt_files:
        basename = os.path.basename(txt_file)
        out_txt_path = os.path.join(audio_dir, basename)
        
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        # Tách chuỗi thành từng ký tự (âm vị) và bỏ dấu cách cũ
        raw_phonemes = list(content.replace(" ", ""))
        
        # Ánh xạ sang chuẩn MFA
        mapped_phonemes = [map_phoneme(p) for p in raw_phonemes]
        
        # Ghi đè vào thư mục data/audio/ dưới dạng các âm vị cách nhau bằng dấu cách
        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.write(" ".join(mapped_phonemes))
            
        # Lưu lại các âm vị duy nhất để tạo từ điển
        unique_phonemes.update(mapped_phonemes)
        
        print(f"Đã xử lý và ghi đè: {out_txt_path}")

    # Tạo từ điển custom mapping mỗi âm vị thành chính nó
    with open(dict_path, "w", encoding="utf-8") as f:
        for p in sorted(list(unique_phonemes)):
            f.write(f"{p}\t{p}\n")
            
    print(f"\nĐã tạo từ điển MFA tại: {dict_path}")
    print("\nBây giờ bạn có thể chạy lệnh MFA:")
    print(f"mfa align --clean {audio_dir} {dict_path} english_mfa data/aligned")

if __name__ == "__main__":
    main()

import json
import csv
import io

content = """Âm mục tiêu (IPA),Cách phát âm sai,Ví dụ từ vựng,Cách người miền Tây hay đọc,Ghi chú kỹ thuật
/θ/,Biến thành t hoặc th,"Thin, Thank","""Tin"", ""Thanh""",Đầu lưỡi rụt vào trong thay vì đặt giữa hai hàm răng.
/ð/,Biến thành d (dờ),"This, Mother","""Dít"", ""Ma-dờ""",Không đặt lưỡi giữa răng và không rung dây thanh.
/s/ (cuối),Bị nuốt (lược bỏ),"Bus, Face","""Ba"", ""Phây""","Lưỡi thả lỏng quá sớm, không đẩy hơi qua khe răng."
/t/ (cuối),Biến thành dấu sắc,"Cat, Bit","""Cét"", ""Bít""",Lưỡi chạm vào vòm miệng cứng quá mạnh và ngắt hơi đột ngột.
/ʃ/ (sh),Biến thành s nhẹ,"She, Shop","""Si"", ""Sóp""","Lưỡi đặt thấp, không uốn cong về phía vòm miệng."
/z/,Biến thành s,"Is, Please","""Ít"", ""Plít""",Dây thanh quản không rung ở cuối từ.
/r/,Biến thành g hoặc y,"Run, River","""Găng"", ""Gí-vờ""",Gốc lưỡi bị gồng (tạo âm G) thay vì cuộn nhẹ (âm R).
/v/,Biến thành d/dz,"Very, Vote","""Dze-ri"", ""Dốt""",Không dùng răng trên chạm môi dưới để tạo độ rung.
/dʒ/,Biến thành d/y,"Job, June","""Dóp"", ""Yun""",Không kết hợp được việc chặn hơi và rung thanh quản.
Trọng âm,Đọc bằng phẳng,"Perfect, Today","""Pơ-phét"", ""Tu-đây""",Cơ hoành không co bóp mạnh để nhấn vào âm tiết chính.
/p/ (bật hơi),Biến thành b,"Pen, Apple","""Ben"", ""Áp-pồ""",Không nén hơi từ bụng để bật mạnh qua môi.
/k/ (bật cuối),Bị mất hoàn toàn,"Like, Check","""Lai"", ""Chết""",Không dùng cơ cổ họng để chặn và bật hơi dứt khoát.
Nối âm,Đọc rời rạc,Get out,"""Gét"" (nghỉ) ""Ao""",Luồng hơi bị ngắt quãng giữa các từ.
"""

# Mapping IPA/Feature to a short ID
id_map = {
    "/θ/": "th_to_t",
    "/ð/": "dh_to_d",
    "/s/ (cuối)": "s_final_omitted",
    "/t/ (cuối)": "t_final_sharp",
    "/ʃ/ (sh)": "sh_to_s",
    "/z/": "z_to_s",
    "/r/": "r_to_g_y",
    "/v/": "v_to_d_dz",
    "/dʒ/": "dj_to_d_y",
    "Trọng âm": "stress_flat",
    "/p/ (bật hơi)": "p_to_b",
    "/k/ (bật cuối)": "k_final_omitted",
    "Nối âm": "linking_broken"
}

label_map = {
    "0": {
        "id": "OK",
        "name": "Chính xác",
        "description": "Phát âm đúng chuẩn"
    }
}

f = io.StringIO(content)
reader = csv.DictReader(f)

index = 1
for row in reader:
    # Skip rows that might be header repeats (though DictReader handles the first one)
    if row['Âm mục tiêu (IPA)'] == 'Âm mục tiêu (IPA)' or row['Âm mục tiêu (IPA)'] == 'Thành phần':
        continue
    
    target = row.get('Âm mục tiêu (IPA)') or row.get('Thành phần')
    error_type = row.get('Cách phát âm sai') or row.get('Lỗi thường gặp')
    
    label_id = id_map.get(target, target.lower().replace("/", "").replace(" ", "_"))
    
    label_map[str(index)] = {
        "id": label_id,
        "target": target,
        "error_type": error_type,
        "examples": row['Ví dụ từ vựng'],
        "local_style": row['Cách người miền Tây hay đọc'],
        "notes": row['Ghi chú kỹ thuật']
    }
    index += 1

with open('d:/fine-tune-2/data/label_map.json', 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("Successfully updated label_map.json")

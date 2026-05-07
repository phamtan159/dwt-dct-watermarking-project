import type {
  CompareSoundsRequest,
  CompareSoundsResponse,
} from "../types/pronunciation.js";

const compareMap: Record<string, CompareSoundsResponse> = {
  "OH:UH": {
    target: "OH",
    predicted: "UH",
    difference: [
      "OH is a diphthong that glides from /o/ toward /ʊ/.",
      "UH is a short central vowel with less lip rounding.",
      "For OH, round your lips more and keep the sound moving.",
    ],
    tips_vi: [
      "Âm OH cần tròn môi hơn.",
      "Không giữ âm đứng yên ở giữa miệng.",
      "Hãy lướt nhẹ từ /o/ sang /ʊ/ thay vì phát âm ngắn như /ʌ/.",
    ],
  },
};

export class CompareSoundsService {
  async compare(
    request: CompareSoundsRequest,
  ): Promise<CompareSoundsResponse> {
    /*
      Real AI integration point:
      - Pull articulatory descriptors from a phoneme knowledge base.
      - Optionally blend these with MDD confusion probabilities and L1 transfer rules.
    */

    const key = `${request.target_phoneme}:${request.predicted_phoneme}`;

    return (
      compareMap[key] ?? {
        target: request.target_phoneme,
        predicted: request.predicted_phoneme,
        difference: [
          `${request.target_phoneme} and ${request.predicted_phoneme} differ in articulation, resonance, and timing.`,
          `Try exaggerating the target sound before returning to natural speed.`,
          `Use mirror practice and slow repetition to stabilize the target.`,
        ],
        tips_vi: [
          `So sánh trực tiếp khẩu hình của âm ${request.target_phoneme} và ${request.predicted_phoneme}.`,
          "Luyện chậm trước rồi mới tăng tốc độ.",
          "Ghi âm lại để kiểm tra xem âm mục tiêu đã rõ hơn chưa.",
        ],
      }
    );
  }
}

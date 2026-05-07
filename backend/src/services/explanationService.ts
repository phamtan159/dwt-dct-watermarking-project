import type { ExplainRequest, ExplainResponse } from "../types/pronunciation.js";

const explanationMap: Record<string, ExplainResponse> = {
  "OH:UH": {
    explanation_en:
      "You pronounced /oʊ/ too low and central, so it sounded like /ʌ/. Try starting with a rounded /o/ sound and glide toward /ʊ/.",
    explanation_vi:
      "Bạn phát âm /oʊ/ hơi thấp và lệch vào giữa miệng nên nghe giống /ʌ/. Hãy bắt đầu bằng âm /o/ với môi hơi tròn, sau đó lướt nhẹ về /ʊ/.",
  },
  "R:L": {
    explanation_en:
      "Your tongue touched too early, so the English /r/ sounded closer to /l/. Pull the tongue slightly back and avoid touching the roof of the mouth.",
    explanation_vi:
      "Lưỡi của bạn chạm quá sớm nên âm /r/ tiếng Anh nghe gần giống /l/. Hãy kéo lưỡi nhẹ về sau và tránh chạm lên vòm miệng.",
  },
};

export class ExplanationService {
  async explain(request: ExplainRequest): Promise<ExplainResponse> {
    /*
      Real AI integration point:
      - Build a prompt using learner level, target phoneme, predicted phoneme,
        stress context, articulation metadata, and L1-specific hints.
      - Call an LLM or agent that generates natural-language correction.
    */

    const key = `${request.target_phoneme}:${request.predicted_phoneme}`;

    return (
      explanationMap[key] ?? {
        explanation_en: `For "${request.word}", you produced /${request.predicted_phoneme}/ instead of /${request.target_phoneme}/. Focus on lip shape, tongue position, and keeping the target sound stable.`,
        explanation_vi: `Ở từ "${request.word}", bạn phát âm /${request.predicted_phoneme}/ thay vì /${request.target_phoneme}/. Hãy chú ý đến khẩu hình môi, vị trí lưỡi và giữ âm mục tiêu rõ ràng hơn.`,
      }
    );
  }
}

import { createMockAssessment } from "../data/mockPronunciation.js";
import type { SentenceAssessment } from "../types/pronunciation.js";

interface EvaluatePronunciationInput {
  sentence: string;
  audioBuffer?: Buffer;
  mimeType?: string;
}

export class PronunciationPipelineService {
  async evaluate(
    input: EvaluatePronunciationInput,
  ): Promise<SentenceAssessment> {
    /*
      Real AI integration point:
      1. Save or stream learner audio.
      2. Run preprocessing and voice activity cleanup.
      3. Convert the prompt sentence into canonical transcript + lexicon entries.
      4. Run MFA forced alignment against that canonical transcript.
      5. Export MFA phone/word timing as the alignment backbone.
      6. Run wav2vec2 phoneme recognition on the learner audio only for acoustic evidence.
      7. Align wav2vec2 evidence back onto MFA timing and score phoneme mismatches.
      8. Crop learner mouth clips and attach visual hints from the visual pipeline.
      9. Aggregate phoneme scores into word-level and sentence-level scores.
      10. Return the JSON contract consumed by the frontend.
    */

    void input.audioBuffer;
    void input.mimeType;

    return {
      ...createMockAssessment(input.sentence),
      pipeline: {
        transcript_source: "prompt_sentence",
        aligner: "MFA",
        acoustic_model: "wav2vec2",
        visual_model: "MediaPipe mouth crop + AV-HuBERT visual encoder",
      },
    };
  }
}

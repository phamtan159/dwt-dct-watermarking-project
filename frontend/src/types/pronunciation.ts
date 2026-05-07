export type PronunciationStatus =
  | "correct"
  | "good"
  | "near_correct"
  | "warning"
  | "wrong"
  | "unrecognized";

export type SupportedExplanationLanguage = "en" | "vi";
export type LearnerLevel = "beginner" | "intermediate" | "advanced";

export interface StressFeedback {
  is_correct: boolean;
  message: string;
}

export interface PipelineMetadata {
  transcript_source: "prompt_sentence";
  aligner: "MFA";
  acoustic_model: "wav2vec2";
  visual_model?: string;
}

export interface PhonemeFeedback {
  target: string;
  predicted: string;
  score: number;
  status: PronunciationStatus;
  message: string;
  audio_target_url?: string;
  audio_user_segment_url?: string;
  explanation?: string;
  alignment_source?: "mfa";
  acoustic_support?: number;
  visual_hint?: string;
}

export interface WordAssessment {
  word: string;
  start_time: number;
  end_time: number;
  score: number;
  status: PronunciationStatus;
  target_phonemes: string[];
  predicted_phonemes: string[];
  stress?: StressFeedback;
  phoneme_feedback: PhonemeFeedback[];
}

export interface SentenceAssessment {
  sentence: string;
  overall_score: number;
  audio_url_user?: string | null;
  words: WordAssessment[];
  pipeline?: PipelineMetadata;
}

export interface ExplainRequest {
  word: string;
  target_phoneme: string;
  predicted_phoneme: string;
  learner_level: LearnerLevel;
  language: SupportedExplanationLanguage;
}

export interface ExplainResponse {
  explanation_en: string;
  explanation_vi?: string;
}

export interface CompareSoundsRequest {
  target_phoneme: string;
  predicted_phoneme: string;
  language: SupportedExplanationLanguage;
}

export interface CompareSoundsResponse {
  target: string;
  predicted: string;
  difference: string[];
  tips_vi?: string[];
}

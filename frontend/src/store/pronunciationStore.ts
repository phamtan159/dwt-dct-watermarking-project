import { create } from "zustand";
import { mockSentence } from "../mocks/mockPronunciation";
import type { SentenceAssessment } from "../types/pronunciation";
import { getLowestScoringWordIndex } from "../utils/score";

interface PronunciationStore {
  sentence: string;
  result: SentenceAssessment | null;
  isEvaluating: boolean;
  selectedWordIndex: number | null;
  autoOpenLowest: boolean;
  error: string | null;
  setSentence: (sentence: string) => void;
  setAutoOpenLowest: (enabled: boolean) => void;
  beginEvaluation: () => void;
  finishEvaluation: (result: SentenceAssessment) => void;
  failEvaluation: (message: string) => void;
  setSelectedWordIndex: (index: number | null) => void;
  clearError: () => void;
}

export const usePronunciationStore = create<PronunciationStore>((set, get) => ({
  sentence: mockSentence,
  result: null,
  isEvaluating: false,
  selectedWordIndex: null,
  autoOpenLowest: true,
  error: null,
  setSentence: (sentence) =>
    set({
      sentence,
      result: null,
      selectedWordIndex: null,
      error: null,
    }),
  setAutoOpenLowest: (enabled) => set({ autoOpenLowest: enabled }),
  beginEvaluation: () => set({ isEvaluating: true, error: null }),
  finishEvaluation: (result) => {
    const shouldAutoOpen = get().autoOpenLowest;
    const selectedWordIndex =
      shouldAutoOpen && result.words.length > 0
        ? getLowestScoringWordIndex(result.words)
        : null;

    set({
      result,
      isEvaluating: false,
      selectedWordIndex,
      error: null,
    });
  },
  failEvaluation: (message) =>
    set({
      isEvaluating: false,
      error: message,
    }),
  setSelectedWordIndex: (selectedWordIndex) => set({ selectedWordIndex }),
  clearError: () => set({ error: null }),
}));

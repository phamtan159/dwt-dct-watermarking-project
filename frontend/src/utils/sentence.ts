export interface SentenceValidationResult {
  isValid: boolean;
  words: string[];
  error: string | null;
}

const ALLOWED_SENTENCE_PATTERN = /^[A-Za-z0-9\s'",.!?;:-]+$/;
const MAX_WORD_COUNT = 24;

export function tokenizeSentence(sentence: string): string[] {
  return sentence.trim().split(/\s+/).filter(Boolean);
}

export function validateSentence(sentence: string): SentenceValidationResult {
  const trimmed = sentence.trim();
  const words = tokenizeSentence(sentence);

  if (!trimmed) {
    return {
      isValid: false,
      words,
      error: "Hãy nhập một câu tiếng Anh trước khi bấm nói.",
    };
  }

  if (!/[A-Za-z]/.test(trimmed)) {
    return {
      isValid: false,
      words,
      error: "Câu mẫu cần có chữ cái tiếng Anh để hệ thống căn chỉnh phát âm.",
    };
  }

  if (!ALLOWED_SENTENCE_PATTERN.test(trimmed)) {
    return {
      isValid: false,
      words,
      error:
        "Chỉ nên dùng chữ cái tiếng Anh, số và dấu câu cơ bản như . , ! ? ' -",
    };
  }

  if (words.length > MAX_WORD_COUNT) {
    return {
      isValid: false,
      words,
      error: `Câu hiện có ${words.length} từ. Hãy giữ dưới ${MAX_WORD_COUNT} từ để phản hồi rõ hơn.`,
    };
  }

  return {
    isValid: true,
    words,
    error: null,
  };
}

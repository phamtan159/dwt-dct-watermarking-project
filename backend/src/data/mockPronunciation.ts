import type { SentenceAssessment, WordAssessment } from "../types/pronunciation.js";

const mockWordTemplates: WordAssessment[] = [
  {
    word: "Okay",
    start_time: 0.12,
    end_time: 0.72,
    score: 70,
    status: "warning",
    target_phonemes: ["OH", "K", "EY"],
    predicted_phonemes: ["UH", "K", "EH"],
    stress: {
      is_correct: true,
      message: "You stressed the right syllable!",
    },
    phoneme_feedback: [
      {
        target: "OH",
        predicted: "UH",
        score: 31,
        status: "wrong",
        message: "You said UH",
        audio_target_url: "/audio/phonemes/oh.mp3",
        audio_user_segment_url: "/segments/user_oh.wav",
        explanation:
          "The target sound /oʊ/ should begin rounded and glide forward. Yours stayed too central.",
      },
      {
        target: "K",
        predicted: "K",
        score: 90,
        status: "correct",
        message: "Awesome!",
        audio_target_url: "/audio/phonemes/k.mp3",
        audio_user_segment_url: "/segments/user_k.wav",
      },
      {
        target: "EY",
        predicted: "EH",
        score: 90,
        status: "near_correct",
        message: "You said EH",
        audio_target_url: "/audio/phonemes/ey.mp3",
        audio_user_segment_url: "/segments/user_ey.wav",
        explanation:
          "The sound /eɪ/ should glide upward. Try not to keep it flat like /ɛ/.",
      },
    ],
  },
  {
    word: "I",
    start_time: 0.73,
    end_time: 0.95,
    score: 95,
    status: "correct",
    target_phonemes: ["AY"],
    predicted_phonemes: ["AY"],
    stress: {
      is_correct: true,
      message: "Clear and natural vowel movement.",
    },
    phoneme_feedback: [
      {
        target: "AY",
        predicted: "AY",
        score: 95,
        status: "correct",
        message: "Excellent diphthong!",
        audio_target_url: "/audio/phonemes/ay.mp3",
        audio_user_segment_url: "/segments/user_ay.wav",
      },
    ],
  },
  {
    word: "will",
    start_time: 0.96,
    end_time: 1.31,
    score: 88,
    status: "good",
    target_phonemes: ["W", "IH", "L"],
    predicted_phonemes: ["W", "IH", "L"],
    stress: {
      is_correct: true,
      message: "Nice control of the final dark L.",
    },
    phoneme_feedback: [
      {
        target: "W",
        predicted: "W",
        score: 87,
        status: "correct",
        message: "Good lip rounding.",
      },
      {
        target: "IH",
        predicted: "IH",
        score: 86,
        status: "good",
        message: "Stable short vowel.",
      },
      {
        target: "L",
        predicted: "L",
        score: 91,
        status: "correct",
        message: "Well released.",
      },
    ],
  },
  {
    word: "say",
    start_time: 1.32,
    end_time: 1.62,
    score: 92,
    status: "correct",
    target_phonemes: ["S", "EY"],
    predicted_phonemes: ["S", "EY"],
    stress: {
      is_correct: true,
      message: "Smooth consonant-to-vowel transition.",
    },
    phoneme_feedback: [
      {
        target: "S",
        predicted: "S",
        score: 93,
        status: "correct",
        message: "Nice airflow.",
      },
      {
        target: "EY",
        predicted: "EY",
        score: 91,
        status: "correct",
        message: "Vowel glide is clear.",
      },
    ],
  },
  {
    word: "some",
    start_time: 1.63,
    end_time: 2.01,
    score: 60,
    status: "warning",
    target_phonemes: ["S", "AH", "M"],
    predicted_phonemes: ["S", "AO", "M"],
    stress: {
      is_correct: true,
      message: "The stress is fine, but the middle vowel needs work.",
    },
    phoneme_feedback: [
      {
        target: "S",
        predicted: "S",
        score: 83,
        status: "good",
        message: "Good start.",
      },
      {
        target: "AH",
        predicted: "AO",
        score: 42,
        status: "wrong",
        message: "You opened too much.",
        explanation:
          "The target /ʌ/ is short and central. Yours drifted toward a more open back vowel.",
      },
      {
        target: "M",
        predicted: "M",
        score: 86,
        status: "correct",
        message: "Lip closure is solid.",
      },
    ],
  },
  {
    word: "wrong",
    start_time: 2.02,
    end_time: 2.49,
    score: 45,
    status: "wrong",
    target_phonemes: ["R", "AO", "NG"],
    predicted_phonemes: ["L", "AA", "NG"],
    stress: {
      is_correct: false,
      message: "You stressed the vowel too heavily. Keep the onset lighter.",
    },
    phoneme_feedback: [
      {
        target: "R",
        predicted: "L",
        score: 29,
        status: "wrong",
        message: "You said L",
        explanation:
          "English /r/ should pull the tongue back without touching the roof. Yours touched too early like /l/.",
      },
      {
        target: "AO",
        predicted: "AA",
        score: 54,
        status: "warning",
        message: "Too open and too long.",
      },
      {
        target: "NG",
        predicted: "NG",
        score: 81,
        status: "good",
        message: "Ending is mostly right.",
      },
    ],
  },
  {
    word: "texts",
    start_time: 2.5,
    end_time: 3.06,
    score: 80,
    status: "good",
    target_phonemes: ["T", "EH", "K", "S", "T", "S"],
    predicted_phonemes: ["T", "EH", "K", "S", "T", "S"],
    stress: {
      is_correct: true,
      message: "Good rhythm on a difficult final cluster.",
    },
    phoneme_feedback: [
      {
        target: "T",
        predicted: "T",
        score: 84,
        status: "good",
        message: "Clean attack.",
      },
      {
        target: "EH",
        predicted: "EH",
        score: 79,
        status: "good",
        message: "Could be a bit shorter.",
      },
      {
        target: "K",
        predicted: "K",
        score: 82,
        status: "good",
        message: "Consonant is audible.",
      },
      {
        target: "S",
        predicted: "S",
        score: 85,
        status: "correct",
        message: "Nice hiss.",
      },
      {
        target: "T",
        predicted: "T",
        score: 76,
        status: "good",
        message: "Keep it crisp.",
      },
      {
        target: "S",
        predicted: "S",
        score: 80,
        status: "good",
        message: "Cluster survived well.",
      },
    ],
  },
];

function stripPunctuation(token: string) {
  return token.replace(/^[^A-Za-z]+|[^A-Za-z]+$/g, "") || token;
}

export function createMockAssessment(sentence: string): SentenceAssessment {
  const fallbackSentence = "Okay, I will say some wrong texts.";
  const workingSentence = sentence.trim() || fallbackSentence;
  const tokens = workingSentence.split(/\s+/).filter(Boolean);
  const wordCount = tokens.length || mockWordTemplates.length;

  const words = Array.from({ length: wordCount }).map((_, index) => {
    const template = mockWordTemplates[index % mockWordTemplates.length];
    const token = tokens[index] ?? template.word;
    const cleanWord = stripPunctuation(token);
    const previousEnd = index === 0 ? 0.08 : undefined;

    return {
      ...template,
      word: cleanWord,
      start_time: previousEnd ?? Number((index * 0.42 + 0.12).toFixed(2)),
      end_time: Number((index * 0.42 + 0.48).toFixed(2)),
    };
  });

  const overallScore = Math.round(
    words.reduce((sum, word) => sum + word.score, 0) / words.length,
  );

  return {
    sentence: workingSentence,
    overall_score: overallScore,
    audio_url_user: null,
    words,
  };
}

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { compareSounds, explainMistake } from "../api/pronunciationApi";
import type {
  CompareSoundsResponse,
  ExplainResponse,
  LearnerLevel,
  SentenceAssessment,
  SupportedExplanationLanguage,
} from "../types/pronunciation";
import { playAudioSource } from "../utils/audio";
import { formatPercent, resolveStatusTheme } from "../utils/score";
import { AudioButton } from "./AudioButton";
import { CompareSoundsPanel } from "./CompareSoundsPanel";
import { ExplainMistakePanel } from "./ExplainMistakePanel";
import { PhonemeFeedbackRow } from "./PhonemeFeedbackRow";
import { ScoreGauge } from "./ScoreGauge";

interface WordDetailModalProps {
  assessment: SentenceAssessment | null;
  selectedWordIndex: number | null;
  learnerAudioUrl?: string | null;
  learnerLevel: LearnerLevel;
  language: SupportedExplanationLanguage;
  autoOpenLowest: boolean;
  onToggleAutoOpen: (enabled: boolean) => void;
  onClose: () => void;
  onSelectWord: (index: number) => void;
}

export function WordDetailModal({
  assessment,
  selectedWordIndex,
  learnerAudioUrl,
  learnerLevel,
  language,
  autoOpenLowest,
  onToggleAutoOpen,
  onClose,
  onSelectWord,
}: WordDetailModalProps) {
  const word =
    typeof selectedWordIndex === "number"
      ? assessment?.words[selectedWordIndex]
      : null;
  const [focusedPhonemeIndex, setFocusedPhonemeIndex] = useState(0);
  const [explainData, setExplainData] = useState<ExplainResponse | null>(null);
  const [compareData, setCompareData] = useState<CompareSoundsResponse | null>(
    null,
  );
  const [isExplainLoading, setIsExplainLoading] = useState(false);
  const [isCompareLoading, setIsCompareLoading] = useState(false);

  useEffect(() => {
    if (!word) return;

    const fallbackIndex =
      word.phoneme_feedback.findIndex((item) => item.score < 80) >= 0
        ? word.phoneme_feedback.findIndex((item) => item.score < 80)
        : 0;

    setFocusedPhonemeIndex(fallbackIndex);
    setExplainData(null);
    setCompareData(null);
  }, [word?.word]);

  if (!assessment || !word || selectedWordIndex === null) {
    return null;
  }

  const activeWord = word;
  const focusedPhoneme =
    activeWord.phoneme_feedback[focusedPhonemeIndex] ??
    activeWord.phoneme_feedback[0];
  const theme = resolveStatusTheme(activeWord.status, activeWord.score);
  const displayTokens = assessment.sentence.trim().split(/\s+/).filter(Boolean);
  const displayWord = displayTokens[selectedWordIndex] ?? activeWord.word;
  const canGoPrev = selectedWordIndex > 0;
  const canGoNext = selectedWordIndex < assessment.words.length - 1;

  async function handleExplainRequest() {
    if (!focusedPhoneme) return;

    setIsExplainLoading(true);

    try {
      const response = await explainMistake({
        word: activeWord.word,
        target_phoneme: focusedPhoneme.target,
        predicted_phoneme: focusedPhoneme.predicted,
        learner_level: learnerLevel,
        language,
      });

      setExplainData(response);
    } finally {
      setIsExplainLoading(false);
    }
  }

  async function handleCompareRequest() {
    if (!focusedPhoneme) return;

    setIsCompareLoading(true);

    try {
      const response = await compareSounds({
        target_phoneme: focusedPhoneme.target,
        predicted_phoneme: focusedPhoneme.predicted,
        language,
      });

      setCompareData(response);
    } finally {
      setIsCompareLoading(false);
    }
  }

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 z-50 flex items-end justify-center bg-slate-950/72 p-0 md:items-center md:p-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          initial={{ y: 48, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 48, opacity: 0 }}
          transition={{ duration: 0.24 }}
          className="card-surface subtle-grid relative h-[92vh] w-full overflow-hidden rounded-t-[2rem] md:h-auto md:max-h-[92vh] md:max-w-5xl md:rounded-[2rem]"
        >
          <div className="absolute inset-x-0 top-0 flex justify-center pt-3 md:hidden">
            <div className="h-1.5 w-14 rounded-full bg-white/15" />
          </div>

          <div className="flex h-full flex-col overflow-hidden">
            <div className="flex items-center justify-between border-b soft-divider px-5 pb-4 pt-6 md:px-8">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Word breakdown
                </div>
                <h2 className="panel-title mt-2 text-2xl font-bold text-white md:text-3xl">
                  {displayWord}
                </h2>
              </div>

              <button
                type="button"
                onClick={onClose}
                className="flex h-11 w-11 items-center justify-center rounded-full border border-slate-600/40 bg-slate-900/60 text-xl text-slate-100 transition hover:bg-slate-800/65"
              >
                ×
              </button>
            </div>

            <div className="overflow-y-auto px-5 pb-8 pt-5 md:px-8">
              <div className="grid gap-6 xl:grid-cols-[0.95fr,1.35fr]">
                <section className="space-y-5">
                  <div className="rounded-[1.8rem] border border-slate-700/45 bg-slate-900/62 p-5">
                    <div className="flex justify-center">
                      <ScoreGauge score={word.score} size={188} />
                    </div>

                    <div className="mt-5 text-center">
                      <div className="panel-title text-3xl font-bold text-white">
                        {activeWord.word}
                      </div>
                      <div className="mt-3 flex flex-wrap justify-center gap-2">
                        {word.phoneme_feedback.map((phoneme) => {
                          const phonemeTheme = resolveStatusTheme(
                            phoneme.status,
                            phoneme.score,
                          );

                          return (
                            <span
                              key={`${word.word}-${phoneme.target}`}
                              className={`rounded-full px-3 py-1.5 text-sm font-semibold ${phonemeTheme.badge}`}
                            >
                              {phoneme.target}
                            </span>
                          );
                        })}
                      </div>
                    </div>

                    <div className={`mt-5 rounded-[1.4rem] border p-4 ${theme.chip}`}>
                      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-300">
                        Stress feedback
                      </div>
                      <p className="mt-2 text-sm leading-7 text-slate-100">
                        {activeWord.stress?.message ??
                          "Stress feedback is not available yet."}
                      </p>
                    </div>

                    <div className="mt-5 grid grid-cols-2 gap-3">
                      <AudioButton
                        label="Coach"
                        active
                        onClick={() =>
                          playAudioSource(undefined, word.word, "en-US")
                        }
                      />
                      <AudioButton
                        label="You"
                        onClick={() =>
                          playAudioSource(
                            learnerAudioUrl ?? assessment.audio_url_user,
                            undefined,
                          )
                        }
                      />
                    </div>

                    <div className="mt-5 rounded-[1.4rem] border border-slate-700/45 bg-slate-950/50 p-4">
                      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                        Target vs. you
                      </div>
                      <div className="mt-3 grid gap-3 sm:grid-cols-2">
                        <div className="rounded-2xl bg-white/[0.03] p-3">
                          <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                            Target
                          </div>
                          <div className="mt-2 text-base font-semibold text-white">
                            {activeWord.target_phonemes.join("  ")}
                          </div>
                        </div>
                        <div className="rounded-2xl bg-white/[0.03] p-3">
                          <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                            You said
                          </div>
                          <div className="mt-2 text-base font-semibold text-white">
                            {activeWord.predicted_phonemes.join("  ")}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-[1.8rem] border border-slate-700/45 bg-slate-900/62 p-5">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                          Focused phoneme
                        </div>
                        <div className="mt-2 text-lg font-semibold text-white">
                          /{focusedPhoneme.target}/
                        </div>
                      </div>
                      <span
                        className={`rounded-full px-3 py-1.5 text-sm font-semibold ${resolveStatusTheme(focusedPhoneme.status, focusedPhoneme.score).badge}`}
                      >
                        {formatPercent(focusedPhoneme.score)}
                      </span>
                    </div>

                    <p className="mt-3 text-sm leading-7 text-slate-300">
                      {focusedPhoneme.message}. You likely produced /
                      {focusedPhoneme.predicted}/ instead.
                    </p>
                  </div>
                </section>

                <section className="space-y-5">
                  <div className="rounded-[1.8rem] border border-slate-700/45 bg-slate-900/62 p-5">
                    <div>
                      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                        Phoneme-level feedback
                      </div>
                      <p className="mt-2 text-sm text-slate-400">
                        Tap a row to focus that sound, then ask for AI explanation or compare articulation.
                      </p>
                    </div>

                    <div className="mt-4 space-y-3">
                      {activeWord.phoneme_feedback.map((feedback, index) => (
                        <PhonemeFeedbackRow
                          key={`${word.word}-${feedback.target}-${index}`}
                          feedback={feedback}
                          isFocused={focusedPhonemeIndex === index}
                          onFocus={() => setFocusedPhonemeIndex(index)}
                          onPlayTarget={() =>
                            playAudioSource(
                              feedback.audio_target_url,
                              feedback.target,
                              "en-US",
                            )
                          }
                          onPlayUser={() =>
                            playAudioSource(
                              feedback.audio_user_segment_url ??
                                learnerAudioUrl ??
                                assessment.audio_url_user,
                            )
                          }
                        />
                      ))}
                    </div>
                  </div>

                  <ExplainMistakePanel
                    loading={isExplainLoading}
                    data={explainData}
                    onRequest={handleExplainRequest}
                    word={activeWord.word}
                    phonemeLabel={focusedPhoneme.target}
                  />

                  <CompareSoundsPanel
                    loading={isCompareLoading}
                    data={compareData}
                    onRequest={handleCompareRequest}
                  />
                </section>
              </div>

              <div className="mt-6 flex flex-col gap-4 rounded-[1.6rem] border border-slate-700/45 bg-slate-900/62 p-4 md:flex-row md:items-center md:justify-between">
                <label className="flex items-center gap-3 text-sm text-slate-200">
                  <button
                    type="button"
                    onClick={() => onToggleAutoOpen(!autoOpenLowest)}
                    className={`relative h-7 w-12 rounded-full transition ${
                      autoOpenLowest ? "bg-emerald-400/70" : "bg-slate-600/55"
                    }`}
                  >
                    <span
                      className={`absolute top-1 h-5 w-5 rounded-full bg-white transition ${
                        autoOpenLowest ? "left-6" : "left-1"
                      }`}
                    />
                  </button>
                  <span>Show this screen automatically</span>
                </label>

                <div className="flex gap-3">
                  <button
                    type="button"
                    disabled={!canGoPrev}
                    onClick={() => onSelectWord(selectedWordIndex - 1)}
                    className="rounded-full border border-slate-600/40 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-slate-800/60 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Previous word
                  </button>
                  <button
                    type="button"
                    disabled={!canGoNext}
                    onClick={() => onSelectWord(selectedWordIndex + 1)}
                    className="rounded-full border border-slate-600/40 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-slate-800/60 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Next word
                  </button>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

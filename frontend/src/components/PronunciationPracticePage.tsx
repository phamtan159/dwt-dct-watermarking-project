import { useState } from "react";
import { evaluatePronunciation } from "../api/pronunciationApi";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { usePronunciationStore } from "../store/pronunciationStore";
import type {
  LearnerLevel,
  SupportedExplanationLanguage,
} from "../types/pronunciation";
import { validateSentence } from "../utils/sentence";
import {
  formatPercent,
  getLowestScoringWordIndex,
  resolveStatusTheme,
} from "../utils/score";
import { RecordButton } from "./RecordButton";
import { SentenceWordRow } from "./SentenceWordRow";
import { WordDetailModal } from "./WordDetailModal";

export function PronunciationPracticePage() {
  const {
    sentence,
    result,
    isEvaluating,
    selectedWordIndex,
    autoOpenLowest,
    error,
    setSentence,
    setAutoOpenLowest,
    beginEvaluation,
    finishEvaluation,
    failEvaluation,
    setSelectedWordIndex,
    clearError,
  } = usePronunciationStore();
  const recorder = useAudioRecorder();
  const [learnerLevel, setLearnerLevel] = useState<LearnerLevel>("beginner");
  const [language, setLanguage] =
    useState<SupportedExplanationLanguage>("vi");
  const sentenceValidation = validateSentence(sentence);
  const canEvaluate = sentenceValidation.isValid && !isEvaluating;

  const weakestWordIndex = result?.words.length
    ? getLowestScoringWordIndex(result.words)
    : null;
  const weakestWord =
    weakestWordIndex !== null && result ? result.words[weakestWordIndex] : null;

  async function runEvaluation(audioBlob?: Blob | null) {
    if (!sentenceValidation.isValid) {
      failEvaluation(sentenceValidation.error ?? "Sentence is invalid.");
      return;
    }

    beginEvaluation();

    try {
      const assessment = await evaluatePronunciation({
        sentence,
        audioBlob,
      });

      finishEvaluation({
        ...assessment,
        sentence,
        audio_url_user:
          recorder.audioUrl ?? assessment.audio_url_user ?? undefined,
      });
    } catch (evaluationError) {
      const message =
        evaluationError instanceof Error
          ? evaluationError.message
          : "Pronunciation scoring failed.";

      failEvaluation(message);
    }
  }

  async function handleRecordStart() {
    clearError();
    recorder.clearRecording();
    await recorder.startRecording();
  }

  async function handleRecordStop() {
    const blob = await recorder.stopRecording();
    await runEvaluation(blob);
  }

  function handleUseDemo() {
    clearError();
    recorder.clearRecording();
    void runEvaluation(null);
  }

  const scoreTheme =
    result != null ? resolveStatusTheme(undefined, result.overall_score) : null;

  return (
    <main className="min-h-screen px-4 py-5 text-white md:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-6">
        <section className="card-surface subtle-grid overflow-hidden rounded-[2rem] border px-5 py-6 md:px-8 md:py-8">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex rounded-full border border-emerald-300/28 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-200">
                AI pronunciation assessment
              </div>
              <h1 className="panel-title mt-4 text-4xl font-bold leading-tight text-white md:text-5xl">
                English speaking practice with word-by-word and phoneme-by-phoneme feedback.
              </h1>
              <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300 md:text-base">
                Learners record one sentence, the backend scores each word, compares phonemes, and opens a coaching sheet with detailed correction tips for Vietnamese learners.
              </p>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 lg:min-w-[360px]">
              <div className="rounded-[1.4rem] border border-white/10 bg-white/[0.04] p-4">
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">
                  Flow
                </div>
                <div className="mt-2 text-sm font-semibold text-white">
                  Record → Align → Score → Explain
                </div>
              </div>
              <div className="rounded-[1.4rem] border border-white/10 bg-white/[0.04] p-4">
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">
                  Target
                </div>
                <div className="mt-2 text-sm font-semibold text-white">
                  Phoneme-level assessment
                </div>
              </div>
              <div className="rounded-[1.4rem] border border-white/10 bg-white/[0.04] p-4">
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">
                  UX
                </div>
                <div className="mt-2 text-sm font-semibold text-white">
                  Tap any word for detail
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.2fr,0.8fr]">
          <div className="space-y-6">
            <div className="card-surface rounded-[1.8rem] border p-5 md:p-6">
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Practice sentence
                  </div>
                  <p className="mt-2 text-sm text-slate-400">
                    Gõ câu mẫu vào ô bên dưới. Mỗi từ bạn gõ sẽ tách thành từng ô ngay lập tức để người học nhìn rõ từng từ trước khi bấm nói.
                  </p>
                </div>

                <label className="flex items-center gap-3 text-sm text-slate-200">
                  <button
                    type="button"
                    onClick={() => setAutoOpenLowest(!autoOpenLowest)}
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
              </div>

              <textarea
                value={sentence}
                onChange={(event) => setSentence(event.target.value)}
                rows={3}
                spellCheck={false}
                className="mt-4 w-full rounded-[1.4rem] border border-slate-600/35 bg-slate-950/52 px-4 py-4 text-lg font-medium text-white outline-none transition focus:border-sky-300/50 focus:ring-2 focus:ring-sky-400/18"
              />

              <div className="mt-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                <div className="text-sm text-slate-300">
                  <span className="font-semibold text-white">
                    Live word boxes:
                  </span>{" "}
                  cac o ben duoi se doi theo tung tu ban go.
                </div>
                <div className="rounded-full border border-slate-600/35 bg-slate-900/55 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-300">
                  {sentenceValidation.words.length} words
                </div>
              </div>

              {sentenceValidation.error ? (
                <div className="mt-3 rounded-[1.2rem] border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                  {sentenceValidation.error}
                </div>
              ) : (
                <div className="mt-3 rounded-[1.2rem] border border-emerald-400/20 bg-emerald-500/8 px-4 py-3 text-sm text-emerald-50/90">
                  Cau hop le. Ban co the bam nut noi de ghi am va cham diem.
                </div>
              )}

              <div className="mt-4">
                <SentenceWordRow
                  sentence={sentence}
                  words={result?.words}
                  selectedWordIndex={selectedWordIndex}
                  onSelectWord={(index) => setSelectedWordIndex(index)}
                />
              </div>
            </div>

            <div className="card-surface rounded-[1.8rem] border p-5 md:p-6">
              <div className="grid gap-5 lg:grid-cols-[0.88fr,1.12fr]">
                <div className="rounded-[1.6rem] border border-slate-700/40 bg-slate-900/52 p-5">
                  <div className="rounded-[1.35rem] border border-sky-300/18 bg-sky-400/8 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-200/80">
                      Step 2
                    </div>
                    <div className="mt-2 text-xl font-semibold text-white">
                      Bam nut nay de noi
                    </div>
                    <p className="mt-2 text-sm leading-7 text-slate-300">
                      1. Bam de bat dau ghi am. 2. Doc dung cau mau. 3. Bam lai
                      de dung. He thong se tu cham diem tung tu.
                    </p>
                  </div>

                  <div className="mt-5">
                  <RecordButton
                    isRecording={recorder.isRecording}
                    isBusy={isEvaluating}
                    disabled={!sentenceValidation.isValid}
                    disabledReason={sentenceValidation.error}
                    onStart={() => void handleRecordStart()}
                    onStop={() => void handleRecordStop()}
                  />
                  </div>

                  <div className="mt-5 flex flex-wrap gap-3">
                    <button
                      type="button"
                      onClick={handleUseDemo}
                      disabled={!canEvaluate}
                      className="rounded-full border border-slate-600/40 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:bg-slate-800/60 disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      Load demo result
                    </button>

                    {recorder.audioUrl ? (
                      <audio
                        controls
                        src={recorder.audioUrl}
                        className="w-full"
                      />
                    ) : null}
                  </div>

                  {(error || recorder.error) && (
                    <div className="mt-4 rounded-[1.2rem] border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
                      {error ?? recorder.error}
                    </div>
                  )}
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-[1.6rem] border border-slate-700/40 bg-slate-900/52 p-5">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      Learner level
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {(["beginner", "intermediate", "advanced"] as const).map(
                        (item) => (
                          <button
                            key={item}
                            type="button"
                            onClick={() => setLearnerLevel(item)}
                            className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                              learnerLevel === item
                                ? "bg-emerald-400/18 text-emerald-100"
                                : "bg-white/[0.05] text-slate-200 hover:bg-white/[0.08]"
                            }`}
                          >
                            {item}
                          </button>
                        ),
                      )}
                    </div>
                  </div>

                  <div className="rounded-[1.6rem] border border-slate-700/40 bg-slate-900/52 p-5">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      Feedback language
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {(["vi", "en"] as const).map((item) => (
                        <button
                          key={item}
                          type="button"
                          onClick={() => setLanguage(item)}
                          className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                            language === item
                              ? "bg-sky-400/18 text-sky-100"
                              : "bg-white/[0.05] text-slate-200 hover:bg-white/[0.08]"
                          }`}
                        >
                          {item === "vi" ? "Vietnamese" : "English"}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-[1.6rem] border border-slate-700/40 bg-slate-900/52 p-5 sm:col-span-2">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                      UX notes
                    </div>
                    <ul className="mt-3 space-y-2 text-sm leading-7 text-slate-200">
                      <li>Words are color-coded by score and stay tappable after evaluation.</li>
                      <li>Neu ban sua cau mau, score cu se tu xoa de tranh lech du lieu.</li>
                      <li>The modal can auto-open the weakest word to reduce learner friction.</li>
                      <li>Explain and Compare buttons map directly to backend AI endpoints.</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <aside className="space-y-6">
            <div className="card-surface rounded-[1.8rem] border p-5 md:p-6">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Sentence score
                  </div>
                  <h2 className="panel-title mt-2 text-3xl font-bold text-white">
                    {result ? formatPercent(result.overall_score) : "--"}
                  </h2>
                </div>
                {scoreTheme ? (
                  <span
                    className={`rounded-full px-3 py-1.5 text-sm font-semibold ${scoreTheme.badge}`}
                  >
                    {scoreTheme.status.replace("_", " ")}
                  </span>
                ) : null}
              </div>

              <p className="mt-4 text-sm leading-7 text-slate-300">
                {result
                  ? "The sentence has been scored. Tap any word chip to inspect phonemes, stress, and personalized coaching."
                  : "After recording, the backend returns overall score, per-word scores, phoneme comparisons, and coaching text."}
              </p>

              {weakestWord ? (
                <div className="mt-5 rounded-[1.4rem] border border-rose-400/20 bg-rose-500/8 p-4">
                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-200/80">
                    Lowest scoring word
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-3">
                    <div>
                      <div className="text-lg font-semibold text-white">
                        {weakestWord.word}
                      </div>
                      <div className="text-sm text-rose-100/75">
                        {formatPercent(weakestWord.score)} • tap for articulatory feedback
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setSelectedWordIndex(weakestWordIndex)}
                      className="rounded-full border border-rose-300/35 bg-rose-400/12 px-4 py-2 text-sm font-semibold text-rose-50 transition hover:bg-rose-400/22"
                    >
                      Open detail
                    </button>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="card-surface rounded-[1.8rem] border p-5 md:p-6">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                AI backend contract
              </div>
              <div className="mt-4 space-y-3 text-sm leading-7 text-slate-200">
                <p>
                  `POST /api/pronunciation/evaluate` should return sentence score, word score, phoneme score, stress feedback, and audio segment URLs.
                </p>
                <p>
                  `POST /api/pronunciation/explain` powers the natural-language coaching button.
                </p>
                <p>
                  `POST /api/pronunciation/compare-sounds` powers the articulatory comparison panel for a target phoneme versus learner phoneme.
                </p>
              </div>
            </div>
          </aside>
        </section>
      </div>

      <WordDetailModal
        assessment={result}
        selectedWordIndex={selectedWordIndex}
        learnerAudioUrl={recorder.audioUrl}
        learnerLevel={learnerLevel}
        language={language}
        autoOpenLowest={autoOpenLowest}
        onToggleAutoOpen={setAutoOpenLowest}
        onClose={() => setSelectedWordIndex(null)}
        onSelectWord={(index) => setSelectedWordIndex(index)}
      />
    </main>
  );
}

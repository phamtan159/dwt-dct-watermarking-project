import type { ExplainResponse } from "../types/pronunciation";

interface ExplainMistakePanelProps {
  loading: boolean;
  data: ExplainResponse | null;
  onRequest: () => void;
  word: string;
  phonemeLabel: string;
}

export function ExplainMistakePanel({
  loading,
  data,
  onRequest,
  word,
  phonemeLabel,
}: ExplainMistakePanelProps) {
  return (
    <section className="rounded-[1.5rem] border border-slate-700/45 bg-slate-900/58 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">
            Explain My Mistake
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            Ask the AI coach why <span className="text-white">{word}</span> and
            <span className="text-white"> /{phonemeLabel}/</span> went off track.
          </p>
        </div>

        <button
          type="button"
          onClick={onRequest}
          disabled={loading}
          className="rounded-full border border-sky-300/45 bg-sky-400/14 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:bg-sky-400/22 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Explaining..." : "Explain My Mistake"}
        </button>
      </div>

      {data ? (
        <div className="mt-4 space-y-3 rounded-2xl border border-slate-700/40 bg-slate-950/50 p-4">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
              English
            </div>
            <p className="mt-2 text-sm leading-7 text-slate-100">
              {data.explanation_en}
            </p>
          </div>

          {data.explanation_vi ? (
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                Tiếng Việt
              </div>
              <p className="mt-2 text-sm leading-7 text-slate-100">
                {data.explanation_vi}
              </p>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

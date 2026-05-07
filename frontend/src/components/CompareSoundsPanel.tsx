import type { CompareSoundsResponse } from "../types/pronunciation";

interface CompareSoundsPanelProps {
  loading: boolean;
  data: CompareSoundsResponse | null;
  onRequest: () => void;
}

export function CompareSoundsPanel({
  loading,
  data,
  onRequest,
}: CompareSoundsPanelProps) {
  return (
    <section className="rounded-[1.5rem] border border-slate-700/45 bg-slate-900/58 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-[0.18em] text-slate-300">
            Compare Sounds
          </h3>
          <p className="mt-1 text-sm text-slate-400">
            Contrast the target sound with what the learner likely produced.
          </p>
        </div>

        <button
          type="button"
          onClick={onRequest}
          disabled={loading}
          className="rounded-full border border-emerald-300/45 bg-emerald-400/14 px-4 py-2 text-sm font-semibold text-emerald-100 transition hover:bg-emerald-400/22 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? "Comparing..." : "Compare Sounds"}
        </button>
      </div>

      {data ? (
        <div className="mt-4 rounded-2xl border border-slate-700/40 bg-slate-950/50 p-4">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="rounded-full bg-white/6 px-3 py-1 text-slate-200">
              Target: /{data.target}/
            </span>
            <span className="rounded-full bg-white/6 px-3 py-1 text-slate-200">
              You said: /{data.predicted}/
            </span>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                Difference
              </div>
              <ul className="mt-2 space-y-2 text-sm leading-6 text-slate-100">
                {data.difference.map((item) => (
                  <li key={item} className="rounded-2xl bg-white/[0.03] px-3 py-2">
                    {item}
                  </li>
                ))}
              </ul>
            </div>

            {data.tips_vi?.length ? (
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                  Tips (VI)
                </div>
                <ul className="mt-2 space-y-2 text-sm leading-6 text-slate-100">
                  {data.tips_vi.map((item) => (
                    <li key={item} className="rounded-2xl bg-white/[0.03] px-3 py-2">
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  );
}

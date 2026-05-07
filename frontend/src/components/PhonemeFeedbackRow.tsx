import clsx from "clsx";
import type { PhonemeFeedback } from "../types/pronunciation";
import { formatPercent, resolveStatusTheme } from "../utils/score";
import { AudioButton } from "./AudioButton";

interface PhonemeFeedbackRowProps {
  feedback: PhonemeFeedback;
  isFocused: boolean;
  onFocus: () => void;
  onPlayTarget: () => void;
  onPlayUser: () => void;
}

export function PhonemeFeedbackRow({
  feedback,
  isFocused,
  onFocus,
  onPlayTarget,
  onPlayUser,
}: PhonemeFeedbackRowProps) {
  const theme = resolveStatusTheme(feedback.status, feedback.score);

  return (
    <button
      type="button"
      onClick={onFocus}
      className={clsx(
        "w-full rounded-[1.4rem] border p-4 text-left transition duration-200",
        isFocused
          ? `${theme.chipActive} ring-2 ${theme.ring}`
          : "border-slate-700/45 bg-slate-900/56 hover:border-slate-500/40 hover:bg-slate-800/50",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="panel-title text-xl font-bold text-white">
              {feedback.target}
            </span>
            <span className={clsx("rounded-full px-2.5 py-1 text-xs font-semibold", theme.badge)}>
              {formatPercent(feedback.score)}
            </span>
          </div>
          <p className="mt-2 text-sm text-slate-200">{feedback.message}</p>
          <p className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-500">
            You said /{feedback.predicted}/
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <AudioButton label="Coach" onClick={onPlayTarget} />
          <AudioButton label="You" onClick={onPlayUser} />
        </div>
      </div>

      {feedback.explanation ? (
        <p className="mt-4 rounded-2xl bg-white/[0.04] px-3 py-3 text-sm leading-6 text-slate-200">
          {feedback.explanation}
        </p>
      ) : null}
    </button>
  );
}

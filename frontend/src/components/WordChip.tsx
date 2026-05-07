import clsx from "clsx";
import { formatPercent, resolveStatusTheme } from "../utils/score";

interface WordChipProps {
  label: string;
  score?: number;
  status?: Parameters<typeof resolveStatusTheme>[0];
  active?: boolean;
  onClick?: () => void;
}

export function WordChip({
  label,
  score,
  status,
  active = false,
  onClick,
}: WordChipProps) {
  const theme = resolveStatusTheme(status, score);
  const hasAssessment = typeof score === "number";

  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "group relative min-w-[88px] rounded-[1.35rem] border px-4 py-3 text-left transition duration-200",
        hasAssessment
          ? active
            ? theme.chipActive
            : theme.chip
          : "border-slate-600/28 bg-slate-900/55 text-slate-100 hover:border-slate-400/30 hover:bg-slate-800/60",
        active && "ring-2 ring-offset-0",
        hasAssessment && theme.ring,
      )}
    >
      {hasAssessment && score! < 85 ? (
        <span
          className={clsx(
            "absolute right-2.5 top-2.5 h-2.5 w-2.5 rounded-full",
            score! < 50 ? "bg-rose-400" : "bg-orange-300",
          )}
        />
      ) : null}

      <div className="pr-4">
        <div className="text-sm font-semibold">{label}</div>
        <div className="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">
          {hasAssessment ? formatPercent(score!) : "Ready"}
        </div>
      </div>
    </button>
  );
}

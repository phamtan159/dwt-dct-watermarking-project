import { formatPercent, resolveStatusTheme } from "../utils/score";

interface ScoreGaugeProps {
  score: number;
  size?: number;
  strokeWidth?: number;
  label?: string;
}

export function ScoreGauge({
  score,
  size = 176,
  strokeWidth = 14,
  label = "Word score",
}: ScoreGaugeProps) {
  const theme = resolveStatusTheme(undefined, score);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (score / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(148, 163, 184, 0.18)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={theme.gaugeColor}
          strokeLinecap="round"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
        />
      </svg>

      <div className="absolute flex flex-col items-center">
        <span className="panel-title text-4xl font-bold tracking-tight text-white">
          {formatPercent(score)}
        </span>
        <span className="mt-1 text-xs uppercase tracking-[0.22em] text-slate-400">
          {label}
        </span>
      </div>
    </div>
  );
}

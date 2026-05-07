import type { PronunciationStatus } from "../types/pronunciation";

export interface StatusTheme {
  status: PronunciationStatus;
  chip: string;
  chipActive: string;
  badge: string;
  text: string;
  ring: string;
  glow: string;
  gaugeColor: string;
}

const STATUS_THEMES: Record<PronunciationStatus, StatusTheme> = {
  correct: {
    status: "correct",
    chip:
      "border-emerald-400/30 bg-emerald-400/12 text-emerald-100 shadow-[0_18px_40px_rgba(16,185,129,0.14)]",
    chipActive:
      "border-emerald-300/70 bg-emerald-400/18 text-emerald-50 shadow-[0_0_0_1px_rgba(110,231,183,0.32)]",
    badge: "bg-emerald-400/12 text-emerald-200",
    text: "text-emerald-300",
    ring: "ring-emerald-300/50",
    glow: "shadow-glow",
    gaugeColor: "#34d399",
  },
  good: {
    status: "good",
    chip:
      "border-lime-400/30 bg-lime-400/12 text-lime-100 shadow-[0_18px_40px_rgba(132,204,22,0.12)]",
    chipActive:
      "border-lime-300/70 bg-lime-400/18 text-lime-50 shadow-[0_0_0_1px_rgba(190,242,100,0.26)]",
    badge: "bg-lime-400/12 text-lime-200",
    text: "text-lime-300",
    ring: "ring-lime-300/50",
    glow: "shadow-[0_12px_32px_rgba(132,204,22,0.22)]",
    gaugeColor: "#84cc16",
  },
  near_correct: {
    status: "near_correct",
    chip:
      "border-amber-400/30 bg-amber-400/12 text-amber-100 shadow-[0_18px_40px_rgba(245,158,11,0.12)]",
    chipActive:
      "border-amber-300/70 bg-amber-400/18 text-amber-50 shadow-[0_0_0_1px_rgba(252,211,77,0.26)]",
    badge: "bg-amber-400/12 text-amber-200",
    text: "text-amber-300",
    ring: "ring-amber-300/50",
    glow: "shadow-[0_12px_32px_rgba(245,158,11,0.22)]",
    gaugeColor: "#f59e0b",
  },
  warning: {
    status: "warning",
    chip:
      "border-orange-400/30 bg-orange-400/12 text-orange-100 shadow-[0_18px_40px_rgba(249,115,22,0.12)]",
    chipActive:
      "border-orange-300/70 bg-orange-400/18 text-orange-50 shadow-[0_0_0_1px_rgba(253,186,116,0.26)]",
    badge: "bg-orange-400/12 text-orange-200",
    text: "text-orange-300",
    ring: "ring-orange-300/50",
    glow: "shadow-[0_12px_32px_rgba(249,115,22,0.22)]",
    gaugeColor: "#f97316",
  },
  wrong: {
    status: "wrong",
    chip:
      "border-rose-400/35 bg-rose-500/12 text-rose-100 shadow-[0_18px_40px_rgba(244,63,94,0.12)]",
    chipActive:
      "border-rose-300/70 bg-rose-500/18 text-rose-50 shadow-[0_0_0_1px_rgba(253,164,175,0.26)]",
    badge: "bg-rose-500/12 text-rose-200",
    text: "text-rose-300",
    ring: "ring-rose-300/55",
    glow: "shadow-[0_12px_32px_rgba(244,63,94,0.22)]",
    gaugeColor: "#fb7185",
  },
  unrecognized: {
    status: "unrecognized",
    chip:
      "border-slate-500/35 bg-slate-500/12 text-slate-200 shadow-[0_18px_40px_rgba(100,116,139,0.1)]",
    chipActive:
      "border-slate-300/55 bg-slate-500/18 text-slate-50 shadow-[0_0_0_1px_rgba(148,163,184,0.24)]",
    badge: "bg-slate-500/14 text-slate-200",
    text: "text-slate-300",
    ring: "ring-slate-300/40",
    glow: "shadow-[0_12px_32px_rgba(100,116,139,0.18)]",
    gaugeColor: "#94a3b8",
  },
};

export function getStatusFromScore(score: number): PronunciationStatus {
  if (score >= 85) return "correct";
  if (score >= 70) return "good";
  if (score >= 50) return "warning";
  return "wrong";
}

export function resolveStatusTheme(
  status?: PronunciationStatus,
  score?: number,
): StatusTheme {
  if (status) {
    return STATUS_THEMES[status];
  }

  if (typeof score === "number") {
    return STATUS_THEMES[getStatusFromScore(score)];
  }

  return STATUS_THEMES.unrecognized;
}

export function getLowestScoringWordIndex(scores: { score: number }[]): number {
  return scores.reduce(
    (lowestIndex, item, index, arr) =>
      item.score < arr[lowestIndex].score ? index : lowestIndex,
    0,
  );
}

export function formatPercent(score: number) {
  return `${Math.round(score)}%`;
}

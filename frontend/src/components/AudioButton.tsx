import clsx from "clsx";
import type { ReactNode } from "react";

interface AudioButtonProps {
  label: string;
  onClick?: () => void;
  icon?: ReactNode;
  active?: boolean;
  disabled?: boolean;
}

export function AudioButton({
  label,
  onClick,
  icon,
  active = false,
  disabled = false,
}: AudioButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold transition duration-200",
        active
          ? "border-sky-300/60 bg-sky-400/16 text-sky-50 shadow-[0_12px_30px_rgba(56,189,248,0.22)]"
          : "border-slate-500/35 bg-slate-800/55 text-slate-100 hover:border-slate-300/40 hover:bg-slate-700/55",
        disabled && "cursor-not-allowed opacity-55",
      )}
    >
      <span className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-xs">
        {icon ?? "▶"}
      </span>
      <span>{label}</span>
    </button>
  );
}

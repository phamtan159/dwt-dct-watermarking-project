import clsx from "clsx";

interface RecordButtonProps {
  isRecording: boolean;
  isBusy?: boolean;
  disabled?: boolean;
  disabledReason?: string | null;
  onStart: () => void;
  onStop: () => void;
}

export function RecordButton({
  isRecording,
  isBusy = false,
  disabled = false,
  disabledReason = null,
  onStart,
  onStop,
}: RecordButtonProps) {
  const isDisabled = isBusy || disabled;

  return (
    <div className="flex flex-col items-center gap-4">
      <button
        type="button"
        disabled={isDisabled}
        onClick={isRecording ? onStop : onStart}
        className={clsx(
          "relative flex h-32 w-32 items-center justify-center rounded-full border text-white transition duration-200",
          isRecording
            ? "border-rose-300/60 bg-rose-500/22 shadow-[0_0_0_14px_rgba(244,63,94,0.08)]"
            : "border-emerald-300/50 bg-emerald-500/18 shadow-[0_0_0_14px_rgba(16,185,129,0.07)] hover:shadow-[0_0_0_18px_rgba(16,185,129,0.1)]",
          isDisabled && "cursor-not-allowed opacity-70",
        )}
      >
        <span
          className={clsx(
            "absolute inset-0 rounded-full",
            isRecording && "animate-ping bg-rose-400/14",
          )}
        />

        <span className="relative flex h-20 w-20 items-center justify-center rounded-full bg-white/10">
          {isBusy ? (
            <span className="h-7 w-7 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          ) : isRecording ? (
            <span className="h-7 w-7 rounded-md bg-rose-200" />
          ) : (
            <span className="flex h-12 w-12 items-center justify-center rounded-full border-[3px] border-white text-[10px] font-bold tracking-[0.18em]">
              MIC
            </span>
          )}
        </span>
      </button>

      <div className="text-center">
        <div className="text-lg font-semibold text-white">
          {isBusy
            ? "Analyzing pronunciation..."
            : isRecording
              ? "Bam lai de dung va cham diem"
              : "Bam de noi"}
        </div>
        <p className="mt-1 max-w-xs text-sm text-slate-400">
          {isDisabled && disabledReason
            ? disabledReason
            : isRecording
            ? "Read the full sentence naturally. We will align, score, and compare every word."
            : "Use the demo flow if microphone access is unavailable."}
        </p>
      </div>
    </div>
  );
}

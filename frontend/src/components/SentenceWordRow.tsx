import type { WordAssessment } from "../types/pronunciation";
import { WordChip } from "./WordChip";

interface SentenceWordRowProps {
  sentence: string;
  words?: WordAssessment[];
  selectedWordIndex?: number | null;
  onSelectWord?: (index: number) => void;
}

export function SentenceWordRow({
  sentence,
  words,
  selectedWordIndex = null,
  onSelectWord,
}: SentenceWordRowProps) {
  const tokens = sentence.trim().split(/\s+/).filter(Boolean);
  const length = Math.max(tokens.length, words?.length ?? 0);

  if (length === 0) {
    return (
      <div className="rounded-[1.35rem] border border-dashed border-slate-600/40 bg-slate-900/36 px-4 py-4 text-sm text-slate-400">
        Chua co tu nao. Hay go cau mau de he thong tao cac o tu.
      </div>
    );
  }

  return (
    <div className="flex flex-wrap gap-3">
      {Array.from({ length }).map((_, index) => {
        const assessment = words?.[index];
        const label = tokens[index] ?? assessment?.word ?? `Word ${index + 1}`;

        return (
          <WordChip
            key={`${label}-${index}`}
            label={label}
            score={assessment?.score}
            status={assessment?.status}
            active={selectedWordIndex === index}
            onClick={assessment ? () => onSelectWord?.(index) : undefined}
          />
        );
      })}
    </div>
  );
}

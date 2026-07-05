"use client";

import type { Phase } from "@/entities/phase";
import { PHASE_META } from "@/entities/phase";
import type { Sample } from "@/entities/sample";
import type { AnalysisResult } from "@/entities/analysis";
import { segmentAreaFrac } from "@/features/analyze-sample";
import { exportCsv, exportReport } from "@/features/export-report";


export function MetricsPanel({
  sample,
  result,
  selectedPhase,
  onSelectPhase,
}: {
  sample: Sample;
  result: AnalysisResult | null;
  selectedPhase: "all" | Phase;
  onSelectPhase: (p: "all" | Phase) => void;
}) {
  if (!result) {
    return (
      <div className="card flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        {/* <div className="flex h-12 w-12 items-center justify-center rounded-full bg-surface text-xl">
          📊
        </div> */}
        <p className="text-sm text-ink-soft">
          Метрики появятся после анализа образца.
        </p>
      </div>
    );
  }

  // Доля каждой фазы относительно всего кадра (в %), как для талька.
  const framePct: Record<Phase, number> = { common: 0, thin: 0, talc: 0 };
  for (const s of result.segments) framePct[s.phase] += segmentAreaFrac(s) * 100;

  return (
    <div className="card flex flex-col p-5">
      {/* Verdict */}
      <div className="mb-4 rounded-2xl bg-surface px-4 py-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-ink-faint">
          Классификация руды
        </div>
        <div className="text-2xl font-extrabold capitalize text-ink">
          {result.verdict}
        </div>
      </div>

      {/* Legend / filter — доля каждой фазы от кадра показана здесь же */}
      <div className="mb-4 space-y-1.5">
        <FilterRow
          active={selectedPhase === "all"}
          onClick={() => onSelectPhase("all")}
          color="#2E2E48"
          label="Все фазы"
        />
        {(Object.keys(PHASE_META) as Phase[]).map((p) => (
          <FilterRow
            key={p}
            active={selectedPhase === p}
            onClick={() => onSelectPhase(p)}
            color={PHASE_META[p].color}
            label={PHASE_META[p].label}
            value={framePct[p]}
          />
        ))}
      </div>

      {/* Text conclusion */}
      <div className="mb-4 rounded-2xl bg-surface px-4 py-3 text-sm leading-relaxed text-ink">
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-faint">
          Заключение
        </div>
        {result.conclusion}
      </div>

      {/* Export */}
      <div className="flex gap-2">
        <button
          onClick={() => exportCsv(sample, result)}
          className="btn-soft flex-1 justify-center"
        >
          Экспорт CSV
        </button>
        <button
          onClick={() => exportReport(sample, result)}
          className="btn-primary flex-1 !py-2 text-sm"
        >
          Отчёт PDF
        </button>
      </div>
    </div>
  );
}

function FilterRow({
  active,
  onClick,
  color,
  label,
  value,
}: {
  active: boolean;
  onClick: () => void;
  color: string;
  label: string;
  value?: number;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex w-full items-center gap-2.5 rounded-pill px-3 py-1.5 text-sm transition ${
        active
          ? "bg-surface font-semibold text-ink"
          : "text-ink-soft hover:bg-surface/60"
      }`}
    >
      <span className="h-3 w-3 rounded-full" style={{ background: color }} />
      <span className="flex-1 text-left">{label}</span>
      {value !== undefined && (
        <span className="font-mono text-sm font-bold text-ink">
          {value.toFixed(1)}%
        </span>
      )}
    </button>
  );
}

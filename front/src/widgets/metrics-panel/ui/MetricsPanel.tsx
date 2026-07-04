"use client";

import type { Phase } from "@/entities/phase";
import { PHASE_META } from "@/entities/phase";
import type { Sample } from "@/entities/sample";
import type { AnalysisResult } from "@/entities/analysis";
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

  const rows: { phase: Phase; value: number; caption: string }[] = [
    { phase: "common", value: result.commonShare, caption: "от сульфидов" },
    { phase: "thin", value: result.thinShare, caption: "от сульфидов" },
    { phase: "talc", value: result.talcShare, caption: "от кадра" },
  ];

  return (
    <div className="card flex h-full flex-col p-5">
      {/* Verdict */}
      <div className="mb-4 rounded-2xl bg-surface px-4 py-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-ink-faint">
          Классификация руды
        </div>
        <div className="text-2xl font-extrabold capitalize text-ink">
          {result.verdict}
        </div>
      </div>

      {/* Legend / filter */}
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
          />
        ))}
      </div>

      {/* Metrics table */}
      <div className="mb-4 rounded-2xl border border-line">
        <div className="flex items-center justify-between px-4 py-2.5">
          <span className="text-sm text-ink-soft">Доля сульфидов</span>
          <span className="font-mono text-lg font-bold text-ink">
            {result.sulfideShare.toFixed(1)}%
          </span>
        </div>
        {rows.map((r) => (
          <div
            key={r.phase}
            className="flex items-center justify-between border-t border-line px-4 py-2.5"
          >
            <span className="flex items-center gap-2 text-sm text-ink">
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ background: PHASE_META[r.phase].color }}
              />
              {PHASE_META[r.phase].label}
            </span>
            <span className="text-right">
              <span className="font-mono font-bold text-ink">
                {r.value.toFixed(1)}%
              </span>
              <span className="ml-1 text-xs text-ink-faint">{r.caption}</span>
            </span>
          </div>
        ))}
        <div className="flex items-center justify-between border-t border-line px-4 py-2.5">
          <span className="text-sm text-ink-soft">Точность (F1)</span>
          <span className="font-mono font-bold text-brand-dark">
            {result.f1.toFixed(3)}
          </span>
        </div>
      </div>

      {/* Text conclusion */}
      <div className="mb-4 rounded-2xl bg-surface px-4 py-3 text-sm leading-relaxed text-ink">
        <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-faint">
          Заключение
        </div>
        {result.conclusion}
      </div>

      {/* Export */}
      <div className="mt-auto flex gap-2">
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
}: {
  active: boolean;
  onClick: () => void;
  color: string;
  label: string;
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
      {label}
    </button>
  );
}

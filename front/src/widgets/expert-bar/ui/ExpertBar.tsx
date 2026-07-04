// Режим экспертной проверки (active learning) — заглушка UI.
export function ExpertBar({ disabled }: { disabled: boolean }) {
  return (
    <div className="card flex flex-wrap items-center justify-between gap-3 px-4 py-3">
      <div className="flex items-center gap-2 text-sm">
        <span className="rounded-pill bg-surface px-2.5 py-1 text-xs font-semibold text-ink-soft">
          Экспертная проверка
        </span>
        <span className="text-ink-soft">
          Отметьте ошибочные участки для дообучения (active learning)
        </span>
      </div>
      <div className="flex gap-2">
        <button disabled={disabled} className="btn-soft disabled:opacity-40">
          Кисть коррекции
        </button>
        <button disabled={disabled} className="btn-soft disabled:opacity-40">
          В набор дообучения
        </button>
      </div>
    </div>
  );
}

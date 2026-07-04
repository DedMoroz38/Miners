"use client";

import type { Sample } from "@/entities/sample";

// Высота одного ряда (~60px) и вертикальный зазор (8px) — под расчёт скролла.
const ROW_H = 60;
const GAP = 8;
const MAX_VISIBLE = 5;
// Список не выше 5 рядов; дальше — внутренний скролл, карточка не растёт.
const LIST_MAX_H = MAX_VISIBLE * ROW_H + (MAX_VISIBLE - 1) * GAP;

export function SampleQueue({
  samples,
  activeId,
  onSelect,
}: {
  samples: Sample[];
  activeId: string;
  onSelect: (s: Sample) => void;
}) {
  return (
    <div className="card flex flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-bold uppercase tracking-wide text-ink">
          Очередь образцов
        </h2>
        <span className="rounded-pill bg-surface px-2 py-0.5 text-xs font-semibold text-ink-soft">
          {samples.length}
        </span>
      </div>
      <div
        className="-mr-1 flex flex-col gap-2 overflow-y-auto pr-1"
        style={{ maxHeight: LIST_MAX_H }}
      >
        {samples.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelect(s)}
            className={`flex items-center gap-3 rounded-2xl border px-3 py-2.5 text-left transition ${
              activeId === s.id
                ? "border-brand bg-brand/10"
                : "border-line hover:border-ink/20"
            }`}
          >
            {s.imageUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={s.imageUrl}
                alt=""
                className="h-9 w-9 flex-none rounded-lg object-cover"
              />
            ) : (
              <span className="flex h-9 w-9 flex-none items-center justify-center rounded-lg bg-surface text-xs font-bold text-ink-faint">
                {s.id.slice(-3)}
              </span>
            )}
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold text-ink">
                {s.name}
              </span>
              <span className="mt-0.5 block truncate text-xs text-ink-faint">
                {s.meta}
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

// Режим экспертной проверки: локальная правка сегментов (без записи на бэкенд).
export function ExpertBar({
  disabled,
  editMode,
  onToggleEdit,
}: {
  disabled: boolean;
  editMode: boolean;
  onToggleEdit: () => void;
}) {
  return (
    <div className="card flex flex-wrap items-center justify-between gap-3 px-4 py-3">
      <div className="flex items-center gap-2 text-sm">
        <span className="rounded-pill bg-surface px-2.5 py-1 text-xs font-semibold text-ink-soft">
          Экспертная проверка
        </span>
        <span className="text-ink-soft">
          {editMode
            ? "Клик по сегменту — выбрать; цвет — сменить фазу; «+ Контур» — добавить"
            : "Скорректируйте маску вручную перед выгрузкой отчёта"}
        </span>
      </div>
      <div className="flex gap-2">
        <button
          disabled={disabled}
          onClick={onToggleEdit}
          className={`btn-soft disabled:opacity-40 ${
            editMode ? "!border-brand !bg-brand/15 !text-ink" : ""
          }`}
        >
          {editMode ? "Завершить правку" : "Кисть коррекции"}
        </button>
        <button disabled className="btn-soft disabled:opacity-40">
          В набор дообучения
        </button>
      </div>
    </div>
  );
}

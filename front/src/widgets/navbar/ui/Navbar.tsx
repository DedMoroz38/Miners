import { Logo } from "@/shared/ui/logo";

export function Navbar() {
  return (
    <header className="sticky top-4 z-30 mx-auto mb-8 flex max-w-[1400px] items-center justify-between rounded-pill bg-white px-6 py-3">
      <div className="flex items-center gap-8">
        <Logo className="h-7 w-auto text-ink" />
        <nav className="hidden items-center gap-6 text-sm font-medium text-ink lg:flex">
          <span className="cursor-pointer hover:text-brand-dark">Проекты</span>
          <span className="cursor-pointer hover:text-brand-dark">Образцы</span>
          <span className="cursor-pointer hover:text-brand-dark">Отчёты</span>
          <span className="cursor-pointer hover:text-brand-dark">Модель</span>
        </nav>
      </div>
      {/* <div className="flex items-center gap-3">
        <span className="hidden rounded-pill bg-surface px-3 py-1.5 text-xs font-semibold text-ink-soft md:inline">
          Локальный инстанс · v0.1
        </span>
        <button className="btn-ghost !px-4 !py-2 text-sm">
          Экспертный режим
        </button>
      </div> */}
    </header>
  );
}

import type { Sample } from "@/entities/sample";
import type { AnalysisResult } from "@/entities/analysis";

function downloadBlob(content: string, filename: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportCsv(sample: Sample, result: AnalysisResult) {
  const lines = [
    "metric,value,unit",
    `sulfide_share,${result.sulfideShare.toFixed(2)},%`,
    `common_intergrowths,${result.commonShare.toFixed(2)},% of sulfides`,
    `thin_intergrowths,${result.thinShare.toFixed(2)},% of sulfides`,
    `talc_share,${result.talcShare.toFixed(2)},% of frame`,
    `verdict,${result.verdict},`,
    `f1_score,${result.f1.toFixed(3)},`,
  ];
  downloadBlob(
    lines.join("\n"),
    `${sample.id}_metrics.csv`,
    "text/csv;charset=utf-8",
  );
}

// В проде — серверная генерация PDF. Здесь: печатаемый HTML-отчёт.
export function exportReport(sample: Sample, result: AnalysisResult) {
  const html = `<!doctype html><html lang="ru"><head><meta charset="utf-8">
    <title>Отчёт ${sample.id}</title>
    <style>body{font-family:system-ui;color:#2E2E48;padding:40px;max-width:720px}
    h1{text-transform:uppercase}td,th{padding:6px 12px;border-bottom:1px solid #eee;text-align:left}</style>
    </head><body>
    <h1>ШЛИФ · Отчёт по образцу ${sample.id}</h1>
    <p>${sample.meta}</p>
    <h2>Вердикт: ${result.verdict}</h2>
    <p>${result.conclusion}</p>
    <table><tr><th>Метрика</th><th>Значение</th></tr>
    <tr><td>Доля сульфидов</td><td>${result.sulfideShare.toFixed(1)}%</td></tr>
    <tr><td>Обычные срастания</td><td>${result.commonShare.toFixed(1)}%</td></tr>
    <tr><td>Тонкие срастания</td><td>${result.thinShare.toFixed(1)}%</td></tr>
    <tr><td>Тальк</td><td>${result.talcShare.toFixed(1)}%</td></tr>
    <tr><td>F1-score</td><td>${result.f1.toFixed(3)}</td></tr>
    </table></body></html>`;
  const w = window.open("", "_blank");
  if (w) {
    w.document.write(html);
    w.document.close();
    w.print();
  }
}

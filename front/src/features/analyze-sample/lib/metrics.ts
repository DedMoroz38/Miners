import type { Phase } from "@/entities/phase";
import type { AnalysisResult, Segment, Verdict } from "@/entities/analysis";

// Площадь кольца (доля единичного квадрата) по формуле шнурков, координаты 0..1.
export function ringAreaNorm(ring: number[][]): number {
  let a = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const [x1, y1] = ring[i];
    const [x2, y2] = ring[(i + 1) % n];
    a += x1 * y2 - x2 * y1;
  }
  return Math.abs(a) / 2;
}

export function segmentAreaFrac(seg: Pick<Segment, "polygons">): number {
  return seg.polygons.reduce((s, ring) => s + ringAreaNorm(ring), 0);
}

export type Shares = {
  sulfideShare: number;
  commonShare: number;
  thinShare: number;
  talcShare: number;
};

// Доли из площадей сегментов (перекрытия при локальном редактировании игнорируем —
// достаточно для экспертной правки; бэкенд считает по пикселям без перекрытий).
export function computeShares(segments: Segment[]): Shares {
  const area: Record<Phase, number> = { common: 0, thin: 0, talc: 0 };
  for (const s of segments) area[s.phase] += segmentAreaFrac(s);
  const sulfide = area.common + area.thin;
  return {
    sulfideShare: sulfide * 100,
    commonShare: sulfide > 0 ? (area.common / sulfide) * 100 : 0,
    thinShare: sulfide > 0 ? (area.thin / sulfide) * 100 : 0,
    talcShare: area.talc * 100,
  };
}

// Вердикт + текстовое заключение. Единый источник правила (совпадает с бэкендом).
export function classify(s: Shares): { verdict: Verdict; conclusion: string } {
  const { talcShare: talc, commonShare: common, thinShare: thin } = s;
  if (talc > 10) {
    const dom = thin > common ? "тонких" : "обычных";
    return {
      verdict: "оталькованная",
      conclusion: `Руда классифицирована как оталькованная: содержание талька — ${talc.toFixed(0)}%, преобладание ${dom} срастаний — ${Math.max(thin, common).toFixed(0)}%.`,
    };
  }
  if (common >= thin) {
    return {
      verdict: "рядовая",
      conclusion: `Руда классифицирована как рядовая: тальк — ${talc.toFixed(0)}% (≤10%), преобладание обычных срастаний — ${common.toFixed(0)}%.`,
    };
  }
  return {
    verdict: "труднообогатимая",
    conclusion: `Руда классифицирована как труднообогатимая: тальк — ${talc.toFixed(0)}% (≤10%), преобладание тонких срастаний — ${thin.toFixed(0)}%.`,
  };
}

// Пересчёт результата после локальной правки сегментов (доли + вердикт + areaFrac).
export function recomputeResult(base: AnalysisResult, segments: Segment[]): AnalysisResult {
  const withArea = segments.map((s) => ({ ...s, areaFrac: segmentAreaFrac(s) }));
  const shares = computeShares(withArea);
  const { verdict, conclusion } = classify(shares);
  return { ...base, ...shares, verdict, conclusion, segments: withArea };
}

import type { Phase } from "@/entities/phase";
import type { Sample } from "@/entities/sample";
import type { AnalysisResult, Segment } from "@/entities/analysis";
import { recomputeResult } from "./metrics";

// Детерминированный ГПСЧ — воспроизводимое демо для образцов без бэкенда.
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Приближаем «зерно» многоугольником (n-угольник со случайным дрожанием радиуса),
// координаты нормализованные 0..1 — тот же контракт, что у реальной сегментации.
function blob(cx: number, cy: number, r: number, rand: () => number): number[][] {
  const n = 8 + Math.floor(rand() * 4);
  const ring: number[][] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    const rr = r * (0.75 + rand() * 0.5);
    ring.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]);
  }
  return ring;
}

// Мок-инференс для образцов без serverId. Реальные образцы идут через бэкенд.
export function mockAnalyze(sample: Sample): AnalysisResult {
  const rand = mulberry32(sample.seed);
  const weights =
    sample.bias === "common"
      ? { common: 0.7, thin: 0.22, talc: 0.08 }
      : sample.bias === "thin"
        ? { common: 0.32, thin: 0.6, talc: 0.08 }
        : { common: 0.4, thin: 0.32, talc: 0.28 };

  const segments: Segment[] = [];
  const count = 46;
  for (let i = 0; i < count; i++) {
    const roll = rand();
    let phase: Phase;
    if (roll < weights.common) phase = "common";
    else if (roll < weights.common + weights.thin) phase = "thin";
    else phase = "talc";

    const base = phase === "talc" ? 0.014 + rand() * 0.022 : 0.022 + rand() * 0.05;
    const r = phase === "common" ? base * 1.15 : base;
    const cx = 0.06 + rand() * 0.88;
    const cy = 0.06 + rand() * 0.88;
    segments.push({
      id: `${phase}-${i}`,
      phase,
      confidence: 0.72 + rand() * 0.27,
      areaFrac: 0, // пересчитается в recomputeResult
      polygons: [blob(cx, cy, r, rand)],
    });
  }

  const base: AnalysisResult = {
    sulfideShare: 0,
    commonShare: 0,
    thinShare: 0,
    talcShare: 0,
    verdict: "рядовая",
    conclusion: "",
    f1: 0.9 + rand() * 0.07,
    imageWidth: 1000,
    imageHeight: 750,
    segments,
  };
  return recomputeResult(base, segments);
}

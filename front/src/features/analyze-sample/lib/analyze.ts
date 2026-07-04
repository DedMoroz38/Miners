import type { Phase } from "@/entities/phase";
import type { Sample } from "@/entities/sample";
import type { AnalysisResult, Grain, Verdict } from "@/entities/analysis";

// Детерминированный ГПСЧ, чтобы демо было воспроизводимым.
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Мок-инференс. Заменяется вызовом реального ML-API (контракт — AnalysisResult).
export function analyze(sample: Sample): AnalysisResult {
  const rand = mulberry32(sample.seed);
  const grains: Grain[] = [];

  const weights =
    sample.bias === "common"
      ? { common: 0.7, thin: 0.22, talc: 0.08 }
      : sample.bias === "thin"
        ? { common: 0.32, thin: 0.6, talc: 0.08 }
        : { common: 0.4, thin: 0.32, talc: 0.28 };

  const count = 46;
  let commonArea = 0;
  let thinArea = 0;
  let talcArea = 0;

  for (let i = 0; i < count; i++) {
    const roll = rand();
    let phase: Phase;
    if (roll < weights.common) phase = "common";
    else if (roll < weights.common + weights.thin) phase = "thin";
    else phase = "talc";

    const base = phase === "talc" ? 1.4 + rand() * 2.2 : 2.2 + rand() * 5.5;
    const r = phase === "common" ? base * 1.15 : base;
    const cx = 6 + rand() * 88;
    const cy = 6 + rand() * 88;
    const confidence = 0.72 + rand() * 0.27;
    grains.push({ cx, cy, r, phase, confidence });

    const area = Math.PI * r * r;
    if (phase === "common") commonArea += area;
    else if (phase === "thin") thinArea += area;
    else talcArea += area;
  }

  const frameArea = 100 * 100;
  const sulfideArea = commonArea + thinArea;
  const sulfideShare = (sulfideArea / frameArea) * 100;
  const talcShare = (talcArea / frameArea) * 100;
  const commonShare = (commonArea / sulfideArea) * 100;
  const thinShare = (thinArea / sulfideArea) * 100;

  let verdict: Verdict;
  if (talcShare > 10) verdict = "оталькованная";
  else if (commonShare >= thinShare) verdict = "рядовая";
  else verdict = "труднообогатимая";

  const conclusion =
    verdict === "оталькованная"
      ? `Руда классифицирована как оталькованная: содержание талька — ${talcShare.toFixed(0)}%, преобладание ${thinShare > commonShare ? "тонких" : "обычных"} срастаний — ${Math.max(thinShare, commonShare).toFixed(0)}%.`
      : verdict === "труднообогатимая"
        ? `Руда классифицирована как труднообогатимая: тальк — ${talcShare.toFixed(0)}% (≤10%), преобладание тонких срастаний — ${thinShare.toFixed(0)}%.`
        : `Руда классифицирована как рядовая: тальк — ${talcShare.toFixed(0)}% (≤10%), преобладание обычных срастаний — ${commonShare.toFixed(0)}%.`;

  return {
    sulfideShare,
    commonShare,
    thinShare,
    talcShare,
    verdict,
    conclusion,
    f1: 0.9 + rand() * 0.07,
    grains,
  };
}

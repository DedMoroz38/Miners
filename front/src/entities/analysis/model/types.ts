import type { Phase } from "@/entities/phase";

// Зерно/сегмент фазовой маски (в проде — из пиксельной сегментации).
export type Grain = {
  cx: number; // 0..100 (viewBox %)
  cy: number;
  r: number;
  phase: Phase;
  confidence: number; // 0..1
};

export type Verdict = "рядовая" | "труднообогатимая" | "оталькованная";

// Контракт результата анализа — под замену реальным ML-API.
export type AnalysisResult = {
  sulfideShare: number; // % площади (обычные + тонкие)
  commonShare: number; // % от сульфидов
  thinShare: number; // % от сульфидов
  talcShare: number; // % площади всего кадра
  verdict: Verdict;
  conclusion: string;
  f1: number;
  grains: Grain[];
};

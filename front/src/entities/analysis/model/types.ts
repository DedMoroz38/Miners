import type { Phase } from "@/entities/phase";

// Сегмент фазовой маски. Полигоны — в НОРМАЛИЗОВАННЫХ координатах (x,y в 0..1),
// поэтому не зависят от разрешения снимка. Одно кольцо = один замкнутый контур.
export type Segment = {
  id: string;
  phase: Phase;
  confidence: number; // 0..1
  areaFrac: number; // доля площади кадра (0..1)
  polygons: number[][][]; // [ [ [x,y], ... ], ... ]
};

export type Verdict = "рядовая" | "труднообогатимая" | "оталькованная";

// Контракт результата анализа — совпадает с бэкендом (AnalysisResultOut).
export type AnalysisResult = {
  sulfideShare: number; // % площади (обычные + тонкие)
  commonShare: number; // % от сульфидов
  thinShare: number; // % от сульфидов
  talcShare: number; // % площади всего кадра
  verdict: Verdict;
  conclusion: string;
  f1: number;
  imageWidth: number;
  imageHeight: number;
  segments: Segment[];
};

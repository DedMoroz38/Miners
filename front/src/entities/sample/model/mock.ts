import type { Sample } from "./types";

// Демо-очередь. Заменяется списком с бэкенда.
export const SAMPLES: Sample[] = [
  {
    id: "NK-1042",
    name: "NK-1042 · Медно-никелевая руда",
    meta: "OM · 8000×6000 px · масштаб 2.4 мкм/px",
    seed: 7,
    bias: "common",
  },
  {
    id: "NK-1188",
    name: "NK-1188 · Вкрапленная руда",
    meta: "OM · 9600×7200 px · масштаб 1.8 мкм/px",
    seed: 23,
    bias: "thin",
  },
  {
    id: "NK-2317",
    name: "NK-2317 · Оталькованная руда",
    meta: "OM · 10000×10000 px · масштаб 2.0 мкм/px",
    seed: 41,
    bias: "talc",
  },
];

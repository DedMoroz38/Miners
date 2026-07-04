"use client";

import { useEffect, useRef, useState } from "react";
import type { Phase } from "@/entities/phase";
import { PHASE_META } from "@/entities/phase";
import type { Sample } from "@/entities/sample";
import type { AnalysisResult } from "@/entities/analysis";
import { PREPROCESS_STEPS } from "@/entities/analysis";

type Layers = { image: boolean; mask: boolean; heatmap: boolean };

const MIN_ZOOM = 1;
const MAX_ZOOM = 6;

export function SlideViewer({
  sample,
  result,
  selectedPhase,
  status,
  step,
}: {
  sample: Sample;
  result: AnalysisResult | null;
  selectedPhase: "all" | Phase;
  status: "idle" | "processing" | "done";
  step: number;
}) {
  const [layers, setLayers] = useState<Layers>({
    image: true,
    mask: true,
    heatmap: false,
  });
  const [maskOpacity, setMaskOpacity] = useState(0.55);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  // Зеркалим стейт в ref-ы, чтобы нативный wheel-обработчик читал свежие значения.
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  zoomRef.current = zoom;
  panRef.current = pan;

  const toggle = (k: keyof Layers) => setLayers((l) => ({ ...l, [k]: !l[k] }));

  function onDown(e: React.MouseEvent) {
    drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
  }
  function onMove(e: React.MouseEvent) {
    if (!drag.current) return;
    setPan({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y });
  }
  function onUp() {
    drag.current = null;
  }
  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  // Зум колесом с приближением к точке под курсором.
  // Нативный listener с { passive: false } — иначе preventDefault() не работает
  // (React вешает onWheel как passive) и вместе с зумом скроллится вся страница.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;

    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const rect = el!.getBoundingClientRect();
      const curZoom = zoomRef.current;
      const curPan = panRef.current;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const nextZoom = clamp(curZoom * factor, MIN_ZOOM, MAX_ZOOM);
      if (nextZoom === curZoom) return;
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      const ratio = nextZoom / curZoom;
      setPan(
        nextZoom === MIN_ZOOM
          ? { x: 0, y: 0 }
          : { x: cx - (cx - curPan.x) * ratio, y: cy - (cy - curPan.y) * ratio },
      );
      setZoom(nextZoom);
    }

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  function zoomButton(dir: 1 | -1) {
    setZoom((z) => {
      const next = clamp(z + dir * 0.25, MIN_ZOOM, MAX_ZOOM);
      if (next === MIN_ZOOM) setPan({ x: 0, y: 0 });
      return next;
    });
  }

  const visibleGrains =
    result?.grains.filter(
      (g) => selectedPhase === "all" || g.phase === selectedPhase,
    ) ?? [];

  return (
    <div className="card flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <LayerChip
            active={layers.image}
            onClick={() => toggle("image")}
            label="Исходник"
          />
          <LayerChip
            active={layers.mask}
            onClick={() => toggle("mask")}
            label="Маска фаз"
            disabled={!result}
          />
          <LayerChip
            active={layers.heatmap}
            onClick={() => toggle("heatmap")}
            label="Карта уверенности"
            disabled={!result}
          />
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs font-medium text-ink-soft">
            Прозрачность
            <input
              type="range"
              min={0.15}
              max={0.9}
              step={0.05}
              value={maskOpacity}
              onChange={(e) => setMaskOpacity(Number(e.target.value))}
              className="slider"
              style={
                {
                  "--pct": `${((maskOpacity - 0.15) / (0.9 - 0.15)) * 100}%`,
                } as React.CSSProperties
              }
            />
          </label>
          <div className="flex items-center gap-1">
            <IconBtn onClick={() => zoomButton(-1)}>−</IconBtn>
            <span className="w-12 text-center text-xs font-semibold text-ink">
              {Math.round(zoom * 100)}%
            </span>
            <IconBtn onClick={() => zoomButton(1)}>+</IconBtn>
            <button
              onClick={resetView}
              className="ml-1 rounded-pill border border-line px-3 py-1 text-xs font-medium text-ink-soft hover:border-ink/30"
            >
              Сброс
            </button>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div
        ref={canvasRef}
        className="relative aspect-[4/3] w-full cursor-grab overflow-hidden bg-[#0c0c14] active:cursor-grabbing"
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={onUp}
      >
        <div
          className="absolute inset-0"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "center",
            transition: drag.current ? "none" : "transform 0.12s ease-out",
          }}
        >
          {/* Реальное изображение шлифа, если загружено */}
          {layers.image && sample.imageUrl && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={sample.imageUrl}
              alt={sample.name}
              draggable={false}
              className="absolute inset-0 h-full w-full select-none object-cover"
            />
          )}

          <svg
            viewBox="0 0 100 75"
            className="absolute inset-0 h-full w-full"
            preserveAspectRatio="xMidYMid slice"
          >
            <defs>
              <radialGradient id="matrix" cx="40%" cy="35%">
                <stop offset="0%" stopColor="#20202e" />
                <stop offset="100%" stopColor="#0a0a12" />
              </radialGradient>
            </defs>

            {/* Синтетическая матрица + зёрна — только когда нет реального снимка */}
            {layers.image && !sample.imageUrl && (
              <g>
                <rect width="100" height="75" fill="url(#matrix)" />
                {(result?.grains ?? placeholderGrains).map((b, i) => (
                  <circle
                    key={i}
                    cx={b.cx}
                    cy={b.cy * 0.75}
                    r={b.r}
                    fill={
                      "phase" in b && b.phase === "talc" ? "#2c2c3a" : "#c7c9d1"
                    }
                    opacity={"phase" in b && b.phase === "talc" ? 0.7 : 0.9}
                  />
                ))}
              </g>
            )}

            {/* Phase mask overlay */}
            {layers.mask && result && (
              <g opacity={maskOpacity}>
                {visibleGrains.map((g, i) => (
                  <circle
                    key={i}
                    cx={g.cx}
                    cy={g.cy * 0.75}
                    r={g.r}
                    fill={PHASE_META[g.phase].color}
                  />
                ))}
              </g>
            )}

            {/* Confidence heatmap */}
            {layers.heatmap && result && (
              <g>
                {result.grains.map((g, i) => (
                  <circle
                    key={i}
                    cx={g.cx}
                    cy={g.cy * 0.75}
                    r={g.r * 1.8}
                    fill={g.confidence > 0.9 ? "#00E6A6" : "#F59E0B"}
                    opacity={0.18}
                  />
                ))}
              </g>
            )}
          </svg>
        </div>

        {/* Overlay badges */}
        <div className="pointer-events-none absolute left-4 top-4 rounded-pill bg-black/45 px-3 py-1 text-xs font-medium text-white backdrop-blur">
          {sample.meta}
        </div>

        {/* Индикатор предобработки */}
        {status === "processing" && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-64 rounded-2xl bg-black/60 p-5 text-white">
              <div className="mb-3 text-sm font-semibold">Предобработка</div>
              <div className="space-y-2">
                {PREPROCESS_STEPS.map((label, i) => {
                  const done = step > i;
                  const running = step === i;
                  return (
                    <div key={label} className="flex items-center gap-2 text-xs">
                      <span
                        className={`flex h-4 w-4 flex-none items-center justify-center rounded-full text-[10px] ${
                          done
                            ? "bg-brand text-ink"
                            : running
                              ? "border border-brand"
                              : "border border-white/25"
                        }`}
                      >
                        {done && "✓"}
                        {running && (
                          <span className="h-2.5 w-2.5 animate-spin rounded-full border border-white/40 border-t-brand" />
                        )}
                      </span>
                      <span className={done || running ? "" : "text-white/50"}>
                        {label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {status === "idle" && !result && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <span className="rounded-pill bg-black/50 px-4 py-2 text-sm text-white backdrop-blur">
              Запустите анализ, чтобы получить маску фаз
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}

const placeholderGrains: { cx: number; cy: number; r: number }[] = Array.from(
  { length: 30 },
  (_, i) => ({
    cx: ((i * 37) % 96) + 2,
    cy: ((i * 53) % 92) + 2,
    r: 1.5 + ((i * 13) % 5),
  }),
);

function LayerChip({
  active,
  onClick,
  label,
  disabled,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-pill border px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${
        active
          ? "border-brand bg-brand/15 text-ink"
          : "border-line bg-white text-ink-soft hover:border-ink/30"
      }`}
    >
      {label}
    </button>
  );
}

function IconBtn({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex h-7 w-7 items-center justify-center rounded-full border border-line text-lg font-bold text-ink hover:border-ink/30"
    >
      {children}
    </button>
  );
}

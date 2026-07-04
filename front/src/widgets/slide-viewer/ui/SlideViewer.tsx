"use client";

import { useEffect, useRef, useState } from "react";
import type { Phase } from "@/entities/phase";
import { PHASE_META } from "@/entities/phase";
import type { Sample } from "@/entities/sample";
import type { AnalysisResult, Segment } from "@/entities/analysis";
import { PREPROCESS_STEPS } from "@/entities/analysis";
import { recomputeResult } from "@/features/analyze-sample";

type Layers = { image: boolean; mask: boolean; heatmap: boolean };

const MIN_ZOOM = 1;
const MAX_ZOOM = 6;
const PHASES: Phase[] = ["common", "thin", "talc"];

export function SlideViewer({
  sample,
  result,
  selectedPhase,
  status,
  step,
  editMode,
  onResultChange,
}: {
  sample: Sample;
  result: AnalysisResult | null;
  selectedPhase: "all" | Phase;
  status: "idle" | "processing" | "done";
  step: number;
  editMode: boolean;
  onResultChange: (r: AnalysisResult) => void;
}) {
  const [layers, setLayers] = useState<Layers>({
    image: true,
    mask: true,
    heatmap: false,
  });
  const [maskOpacity, setMaskOpacity] = useState(0.55);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // --- editing state ---
  const [drawPhase, setDrawPhase] = useState<Phase>("talc");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState<number[][]>([]);

  useEffect(() => {
    if (!editMode) {
      setSelectedId(null);
      setAdding(false);
      setDraft([]);
    }
  }, [editMode]);

  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  zoomRef.current = zoom;
  panRef.current = pan;

  const toggle = (k: keyof Layers) => setLayers((l) => ({ ...l, [k]: !l[k] }));

  const vbW = result?.imageWidth ?? 100;
  const vbH = result?.imageHeight ?? 75;

  function onDown(e: React.MouseEvent) {
    drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y, moved: false };
  }
  function onMove(e: React.MouseEvent) {
    if (!drag.current) return;
    drag.current.moved = true;
    setPan({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y });
  }
  function onUp() {
    drag.current = null;
  }
  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

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

  // Экранные координаты -> нормализованные (0..1) через матрицу самого SVG:
  // getScreenCTM учитывает и viewBox, и CSS-трансформы зума/панорамирования.
  function clientToNorm(e: React.MouseEvent): number[] | null {
    const svg = svgRef.current;
    if (!svg) return null;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const p = pt.matrixTransform(ctm.inverse());
    return [clamp(p.x / vbW, 0, 1), clamp(p.y / vbH, 0, 1)];
  }

  function onSvgClick(e: React.MouseEvent) {
    if (!adding) return;
    const p = clientToNorm(e);
    if (p) setDraft((d) => [...d, p]);
  }

  function finishAdd() {
    if (result && draft.length >= 3) {
      const seg: Segment = {
        id: `${drawPhase}-user-${Date.now()}`,
        phase: drawPhase,
        confidence: 1,
        areaFrac: 0,
        polygons: [draft],
      };
      onResultChange(recomputeResult(result, [...result.segments, seg]));
    }
    setAdding(false);
    setDraft([]);
  }

  function pickPhase(phase: Phase) {
    setDrawPhase(phase);
    if (result && selectedId && !adding) {
      const segs = result.segments.map((s) =>
        s.id === selectedId ? { ...s, phase } : s,
      );
      onResultChange(recomputeResult(result, segs));
    }
  }

  function deleteSelected() {
    if (!result || !selectedId) return;
    onResultChange(
      recomputeResult(
        result,
        result.segments.filter((s) => s.id !== selectedId),
      ),
    );
    setSelectedId(null);
  }

  const visibleSegments =
    result?.segments.filter(
      (g) => selectedPhase === "all" || g.phase === selectedPhase,
    ) ?? [];

  return (
    <div className="card flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <LayerChip active={layers.image} onClick={() => toggle("image")} label="Исходник" />
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

      {/* Edit toolbar */}
      {editMode && result && (
        <div className="flex flex-wrap items-center gap-3 border-b border-line bg-surface/60 px-5 py-2.5">
          <span className="text-xs font-semibold text-ink-soft">Правка:</span>
          <div className="flex items-center gap-1.5">
            {PHASES.map((p) => (
              <button
                key={p}
                onClick={() => pickPhase(p)}
                title={PHASE_META[p].label}
                className={`h-6 w-6 rounded-full border-2 transition ${
                  drawPhase === p ? "border-ink" : "border-transparent"
                }`}
                style={{ background: PHASE_META[p].color }}
              />
            ))}
          </div>
          <span className="text-xs text-ink-faint">
            {selectedId
              ? "выбран сегмент — клик по цвету меняет фазу"
              : "клик по сегменту — выбрать"}
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={deleteSelected}
              disabled={!selectedId}
              className="btn-soft !py-1 text-xs disabled:opacity-40"
            >
              Удалить
            </button>
            {adding ? (
              <>
                <button onClick={finishAdd} className="btn-primary !py-1 text-xs">
                  Готово ({draft.length})
                </button>
                <button
                  onClick={() => {
                    setAdding(false);
                    setDraft([]);
                  }}
                  className="btn-soft !py-1 text-xs"
                >
                  Отмена
                </button>
              </>
            ) : (
              <button
                onClick={() => {
                  setSelectedId(null);
                  setAdding(true);
                  setDraft([]);
                }}
                className="btn-soft !py-1 text-xs"
              >
                + Контур ({PHASE_META[drawPhase].label})
              </button>
            )}
          </div>
        </div>
      )}

      {/* Canvas */}
      <div
        ref={canvasRef}
        className={`relative aspect-[4/3] w-full overflow-hidden bg-[#0c0c14] ${
          adding ? "cursor-crosshair" : "cursor-grab active:cursor-grabbing"
        }`}
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
            ref={svgRef}
            viewBox={`0 0 ${vbW} ${vbH}`}
            className="absolute inset-0 h-full w-full"
            preserveAspectRatio="xMidYMid slice"
            onClick={onSvgClick}
          >
            <defs>
              <radialGradient id="matrix" cx="40%" cy="35%">
                <stop offset="0%" stopColor="#20202e" />
                <stop offset="100%" stopColor="#0a0a12" />
              </radialGradient>
            </defs>

            {/* Тёмная матрица-подложка, когда реального снимка нет */}
            {layers.image && !sample.imageUrl && (
              <rect width={vbW} height={vbH} fill="url(#matrix)" />
            )}

            {/* Phase mask overlay (polygons) */}
            {layers.mask && result && (
              <g opacity={maskOpacity} style={{ pointerEvents: editMode && !adding ? "auto" : "none" }}>
                {visibleSegments.map((s) =>
                  s.polygons.map((ring, ri) => (
                    <path
                      key={`${s.id}-${ri}`}
                      d={ringToPath(ring, vbW, vbH)}
                      fill={PHASE_META[s.phase].color}
                      stroke={selectedId === s.id ? "#ffffff" : "none"}
                      strokeWidth={selectedId === s.id ? Math.max(vbW, vbH) / 250 : 0}
                      style={{ cursor: editMode ? "pointer" : "default" }}
                      onClick={(e) => {
                        if (editMode && !adding) {
                          e.stopPropagation();
                          setSelectedId(s.id);
                        }
                      }}
                    />
                  )),
                )}
              </g>
            )}

            {/* Confidence heatmap */}
            {layers.heatmap && result && (
              <g style={{ pointerEvents: "none" }}>
                {result.segments.map((s) =>
                  s.polygons.map((ring, ri) => (
                    <path
                      key={`h-${s.id}-${ri}`}
                      d={ringToPath(ring, vbW, vbH)}
                      fill={s.confidence > 0.9 ? "#00E6A6" : "#F59E0B"}
                      opacity={0.28}
                    />
                  )),
                )}
              </g>
            )}

            {/* Draft polygon being drawn */}
            {adding && draft.length > 0 && (
              <g style={{ pointerEvents: "none" }}>
                <polyline
                  points={draft.map(([x, y]) => `${x * vbW},${y * vbH}`).join(" ")}
                  fill={PHASE_META[drawPhase].color}
                  fillOpacity={0.3}
                  stroke={PHASE_META[drawPhase].color}
                  strokeWidth={Math.max(vbW, vbH) / 300}
                />
                {draft.map(([x, y], i) => (
                  <circle
                    key={i}
                    cx={x * vbW}
                    cy={y * vbH}
                    r={Math.max(vbW, vbH) / 200}
                    fill="#ffffff"
                    stroke={PHASE_META[drawPhase].color}
                    strokeWidth={Math.max(vbW, vbH) / 400}
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

function ringToPath(ring: number[][], w: number, h: number): string {
  if (ring.length === 0) return "";
  const pts = ring.map(([x, y]) => `${(x * w).toFixed(2)},${(y * h).toFixed(2)}`);
  return `M${pts.join("L")}Z`;
}

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}

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

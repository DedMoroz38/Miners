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

// Глубина истории «отмены» (Ctrl/⌘+Z). 20 шагов — разумный дефолт.
const UNDO_LIMIT = 20;
// Системная комбинация отмены: ⌘Z на macOS, Ctrl+Z в остальных ОС.
const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/i.test(navigator.userAgent);

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
  // Active vertex drag on the selected segment: which ring / point is being moved.
  // `pushed` guards that a drag adds exactly ONE undo entry (on first move).
  const vDrag = useRef<{ ri: number; pi: number; pushed: boolean } | null>(null);

  // Undo/redo history: snapshots of the whole result taken BEFORE each edit.
  const [undoStack, setUndoStack] = useState<AnalysisResult[]>([]);
  const [redoStack, setRedoStack] = useState<AnalysisResult[]>([]);

  // Drop history when the edited image changes or a new analysis starts.
  useEffect(() => {
    setUndoStack([]);
    setRedoStack([]);
  }, [sample.id]);
  useEffect(() => {
    if (status === "processing") {
      setUndoStack([]);
      setRedoStack([]);
    }
  }, [status]);

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
    if (vDrag.current) return; // editing a vertex — don't pan
    drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y, moved: false };
  }
  function onMove(e: React.MouseEvent) {
    if (vDrag.current || !drag.current) return;
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
      pushHistory();
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
      pushHistory();
      const segs = result.segments.map((s) =>
        s.id === selectedId ? { ...s, phase } : s,
      );
      onResultChange(recomputeResult(result, segs));
    }
  }

  function deleteSelected() {
    if (!result || !selectedId) return;
    pushHistory();
    onResultChange(
      recomputeResult(
        result,
        result.segments.filter((s) => s.id !== selectedId),
      ),
    );
    setSelectedId(null);
  }

  // --- editing the geometry of the SELECTED segment (vertex handles) ---
  const selectedSeg = result?.segments.find((s) => s.id === selectedId) ?? null;

  // Replace the selected segment's polygons and recompute shares/verdict live.
  function commitSelectedPolys(polys: number[][][]) {
    if (!result || !selectedId) return;
    const segs = result.segments.map((s) =>
      s.id === selectedId ? { ...s, polygons: polys } : s,
    );
    onResultChange(recomputeResult(result, segs));
  }

  function onVertexDown(e: React.PointerEvent, ri: number, pi: number) {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId); // keep events while off-handle
    vDrag.current = { ri, pi, pushed: false };
  }
  function onVertexMove(e: React.PointerEvent) {
    if (!vDrag.current || !selectedSeg) return;
    const p = clientToNorm(e);
    if (!p) return;
    if (!vDrag.current.pushed) {
      pushHistory(); // one undo entry per drag, only once it actually moves
      vDrag.current.pushed = true;
    }
    const { ri, pi } = vDrag.current;
    const polys = selectedSeg.polygons.map((ring, r) =>
      r === ri ? ring.map((pt, i) => (i === pi ? p : pt)) : ring,
    );
    commitSelectedPolys(polys);
  }
  function onVertexUp(e: React.PointerEvent) {
    if (!vDrag.current) return;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* capture already released */
    }
    vDrag.current = null;
  }
  function deleteVertex(e: React.MouseEvent, ri: number, pi: number) {
    e.preventDefault();
    e.stopPropagation();
    if (!selectedSeg || selectedSeg.polygons[ri].length <= 3) return; // keep a valid ring
    pushHistory();
    const polys = selectedSeg.polygons.map((ring, r) =>
      r === ri ? ring.filter((_, i) => i !== pi) : ring,
    );
    commitSelectedPolys(polys);
  }
  function insertVertex(e: React.MouseEvent, ri: number, pi: number) {
    e.stopPropagation();
    if (!selectedSeg) return;
    pushHistory();
    const ring = selectedSeg.polygons[ri];
    const a = ring[pi];
    const b = ring[(pi + 1) % ring.length];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const newRing = [...ring.slice(0, pi + 1), mid, ...ring.slice(pi + 1)];
    const polys = selectedSeg.polygons.map((rg, idx) => (idx === ri ? newRing : rg));
    commitSelectedPolys(polys);
  }

  // --- undo (Ctrl/⌘+Z + button) ---------------------------------------------
  // Snapshot the current result BEFORE an edit. Edits are immutable (map/filter/
  // spread), so keeping the reference is a valid snapshot — no deep clone needed.
  function pushHistory() {
    if (!result) return;
    setUndoStack((s) => [...s, result].slice(-UNDO_LIMIT));
    setRedoStack([]); // a fresh edit invalidates the redo branch
  }

  const canUndo = (adding && draft.length > 0) || undoStack.length > 0;
  const canRedo = redoStack.length > 0;

  // Clear selection if a restored state no longer contains the selected segment.
  function reconcileSelection(next: AnalysisResult) {
    if (selectedId && !next.segments.some((seg) => seg.id === selectedId)) {
      setSelectedId(null);
    }
  }

  function undo() {
    // While drawing a contour, step back one draft point first.
    if (adding && draft.length > 0) {
      setDraft((d) => d.slice(0, -1));
      return;
    }
    if (undoStack.length === 0) return;
    const prev = undoStack[undoStack.length - 1];
    setUndoStack((s) => s.slice(0, -1));
    if (result) setRedoStack((r) => [...r, result].slice(-UNDO_LIMIT));
    onResultChange(prev);
    reconcileSelection(prev);
  }

  function redo() {
    if (redoStack.length === 0) return;
    const next = redoStack[redoStack.length - 1];
    setRedoStack((r) => r.slice(0, -1));
    if (result) setUndoStack((s) => [...s, result].slice(-UNDO_LIMIT));
    onResultChange(next);
    reconcileSelection(next);
  }

  // Keep one keydown listener that always calls the latest undo/redo/editMode.
  const undoRef = useRef(undo);
  undoRef.current = undo;
  const redoRef = useRef(redo);
  redoRef.current = redo;
  const editModeRef = useRef(editMode);
  editModeRef.current = editMode;
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!editModeRef.current) return;
      const meta = IS_MAC ? e.metaKey : e.ctrlKey;
      if (!meta || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "z" && !e.shiftKey) {
        e.preventDefault();
        undoRef.current();
      } else if ((k === "y" && !e.shiftKey) || (k === "z" && e.shiftKey)) {
        // Ctrl+Y (Windows/Linux) or Ctrl/⌘+Shift+Z (common / macOS) = redo
        e.preventDefault();
        redoRef.current();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

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
          {/* <span className="text-xs text-ink-faint">
            {selectedId
              ? "тяните точки · клик по ребру — добавить · правый клик по точке — удалить · цвет меняет фазу"
              : "клик по сегменту — выбрать"}
          </span> */}
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={undo}
              disabled={!canUndo}
              aria-label="Отменить"
              title={`Отменить (${IS_MAC ? "⌘Z" : "Ctrl+Z"})`}
              className="flex h-7 w-7 items-center justify-center rounded-full border border-line text-base leading-none text-ink hover:border-ink/30 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:border-line"
            >
              ↶
            </button>
            <button
              onClick={redo}
              disabled={!canRedo}
              aria-label="Повторить"
              title={`Повторить (${IS_MAC ? "⌘⇧Z" : "Ctrl+Y"})`}
              className="flex h-7 w-7 items-center justify-center rounded-full border border-line text-base leading-none text-ink hover:border-ink/30 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:border-line"
            >
              ↷
            </button>
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

            {/* Карта уверенности: цвета сегментов — как в классификации (фазы),
                число по центру сегмента — уверенность модели в %. Прозрачность
                группы завязана на слайдер «прозрачность» (цвета и числа гаснут). */}
            {layers.heatmap && result && (
              <g opacity={maskOpacity} style={{ pointerEvents: "none" }}>
                {visibleSegments.map((s) => {
                  const big = s.polygons.reduce(
                    (a, b) => (b.length > a.length ? b : a),
                    s.polygons[0] ?? [],
                  );
                  const fs = Math.max(vbW, vbH) / 42 / zoom;
                  return (
                    <g key={`h-${s.id}`}>
                      {s.polygons.map((ring, ri) => (
                        <path
                          key={ri}
                          d={ringToPath(ring, vbW, vbH)}
                          fill={PHASE_META[s.phase].color}
                        />
                      ))}
                      {big.length > 0 &&
                        (() => {
                          const [cx, cy] = ringCentroid(big);
                          return (
                            <text
                              x={cx * vbW}
                              y={cy * vbH}
                              fontSize={fs}
                              fill="#ffffff"
                              stroke="#0a0a12"
                              strokeWidth={fs / 7}
                              paintOrder="stroke"
                              textAnchor="middle"
                              dominantBaseline="central"
                              style={{ fontWeight: 700 }}
                            >
                              {Math.round(s.confidence * 100)}%
                            </text>
                          );
                        })()}
                    </g>
                  );
                })}
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

            {/* Vertex handles: reshape the selected segment (drag / insert / delete).
                Radius divided by zoom so handles stay a constant on-screen size. */}
            {editMode && !adding && selectedSeg && (
              <g onMouseDown={(e) => e.stopPropagation()}>
                {selectedSeg.polygons.map((ring, ri) => {
                  const r = Math.max(vbW, vbH) / 150 / zoom;
                  const sw = Math.max(vbW, vbH) / 600 / zoom;
                  return (
                    <g key={`edit-${ri}`}>
                      {/* edge midpoints — click to insert a new vertex */}
                      {ring.map((pt, pi) => {
                        const nb = ring[(pi + 1) % ring.length];
                        return (
                          <circle
                            key={`mid-${pi}`}
                            cx={((pt[0] + nb[0]) / 2) * vbW}
                            cy={((pt[1] + nb[1]) / 2) * vbH}
                            r={r * 0.62}
                            fill="#ffffff"
                            fillOpacity={0.45}
                            stroke="#2E2E48"
                            strokeWidth={sw}
                            style={{ cursor: "copy" }}
                            onClick={(e) => insertVertex(e, ri, pi)}
                          />
                        );
                      })}
                      {/* vertices — drag to move, right-click to delete */}
                      {ring.map((pt, pi) => (
                        <circle
                          key={`vtx-${pi}`}
                          cx={pt[0] * vbW}
                          cy={pt[1] * vbH}
                          r={r}
                          fill={PHASE_META[selectedSeg.phase].color}
                          stroke="#ffffff"
                          strokeWidth={sw}
                          style={{ cursor: "grab", touchAction: "none" }}
                          onPointerDown={(e) => onVertexDown(e, ri, pi)}
                          onPointerMove={onVertexMove}
                          onPointerUp={onVertexUp}
                          onContextMenu={(e) => deleteVertex(e, ri, pi)}
                        />
                      ))}
                    </g>
                  );
                })}
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

// Средняя точка кольца (нормализованные координаты) — для подписи % по центру.
function ringCentroid(ring: number[][]): [number, number] {
  let x = 0;
  let y = 0;
  for (const [px, py] of ring) {
    x += px;
    y += py;
  }
  return [x / ring.length, y / ring.length];
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

"use client";

import { useEffect, useRef, useState } from "react";
import type { Phase } from "@/entities/phase";
import { PHASE_META } from "@/entities/phase";
import type { Sample } from "@/entities/sample";
import type { AnalysisResult, Segment } from "@/entities/analysis";
import { PREPROCESS_STEPS } from "@/entities/analysis";
import { recomputeResult } from "@/features/analyze-sample";

type Layers = { image: boolean; mask: boolean; heatmap: boolean };

const PHASES: Phase[] = ["common", "thin", "talc"];

// Глубина истории «отмены» (Ctrl/⌘+Z). 20 шагов — разумный дефолт.
const UNDO_LIMIT = 20;
// Максимум одновременно рисуемых ручек-вершин (culling по вьюпорту + прореживание).
const MAX_HANDLES = 300;
// Цель по числу точек для кнопки «Упростить» (Douglas–Peucker).
const SIMPLIFY_TARGET = 60;
type Bounds = { x0: number; y0: number; x1: number; y1: number };
const FULL_BOUNDS: Bounds = { x0: 0, y0: 0, x1: 1, y1: 1 };
// Системная комбинация отмены: ⌘Z на macOS, Ctrl+Z в остальных ОС.
const IS_MAC =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/i.test(navigator.userAgent);

/* eslint-disable @typescript-eslint/no-explicit-any */

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

  // --- OpenSeadragon (гигапиксельный зум через тайлы) ---
  const osdRef = useRef<HTMLDivElement>(null); // контейнер OSD
  const canvasWrapRef = useRef<HTMLDivElement>(null); // обёртка канваса (для wheel-зума)
  const viewerRef = useRef<any>(null); // экземпляр OSD.Viewer
  const osdLibRef = useRef<any>(null); // сам модуль OpenSeadragon (для OSD.Point)
  const overlayGroupRef = useRef<SVGGElement>(null); // <g>, трансформируемая под вьюпорт
  const svgRef = useRef<SVGSVGElement>(null); // fallback-SVG (когда нет картинки)
  const [contentSize, setContentSize] = useState<{ x: number; y: number } | null>(null);
  const [overlayScale, setOverlayScale] = useState(1); // экранных px на 1 px изображения
  const [zoomPct, setZoomPct] = useState(100);
  const [viewBounds, setViewBounds] = useState<Bounds>(FULL_BOUNDS); // видимая область (норм. коорд.)

  // --- editing state ---
  const [drawPhase, setDrawPhase] = useState<Phase>("talc");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState<number[][]>([]);
  // Active vertex drag: which ring / point is moved. `pushed` = один undo на drag.
  const vDrag = useRef<{ ri: number; pi: number; pushed: boolean } | null>(null);

  // Undo/redo history: snapshots of the whole result taken BEFORE each edit.
  const [undoStack, setUndoStack] = useState<AnalysisResult[]>([]);
  const [redoStack, setRedoStack] = useState<AnalysisResult[]>([]);

  const hasImage = !!(sample.tilesUrl || sample.imageUrl);

  // Размеры холста в пикселях изображения: берём из OSD (contentSize), иначе из result.
  const vbW = contentSize?.x ?? result?.imageWidth ?? 100;
  const vbH = contentSize?.y ?? result?.imageHeight ?? 75;

  // Сбросить историю/контент при смене образца или новом анализе.
  useEffect(() => {
    setUndoStack([]);
    setRedoStack([]);
    setContentSize(null);
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

  const toggle = (k: keyof Layers) => setLayers((l) => ({ ...l, [k]: !l[k] }));

  // --- OSD init/teardown (пересоздаём при смене источника) ---
  useEffect(() => {
    if (!hasImage || !osdRef.current) return;
    let destroyed = false;
    let viewer: any;
    (async () => {
      const mod: any = await import("openseadragon");
      const OSD = mod.default || mod;
      if (destroyed || !osdRef.current) return;
      osdLibRef.current = OSD;
      const tileSources = sample.tilesUrl
        ? sample.tilesUrl
        : { type: "image", url: sample.imageUrl! };
      viewer = OSD({
        element: osdRef.current,
        tileSources,
        showNavigationControl: false,
        // Свой wheel-зум (см. эффект ниже) — нативный отключаем, чтобы оверлей
        // не мог «съесть» событие и зум колёсиком работал всегда.
        gestureSettingsMouse: { scrollToZoom: false, clickToZoom: false, dblClickToZoom: false },
        minZoomImageRatio: 0.9,
        maxZoomPixelRatio: 5,
        visibilityRatio: 1,
        constrainDuringPan: true,
        animationTime: 0.3,
        springStiffness: 9,
      });
      viewerRef.current = viewer;

      const measure = () => {
        if (!viewer.world.getItemCount()) return null;
        const vp = viewer.viewport;
        const cs = viewer.world.getItemAt(0).getContentSize();
        const o = vp.imageToViewerElementCoordinates(new OSD.Point(0, 0));
        const x1 = vp.imageToViewerElementCoordinates(new OSD.Point(cs.x, 0));
        return { o, cs, scale: (x1.x - o.x) / cs.x };
      };
      const updateTransform = () => {
        const g = overlayGroupRef.current;
        const m = measure();
        if (g && m) g.setAttribute("transform", `translate(${m.o.x} ${m.o.y}) scale(${m.scale})`);
      };
      const syncScale = () => {
        const m = measure();
        if (!m) return;
        setOverlayScale(m.scale);
        const vp = viewer.viewport;
        setZoomPct(Math.round((vp.getZoom(true) / vp.getHomeZoom()) * 100));
        // Видимая область в нормализованных координатах — для culling ручек-вершин.
        const r = vp.viewportToImageRectangle(vp.getBounds(true));
        setViewBounds({
          x0: r.x / m.cs.x,
          y0: r.y / m.cs.y,
          x1: (r.x + r.width) / m.cs.x,
          y1: (r.y + r.height) / m.cs.y,
        });
      };
      const onOpen = () => {
        const cs = viewer.world.getItemAt(0)?.getContentSize();
        if (cs) setContentSize({ x: cs.x, y: cs.y });
        viewer.world.getItemAt(0)?.setOpacity(layers.image ? 1 : 0);
        updateTransform();
        syncScale();
      };
      viewer.addHandler("open", onOpen);
      viewer.addHandler("update-viewport", updateTransform);
      viewer.addHandler("animation-finish", syncScale);
      viewer.addHandler("resize", () => {
        updateTransform();
        syncScale();
      });
    })();

    return () => {
      destroyed = true;
      if (viewer) viewer.destroy();
      viewerRef.current = null;
      osdLibRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sample.id, sample.tilesUrl, sample.imageUrl, hasImage]);

  // Слой «Исходник»: гасим/показываем базовое изображение OSD.
  useEffect(() => {
    const item = viewerRef.current?.world?.getItemAt?.(0);
    if (item) item.setOpacity(layers.image ? 1 : 0);
  }, [layers.image, contentSize]);

  function zoomBy(factor: number) {
    const v = viewerRef.current;
    if (!v) return;
    v.viewport.zoomBy(factor);
    v.viewport.applyConstraints();
  }
  function resetView() {
    viewerRef.current?.viewport.goHome();
  }

  // Свой зум колёсиком: зумим вьюпорт OSD к точке под курсором (нативный OSD
  // scroll-zoom отключён, чтобы SVG-оверлей не перехватывал событие).
  useEffect(() => {
    const el = canvasWrapRef.current;
    if (!el) return;
    function onWheel(e: WheelEvent) {
      const v = viewerRef.current;
      const OSD = osdLibRef.current;
      if (!v || !OSD) return;
      e.preventDefault();
      const rect = el!.getBoundingClientRect();
      const ref = v.viewport.pointFromPixel(
        new OSD.Point(e.clientX - rect.left, e.clientY - rect.top),
      );
      v.viewport.zoomBy(e.deltaY < 0 ? 1.2 : 1 / 1.2, ref);
      v.viewport.applyConstraints();
    }
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [hasImage]);

  // Экранные координаты -> нормализованные (0..1). В OSD-режиме через вьюпорт,
  // иначе (нет картинки) — через матрицу fallback-SVG.
  function clientToNorm(e: { clientX: number; clientY: number }): number[] | null {
    const OSD = osdLibRef.current;
    const v = viewerRef.current;
    if (v && OSD && osdRef.current) {
      const rect = osdRef.current.getBoundingClientRect();
      const p = v.viewport.viewerElementToImageCoordinates(
        new OSD.Point(e.clientX - rect.left, e.clientY - rect.top),
      );
      return [clamp(p.x / vbW, 0, 1), clamp(p.y / vbH, 0, 1)];
    }
    const svg = svgRef.current;
    if (!svg) return null;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const q = pt.matrixTransform(ctm.inverse());
    return [clamp(q.x / vbW, 0, 1), clamp(q.y / vbH, 0, 1)];
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

  function commitSelectedPolys(polys: number[][][]) {
    if (!result || !selectedId) return;
    const segs = result.segments.map((s) =>
      s.id === selectedId ? { ...s, polygons: polys } : s,
    );
    onResultChange(recomputeResult(result, segs));
  }

  function onVertexDown(e: React.PointerEvent, ri: number, pi: number) {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    viewerRef.current?.setMouseNavEnabled(false); // не панорамируем во время правки точки
    vDrag.current = { ri, pi, pushed: false };
  }
  function onVertexMove(e: React.PointerEvent) {
    if (!vDrag.current || !selectedSeg) return;
    const p = clientToNorm(e);
    if (!p) return;
    if (!vDrag.current.pushed) {
      pushHistory(); // один undo на перетаскивание, при первом движении
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
    viewerRef.current?.setMouseNavEnabled(true);
    vDrag.current = null;
  }
  function deleteVertex(e: React.MouseEvent, ri: number, pi: number) {
    e.preventDefault();
    e.stopPropagation();
    if (!selectedSeg || selectedSeg.polygons[ri].length <= 3) return;
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

  // «Упростить контур»: Douglas–Peucker до ~SIMPLIFY_TARGET точек на кольцо.
  function simplifySelected() {
    if (!selectedSeg) return;
    const before = selectedSeg.polygons.reduce((n, r) => n + r.length, 0);
    const polys = selectedSeg.polygons.map((r) => simplifyRing(r, SIMPLIFY_TARGET));
    const after = polys.reduce((n, r) => n + r.length, 0);
    if (after >= before) return; // упрощать нечего
    pushHistory();
    commitSelectedPolys(polys);
  }

  const selectedPointCount =
    selectedSeg?.polygons.reduce((n, r) => n + r.length, 0) ?? 0;

  // --- undo / redo -----------------------------------------------------------
  function pushHistory() {
    if (!result) return;
    setUndoStack((s) => [...s, result].slice(-UNDO_LIMIT));
    setRedoStack([]);
  }

  const canUndo = (adding && draft.length > 0) || undoStack.length > 0;
  const canRedo = redoStack.length > 0;

  function reconcileSelection(next: AnalysisResult) {
    if (selectedId && !next.segments.some((seg) => seg.id === selectedId)) {
      setSelectedId(null);
    }
  }
  function undo() {
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

  // Размеры маркеров/подписей в пикселях ИЗОБРАЖЕНИЯ, дающие ~постоянный размер
  // на экране: px = (пикселей изображения на 1 экранный px).
  const px = hasImage ? 1 / overlayScale : Math.max(vbW, vbH) / 320;
  const R = 8 * px;
  const SW = 1.6 * px;
  const SELSW = 2 * px;
  const DOTR = 6 * px;
  const DRAFTSW = 2.2 * px;
  const labelFS = 15 * px;

  // Общий контент оверлея (одинаков в OSD- и fallback-режиме; координаты — в px изображения).
  const overlay = (
    <>
      {/* Phase mask overlay */}
      {layers.mask && result && (
        <g opacity={maskOpacity} style={{ pointerEvents: editMode && !adding ? "auto" : "none" }}>
          {visibleSegments.map((s) =>
            s.polygons.map((ring, ri) => (
              <path
                key={`${s.id}-${ri}`}
                d={ringToPath(ring, vbW, vbH)}
                fill={PHASE_META[s.phase].color}
                stroke={selectedId === s.id ? "#ffffff" : "none"}
                strokeWidth={selectedId === s.id ? SELSW : 0}
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

      {/* Карта уверенности: цвета — как в классификации, число по центру = уверенность. */}
      {layers.heatmap && result && (
        <g opacity={maskOpacity} style={{ pointerEvents: "none" }}>
          {visibleSegments.map((s) => {
            const big = s.polygons.reduce(
              (a, b) => (b.length > a.length ? b : a),
              s.polygons[0] ?? [],
            );
            return (
              <g key={`h-${s.id}`}>
                {s.polygons.map((ring, ri) => (
                  <path key={ri} d={ringToPath(ring, vbW, vbH)} fill={PHASE_META[s.phase].color} />
                ))}
                {big.length > 0 &&
                  (() => {
                    const [cx, cy] = ringCentroid(big);
                    return (
                      <text
                        x={cx * vbW}
                        y={cy * vbH}
                        fontSize={labelFS}
                        fill="#ffffff"
                        stroke="#0a0a12"
                        strokeWidth={labelFS / 7}
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
            strokeWidth={DRAFTSW}
          />
          {draft.map(([x, y], i) => (
            <circle
              key={i}
              cx={x * vbW}
              cy={y * vbH}
              r={DOTR}
              fill="#ffffff"
              stroke={PHASE_META[drawPhase].color}
              strokeWidth={SW}
            />
          ))}
        </g>
      )}

      {/* Vertex handles: reshape the selected segment (drag / insert / delete).
          Only vertices in the current viewport are drawn, capped at MAX_HANDLES
          (decimated if denser) — так тысячеточечные ML-контуры не тормозят. */}
      {editMode && !adding && selectedSeg && (
        <g style={{ pointerEvents: "auto" }} onMouseDown={(e) => e.stopPropagation()}>
          {selectedSeg.polygons.map((ring, ri) => {
            const n = ring.length;
            const h = visibleHandles(ring, viewBounds, MAX_HANDLES);
            // Не терять перетаскиваемую вершину, если она ушла к краю кадра.
            if (vDrag.current && vDrag.current.ri === ri && !h.vtx.includes(vDrag.current.pi)) {
              h.vtx.push(vDrag.current.pi);
            }
            return (
              <g key={`edit-${ri}`}>
                {h.mids.map((pi) => {
                  const a = ring[pi];
                  const b = ring[(pi + 1) % n];
                  return (
                    <circle
                      key={`mid-${pi}`}
                      cx={((a[0] + b[0]) / 2) * vbW}
                      cy={((a[1] + b[1]) / 2) * vbH}
                      r={R * 0.62}
                      fill="#ffffff"
                      fillOpacity={0.45}
                      stroke="#2E2E48"
                      strokeWidth={SW}
                      style={{ cursor: "copy" }}
                      onClick={(e) => insertVertex(e, ri, pi)}
                    />
                  );
                })}
                {h.vtx.map((pi) => {
                  const pt = ring[pi];
                  return (
                    <circle
                      key={`vtx-${pi}`}
                      cx={pt[0] * vbW}
                      cy={pt[1] * vbH}
                      r={R}
                      fill={PHASE_META[selectedSeg.phase].color}
                      stroke="#ffffff"
                      strokeWidth={SW}
                      style={{ cursor: "grab", touchAction: "none" }}
                      onPointerDown={(e) => onVertexDown(e, ri, pi)}
                      onPointerMove={onVertexMove}
                      onPointerUp={onVertexUp}
                      onContextMenu={(e) => deleteVertex(e, ri, pi)}
                    />
                  );
                })}
              </g>
            );
          })}
        </g>
      )}
    </>
  );

  return (
    <div className="card flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <LayerChip active={layers.image} onClick={() => toggle("image")} label="Исходник" disabled={!hasImage} />
          <LayerChip active={layers.mask} onClick={() => toggle("mask")} label="Маска фаз" disabled={!result} />
          <LayerChip active={layers.heatmap} onClick={() => toggle("heatmap")} label="Карта уверенности" disabled={!result} />
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
              style={{ "--pct": `${((maskOpacity - 0.15) / (0.9 - 0.15)) * 100}%` } as React.CSSProperties}
            />
          </label>
          <div className="flex items-center gap-1">
            <IconBtn onClick={() => zoomBy(1 / 1.4)}>−</IconBtn>
            <span className="w-12 text-center text-xs font-semibold text-ink">{zoomPct}%</span>
            <IconBtn onClick={() => zoomBy(1.4)}>+</IconBtn>
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
            <button
              onClick={simplifySelected}
              disabled={!selectedId || selectedPointCount <= SIMPLIFY_TARGET}
              title="Упростить контур (Douglas–Peucker)"
              className="btn-soft !py-1 text-xs disabled:opacity-40"
            >
              Упростить
              {selectedId && selectedPointCount > SIMPLIFY_TARGET ? ` (${selectedPointCount})` : ""}
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
      <div ref={canvasWrapRef} className="relative aspect-[4/3] w-full overflow-hidden bg-[#0c0c14]">
        {hasImage ? (
          <>
            <div ref={osdRef} className="absolute inset-0" />
            {/* SVG-оверлей поверх OSD: <g> трансформируется под вьюпорт (imperative).
                pointer-events: none, чтобы OSD получал зум/пан; интерактивные дети
                включают события сами. В режиме рисования ловим клики всем оверлеем. */}
            <svg
              className="absolute inset-0 h-full w-full"
              style={{ pointerEvents: adding ? "auto" : "none", cursor: adding ? "crosshair" : "default" }}
              onClick={onSvgClick}
            >
              <g ref={overlayGroupRef}>{overlay}</g>
            </svg>
          </>
        ) : (
          <svg
            ref={svgRef}
            viewBox={`0 0 ${vbW} ${vbH}`}
            className="absolute inset-0 h-full w-full"
            preserveAspectRatio="xMidYMid slice"
            onClick={onSvgClick}
            style={{ cursor: adding ? "crosshair" : "default" }}
          >
            <defs>
              <radialGradient id="matrix" cx="40%" cy="35%">
                <stop offset="0%" stopColor="#20202e" />
                <stop offset="100%" stopColor="#0a0a12" />
              </radialGradient>
            </defs>
            <rect width={vbW} height={vbH} fill="url(#matrix)" />
            {overlay}
          </svg>
        )}

        {/* Overlay badges */}
        <div className="pointer-events-none absolute left-4 top-4 rounded-pill bg-black/45 px-3 py-1 text-xs font-medium text-white backdrop-blur">
          {sample.meta}
        </div>

        {layers.heatmap && result && (
          <div className="pointer-events-none absolute bottom-4 right-4 rounded-pill bg-black/55 px-3 py-1 text-[11px] font-medium text-white backdrop-blur">
            % — уверенность модели
          </div>
        )}

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
                      <span className={done || running ? "" : "text-white/50"}>{label}</span>
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

// Индексы вершин, для которых рисуем ручки: только попавшие во вьюпорт `b`,
// и не более `cap` (иначе берём каждую k-ю). Midpoint-ы (для вставки) — только
// между СОСЕДНИМИ показанными вершинами и только без прореживания.
function visibleHandles(
  ring: number[][],
  b: Bounds,
  cap: number,
): { vtx: number[]; mids: number[] } {
  const n = ring.length;
  const m = 0.03; // небольшой запас за краем кадра
  let idx: number[] = [];
  for (let i = 0; i < n; i++) {
    const p = ring[i];
    if (p[0] >= b.x0 - m && p[0] <= b.x1 + m && p[1] >= b.y0 - m && p[1] <= b.y1 + m) {
      idx.push(i);
    }
  }
  let step = 1;
  if (idx.length > cap) {
    step = Math.ceil(idx.length / cap);
    idx = idx.filter((_, k) => k % step === 0);
  }
  const set = new Set(idx);
  const mids = step === 1 ? idx.filter((i) => set.has((i + 1) % n)) : [];
  return { vtx: idx, mids };
}

// Перпендикулярное расстояние точки p до отрезка a—b (нормализованные координаты).
function perpDist(p: number[], a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  const t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
  const tc = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (a[0] + tc * dx), p[1] - (a[1] + tc * dy));
}

// Douglas–Peucker для открытой ломаной.
function rdp(points: number[][], eps: number): number[][] {
  if (points.length < 3) return points;
  let dmax = 0;
  let idx = 0;
  const a = points[0];
  const b = points[points.length - 1];
  for (let i = 1; i < points.length - 1; i++) {
    const d = perpDist(points[i], a, b);
    if (d > dmax) {
      dmax = d;
      idx = i;
    }
  }
  if (dmax > eps) {
    const left = rdp(points.slice(0, idx + 1), eps);
    const right = rdp(points.slice(idx), eps);
    return left.slice(0, -1).concat(right);
  }
  return [a, b];
}

// Упростить кольцо до ~targetMax точек, наращивая допуск, пока не уложимся.
function simplifyRing(ring: number[][], targetMax: number): number[][] {
  if (ring.length <= targetMax) return ring;
  let eps = 0.0008;
  let out = rdp(ring, eps);
  while (out.length > targetMax && eps < 0.2) {
    eps *= 1.6;
    out = rdp(ring, eps);
  }
  return out.length >= 3 ? out : ring;
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

function IconBtn({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex h-7 w-7 items-center justify-center rounded-full border border-line text-lg font-bold text-ink hover:border-ink/30"
    >
      {children}
    </button>
  );
}

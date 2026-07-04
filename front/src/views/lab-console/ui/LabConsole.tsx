"use client";

import { useState } from "react";
import type { Phase } from "@/entities/phase";
import { SAMPLES, type Sample } from "@/entities/sample";
import type { AnalysisResult } from "@/entities/analysis";
import { analyzeSample } from "@/features/analyze-sample";
import { UploadZone } from "@/features/upload-sample";
import { Navbar } from "@/widgets/navbar";
import { SampleQueue } from "@/widgets/sample-queue";
import { SlideViewer } from "@/widgets/slide-viewer";
import { MetricsPanel } from "@/widgets/metrics-panel";
import { ExpertBar } from "@/widgets/expert-bar";
import { Spinner } from "@/shared/ui/spinner";

export function LabConsole() {
  const [samples, setSamples] = useState<Sample[]>(SAMPLES);
  const [active, setActive] = useState<Sample>(SAMPLES[0]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<"idle" | "processing" | "done">("idle");
  const [step, setStep] = useState(0);
  const [selectedPhase, setSelectedPhase] = useState<"all" | Phase>("all");
  const [editMode, setEditMode] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function selectSample(s: Sample) {
    setActive(s);
    setResult(null);
    setStatus("idle");
    setStep(0);
    setSelectedPhase("all");
    setEditMode(false);
    setError(null);
  }

  function onUpload(newOnes: Sample[]) {
    if (newOnes.length === 0) return;
    setSamples((prev) => [...newOnes, ...prev]);
    selectSample(newOnes[0]);
  }

  async function runAnalysis() {
    setStatus("processing");
    setResult(null);
    setEditMode(false);
    setError(null);
    setStep(0);
    // Пошаговый индикатор предобработки, пока идёт инференс.
    const perStep = 420;
    const timers = [1, 2, 3, 4].map((i) =>
      setTimeout(() => setStep(i), perStep * i),
    );
    try {
      const res = await analyzeSample(active);
      setResult(res);
      setStatus("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Не удалось выполнить анализ.");
      setStatus("idle");
    } finally {
      timers.forEach(clearTimeout);
    }
  }

  return (
    <main className="min-h-screen px-4 pb-16 pt-4">
      <Navbar />

      <div className="mx-auto max-w-[1400px]">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="heading text-4xl leading-none md:text-5xl">
              Lab Console
            </h1>
            <p className="mt-2 max-w-xl text-sm text-ink-soft">
              Автоматическая классификация руд по панорамным микрофотографиям
              полированных шлифов — сегментация фаз, метрики и интерпретируемый
              вывод.
            </p>
          </div>
          <button
            onClick={runAnalysis}
            disabled={status === "processing"}
            className="btn-primary"
          >
            {status === "processing" ? (
              <>
                <Spinner /> Анализ…
              </>
            ) : (
              <>Анализировать образец</>
            )}
          </button>
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[300px_1fr_360px] lg:grid-rows-[auto_auto]">
          {/* Left: upload + queue — только ряд 1, низ вровень с вьюером */}
          <aside className="flex flex-col gap-5 lg:col-start-1 lg:row-start-1">
            <UploadZone onUpload={onUpload} />
            <SampleQueue
              samples={samples}
              activeId={active.id}
              onSelect={selectSample}
            />
          </aside>

          {/* Center: viewer (ряд 1) */}
          <div className="lg:col-start-2 lg:row-start-1">
            <SlideViewer
              sample={active}
              result={result}
              selectedPhase={selectedPhase}
              status={status}
              step={step}
              editMode={editMode}
              onResultChange={setResult}
            />
            {error && (
              <p className="mt-2 rounded-xl bg-phase-thin/10 px-3 py-2 text-xs text-phase-thin">
                {error}
              </p>
            )}
          </div>

          {/* Center: expert bar (ряд 2) */}
          <div className="lg:col-start-2 lg:row-start-2">
            <ExpertBar
              disabled={!result}
              editMode={editMode}
              onToggleEdit={() => setEditMode((v) => !v)}
            />
          </div>

          {/* Right: metrics — на всю высоту (оба ряда) */}
          <div className="lg:col-start-3 lg:row-span-2 lg:row-start-1">
            <MetricsPanel
              sample={active}
              result={result}
              selectedPhase={selectedPhase}
              onSelectPhase={setSelectedPhase}
            />
          </div>
        </div>
      </div>
    </main>
  );
}

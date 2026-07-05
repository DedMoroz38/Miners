"use client";

import { useState } from "react";
import type { Phase } from "@/entities/phase";
import { SAMPLES, type Sample } from "@/entities/sample";
import type { AnalysisResult } from "@/entities/analysis";
import { analyzeSample } from "@/features/analyze-sample";
import { UploadZone } from "@/features/upload-sample";
import { SampleQueue } from "@/widgets/sample-queue";
import { SlideViewer } from "@/widgets/slide-viewer";
import { MetricsPanel } from "@/widgets/metrics-panel";

export function LabConsole() {
  const [samples, setSamples] = useState<Sample[]>(SAMPLES);
  const [active, setActive] = useState<Sample | null>(SAMPLES[0] ?? null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<"idle" | "processing" | "done">("idle");
  const [step, setStep] = useState(0);
  const [selectedPhase, setSelectedPhase] = useState<"all" | Phase>("all");
  const [editMode, setEditMode] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Боковые панели в стиле VS Code: по дефолту открыты, обе можно свернуть.
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

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
    if (!active) return;
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
    <main className="flex h-screen w-full gap-3 overflow-hidden p-3">
      {/* Left panel: upload + queue */}
      {leftOpen && (
        <aside className="flex w-[300px] flex-none flex-col gap-3 overflow-y-auto">
          <UploadZone onUpload={onUpload} />
          <SampleQueue
            samples={samples}
            activeId={active?.id ?? ""}
            onSelect={selectSample}
          />
        </aside>
      )}

      {/* Center: viewer на весь экран */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2">
        {active ? (
          <SlideViewer
            sample={active}
            result={result}
            selectedPhase={selectedPhase}
            status={status}
            step={step}
            editMode={editMode}
            leftOpen={leftOpen}
            onToggleLeft={() => setLeftOpen((v) => !v)}
            rightOpen={rightOpen}
            onToggleRight={() => setRightOpen((v) => !v)}
            onToggleEdit={() => setEditMode((v) => !v)}
            onAnalyze={runAnalysis}
            onResultChange={setResult}
          />
        ) : (
          <div className="card flex flex-1 items-center justify-center p-8 text-center">
            <p className="text-sm text-ink-soft">
              Загрузите образец, чтобы начать анализ.
            </p>
          </div>
        )}
        {error && (
          <p className="flex-none rounded-xl bg-phase-thin/10 px-3 py-2 text-xs text-phase-thin">
            {error}
          </p>
        )}
      </div>

      {/* Right panel: classification / metrics */}
      {rightOpen && (
        <aside className="w-[340px] flex-none overflow-y-auto">
          {active ? (
            <MetricsPanel
              sample={active}
              result={result}
              selectedPhase={selectedPhase}
              onSelectPhase={setSelectedPhase}
            />
          ) : (
            <div className="card p-8 text-center text-sm text-ink-soft">
              Нет выбранного образца.
            </div>
          )}
        </aside>
      )}
    </main>
  );
}

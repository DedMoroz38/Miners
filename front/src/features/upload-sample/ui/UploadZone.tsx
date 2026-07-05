"use client";

import { useRef, useState } from "react";
import {
  ACCEPTED_EXT,
  validateFile,
  type FileError,
  type Sample,
} from "@/entities/sample";
import { uploadAndBuild } from "../lib/upload";

export function UploadZone({ onUpload }: { onUpload: (s: Sample[]) => void }) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handle(fileList: FileList | null) {
    if (!fileList || fileList.length === 0) return;
    const valid: File[] = [];
    const rejected: string[] = [];
    for (const f of Array.from(fileList)) {
      const err: FileError | null = validateFile(f);
      if (err) rejected.push(f.name);
      else valid.push(f);
    }

    setBusy(true);
    const built = await Promise.all(valid.map((f) => uploadAndBuild(f)));
    setBusy(false);

    const accepted = built.map((b) => b.sample);
    if (accepted.length) onUpload(accepted);

    const offline = built.some((b) => b.offline);
    setError(
      rejected.length
        ? `Пропущено (неподдерживаемый формат): ${rejected.join(", ")}`
        : offline
          ? "Бэкенд недоступен — образец добавлен локально (демо-анализ)."
          : null,
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handle(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      className={`card cursor-pointer border-2 border-dashed p-6 text-center transition ${
        dragging
          ? "border-brand bg-brand/10"
          : "border-brand/50 bg-brand/[0.04] hover:border-brand"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXT.join(",")}
        multiple
        className="hidden"
        onChange={(e) => {
          handle(e.target.files);
          e.target.value = "";
        }}
      />
      <p className="text-sm font-semibold text-ink">
        {busy
          ? "Загрузка…"
          : dragging
            ? "Отпустите для загрузки"
            : "Перетащите панорамный шлиф"}
      </p>
      <p className="mt-1 text-xs text-ink-faint">
        TIFF · PNG · JPEG · до 10000×10000 px
      </p>
      <span className="btn-soft pointer-events-none mt-3">Выбрать файл</span>
      {error && (
        <p className="mt-3 rounded-xl bg-phase-thin/10 px-3 py-2 text-xs text-phase-thin">
          {error}
        </p>
      )}
    </div>
  );
}

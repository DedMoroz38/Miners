import { sampleFromFile, type Sample } from "@/entities/sample";
import { uploadSample } from "@/shared/api";

// Загружает файл на бэкенд и строит Sample с serverId. Локальный blob-URL
// оставляем для мгновенного показа. Если бэкенд недоступен — образец остаётся
// локальным (serverId нет -> сработает мок-анализ), чтобы демо не падало.
export async function uploadAndBuild(
  file: File,
): Promise<{ sample: Sample; offline: boolean }> {
  const local = sampleFromFile(file);
  try {
    const res = await uploadSample(file);
    return {
      sample: {
        ...local,
        serverId: res.id,
        meta: res.meta || local.meta,
        tilesUrl: res.tiles_url ?? undefined,
      },
      offline: false,
    };
  } catch {
    return { sample: local, offline: true };
  }
}

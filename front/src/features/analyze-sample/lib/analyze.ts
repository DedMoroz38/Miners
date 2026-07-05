import type { Sample } from "@/entities/sample";
import type { AnalysisResult } from "@/entities/analysis";
import { getJob, startAnalysis } from "@/shared/api";
import { mockAnalyze } from "./mock";

const POLL_MS = 1200;
const POLL_TIMEOUT_MS = 20 * 60 * 1000; // 20 минут на тяжёлый инференс

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Реальный анализ через бэкенд: запуск задания + опрос до готовности.
async function analyzeViaApi(serverId: string): Promise<AnalysisResult> {
  const { job_id } = await startAnalysis(serverId);
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  for (;;) {
    const job = await getJob<AnalysisResult>(serverId, job_id);
    if (job.status === "done" && job.result) return job.result;
    if (job.status === "error") {
      throw new Error(job.error ?? "Анализ завершился с ошибкой.");
    }
    if (Date.now() > deadline) throw new Error("Превышено время ожидания анализа.");
    await sleep(POLL_MS);
  }
}

// Единая точка входа: реальный образец (serverId) -> бэкенд, иначе -> мок.
export async function analyzeSample(sample: Sample): Promise<AnalysisResult> {
  if (sample.serverId) return analyzeViaApi(sample.serverId);
  await sleep(600); // имитация работы для демо-образцов
  return mockAnalyze(sample);
}

export { mockAnalyze };

// Клиент бэкенда лаборатории. База — из NEXT_PUBLIC_API_URL (по умолчанию localhost:8000).
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export type SampleUploadResponse = {
  id: string;
  name: string;
  meta: string;
  size_bytes: number;
  content_type: string;
  width?: number | null;
  height?: number | null;
  image_url: string;
  tiles_url?: string | null;
};

// Бэкенд отдаёт результат в форме, совпадающей с AnalysisResult (entities/analysis).
export type JobStatusResponse<R = unknown> = {
  job_id: string;
  status: "pending" | "running" | "done" | "error";
  result: R | null;
  error: string | null;
};

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json())?.detail ?? detail;
    } catch {
      /* тело не JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadSample(file: File): Promise<SampleUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/samples`, { method: "POST", body: form });
  return asJson<SampleUploadResponse>(res);
}

export async function startAnalysis(sampleId: string): Promise<{ job_id: string }> {
  const res = await fetch(`${API_BASE}/api/samples/${sampleId}/analyze`, {
    method: "POST",
  });
  return asJson<{ job_id: string; status: string }>(res);
}

export async function getJob<R = unknown>(
  sampleId: string,
  jobId: string,
): Promise<JobStatusResponse<R>> {
  const res = await fetch(`${API_BASE}/api/samples/${sampleId}/analyze/${jobId}`);
  return asJson<JobStatusResponse<R>>(res);
}

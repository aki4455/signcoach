// Small typed client for the SignCoach API (CLAUDE.md §12). Errors carry an i18n error key.
import type { JobStatus, LessonSummary, Report } from "./types";

export class ApiError extends Error {
  constructor(public key: string, public params: Record<string, unknown> = {}) {
    super(key);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new ApiError("error.upload_incomplete");
  }
  if (!response.ok) {
    let key = "error.processing_failed";
    let params = {};
    try {
      const body = await response.json();
      key = body.error_key ?? key;
      params = body.params ?? {};
    } catch {
      /* not JSON */
    }
    throw new ApiError(key, params);
  }
  return response.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export interface UploadForm {
  student: string; lesson_label: string; consent: boolean;
  camera_height_m?: number | null; hfov_deg?: number | null;
}

/** Upload a file in chunks (retrying each chunk), then queue the job. Returns the job id. */
export async function uploadVideo(file: File, form: UploadForm, onProgress: (fraction: number) => void) {
  const { upload_id, chunk_bytes } = await request<{ upload_id: string; chunk_bytes: number }>(
    "/api/uploads/init", json({ filename: file.name, size_bytes: file.size }));
  const n = Math.ceil(file.size / chunk_bytes);
  for (let i = 0; i < n; i++) {
    const body = file.slice(i * chunk_bytes, Math.min(file.size, (i + 1) * chunk_bytes));
    for (let attempt = 1; ; attempt++) {
      try {
        await request(`/api/uploads/${upload_id}/chunk?index=${i}`, { method: "PUT", body });
        break;
      } catch (e) {
        if (attempt >= 4) throw e;
        await new Promise((r) => setTimeout(r, 1000 * attempt)); // connection hiccup: wait, then resend
      }
    }
    onProgress((i + 1) / n);
  }
  const { job_id } = await request<{ job_id: string }>(`/api/uploads/${upload_id}/complete`, json(form));
  return job_id;
}

export const getJob = (id: string) => request<JobStatus>(`/api/jobs/${id}`);
export const getReport = (id: string) => request<Report>(`/api/lessons/${id}/report`);
export const listLessons = (ids: string[]) =>
  request<LessonSummary[]>(`/api/lessons?ids=${encodeURIComponent(ids.join(","))}`);

// The lessons this browser uploaded (ids are unguessable links; nothing is listed publicly).
const KEY = "signcoach.myLessons";
export function myLessons(): string[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]");
  } catch {
    return [];
  }
}
export function rememberLesson(id: string) {
  try {
    localStorage.setItem(KEY, JSON.stringify([id, ...myLessons().filter((x) => x !== id)].slice(0, 50)));
  } catch {
    /* private mode: the lesson link still works */
  }
}

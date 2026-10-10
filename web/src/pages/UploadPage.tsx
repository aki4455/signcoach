import { useEffect, useState } from "react";
import { ApiError, getJob, rememberLesson, uploadVideo } from "../api";
import { t } from "../i18n";
import { navigate } from "../router";
import type { JobStatus } from "../types";

/** Upload a dashcam video (chunked), then follow the processing job until the report is ready. */
export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [student, setStudent] = useState("");
  const [label, setLabel] = useState("");
  const [consent, setConsent] = useState(false);
  const [uploaded, setUploaded] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      try {
        const status = await getJob(jobId);
        setJob(status);
        if (status.status === "done" && status.lesson_id) {
          clearInterval(timer);
          rememberLesson(status.lesson_id);
          navigate(`/lessons/${status.lesson_id}`);
        } else if (status.status === "failed") {
          clearInterval(timer);
          setError(t(status.error_key ?? "error.processing_failed", { max_minutes: 3 }));
        }
      } catch {
        /* keep polling through short network hiccups */
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [jobId]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setUploaded(0);
    try {
      setJobId(await uploadVideo(file, { student, lesson_label: label, consent }, setUploaded));
    } catch (err) {
      setUploaded(null);
      setError(err instanceof ApiError ? t(err.key, { max_minutes: 3, ...err.params }) : t("error.processing_failed"));
    }
  }

  const busy = uploaded !== null && !error;
  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold">Upload a lesson video</h1>
        <p className="mt-1 text-slate-600">
          Upload the dashcam video of a driving lesson. You get a report of every road sign, the rules in force and
          where they were broken. Faces and number plates are blurred; sound is removed.
        </p>
      </section>
      <form onSubmit={submit} className="space-y-4 rounded-xl border border-slate-200 bg-white p-4">
        <label className="block">
          <span className="text-sm font-medium">Dashcam video (up to 3 minutes)</span>
          <input type="file" accept="video/*,.mov,.mkv,.avi" disabled={busy} required
                 onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                 className="mt-1 block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-slate-900 file:px-3 file:py-2 file:text-white" />
        </label>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block">
            <span className="text-sm font-medium">Student name</span>
            <input value={student} onChange={(e) => setStudent(e.target.value)} required maxLength={80} disabled={busy}
                   className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2" />
          </label>
          <label className="block">
            <span className="text-sm font-medium">Lesson label</span>
            <input value={label} onChange={(e) => setLabel(e.target.value)} maxLength={120} disabled={busy}
                   placeholder="e.g. City route, lesson 3" className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2" />
          </label>
        </div>
        <label className="flex items-start gap-2 text-sm">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} disabled={busy}
                 className="mt-1" />
          <span>I may upload this video. Faces and number plates will be blurred.</span>
        </label>
        <button type="submit" disabled={!file || !consent || !student || busy}
                className="rounded-md bg-slate-900 px-4 py-2 font-medium text-white disabled:opacity-40">
          Upload and analyse
        </button>
      </form>

      {uploaded !== null && !jobId && <Progress label={`Uploading… ${Math.round(uploaded * 100)}%`} value={uploaded} />}
      {job && job.status !== "failed" && (
        <Progress
          label={`${t(`stage.${job.stage}`)}${job.status === "queued" && job.queue_position > 0
            ? ` — ${job.queue_position} video(s) ahead` : ""}`}
          value={job.progress} />
      )}
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-800">{error}</p>}
    </div>
  );
}

function Progress({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4">
      <p className="text-sm font-medium">{label}</p>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
        <div className="h-full bg-slate-900 transition-all" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

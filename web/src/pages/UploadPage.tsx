import { useEffect, useState } from "react";
import { ApiError, getJob, listLessons, rememberLesson, uploadVideo } from "../api";
import { t } from "../i18n";
import { linkProps, navigate } from "../router";
import type { JobStatus, LessonSummary } from "../types";

/** Upload a dashcam video (chunked), then follow the processing job until the report is ready. */
export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [student, setStudent] = useState("");
  const [label, setLabel] = useState("");
  const [height, setHeight] = useState("");
  const [fov, setFov] = useState("");
  const [consent, setConsent] = useState(false);
  const [uploaded, setUploaded] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<LessonSummary[]>([]);

  useEffect(() => {
    listLessons([]).then((all) => setSamples(all.filter((l) => l.is_sample))).catch(() => undefined);
  }, []);

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
    setJob(null);
    setUploaded(0);
    try {
      setJobId(await uploadVideo(file, {
        student, lesson_label: label, consent,
        camera_height_m: height ? Number(height) : null, hfov_deg: fov ? Number(fov) : null,
      }, setUploaded));
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
          Upload the dashcam video of a driving lesson. SignCoach finds every road sign, explains it, keeps track of
          the rules in force and points out where they were broken — with the video moments as evidence.{" "}
          <a {...linkProps("/lessons")} className="underline">See lessons</a>.
        </p>
      </section>

      <form onSubmit={submit} className="space-y-4 rounded-xl border border-slate-200 bg-white p-4">
        <label onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
               onDrop={(e) => { e.preventDefault(); setDragging(false); if (!busy && e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); }}
               className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center ${dragging ? "border-slate-900 bg-slate-50" : "border-slate-300"}`}>
          <input type="file" accept="video/*,.mov,.mkv,.avi" disabled={busy} className="sr-only"
                 onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <span className="font-medium">{file ? file.name : "Drop the dashcam video here or tap to choose"}</span>
          <span className="mt-1 text-sm text-slate-500">
            {file ? `${(file.size / 1e6).toFixed(1)} MB` : "MP4, MOV, MKV or AVI, up to 3 minutes"}
          </span>
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

        <details className="text-sm">
          <summary className="cursor-pointer text-slate-600">Camera details (optional, improves the speed estimate)</summary>
          <div className="mt-2 grid gap-4 sm:grid-cols-2">
            <label className="block">
              <span className="font-medium">Camera height above the road (m)</span>
              <input type="number" step="0.05" min="0.5" max="3" value={height} onChange={(e) => setHeight(e.target.value)}
                     placeholder="1.3" disabled={busy} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2" />
            </label>
            <label className="block">
              <span className="font-medium">Horizontal field of view (°)</span>
              <input type="number" step="1" min="30" max="170" value={fov} onChange={(e) => setFov(e.target.value)}
                     placeholder="70" disabled={busy} className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2" />
            </label>
          </div>
        </details>

        <label className="flex items-start gap-2 text-sm">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} disabled={busy} className="mt-1" />
          <span>I may upload this video. Faces and number plates will be blurred and the sound removed.</span>
        </label>
        <button type="submit" disabled={!file || !consent || !student || busy}
                className="w-full rounded-md bg-slate-900 px-4 py-3 font-medium text-white disabled:opacity-40 sm:w-auto">
          Upload and analyse
        </button>
      </form>

      {uploaded !== null && !jobId && <Progress label={`Uploading… ${Math.round(uploaded * 100)}%`} value={uploaded} />}
      {job && job.status !== "failed" && (
        <Progress value={job.progress}
                  label={`${t(`stage.${job.stage}`)}${job.status === "queued" && job.queue_position > 0
                    ? ` — ${job.queue_position} video(s) ahead` : ""}`} />
      )}
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-800">{error}</p>}

      {samples.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Or open a sample lesson</h2>
          <div className="grid gap-3 sm:grid-cols-3">
            {samples.map((l) => (
              <a key={l.id} {...linkProps(`/lessons/${l.id}`)}
                 className="overflow-hidden rounded-xl border border-slate-200 bg-white hover:border-slate-400">
                <img src={`/media/${l.id}/poster.jpg`} alt="" loading="lazy" className="aspect-video w-full bg-slate-200 object-cover" />
                <span className="block p-2 text-sm font-medium">{l.label}</span>
              </a>
            ))}
          </div>
        </section>
      )}
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

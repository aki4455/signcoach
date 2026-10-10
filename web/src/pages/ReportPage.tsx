import { useEffect, useRef, useState } from "react";
import { ApiError, getReport } from "../api";
import { signName, t } from "../i18n";
import type { LessonEvent, Report } from "../types";

const SEVERITY_STYLE: Record<string, string> = {
  info: "bg-slate-100 text-slate-700", violation: "bg-red-100 text-red-800",
  possible: "bg-amber-100 text-amber-800", complied: "bg-green-100 text-green-800",
};

function eventText(e: LessonEvent): string {
  return t(e.title_key, { ...e.detail, sign_name: signName(e.sign_code, e.value) });
}

function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Lesson report (M1: bare version — M4 adds the timeline, voice prompts and the full layout). */
export default function ReportPage({ id }: { id: string }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    getReport(id).then(setReport).catch((e) => setError(t(e instanceof ApiError ? e.key : "error.not_found")));
  }, [id]);

  if (error) return <p className="rounded-lg bg-red-50 p-3 text-red-800">{error}</p>;
  if (!report) return <p>Loading…</p>;

  const seek = (time: number) => {
    if (!video.current) return;
    video.current.currentTime = time;
    void video.current.play().catch(() => undefined);
  };
  const { summary } = report;
  return (
    <div className="space-y-5">
      <section>
        <h1 className="text-2xl font-semibold">{report.lesson.label || "Lesson report"}</h1>
        <p className="text-slate-600">{report.lesson.student} · {clock(report.lesson.duration_s)}</p>
      </section>

      <video ref={video} src={report.media.annotated_video_url} poster={report.media.poster_url} controls playsInline
             className="max-h-[70vh] w-full rounded-xl bg-black" />

      {report.quality_warnings.length > 0 && (
        <ul className="space-y-1 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">
          {report.quality_warnings.map((w) => <li key={w}>{t(w)}</li>)}
        </ul>
      )}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Card label="Score (indicative)" value={summary.score} />
        <Card label="Signs passed" value={summary.signs_passed} />
        <Card label="Violations" value={summary.violations} />
        <Card label="Possible" value={summary.possible} />
      </section>

      <section>
        <h2 className="mb-2 text-lg font-semibold">Events</h2>
        {report.events.length === 0 && <p className="text-slate-600">No events in this lesson.</p>}
        <ul className="space-y-2">
          {report.events.map((e) => (
            <li key={e.id}>
              <button onClick={() => seek(e.clip[0])}
                      className="flex w-full items-center gap-3 rounded-xl border border-slate-200 bg-white p-2 text-left hover:border-slate-400">
                {e.thumbnail ? <img src={e.thumbnail} alt="" className="h-14 w-14 rounded object-cover" />
                  : <span className="h-14 w-14 rounded bg-slate-100" />}
                <span className="flex-1">
                  <span className="block text-sm text-slate-500">{clock(e.t)}</span>
                  <span className="block font-medium">{eventText(e)}</span>
                </span>
                <span className={`rounded-full px-2 py-0.5 text-xs ${SEVERITY_STYLE[e.severity]}`}>{t(`severity.${e.severity}`)}</span>
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="mb-2 text-lg font-semibold">Every sign detected</h2>
        <ul className="space-y-1 text-sm">
          {report.signs.map((s) => (
            <li key={s.id}>
              <button onClick={() => seek(Math.max(0, s.t_first - 1))} className="text-left hover:underline">
                {clock(s.t_pass)} · <b>{signName(s.sign_code, s.value)}</b> ({s.relevant ? "applies to us" : "not for us"}: {s.relevance_reason})
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-600">
        <h2 className="mb-1 font-semibold text-slate-800">How this was computed</h2>
        <p>Detector: {String(report.processing.model)} · device: {String(report.processing.device)} · processing time: {String(report.processing.total_seconds)} s for {String(report.processing.video_seconds)} s of video · speed source: {report.speed_source}</p>
      </section>
    </div>
  );
}

function Card({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-2xl font-semibold">{value}</p>
    </div>
  );
}

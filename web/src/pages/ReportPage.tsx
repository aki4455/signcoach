import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getReport } from "../api";
import SpeedChart from "../components/SpeedChart";
import Timeline from "../components/Timeline";
import { SEVERITY_PILL, clock, eventDetail, eventTitle } from "../format";
import { signEntry, signName, t } from "../i18n";
import type { LessonEvent, Report } from "../types";
import { VoicePrompts } from "../voice";

const VOICE_PREF = "signcoach.voice";

/** The lesson report (CLAUDE.md §13): video + timeline + voice prompts, summary, events, signs, method. */
export default function ReportPage({ id }: { id: string }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const [voiceOn, setVoiceOn] = useState(() => {
    try {
      return localStorage.getItem(VOICE_PREF) !== "off";
    } catch {
      return true;
    }
  });
  const video = useRef<HTMLVideoElement>(null);
  const lastTime = useRef(0);
  const voice = useMemo(() => new VoicePrompts(), []);

  useEffect(() => {
    getReport(id).then(setReport).catch((e) => setError(t(e instanceof ApiError ? e.key : "error.not_found")));
    return () => voice.stop();
  }, [id, voice]);

  useEffect(() => {
    voice.enabled = voiceOn;
    if (!voiceOn) voice.stop();
    try {
      localStorage.setItem(VOICE_PREF, voiceOn ? "on" : "off");
    } catch {
      /* the choice just isn't remembered */
    }
  }, [voiceOn, voice]);

  if (error) return <p className="rounded-lg bg-red-50 p-3 text-red-800">{error}</p>;
  if (!report) return <p className="text-slate-600">Loading the report…</p>;

  const duration = report.lesson.duration_s;
  const seek = (to: number) => {
    const v = video.current;
    if (!v) return;
    voice.stop();
    v.currentTime = Math.max(0, to);
    lastTime.current = v.currentTime;
    void v.play().catch(() => undefined);
  };
  // Play the prompts whose events normal playback just crossed (not when the user jumps around);
  // several can fall in one tick (signs on one pole) — VoicePrompts queues them.
  const onTimeUpdate = () => {
    const now = video.current?.currentTime ?? 0;
    const prev = lastTime.current;
    if (now > prev && now - prev < 1.5) {
      for (const e of report.events) {
        if (e.voice_key && e.t > prev && e.t <= now) voice.play(e.voice_key);
      }
    }
    lastTime.current = now;
    setTime(now);
  };
  const { summary } = report;
  // Every sign is voiced each time we pass it; a repeat of a sign already explained is not listed again.
  const listed = report.events.filter((e) => !e.detail?.repeat);
  const speedLabel = t(`ui.speed_source.${report.speed_source}`);

  return (
    <div className="space-y-5">
      <section>
        <h1 className="text-2xl font-semibold">{report.lesson.label || t("ui.lesson_report")}</h1>
        <p className="text-slate-600">
          {report.lesson.student} · {clock(duration)} · speed {speedLabel}
          {report.lesson.is_sample && <span className="ml-2 rounded bg-slate-200 px-2 py-0.5 text-xs">sample lesson</span>}
        </p>
      </section>

      <section className="space-y-2">
        <video ref={video} src={report.media.annotated_video_url} poster={report.media.poster_url} controls playsInline
               preload="metadata" onPlay={() => voice.unlock()} onTimeUpdate={onTimeUpdate}
               onSeeked={() => {
                 voice.stop();
                 lastTime.current = video.current?.currentTime ?? 0;
               }}
               className="max-h-[70vh] w-full rounded-xl bg-black" />
        <Timeline duration={duration} events={listed} timeline={report.timeline} time={time} onSeek={seek} />
        <SpeedChart duration={duration} motion={report.motion} timeline={report.timeline} time={time} onSeek={seek} />
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <Legend />
          <button onClick={() => setVoiceOn(!voiceOn)}
                  className={`rounded-full border px-3 py-1 ${voiceOn ? "border-slate-900 bg-slate-900 text-white" : "border-slate-300 text-slate-700"}`}>
            {voiceOn ? `🔊 ${t("ui.voice_on")}` : `🔇 ${t("ui.voice_off")}`}
          </button>
        </div>
      </section>

      {report.quality_warnings.length > 0 && (
        <ul className="space-y-1 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          {report.quality_warnings.map((w) => <li key={w}>⚠ {t(w)}</li>)}
        </ul>
      )}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Card label={t("ui.score_indicative")} value={summary.score} />
        <Card label={t("ui.signs_passed")} value={summary.signs_passed} />
        <Card label={t("ui.violations")} value={summary.violations} tone="text-red-700" />
        <Card label={t("ui.possible")} value={summary.possible} tone="text-amber-700" />
        <Card label={t("ui.done_right")} value={summary.complied} tone="text-green-700" />
      </section>

      <section>
        <h2 className="mb-2 text-lg font-semibold">{t("ui.events")}</h2>
        {listed.length === 0 && <p className="text-slate-600">{t("ui.no_events")}</p>}
        <ul className="space-y-2">
          {listed.map((e) => <EventCard key={e.id} event={e} onSeek={seek} />)}
        </ul>
      </section>

      {report.signs_not_met.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">{t("ui.signs_not_met")}</h2>
          <div className="flex flex-wrap gap-2">
            {report.signs_not_met.map((code) => (
              <span key={code} className="rounded-full border border-slate-300 bg-white px-3 py-1 text-sm">
                {signName(code)} <span className="text-slate-400">{code}</span>
              </span>
            ))}
          </div>
        </section>
      )}

      <details className="rounded-xl border border-slate-200 bg-white p-3">
        <summary className="cursor-pointer font-semibold">{t("ui.all_signs")} ({report.signs.length})</summary>
        <ul className="mt-2 space-y-1 text-sm">
          {report.signs.map((s) => (
            <li key={s.id}>
              <button onClick={() => seek(s.t_first - 1)} className="text-left hover:underline">
                {clock(s.t_pass)} · <b>{signName(s.sign_code, s.value)}</b>{" "}
                <span className={s.relevant ? "text-green-700" : "text-slate-500"}>
                  ({s.relevant ? t("ui.applies") : t("ui.not_for_us")}: {s.relevance_reason})
                </span>
              </button>
            </li>
          ))}
        </ul>
      </details>

      <HowComputed report={report} />
    </div>
  );
}

function EventCard({ event: e, onSeek }: { event: LessonEvent; onSeek: (t: number) => void }) {
  const sign = e.type === "sign_explained" ? signEntry(e.sign_code) : undefined;
  const detail = eventDetail(e);
  return (
    <li className="rounded-xl border border-slate-200 bg-white">
      <button onClick={() => onSeek(e.clip[0])} className="flex w-full items-start gap-3 p-2 text-left">
        {e.thumbnail ? <img src={e.thumbnail} alt="" className="h-16 w-16 shrink-0 rounded-lg object-cover" />
          : <span className="flex h-16 w-16 shrink-0 items-center justify-center rounded-lg bg-slate-100 text-xs text-slate-500">{e.sign_code ?? "—"}</span>}
        <span className="min-w-0 flex-1">
          <span className="block text-xs text-slate-500">{clock(e.t)}</span>
          <span className="block font-medium">{eventTitle(e)}</span>
          {detail && <span className="block text-sm text-slate-600">{detail}</span>}
          {sign && <span className="block text-sm text-slate-600">{sign.meaning}</span>}
        </span>
        <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs ${SEVERITY_PILL[e.severity]}`}>{t(`severity.${e.severity}`)}</span>
      </button>
      {sign && (
        <details className="border-t border-slate-100 px-3 py-2 text-sm text-slate-600">
          <summary className="cursor-pointer">{t("ui.official_text")}</summary>
          <p className="mt-1">{sign.code} «{sign.official.name_ru}». {sign.official.meaning_ru}</p>
          {sign.official.notes_ru.map((n) => <p key={n} className="mt-1">{n}</p>)}
          <p className="mt-1 text-xs">
            {t("ui.source")}: <a href={sign.source.url} target="_blank" rel="noreferrer" className="underline">{sign.source.section}</a>, {sign.source.edition} · {sign.verified ? t("ui.verified") : t("ui.draft")}
          </p>
        </details>
      )}
    </li>
  );
}

function HowComputed({ report }: { report: Report }) {
  const p = report.processing as Record<string, unknown>;
  const stages = (p.seconds_per_stage ?? {}) as Record<string, number>;
  return (
    <details className="rounded-xl border border-slate-200 bg-white p-3 text-sm text-slate-700">
      <summary className="cursor-pointer font-semibold">{t("ui.how_computed")}</summary>
      <ul className="mt-2 list-disc space-y-1 pl-5">
        <li>Road signs: detector <b>{String(p.model)}</b>{p.model_version ? ` (${String(p.model_version)})` : ""}, tracked across frames; a sign counts only if seen in several frames (track voting).</li>
        <li>Whether a sign applies to us: side of the road, which way it faces, and whether we approach it.</li>
        <li>Rules and sign texts: Russian Road Traffic Rules (ПДД РФ), checked by the team.</li>
        <li>Speed: {t(`ui.speed_source.${report.speed_source}`)}{report.speed_source === "none" ? " — speeding and stops were not judged." : "."}</li>
        <li>Processing: {String(p.video_seconds)} s of video in {String(p.total_seconds)} s on {String(p.device)} ({Object.entries(stages).map(([k, v]) => `${k} ${v} s`).join(", ")}); {String(p.frames_analysed)} frames analysed.</li>
        <li>Privacy: faces and number plates are pixelated in every frame and thumbnail; the sound is removed; the original upload is deleted.</li>
        <li>The score is indicative: 100 minus 10 per violation and 3 per possible violation.</li>
      </ul>
    </details>
  );
}

function Card({ label, value, tone = "text-slate-900" }: { label: string; value: number; tone?: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`text-2xl font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

function Legend() {
  const items: [string, string][] = [["bg-red-600", "violation"], ["bg-amber-500", "possible"], ["bg-green-600", "done right"], ["bg-slate-400", "info"]];
  return (
    <span className="flex flex-wrap gap-3 text-xs text-slate-600">
      {items.map(([c, l]) => <span key={l} className="flex items-center gap-1"><span className={`h-3 w-3 rounded-full ${c}`} />{l}</span>)}
    </span>
  );
}

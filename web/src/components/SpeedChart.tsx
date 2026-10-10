import type { MotionSample, RuleState } from "../types";

/** Estimated speed over time (line) against the speed limit in force (dashed steps), red where over.
 *  Gaps = moments where the speed could not be measured from the video. Tap to jump there. */
export default function SpeedChart({ duration, motion, timeline, time, onSeek }: {
  duration: number; motion: MotionSample[]; timeline: RuleState[]; time: number; onSeek: (t: number) => void;
}) {
  const measured = motion.filter((m) => m.speed_kmh != null);
  if (measured.length < 2) return null;
  const W = 1000, H = 120, top = 10;
  const maxV = Math.max(40, ...measured.map((m) => m.speed_kmh ?? 0),
    ...timeline.map((s) => s.speed_limit_kmh ?? 0)) * 1.1;
  const x = (t: number) => (t / Math.max(duration, 0.001)) * W;
  const y = (v: number) => H - (v / maxV) * (H - top);
  const limitAt = (t: number) => [...timeline].reverse().find((s) => s.t <= t)?.speed_limit_kmh ?? null;

  // Split the line where samples are missing (> 1.5 s apart) so gaps stay visible.
  const runs: MotionSample[][] = [];
  for (const m of measured) {
    const run = runs[runs.length - 1];
    if (run && m.t - run[run.length - 1].t <= 1.5) run.push(m);
    else runs.push([m]);
  }
  const limitPath = timeline.map((s, i) => {
    const end = timeline[i + 1]?.t ?? duration;
    return s.speed_limit_kmh ? `M${x(s.t)},${y(s.speed_limit_kmh)} H${x(end)}` : "";
  }).join(" ");

  return (
    <figure className="rounded-xl border border-slate-200 bg-white p-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-24 w-full cursor-pointer" preserveAspectRatio="none"
           onClick={(e) => {
             const r = e.currentTarget.getBoundingClientRect();
             onSeek(((e.clientX - r.left) / r.width) * duration);
           }}>
        <path d={limitPath} stroke="#64748b" strokeWidth={2} strokeDasharray="6 4" fill="none" vectorEffect="non-scaling-stroke" />
        {runs.map((run, i) => (
          <g key={i}>
            {run.slice(1).map((m, j) => {
              const prev = run[j];
              const lim = limitAt(m.t);
              const over = lim != null && (m.speed_kmh ?? 0) > lim;
              return <line key={j} x1={x(prev.t)} y1={y(prev.speed_kmh ?? 0)} x2={x(m.t)} y2={y(m.speed_kmh ?? 0)}
                           stroke={over ? "#dc2626" : "#0f172a"} strokeWidth={2} vectorEffect="non-scaling-stroke" />;
            })}
          </g>
        ))}
        <line x1={x(time)} x2={x(time)} y1={0} y2={H} stroke="#0f172a" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption className="mt-1 flex flex-wrap gap-3 text-xs text-slate-600">
        <span>Speed estimated from the video (km/h, up to {Math.round(maxV / 1.1)})</span>
        <span>— — speed limit in force</span>
        <span className="text-red-700">red: over the limit</span>
        <span>gaps: not measured</span>
      </figcaption>
    </figure>
  );
}

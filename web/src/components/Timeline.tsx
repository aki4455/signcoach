import { SEVERITY_COLOR, clock, eventTitle } from "../format";
import type { LessonEvent, RuleState } from "../types";

/** Bar under the video: speed limit in force over time, event markers by severity, playhead. */
export default function Timeline({ duration, events, timeline, time, onSeek }: {
  duration: number; events: LessonEvent[]; timeline: RuleState[]; time: number; onSeek: (t: number) => void;
}) {
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / Math.max(duration, 0.001)) * 100))}%`;
  // One band segment per speed limit (consecutive states with the same limit are merged).
  const changes = timeline.filter((s, i) => i === 0 || s.speed_limit_kmh !== timeline[i - 1].speed_limit_kmh);
  const segments = changes.map((s, i) => ({ ...s, end: changes[i + 1]?.t ?? duration }));
  return (
    <div className="select-none" aria-label="Lesson timeline">
      <div className="relative h-5 text-[10px] text-slate-600">
        {segments.map((s) => (
          <div key={s.t} className="absolute top-0 h-5 truncate border-l border-slate-300 pl-1 leading-5"
               style={{ left: pct(s.t), width: `calc(${pct(s.end)} - ${pct(s.t)})` }}>
            {s.speed_limit_kmh ? `${s.speed_limit_kmh} km/h` : ""}
          </div>
        ))}
      </div>
      <div className="relative h-8 cursor-pointer"
           onClick={(e) => {
             const r = e.currentTarget.getBoundingClientRect();
             onSeek(((e.clientX - r.left) / r.width) * duration);
           }}>
        <div className="absolute inset-x-0 top-3 h-2 rounded-full bg-slate-200" />
        {events.map((e) => (
          <button key={e.id} title={`${clock(e.t)} ${eventTitle(e)}`}
                  onClick={(ev) => { ev.stopPropagation(); onSeek(e.clip[0]); }}
                  className={`absolute top-1.5 h-5 w-5 -translate-x-1/2 rounded-full border-2 border-white shadow ${SEVERITY_COLOR[e.severity]}`}
                  style={{ left: pct(e.t) }} />
        ))}
        <div className="pointer-events-none absolute top-0 h-8 w-0.5 bg-slate-900" style={{ left: pct(time) }} />
      </div>
    </div>
  );
}

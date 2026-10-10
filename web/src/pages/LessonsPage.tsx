import { useEffect, useState } from "react";
import { listLessons, myLessons } from "../api";
import { linkProps } from "../router";
import type { LessonSummary } from "../types";

/** Sample lessons plus the lessons uploaded from this browser. */
export default function LessonsPage() {
  const [lessons, setLessons] = useState<LessonSummary[] | null>(null);
  useEffect(() => {
    listLessons(myLessons()).then(setLessons).catch(() => setLessons([]));
  }, []);
  if (lessons === null) return <p>Loading…</p>;
  return (
    <div className="space-y-3">
      <h1 className="text-2xl font-semibold">Lessons</h1>
      {lessons.length === 0 && <p className="text-slate-600">No lessons yet. Upload a video to get the first report.</p>}
      <ul className="space-y-2">
        {lessons.map((l) => (
          <li key={l.id}>
            <a {...linkProps(`/lessons/${l.id}`)}
               className="block rounded-xl border border-slate-200 bg-white p-3 hover:border-slate-400">
              <span className="font-medium">{l.label || "Lesson"}</span>
              {l.is_sample && <span className="ml-2 rounded bg-slate-100 px-2 py-0.5 text-xs">sample</span>}
              <span className="block text-sm text-slate-600">
                {l.student} · {l.duration_s ? `${Math.round(l.duration_s)} s` : ""} · {new Date(l.created_at).toLocaleString()}
              </span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

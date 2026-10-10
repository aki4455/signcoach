import { useEffect, useState } from "react";

type Health = { ok: boolean; git_sha: string; model: string; device: string; queue_length: number };

/** Placeholder home page (M0). The upload and report pages arrive in M1 and M4. */
export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    fetch("/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <main className="mx-auto max-w-xl px-4 py-12 font-sans text-slate-800">
      <h1 className="text-3xl font-semibold">SignCoach</h1>
      <p className="mt-3 text-lg">
        AI co-instructor for driving schools. Upload a lesson video and get a report of every road sign, the rules
        in force and where they were broken.
      </p>
      <p className="mt-6 rounded-lg bg-amber-50 p-4 text-amber-900">The app is being built. Uploads open soon.</p>
      <p className="mt-6 text-sm text-slate-500">
        {health ? `Version ${health.git_sha.slice(0, 7)} · model ${health.model} · ${health.device}` : "Checking service…"}
      </p>
    </main>
  );
}

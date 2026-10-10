"""Processing speed and cost per lesson (CLAUDE.md §15).

Runs the complete pipeline — exactly what an upload goes through — on the given videos with the
configured detector and device, in a throw-away instance (nothing appears on the website), and
reads the timing the pipeline records in every report: seconds per stage, frames analysed per
second, peak GPU memory. It also samples the GPU's board power with nvidia-smi while each video is
processed. From that: seconds of processing per minute of video and, with the price inputs in settings
(`cost.*`; the measured power is used when `cost.gpu_power_w` is empty), cost per lesson-hour.
Writes eval/results/benchmark.{json,md}.

    SIGNCOACH_DEVICE=cuda:0 uv run python eval/benchmark.py ~/signcoach-data/datasets/own/ru_0*.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from app.settings import REPO_DIR, get_settings
from pipeline.catalogue import Catalogue
from pipeline.run import JobInput, Models, process_job


NVIDIA_SMI = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"


class PowerSampler:
    """Average GPU board power (W) while a block runs, from `nvidia-smi` every 0.5 s (None without it)."""

    def __enter__(self):
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            try:
                out = subprocess.run([NVIDIA_SMI, "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.split()
                self.samples.append(float(out[0]))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                return

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def mean_w(self) -> float | None:
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    args = ap.parse_args()
    s = get_settings()
    cfg = s.cfg
    t = time.monotonic()
    models = Models.load(cfg, s.models_dir, s.device)
    load_s = time.monotonic() - t
    work = Path(tempfile.mkdtemp(prefix="signcoach-bench-"))
    rows = []
    try:
        for i, video in enumerate(args.videos):
            upload = work / "jobs" / f"b{i}" / f"upload{video.suffix}"
            upload.parent.mkdir(parents=True)
            shutil.copy(video, upload)
            with PowerSampler() as power:
                report = process_job(JobInput(f"b{i}", f"bench{i}", upload, "benchmark", video.stem, "bench"), cfg,
                                     models, Catalogue.load(), work / "media", work / "jobs", s.models_dir, s.git_sha,
                                     lambda stage, p: None, time.monotonic() + 3600)
            p = report.processing
            minutes = p["video_seconds"] / 60
            rows.append({"video": video.name, "video_s": p["video_seconds"], "total_s": p["total_seconds"],
                         "s_per_video_min": round(p["total_seconds"] / minutes, 1), "realtime_factor": p["realtime_factor"],
                         "fps_analysed": p["fps_analysed"], "frames": p["frames_analysed"],
                         "stages_s": p["seconds_per_stage"], "peak_vram_mb": p["peak_vram_mb"], "device": p["device"],
                         "model": p["model"], "gpu_power_w": power.mean_w})
            print(rows[-1])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    s_per_min = sum(r["total_s"] for r in rows) / (sum(r["video_s"] for r in rows) / 60)
    gpu_hours_per_lesson_hour = s_per_min * 60 / 3600
    c = cfg.get("cost", {})
    measured = [(r["gpu_power_w"], r["total_s"]) for r in rows if r["gpu_power_w"]]
    measured_w = round(sum(w * t for w, t in measured) / sum(t for _, t in measured), 1) if measured else None
    power_w = c.get("gpu_power_w") or measured_w
    cost = {"processing_hours_per_lesson_hour": round(gpu_hours_per_lesson_hour, 3),
            "gpu_power_w_measured": measured_w,
            "energy_kwh_per_lesson_hour": round(gpu_hours_per_lesson_hour * power_w / 1000, 3) if power_w else None}
    if power_w and c.get("electricity_price_per_kwh"):
        cost["electricity_per_lesson_hour"] = round(gpu_hours_per_lesson_hour * power_w / 1000
                                                    * c["electricity_price_per_kwh"], 4)
    if c.get("cloud_gpu_price_per_hour"):
        cost["cloud_gpu_per_lesson_hour"] = round(gpu_hours_per_lesson_hour * c["cloud_gpu_price_per_hour"], 3)
    result = {"device": s.device, "model_load_s": round(load_s, 1), "seconds_per_video_minute": round(s_per_min, 1),
              "cost": cost, "cost_inputs": c, "runs": rows}
    out = REPO_DIR / "eval" / "results"
    (out / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Processing speed and cost", "", f"Generated by `eval/benchmark.py` on `{s.device}` (full pipeline per upload).", "",
             "| Video | Length s | Processing s | s per video minute | × real time | frames/s | peak VRAM MB | GPU power W |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['video']} | {r['video_s']} | {r['total_s']} | {r['s_per_video_min']} | {r['realtime_factor']} | "
              f"{r['fps_analysed']} | {r['peak_vram_mb']} | {r['gpu_power_w']} |" for r in rows]
    priced = {k: v for k, v in cost.items() if k.endswith("_per_lesson_hour") and k not in
              ("processing_hours_per_lesson_hour", "energy_kwh_per_lesson_hour")}
    lines += ["", f"Average: **{result['seconds_per_video_minute']} s of processing per minute of video**; "
                  f"{cost['processing_hours_per_lesson_hour']} GPU-hours per lesson-hour; average GPU board power while "
                  f"processing {measured_w} W (nvidia-smi) -> {cost['energy_kwh_per_lesson_hour']} kWh per lesson-hour.",
              "Cost per lesson-hour: " + (", ".join(f"{k} = {v}" for k, v in priced.items())
                                         or "fill the prices in `cost.*` (config/settings.yaml) to compute it.")]
    (out / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

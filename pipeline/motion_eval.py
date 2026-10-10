"""Run only the motion part of the pipeline (normalize -> frames -> vehicles -> ego-motion) on a video.

Used by scripts/calibrate_speed.py and eval/eval_speed.py, so calibration and evaluation measure
exactly what the real pipeline computes (same normalization, frame step, masks and settings).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pipeline import egomotion, frames, ingest
from pipeline.types import MotionSample, TurnEvent
from pipeline.vehicles import VehicleDetector


def motion_for_video(video: Path, cfg: dict, models_dir: Path, device: str, hfov_deg: float | None = None,
                     height_m: float | None = None) -> tuple[list[MotionSample], list[TurnEvent], dict]:
    work = Path(tempfile.mkdtemp(prefix="signcoach-motion-"))
    try:
        info = ingest.probe(video)
        normalized = work / "normalized.mp4"
        ninfo = ingest.normalize(video, normalized, info, cfg)
        step = frames.analysis_step(ninfo.fps, float(cfg["analysis_fps"]))
        fps = ninfo.fps / step
        grays = egomotion.sample_grays(normalized, cfg["egomotion"]["sample_frames"])
        horizon, method = egomotion.estimate_horizon(grays, cfg["camera"]["horizon_fallback_frac"])
        hfov = hfov_deg or cfg["camera"]["hfov_deg"]
        cam = egomotion.Camera(ninfo.width, ninfo.height, egomotion.focal_length(ninfo.width, hfov), ninfo.width / 2,
                               horizon, height_m or cfg["camera"]["height_m"])
        ego = egomotion.EgoMotion(cam, egomotion.static_mask(grays, cfg["egomotion"]["static_std"]), fps, cfg)
        v = cfg["models"]["vehicles"]
        vehicles = VehicleDetector(models_dir / v["weights"], v["imgsz"], v["conf"], device, cfg["detect"].get("half", True))
        for _, t, frame in frames.iter_frames(normalized, ninfo.width, ninfo.height, ninfo.fps, step):
            ego.update(frame, t, [o.box for o in vehicles.detect(frame)])
        motion, turns = ego.finish()
        return motion, turns, {"horizon_y": horizon, "horizon_method": method, "hfov_deg": hfov, "fps": fps,
                               "width": ninfo.width, "height": ninfo.height}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def read_truth(path: Path) -> list[tuple[float, float]]:
    """Ground-truth speed CSV (t_seconds,speed_kmh) -> [(t, km/h)], skipping rows left empty."""
    import csv
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("speed_kmh") or "").strip():
                rows.append((float(r["t_seconds"]), float(r["speed_kmh"])))
    return rows


def paired(motion: list[MotionSample], truth: list[tuple[float, float]], window_s: float = 0.5
           ) -> list[tuple[float, float, float]]:
    """(t, true km/h, estimated km/h) for every truth row with an estimate within ±window_s."""
    import numpy as np
    out = []
    for t, v in truth:
        near = [m.speed_kmh for m in motion if abs(m.t - t) <= window_s and m.speed_kmh is not None]
        if near:
            out.append((t, v, float(np.median(near))))
    return out

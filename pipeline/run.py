"""Process one uploaded video end to end (CLAUDE.md §6):

    ingest -> analysis pass (signs + tracking, vehicles/people) -> voting -> relevance
           -> [ego-motion: M3] -> rules engine -> render pass (blur + overlays) -> report.json

`process_job` is called by the single worker thread with models that stay loaded between jobs.
It reports progress as (stage, 0..1) and checks a deadline between frames, because a Python thread
cannot be killed from outside (CLAUDE.md §12). Unblurred material (the upload and the normalized
video) stays in the job folder and is deleted at the end; only blurred media reach media/<lesson>/.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline import egomotion, frames, ingest, relevance, render, report, rules_engine, signs
from pipeline.catalogue import Catalogue
from pipeline.detect import ClassMap, SignDetector, build_detector
from pipeline.privacy import FaceDetector
from pipeline.types import LessonEvent, Report
from pipeline.vehicles import PlateDetector, VehicleDetector

I18N = Path(__file__).resolve().parents[1] / "config" / "i18n" / "en.json"


class JobTimeout(Exception):
    error_key = "error.timeout"


@dataclass
class Models:
    """Everything that is slow to load; loaded once by the worker and reused for every job."""

    signs: SignDetector
    detectable: set[str]          # catalogue codes the sign detector can output
    vehicles: VehicleDetector
    plates: PlateDetector
    faces: FaceDetector
    device: str

    @classmethod
    def load(cls, cfg: dict, models_dir: Path, device: str, replay_file: Path | None = None) -> "Models":
        detector = build_detector(cfg, models_dir, device, replay_file)
        vcfg, pcfg = cfg["models"]["vehicles"], cfg["models"]["plates"]
        return cls(signs=detector, detectable=ClassMap.for_model(cfg["detect"]["model"]).codes,
                   vehicles=VehicleDetector(models_dir / vcfg["weights"], vcfg["imgsz"], vcfg["conf"], device,
                                            cfg["detect"].get("half", True)),
                   plates=PlateDetector(models_dir / pcfg["weights"], pcfg["imgsz"], pcfg["conf"], device,
                                        cfg["detect"].get("half", True)),
                   faces=FaceDetector(models_dir / cfg["models"]["faces"]["weights"],
                                      cfg["privacy"]["face_score_threshold"]),
                   device=device)


@dataclass
class JobInput:
    job_id: str
    lesson_id: str
    upload_path: Path
    student: str
    label: str
    created_at: str
    camera_height_m: float | None = None
    hfov_deg: float | None = None


def event_text(e: LessonEvent, catalogue: Catalogue, strings: dict) -> str:
    """English sentence for an event (same templates as the website)."""
    params = {k: v for k, v in e.detail.items()}
    if e.sign_code:
        params.setdefault("sign_name", catalogue.label(e.sign_code, e.value))
    try:
        return strings.get(e.title_key, e.title_key).format(**params)
    except (KeyError, IndexError):
        return strings.get(e.title_key, e.title_key)


def _model_version(path: Path) -> str | None:
    sha = path.with_name(path.name + ".sha256")
    return sha.read_text().split()[0][:12] if sha.exists() else None


def process_job(job: JobInput, cfg: dict, models: Models, catalogue: Catalogue, media_root: Path,
                jobs_root: Path, models_dir: Path, git_sha: str, progress: Callable[[str, float], None],
                deadline: float) -> Report:
    def check_deadline():
        if time.monotonic() > deadline:
            raise JobTimeout()

    timings: dict[str, float] = {}
    job_dir = jobs_root / job.job_id
    out_dir = media_root / job.lesson_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    try:
        # 1. Ingest -------------------------------------------------------------------------------
        progress("ingest", 0.02)
        t = time.monotonic()
        info = ingest.probe(job.upload_path)
        ingest.validate(info, cfg)
        normalized = job_dir / "normalized.mp4"
        ninfo = ingest.normalize(job.upload_path, normalized, info, cfg)
        warnings = ingest.quality_warnings(normalized, ninfo, cfg)
        timings["ingest"] = time.monotonic() - t

        # 2. Analysis pass: signs (with tracking) and people/vehicles, frame by frame ---------------
        step = frames.analysis_step(ninfo.fps, float(cfg["analysis_fps"]))  # keep every step-th frame
        fps = ninfo.fps / step
        expected = max(1, int(ninfo.duration_s * fps))
        # Camera model and masks for speed/turns (horizon and static overlays need the whole video).
        ego_cfg = cfg["egomotion"]
        grays = egomotion.sample_grays(normalized, ego_cfg["sample_frames"])
        horizon, horizon_method = egomotion.estimate_horizon(grays, cfg["camera"]["horizon_fallback_frac"])
        hfov = job.hfov_deg or cfg["camera"]["hfov_deg"]
        cam_height = job.camera_height_m or cfg["camera"]["height_m"]
        cam = egomotion.Camera(ninfo.width, ninfo.height, egomotion.focal_length(ninfo.width, hfov), ninfo.width / 2,
                               horizon, cam_height)
        ego = egomotion.EgoMotion(cam, egomotion.static_mask(grays, ego_cfg["static_std"]), fps, cfg)
        models.signs.reset()
        detections, objects_by_frame = [], []
        t = time.monotonic()
        if models.device.startswith("cuda"):
            import torch
            torch.cuda.reset_peak_memory_stats()
        for idx, ts, frame in frames.iter_frames(normalized, ninfo.width, ninfo.height, ninfo.fps, step):
            check_deadline()
            detections += models.signs.detect(frame, ts, idx)
            objects = models.vehicles.detect(frame)
            ego.update(frame, ts, [o.box for o in objects])
            objects_by_frame.append(objects + models.plates.detect(frame))
            if idx % 10 == 0:
                progress("analysis", 0.1 + 0.55 * min(1.0, idx / expected))
        timings["analysis"] = time.monotonic() - t
        n_frames = len(objects_by_frame)
        # A plate found in a neighbouring frame is hidden here too (no single-frame flashes).
        k = cfg["privacy"]["plate_neighbour_frames"]
        plates = [[o for o in objs if o.kind == "plate"] for objs in objects_by_frame]
        objects_by_frame = [objs + [o for j in range(max(0, i - k), min(n_frames, i + k + 1)) if j != i for o in plates[j]]
                            for i, objs in enumerate(objects_by_frame)]

        # 3. Voting and relevance -------------------------------------------------------------------
        t = time.monotonic()
        voted = signs.vote(detections, cfg, ninfo.width)
        plates = [v for v in voted if catalogue.is_plate(v.sign_code)]
        sign_votes = [v for v in voted if not catalogue.is_plate(v.sign_code)]
        sign_events = [relevance.assess(v, f"s{i}", plates, ninfo.width, ninfo.height,
                                        catalogue.shape(v.sign_code), cfg) for i, v in enumerate(sign_votes, 1)]

        # 4. Ego-motion: speed, turns and stops from the video itself.
        motion, turns = ego.finish()
        measured = sum(m.speed_kmh is not None for m in motion) / max(1, len(motion))
        speed_source = "video" if measured > 0 else "none"
        if measured == 0:
            warnings.append("warning.speed_not_checked")
        elif measured < 0.8:
            warnings.append("warning.speed_not_checked_partial")
        if speed_source == "video" and not cfg["speed"].get("calibrated", False):
            warnings.append("warning.camera_uncalibrated")

        # 5. Rules ------------------------------------------------------------------------------------
        # Each sign is announced when it is confirmed — the vote.min_frames-th sighting of its final
        # class — which is when a system in the car could have spoken; its rule applies when we pass it.
        announce_at = {}
        if cfg.get("voice", {}).get("announce_at", "confirmed") == "confirmed":
            k = cfg["vote"]["min_frames"]
            for e, v in zip(sign_events, sign_votes):
                seen = sorted(d.t for d in v.winning)
                announce_at[e.id] = seen[min(k, len(seen)) - 1]
        events, timeline = rules_engine.run(sign_events, motion, turns, catalogue, cfg, duration_s=ninfo.duration_s,
                                            announce_at=announce_at)
        if not any(s.relevant for s in sign_events):
            warnings.append("warning.no_signs")
        timings["signs_and_rules"] = time.monotonic() - t

        # 6. Render pass ------------------------------------------------------------------------------
        t = time.monotonic()
        strings = json.loads(I18N.read_text(encoding="utf-8"))
        sign_boxes: dict[int, list] = {}
        for e, v in zip(sign_events, sign_votes):
            label = catalogue.label(e.sign_code, e.value)
            for d in v.detections:
                sign_boxes.setdefault(d.frame_idx, []).append((d.bbox, label, e.relevant))
        thumbs: dict[int, list] = {}
        for e in sign_events:
            if any(ev.sign_event_id == e.id for ev in events):
                thumbs.setdefault(e.best_frame_idx, []).append((f"{e.id}.jpg", e.best_bbox))
        subtitles = render.merge_subtitles([(e.t, e.t + 3.0, event_text(e, catalogue, strings)) for e in events
                                            if e.voice_key or e.severity in ("violation", "possible")])
        poster = events[0].t if events else ninfo.duration_s / 3
        saved = render.render(
            render.RenderJob(normalized=normalized, width=ninfo.width, height=ninfo.height, video_fps=ninfo.fps, step=step,
                             objects_by_frame=objects_by_frame, sign_boxes=sign_boxes, timeline=timeline,
                             motion=motion, subtitles=subtitles, thumbnails=thumbs,
                             poster_frame=min(n_frames - 1, int(poster * fps)) if n_frames else 0),
            out_dir, job_dir / "render.log", models.faces, cfg, check_deadline,
            lambda p: progress("render", 0.7 + 0.27 * p))
        for e in events:
            if e.sign_event_id and f"{e.sign_event_id}.jpg" in saved:
                e.thumbnail = f"/media/{job.lesson_id}/{saved[e.sign_event_id + '.jpg']}"
        timings["render"] = time.monotonic() - t

        # 7. Report -----------------------------------------------------------------------------------
        progress("report", 0.98)
        peak_vram = None
        if models.device.startswith("cuda"):
            import torch
            peak_vram = round(torch.cuda.max_memory_allocated() / 2**20)
        seen = {s.sign_code for s in sign_events if s.relevant}
        total = time.monotonic() - t_start
        det_cfg = cfg["detect"]
        weights = models_dir / cfg["models"].get(det_cfg["model"], {}).get("weights", "")
        result = report.build(
            lesson={"id": job.lesson_id, "student": job.student, "label": job.label, "created_at": job.created_at,
                    "duration_s": round(ninfo.duration_s, 2), "is_sample": False},
            media={"annotated_video_url": f"/media/{job.lesson_id}/annotated.mp4",
                   "poster_url": f"/media/{job.lesson_id}/poster.jpg"},
            speed_source=speed_source,
            calibration={"scale_k": cfg["speed"]["scale_k"], "err_model": cfg["speed"]["err_model"],
                         "camera_height_m": cam_height, "hfov_deg": hfov,
                         "calibrated": bool(cfg["speed"].get("calibrated", False)),
                         "horizon_y": round(horizon, 1), "horizon_method": horizon_method,
                         "speed_measured_fraction": round(measured, 3), "turns_detected": len(turns)},
            events=events, timeline=timeline, motion=motion, signs=sign_events,
            signs_not_met=[c for c in catalogue.explainable_codes(models.detectable) if c not in seen],
            quality_warnings=warnings, score_weights=cfg["score"]["weights"],
            processing={"model": models.signs.name, "model_version": None if models.signs.name.startswith("replay") else _model_version(weights),
                        "device": models.device, "git_sha": git_sha, "analysis_fps": fps,
                        "frames_analysed": n_frames, "video_seconds": round(ninfo.duration_s, 2),
                        "seconds_per_stage": {k: round(v, 2) for k, v in timings.items()},
                        "total_seconds": round(total, 2),
                        "fps_analysed": round(n_frames / timings["analysis"], 2) if timings["analysis"] else None,
                        "realtime_factor": round(total / ninfo.duration_s, 2) if ninfo.duration_s else None,
                        "peak_vram_mb": peak_vram, "source_video": {"fps": round(info.fps, 3), "width": info.width,
                                                                    "height": info.height, "codec": info.codec,
                                                                    "hdr": info.is_hdr}})
        (out_dir / "report.json").write_text(result.model_dump_json(indent=1), encoding="utf-8")
        return result
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)   # never leave half-written media behind
        raise
    finally:
        # Unblurred material never leaves the job folder and is deleted after processing (§10.9).
        if cfg["privacy"].get("delete_unblurred_after_processing", True):
            shutil.rmtree(job_dir, ignore_errors=True)
            job.upload_path.unlink(missing_ok=True)

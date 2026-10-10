"""ByteTrack vs our SignTracker on whole clips: how many signs survive voting with each.

Both paths use the same detector, settings and voting; only the tracker differs:
- `bytetrack`: Ultralytics' ByteTrack (`model.track`, class-aware NMS) — the app before Sunday 03:30;
- `signcoach`: `pipeline/tracking.py` (`model.predict` with class-agnostic NMS + motion prediction).

There is no ground truth for these clips, so the script counts signs kept after voting (and the ones
judged relevant) and lists the differences; `--sheet` writes a crop sheet of the signs only one
tracker kept, for a person to check which are real. Detections are cached per clip (`--cache`), so
re-running the comparison does not need the GPU.

    uv run python eval/compare_trackers.py --cache ~/signcoach-data/tracker_cache clip1.mp4 clip2.mp4 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.settings import get_settings
from pipeline import frames, ingest, relevance, signs
from pipeline.catalogue import Catalogue
from pipeline.detect import ClassMap, UltralyticsDetector
from pipeline.types import Detection

RESULTS = Path(__file__).resolve().parent / "results"
TRACKERS = {"bytetrack": "bytetrack.yaml", "signcoach": "signcoach"}


def detect_clip(clip: Path, cfg: dict, detectors: dict) -> dict:
    info = ingest.probe(clip)
    step = frames.analysis_step(info.fps, cfg["analysis_fps"])
    out = {"width": info.width, "height": info.height, "fps": info.fps / step}
    for d in detectors.values():
        d.reset()
    found = {name: [] for name in detectors}
    for idx, t, frame in frames.iter_frames(clip, info.width, info.height, info.fps, step):
        for name, d in detectors.items():
            found[name] += d.detect(frame, t, idx)
    return out | {name: [x.model_dump(mode="json") for x in dets] for name, dets in found.items()}


def judged(dets: list[Detection], cfg: dict, catalogue: Catalogue, W: int, H: int):
    voted = signs.vote(dets, cfg, W)
    plates = [v for v in voted if catalogue.is_plate(v.sign_code)]
    kept = [v for v in voted if not catalogue.is_plate(v.sign_code)]
    return [(v, relevance.assess(v, f"s{i}", plates, W, H, catalogue.shape(v.sign_code), cfg))
            for i, v in enumerate(kept, 1)]


def label(v) -> str:
    return f"{v.sign_code}{'/' + str(v.value) if v.value else ''}"


def only_in(a, b, window_s: float = 1.5):
    """Signs in a with no sign of the same code and value in b within window_s."""
    return [(v, e) for v, e in a if not any((label(w) == label(v)) and abs(w.t_last - v.t_last) <= window_s for w, _ in b)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--cache", type=Path, required=True, help="folder for per-clip detections (outside git)")
    ap.add_argument("--name", default="tracker_comparison", help="output name in eval/results/")
    args = ap.parse_args()
    s = get_settings()
    cfg, det = s.cfg, s.cfg["detect"]
    catalogue = Catalogue.load()
    args.cache.mkdir(parents=True, exist_ok=True)
    detectors = None

    rows, total = [], {f"{k}_{n}": 0 for k in TRACKERS for n in ("signs", "relevant")}
    for clip in args.clips:
        cached = args.cache / f"{clip.stem}.json"
        if not cached.exists():
            if detectors is None:
                weights = s.models_dir / cfg["models"][det["model"]]["weights"]
                detectors = {name: UltralyticsDetector(det["model"], weights, ClassMap.for_model(det["model"]), det["imgsz"],
                                                       det["conf"], s.device, det.get("half", True), tracker, cfg["track"])
                             for name, tracker in TRACKERS.items()}
            cached.write_text(json.dumps(detect_clip(clip, cfg, detectors)))
        data = json.loads(cached.read_text())
        W, H = data["width"], data["height"]
        res = {k: judged([Detection.model_validate(x) for x in data[k]], cfg, catalogue, W, H) for k in TRACKERS}
        row = {"clip": clip.stem, "boxes": {k: len(data[k]) for k in TRACKERS}}
        for k, signs_ in res.items():
            row[k] = {"signs": len(signs_), "relevant": sum(e.relevant for _, e in signs_)}
            total[f"{k}_signs"] += row[k]["signs"]
            total[f"{k}_relevant"] += row[k]["relevant"]
        row["only_signcoach"] = [f"{label(v)}@{v.t_last:.0f}s{'' if e.relevant else ' (not ours)'}"
                                 for v, e in only_in(res["signcoach"], res["bytetrack"])]
        row["only_bytetrack"] = [f"{label(v)}@{v.t_last:.0f}s{'' if e.relevant else ' (not ours)'}"
                                 for v, e in only_in(res["bytetrack"], res["signcoach"])]
        rows.append(row)
        print(f"{clip.stem}: ByteTrack {row['bytetrack']} | ours {row['signcoach']}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps({"rows": rows, "total": total}, indent=1), encoding="utf-8")
    lines = [
        "# Tracker comparison: ByteTrack vs our SignTracker", "",
        f"Generated by `eval/compare_trackers.py` ({det['model']} at {det['imgsz']} px, conf {det['conf']}, "
        f"{cfg['analysis_fps']} analysed frames/s target; voting: ≥ {cfg['vote']['min_frames']} frames, mean conf ≥ "
        f"{cfg['vote']['min_conf']}). No ground truth: counts of signs kept after voting; differences listed for a "
        "person to check.", "",
        "| Clip | Boxes detected | Signs kept: ByteTrack | Signs kept: ours | Relevant: ByteTrack | Relevant: ours |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['clip']} | {r['boxes']['signcoach']} | {r['bytetrack']['signs']} | {r['signcoach']['signs']} | "
                     f"{r['bytetrack']['relevant']} | {r['signcoach']['relevant']} |")
    lines.append(f"| **Total** | | **{total['bytetrack_signs']}** | **{total['signcoach_signs']}** | "
                 f"**{total['bytetrack_relevant']}** | **{total['signcoach_relevant']}** |")
    lines += ["", "Signs kept by only one tracker (same code and value not found by the other within 1.5 s):", ""]
    for r in rows:
        if r["only_signcoach"] or r["only_bytetrack"]:
            lines.append(f"- **{r['clip']}** — only ours: {', '.join(r['only_signcoach']) or '–'}; "
                         f"only ByteTrack: {', '.join(r['only_bytetrack']) or '–'}")
    (RESULTS / f"{args.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(total)


if __name__ == "__main__":
    main()

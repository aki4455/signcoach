"""Prepare the Dognellaf "russian-road-signs" dataset for YOLO training (training track T1–T3).

What it does, in order:

T1  Completeness check. Pairs every image in ``images/part_*/`` with its label file and reports
    images without labels, labels without images, empty label files, broken label lines and
    duplicate names. Only complete image+label pairs are used. The Hugging Face repo also has
    copies of some files directly in ``images/`` and ``labels/`` (outside the part folders);
    those are duplicates of part files and are ignored.

T2  Inventory. Counts instances, frames and distinct source videos per class, lists which signs
    from CLAUDE.md §11.1 are present or missing, and writes
    ``eval/results/dataset_inventory.{json,md}``.

T3  Split by source video. Derives a video id from each file name, holds out about 15% of the
    videos for validation (a video is never split between train and val, otherwise near-identical
    frames would leak), keeps every 2nd frame of each video (consecutive frames are near-duplicates)
    plus every frame that contains a rare class, and writes a YOLO dataset: symlinked images,
    cleaned label copies, ``data.yaml`` and a small ``data_smoke.yaml`` for the smoke test.

Why symlinks: Ultralytics finds a label by replacing ``/images/`` with ``/labels/`` in the image
path, so training needs its own images/ + labels/ tree; symlinks avoid copying ~18 GB of JPEGs.

Only the standard library is used. Run it with ``python -I`` (isolated mode), passing data paths
as arguments, so nothing inside the downloaded dataset can be imported as code:

    python3 -I training/prepare_dognellaf.py \
        --images ~/signcoach-data/datasets/dognellaf/images \
        --labels ~/signcoach-data/datasets/dognellaf_labels \
        --classes ~/signcoach-data/datasets/dognellaf/classes.txt \
        --out ~/signcoach-data/datasets/dognellaf_yolo
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Signs the rules engine needs (CLAUDE.md §11.1), as Russian GOST codes used by the dataset.
# "3.24" stands for every speed-limit class (the dataset encodes the value as 3.24.N).
REQUIRED_SIGNS = {
    "3.24": "speed_limit", "3.25": "end_speed_limit", "3.31": "end_all_restrictions",
    "3.1": "no_entry", "3.18.1": "no_right_turn", "3.18.2": "no_left_turn", "3.19": "no_u_turn",
    "3.20": "no_overtaking", "3.21": "end_no_overtaking", "3.27": "no_stopping", "3.28": "no_parking",
    "2.5": "stop", "2.4": "give_way", "2.1": "main_road", "2.2": "end_main_road",
    "4.1.1": "mandatory_straight", "4.1.2": "mandatory_right", "4.1.3": "mandatory_left",
    "4.1.4": "mandatory_straight_or_right", "4.1.5": "mandatory_straight_or_left",
    "4.1.6": "mandatory_right_or_left", "4.3": "roundabout",
    "5.19.1": "pedestrian_crossing", "5.19.2": "pedestrian_crossing",
    "1.22": "pedestrian_crossing_ahead", "1.23": "children", "5.20": "speed_bump",
    "1.17": "speed_bump_ahead", "5.23.1": "settlement_start", "5.23.2": "settlement_start",
    "5.24.1": "settlement_end", "5.24.2": "settlement_end",
}
# Rule of thumb from CLAUDE.md §14.2: a core sign is "enough" with >= 150 instances from >= 8 videos.
MIN_INSTANCES, MIN_VIDEOS = 150, 8


def read_classes(path: Path) -> list[str]:
    """Class names, one per line. The file has Windows line endings, so strip '\\r' too."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def video_id(stem: str) -> str:
    """Best guess of the source recording a frame came from, derived from its file name.

    Examples seen in the dataset:
      PXL_20231212_144313113-3__img317Frames_...   -> PXL_20231212_144313113   (segments of one recording)
      00030002__2_-output__frame-0-00-27.67        -> 00030002
      0_inter-n-13__inter-n-13__10_                 -> inter-n-13
      2023_1123_064652_926.__frame-...              -> dashcam_2023_1123_06     (1-minute dashcam clips of one drive)
      asmr-251-252__frame-2-32-16.69                -> asmr_block10             (segments of one long drive)
    """
    head = stem.split("__", 1)[0]
    head = re.sub(r"-output.*$", "", head)
    head = re.sub(r"^0_", "", head)
    if re.match(r"^(PXL_|VID|IMG_)", head):
        head = re.sub(r"-\d+$", "", head)
    if re.match(r"^\d+(-\d+)+$", head):
        head = head.split("-")[0]
    m = re.match(r"^(\d{4}_\d{4}_\d{2})\d{4}_\d+", head)
    if m:
        return "dashcam_" + m.group(1)
    m = re.match(r"^asmr-(\d+)", head)
    if m:
        return f"asmr_block{int(m.group(1)) // 25}"
    return head


def natural_key(text: str) -> list:
    """Sort key that orders 'frame 9' before 'frame 10'."""
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", text)]


def parse_label(path: Path, n_classes: int) -> tuple[list[tuple[int, float, float, float, float]], int, int]:
    """Read a YOLO label file. Returns (boxes, dropped_lines, clipped_boxes).

    Lines with a bad class id or a non-positive size are dropped. Coordinates slightly outside
    [0, 1] are clipped, because Ultralytics would otherwise reject the whole image as corrupt.
    """
    boxes, dropped, clipped = [], 0, 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            dropped += 1
            continue
        if not 0 <= cls < n_classes or len(parts) != 5:
            dropped += 1
            continue
        # Convert to corners, clip to the image, convert back.
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            clipped += 1
        if cx2 - cx1 <= 1e-6 or cy2 - cy1 <= 1e-6:
            dropped += 1
            continue
        boxes.append((cls, (cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1))
    return boxes, dropped, clipped


def check_completeness(images_root: Path, labels_root: Path) -> dict:
    """T1: pair images and labels inside the part_* folders only."""
    images, dup_names = {}, []
    for part in sorted(p for p in images_root.iterdir() if p.is_dir() and p.name.startswith("part_")):
        for img in part.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            if img.stem in images:
                dup_names.append(img.stem)
            images[img.stem] = (img, labels_root / part.name / f"{img.stem}.txt")
    label_stems = {
        lbl.stem for part in labels_root.iterdir() if part.is_dir() and part.name.startswith("part_")
        for lbl in part.glob("*.txt")
    }
    pairs = {stem: v for stem, v in images.items() if v[1].is_file()}
    return {
        "pairs": pairs,
        "n_images": len(images),
        "n_labels": len(label_stems),
        "images_without_labels": sorted(set(images) - set(pairs)),
        "labels_without_images": sorted(label_stems - set(images)),
        "duplicate_names": sorted(set(dup_names)),
    }


def choose_val_videos(frames_by_video: dict, counts_by_video: dict, n_classes: int,
                      val_fraction: float, seed: int, tries: int = 400) -> set[str]:
    """T3: pick validation videos so that ~val_fraction of the frames AND of each class are held out.

    Random search: shuffle videos, fill val up to the target frame share, score how far each
    class's val share is from the target, keep the best of `tries` shuffles.
    """
    rng = random.Random(seed)
    videos = sorted(frames_by_video)
    total_frames = sum(len(f) for f in frames_by_video.values())
    class_totals = [sum(c[k] for c in counts_by_video.values()) for k in range(n_classes)]
    best, best_score = set(), float("inf")
    for _ in range(tries):
        rng.shuffle(videos)
        val, n_val = set(), 0
        for v in videos:
            if n_val >= val_fraction * total_frames:
                break
            val.add(v)
            n_val += len(frames_by_video[v])
        score = 0.0
        for k in range(n_classes):
            if class_totals[k] < 20:
                continue  # too rare to balance; it lands wherever its videos land
            share = sum(counts_by_video[v][k] for v in val) / class_totals[k]
            score += abs(share - val_fraction) + (1.0 if share == 0 else 0.0)
        if score < best_score:
            best, best_score = set(val), score
    return best


def write_yolo_dataset(out: Path, split_frames: dict, labels: dict, class_names: list[str]) -> None:
    """Write images/{train,val} symlinks, cleaned labels/{train,val} files, data.yaml and data_smoke.yaml."""
    if out.exists():
        shutil.rmtree(out)  # the output is fully generated by this script
    for split, stems in split_frames.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for stem in stems:
            img_path, boxes = labels[stem]
            (out / "images" / split / img_path.name).symlink_to(img_path)
            (out / "labels" / split / f"{stem}.txt").write_text(
                "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes),
                encoding="utf-8")
    names = "\n".join(f"  {i}: {json.dumps(n)}" for i, n in enumerate(class_names))
    (out / "data.yaml").write_text(
        f"# Generated by training/prepare_dognellaf.py — do not edit by hand.\n"
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    # Smoke test (T4): ~5% of each split, as image lists (labels are found next to them).
    rng = random.Random(0)
    for split in ("train", "val"):
        files = sorted(str(out / "images" / split / labels[s][0].name) for s in split_frames[split])
        pick = rng.sample(files, max(1, len(files) // 20))
        (out / f"smoke_{split}.txt").write_text("\n".join(sorted(pick)) + "\n", encoding="utf-8")
    (out / "data_smoke.yaml").write_text(
        f"# 5% subset for the smoke test (T4). Generated — do not edit.\n"
        f"path: {out}\ntrain: smoke_train.txt\nval: smoke_val.txt\nnames:\n{names}\n", encoding="utf-8")
    # Full run (T5): validating 3k images after every epoch costs ~2.7 min; a fixed 1,000-image
    # subset keeps best.pt selection honest at a third of the cost. Final numbers use the full val split.
    val_files = sorted(str(out / "images" / "val" / labels[s][0].name) for s in split_frames["val"])
    subset = sorted(random.Random(1).sample(val_files, min(1000, len(val_files))))
    (out / "val_subset.txt").write_text("\n".join(subset) + "\n", encoding="utf-8")
    (out / "data_train.yaml").write_text(
        f"# Full training run (T5): all train frames, 1,000-frame val subset per epoch. Generated — do not edit.\n"
        f"path: {out}\ntrain: images/train\nval: val_subset.txt\nnames:\n{names}\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="dataset images/ folder (with part_* subfolders)")
    ap.add_argument("--labels", type=Path, required=True, help="dataset labels/ folder (with part_* subfolders)")
    ap.add_argument("--classes", type=Path, required=True, help="classes.txt")
    ap.add_argument("--out", type=Path, required=True, help="output YOLO dataset folder (recreated)")
    ap.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[1] / "eval" / "results")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--keep-every", type=int, default=2, help="keep every Nth frame of a video")
    ap.add_argument("--rare-threshold", type=int, default=300,
                    help="frames containing a class with fewer instances than this are always kept")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inventory-only", action="store_true", help="run T1+T2 only, write no dataset")
    args = ap.parse_args()

    class_names = read_classes(args.classes)
    n_classes = len(class_names)

    # ---- T1: completeness -------------------------------------------------------------------
    t1 = check_completeness(args.images.expanduser(), args.labels.expanduser())
    labels, empty, dropped_total, clipped_total = {}, 0, 0, 0
    for stem, (img, lbl) in t1["pairs"].items():
        boxes, dropped, clipped = parse_label(lbl, n_classes)
        labels[stem] = (img, boxes)
        empty += not boxes
        dropped_total += dropped
        clipped_total += clipped
    print(f"T1  images={t1['n_images']} labels={t1['n_labels']} complete_pairs={len(labels)} "
          f"images_without_labels={len(t1['images_without_labels'])} "
          f"labels_without_images={len(t1['labels_without_images'])} "
          f"duplicate_names={len(t1['duplicate_names'])} empty_label_files={empty} "
          f"dropped_lines={dropped_total} clipped_boxes={clipped_total}")

    # ---- T2: inventory ----------------------------------------------------------------------
    frames_by_video = collections.defaultdict(list)
    counts_by_video = collections.defaultdict(lambda: [0] * n_classes)
    inst, frames, videos = [0] * n_classes, [0] * n_classes, [set() for _ in range(n_classes)]
    for stem, (_, boxes) in labels.items():
        vid = video_id(stem)
        frames_by_video[vid].append(stem)
        for k in {b[0] for b in boxes}:
            frames[k] += 1
            videos[k].add(vid)
        for b in boxes:
            inst[b[0]] += 1
            counts_by_video[vid][b[0]] += 1
    per_class = [{"id": k, "name": class_names[k], "instances": inst[k], "frames": frames[k],
                  "videos": len(videos[k])} for k in range(n_classes)]

    required = []
    for code, key in REQUIRED_SIGNS.items():
        ks = [k for k, n in enumerate(class_names) if n == code or (code == "3.24" and n.startswith("3.24"))]
        n_inst = sum(inst[k] for k in ks)
        n_vid = len(set().union(*(videos[k] for k in ks))) if ks else 0
        status = "missing" if not ks else ("low" if n_inst < MIN_INSTANCES or n_vid < MIN_VIDEOS else "ok")
        required.append({"code": code, "key": key, "classes": [class_names[k] for k in ks],
                         "instances": n_inst, "videos": n_vid, "status": status})
    print(f"T2  source videos={len(frames_by_video)}  required signs: "
          f"ok={sum(r['status'] == 'ok' for r in required)} low={sum(r['status'] == 'low' for r in required)} "
          f"missing={[r['code'] for r in required if r['status'] == 'missing']}")

    inventory = {
        "t1": {k: (len(v) if isinstance(v, list) else v) for k, v in t1.items() if k != "pairs"}
              | {"complete_pairs": len(labels), "empty_label_files": empty,
                 "dropped_label_lines": dropped_total, "clipped_boxes": clipped_total,
                 "images_without_labels_sample": t1["images_without_labels"][:20]},
        "source_videos": len(frames_by_video),
        "classes": per_class,
        "required_signs": required,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "dataset_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    md = ["# Dognellaf dataset inventory", "",
          "Generated by `training/prepare_dognellaf.py` (T1–T2). Codes are Russian GOST numbers.", "",
          f"- Complete image+label pairs: **{len(labels)}** of {t1['n_images']} images "
          f"({len(t1['images_without_labels'])} images have no label file and are not used)",
          f"- Empty label files (frames with no objects): {empty}; dropped label lines: {dropped_total}; "
          f"boxes clipped to the image: {clipped_total}",
          f"- Distinct source videos (derived from file names): {len(frames_by_video)}", "",
          "## Signs required by the rules engine (CLAUDE.md §11.1)", "",
          f"Status `low` = fewer than {MIN_INSTANCES} instances or fewer than {MIN_VIDEOS} source videos.", "",
          "| Code | Key | Instances | Videos | Status |", "|---|---|---:|---:|---|"]
    md += [f"| {r['code']} | {r['key']} | {r['instances']} | {r['videos']} | {r['status']} |" for r in required]
    md += ["", "## All classes", "", "| Id | Class | Instances | Frames | Videos |", "|---:|---|---:|---:|---:|"]
    md += [f"| {c['id']} | {c['name']} | {c['instances']} | {c['frames']} | {c['videos']} |"
           for c in sorted(per_class, key=lambda c: -c["instances"])]
    (args.report_dir / "dataset_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if args.inventory_only:
        return 0

    # ---- T3: split by video + subsample -----------------------------------------------------
    val_videos = choose_val_videos(frames_by_video, counts_by_video, n_classes, args.val_fraction, args.seed)
    rare = {k for k in range(n_classes) if inst[k] < args.rare_threshold}
    split_frames = {"train": [], "val": []}
    for vid, stems in frames_by_video.items():
        for i, stem in enumerate(sorted(stems, key=natural_key)):
            has_rare = any(b[0] in rare for b in labels[stem][1])
            if i % args.keep_every == 0 or has_rare:
                split_frames["val" if vid in val_videos else "train"].append(stem)
    out = args.out.expanduser()
    write_yolo_dataset(out, split_frames, labels, class_names)
    (out / "split.json").write_text(json.dumps(
        {"val_videos": sorted(val_videos), "seed": args.seed, "val_fraction": args.val_fraction,
         "keep_every": args.keep_every, "rare_threshold": args.rare_threshold,
         "rare_classes": sorted(class_names[k] for k in rare)}, indent=2), encoding="utf-8")
    n_tr, n_va = len(split_frames["train"]), len(split_frames["val"])
    print(f"T3  val videos={len(val_videos)}/{len(frames_by_video)}  train frames={n_tr}  val frames={n_va} "
          f"({n_va / max(1, n_tr + n_va):.1%})  rare classes always kept={len(rare)}  -> {out}/data.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())

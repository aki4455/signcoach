"""Export a grid of example crops per class, so a human can check what a class really shows.

Why: the Dognellaf dataset encodes speed limits as classes 3.24.1 … 3.24.11 without saying which
km/h value each one is. The class map (config/class_maps/dognellaf_y11s.yaml) must get those
values from evidence, not from guessing — these grids are that evidence.

For each requested class it takes the largest boxes (most legible), at most one per source video
first, crops them with a margin and tiles them into one PNG per class.

    uv run python -I training/class_crops.py --data ~/signcoach-data/datasets/dognellaf_yolo \
        --classes 3.24.1 3.24.2 --out /tmp/crops
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE = 112  # px per crop in the grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="dataset images/ folder with part_* subfolders")
    ap.add_argument("--labels", type=Path, required=True, help="dataset labels/ folder with part_* subfolders")
    ap.add_argument("--class-file", type=Path, required=True, help="classes.txt")
    ap.add_argument("--classes", nargs="+", required=True, help="class names to export")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12, help="crops per class")
    args = ap.parse_args()

    names = [l.strip() for l in args.class_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    wanted = {names.index(c): c for c in args.classes}
    found: dict[int, list[tuple[float, Path, tuple[float, float, float, float]]]] = {k: [] for k in wanted}
    for part in sorted(args.labels.iterdir()):
        if not part.name.startswith("part_"):
            continue
        for lbl in part.glob("*.txt"):
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) == 5 and int(p[0]) in wanted:
                    x, y, w, h = map(float, p[1:])
                    img = args.images / part.name / f"{lbl.stem}.jpg"
                    found[int(p[0])].append((w * h, img, (x, y, w, h)))

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for k, cname in wanted.items():
        # Largest boxes first, but spread over different source frames' name prefixes.
        items, seen_prefix = [], set()
        for area, img, box in sorted(found[k], key=lambda t: -t[0]):
            prefix = img.stem.split("__")[0]
            if prefix in seen_prefix and len(found[k]) > args.n * 3:
                continue
            seen_prefix.add(prefix)
            items.append((img, box))
            if len(items) == args.n:
                break
        tiles = []
        for img_path, (x, y, w, h) in items:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            H, W = im.shape[:2]
            m = 0.25  # margin around the box
            x1, y1 = int(max(0, (x - w / 2 - w * m) * W)), int(max(0, (y - h / 2 - h * m) * H))
            x2, y2 = int(min(W, (x + w / 2 + w * m) * W)), int(min(H, (y + h / 2 + h * m) * H))
            tiles.append(cv2.resize(im[y1:y2, x1:x2], (TILE, TILE)))
        while len(tiles) % 6:
            tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
        grid = np.vstack(rows) if rows else np.zeros((TILE, TILE * 6, 3), np.uint8)
        out_file = args.out / f"{cname}.png"
        cv2.imwrite(str(out_file), grid)
        summary[cname] = {"instances": len(found[k]), "shown": len(items), "file": str(out_file)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Privacy spot-check gallery (CLAUDE.md M5, hard rule 4).

For every lesson of an instance, makes one contact sheet: frames spread over the annotated video
plus every thumbnail — exactly what the website serves. A person looks through the sheets for any
readable face or number plate before sample lessons go public. The sheets show blurred output only
and are written to $SIGNCOACH_DATA/privacy_gallery/ (not to git).

    SIGNCOACH_INSTANCE=prod uv run python eval/privacy_gallery.py [--frames 12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.settings import get_settings

TILE_W = 640


def sheet(lesson_dir: Path, n_frames: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(lesson_dir / "annotated.mp4"))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for i in np.linspace(0, max(0, total - 1), num=min(n_frames, max(total, 1))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            tiles.append(cv2.resize(frame, (TILE_W, int(TILE_W * frame.shape[0] / frame.shape[1]))))
    cap.release()
    if not tiles:
        return None
    rows = [np.hstack(tiles[i:i + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[i:i + 2])))
            for i in range(0, len(tiles), 2)]
    thumbs = [cv2.resize(cv2.imread(str(p)), (160, 160)) for p in sorted((lesson_dir / "thumbs").glob("*.jpg"))]
    if thumbs:
        per_row = (2 * TILE_W) // 160
        for i in range(0, len(thumbs), per_row):
            chunk = thumbs[i:i + per_row]
            chunk += [np.zeros((160, 160, 3), np.uint8)] * (per_row - len(chunk))
            rows.append(np.hstack(chunk))
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()
    s = get_settings()
    out = s.data_dir / "privacy_gallery" / s.instance
    out.mkdir(parents=True, exist_ok=True)
    for lesson_dir in sorted(p for p in s.media_dir.iterdir() if (p / "annotated.mp4").exists()):
        img = sheet(lesson_dir, args.frames)
        if img is not None:
            cv2.imwrite(str(out / f"{lesson_dir.name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{lesson_dir.name}: {out / (lesson_dir.name + '.jpg')}")


if __name__ == "__main__":
    main()

"""Download every pretrained model file into $SIGNCOACH_DATA/models and check its sha256 (CLAUDE.md §14).

Weights are never committed to git (hard rule 8); this script makes a fresh machine reproducible.
Each file has a pinned sha256 (recorded in DATA_SOURCES.md); a mismatch stops the script.
The fine-tuned sign detector is produced by training/ and is not downloaded here.

    uv run python scripts/download_models.py            # download what is missing
    uv run python scripts/download_models.py --names    # also print the placeholder's class names
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from app.settings import get_settings

MODELS = {
    "yolo11s.pt": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
                   "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"),
    "yolo11n.pt": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                   "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    "streetsignsense-yolo12s.pt": (
        "https://huggingface.co/AlessandroFerrante/StreetSignSenseY12s/resolve/main/streetsignsense-yolo12s.pt",
        "ea716000d066b3c48de91583210e8934de1865cb1956857c9c0fb89f7ec9ef9e"),
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", action="store_true", help="print the placeholder detector's class names")
    args = ap.parse_args()
    models_dir = get_settings().models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODELS.items():
        path = models_dir / name
        if not path.exists():
            print(f"downloading {name} ...")
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        digest = sha256(path)
        if digest != expected:
            raise SystemExit(f"sha256 mismatch for {name}: {digest} != {expected}")
        path.with_name(name + ".sha256").write_text(f"{digest}  {name}\n")
        print(f"ok  {name}  {digest[:12]}")
    if args.names:
        from ultralytics import YOLO
        print(YOLO(str(models_dir / "streetsignsense-yolo12s.pt")).names)


if __name__ == "__main__":
    main()

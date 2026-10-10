"""Load the SYNTHETIC sample report (tests/fixtures/report.sample.json) as a lesson in a dev instance,
so the report page can be checked with every event type. Refuses to run on production: the fixture
is made up and must never be shown as a real lesson (CLAUDE.md hard rule 1).

    SIGNCOACH_INSTANCE=dev uv run python scripts/dev_fixture_lesson.py   # -> /lessons/sample-fixture
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app import db
from app.settings import get_settings

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "report.sample.json"


def main() -> None:
    s = get_settings()
    if s.instance == "prod":
        raise SystemExit("refusing to load a synthetic fixture into the production instance")
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    lesson_id = report["lesson"]["id"]
    out = s.media_dir / lesson_id
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    duration = report["lesson"]["duration_s"]
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size=1280x720:rate=15:duration={duration}", "-vf",
                    "drawtext=text='SYNTHETIC FIXTURE':fontsize=48:fontcolor=white:x=40:y=40:box=1:boxcolor=black@0.6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "annotated.mp4")],
                   check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", "5", "-i", str(out / "annotated.mp4"),
                    "-frames:v", "1", str(out / "poster.jpg")], check=True)
    for e in report["events"]:
        if e["thumbnail"]:
            img = np.full((128, 128, 3), 235, np.uint8)
            cv2.putText(img, e["sign_code"] or "", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
            cv2.imwrite(str(out / "thumbs" / Path(e["thumbnail"]).name), img)
    (out / "report.json").write_text(json.dumps(report), encoding="utf-8")
    db.init_db(s.db_path)
    db.add_lesson(s.db_path, report["lesson"], out / "report.json", out / "annotated.mp4")
    print(f"dev lesson ready: /lessons/{lesson_id}  (instance {s.instance})")


if __name__ == "__main__":
    main()

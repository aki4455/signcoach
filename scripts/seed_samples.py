"""Create the sample lessons everyone sees on the website (CLAUDE.md M8, hard rule 1).

Each sample video goes through the real pipeline — `process_job`, exactly like an upload — and the
lesson is marked as a sample. Nothing in the reports is edited by hand. After a full run the site's
samples are exactly SAMPLES: older samples (also ones no longer listed) are removed with their
generated media; `--only` re-seeds just the named files. Run it against the server's instance, with
the GPU free:

    SIGNCOACH_INSTANCE=prod SIGNCOACH_DEVICE=cuda:0 uv run python scripts/seed_samples.py

Then check the blur on every frame sheet: SIGNCOACH_INSTANCE=prod uv run python eval/privacy_gallery.py
"""

from __future__ import annotations

import argparse
import secrets
import shutil
import time
from pathlib import Path

from app import db
from app.settings import get_settings
from pipeline.catalogue import Catalogue
from pipeline.run import JobInput, Models, process_job

CLIPS = Path("~/signcoach-data/datasets/own").expanduser()
SAMPLES = [  # file under CLIPS, lesson label — Tashkent clips used with the owners' consent (DATA_SOURCES.md)
    ("tashkent/UZB_Toshkent_Qf6QPtOhC7I_s17-137.mp4", "Tashkent, Islam Karimov street"),
    ("tashkent/UZB_Toshkent_nxg4Fm4VazQ_s0-120.mp4", "Tashkent, Alisher Navoi street"),
    ("tashkent/UZB_Toshkent_EG0g-sB1lOk_s0-120.mp4", "Tashkent, downtown Presidential track"),
    ("tashkent/UZB_Toshkent_Fr526_8i3r8_s7-127.mp4", "Tashkent, Qoraqamish"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="file names to (re)seed; default: all samples")
    args = ap.parse_args()
    s = get_settings()
    s.ensure_dirs()
    db.init_db(s.db_path)
    models = Models.load(s.cfg, s.models_dir, s.device)
    wanted = [x for x in SAMPLES if not args.only or x[0] in args.only]

    with db.session(s.db_path) as con:  # remove the samples being replaced (all of them on a full run)
        for row in con.execute("SELECT id, label FROM lessons WHERE is_sample = 1").fetchall():
            if not args.only or any(row["label"] == label for _, label in wanted):
                shutil.rmtree(s.media_dir / row["id"], ignore_errors=True)
                con.execute("DELETE FROM lessons WHERE id = ?", (row["id"],))

    for name, label in wanted:
        job_id, lesson_id = f"sample-{secrets.token_hex(4)}", secrets.token_urlsafe(16)
        upload = s.jobs_dir / job_id / f"upload{Path(name).suffix}"
        upload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(CLIPS / name, upload)
        t = time.monotonic()
        report = process_job(JobInput(job_id, lesson_id, upload, "Sample lesson", label, db.now()), s.cfg, models,
                             Catalogue.load(), s.media_dir, s.jobs_dir, s.models_dir, s.git_sha,
                             lambda stage, p: None, time.monotonic() + 3600)
        report.lesson["is_sample"] = True
        (s.media_dir / lesson_id / "report.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
        db.add_lesson(s.db_path, report.lesson, s.media_dir / lesson_id / "report.json",
                      s.media_dir / lesson_id / "annotated.mp4", is_sample=True)
        print(f"{label}: lesson {lesson_id} in {time.monotonic() - t:.0f} s — {report.summary}")


if __name__ == "__main__":
    main()

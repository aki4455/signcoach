"""Generate ``tests/fixtures/report.sample.json`` — a SYNTHETIC report for frontend development.

Why: the report page (M4) must be built before the pipeline (M1) produces real reports. This
fixture validates against ``pipeline.types.Report`` and contains every event type, so the frontend
can render all of them. It is built from the Pydantic models (not typed by hand) so it is always
valid and can be regenerated after a contract change:

    uv run python tests/fixtures/make_report_sample.py

Everything in it is made up — a 95-second lesson with a stop sign not respected, speeding, a turn
against a "no right turn" sign, etc. It must never be served as a lesson or quoted as a result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pipeline.types import LessonEvent, MotionSample, Report, RuleState, SignEvent

DURATION = 95.0
LESSON = "sample-fixture"
ERR_A, ERR_B = 3.0, 0.10  # speed error model a + b·v (km/h)


def speed_at(t: float) -> float | None:
    """Made-up speed profile (km/h) that triggers every rule once."""
    pieces = [  # (t_start, t_end, v_start, v_end)
        (0, 3, 0, 40), (3, 16, 42, 47), (16, 19, 45, 14), (19, 22, 14, 55), (22, 27, 55, 72),
        (27, 34, 72, 78), (34, 39, 76, 64), (39, 45, 64, 66), (45, 53, 60, 20), (53, 56, 19, 19),
        (56, 62, 20, 40), (62, 79, 38, 42), (79, 80.5, 40, 0), (80.5, 82.5, 0, 0), (82.5, 85, 0, 25),
    ]
    for t0, t1, v0, v1 in pieces:
        if t0 <= t < t1:
            return round(v0 + (v1 - v0) * (t - t0) / (t1 - t0), 1)
    return None  # 85–95 s: estimate too poor -> "speed not checked"


def motion() -> list[MotionSample]:
    samples, heading = [], 0.0
    for i in range(int(DURATION * 2)):
        t = i / 2
        v = speed_at(t)
        yaw = 28.0 if 53 <= t < 56 else 0.0  # a right turn of ~84°
        heading += yaw * 0.5
        samples.append(MotionSample(
            t=t, speed_kmh=v, speed_err_kmh=None if v is None else round(ERR_A + ERR_B * v, 1),
            source="video", quality=0.2 if v is None else round(0.75 + 0.1 * math.sin(t), 2),
            yaw_rate_dps=yaw, heading_deg=round(heading, 1), stopped=v is not None and v < 2.0))
    return samples


def sign(i: int, code: str, value: int | None, t_pass: float, side="right", relevant=True,
         reason="right side, facing us, approaching") -> SignEvent:
    return SignEvent(id=f"s{i}", sign_code=code, value=value, t_first=round(t_pass - 2.6, 1),
                     t_last=t_pass, t_pass=t_pass, conf=0.82, n_frames=39, side=side, relevant=relevant,
                     relevance_reason=reason, conditional=False, best_frame_idx=int((t_pass - 0.8) * 15),
                     best_bbox=(1480.0, 310.0, 1560.0, 392.0))


SIGNS = [
    sign(1, "3.24", 60, 4.2), sign(2, "5.19.1", None, 12.0), sign(3, "2.5", None, 18.5),
    sign(4, "1.23", None, 33.0), sign(5, "3.18.1", None, 48.0),
    sign(6, "3.27", None, 52.0, side="left", relevant=False, reason="left side and facing a side road"),
    sign(7, "4.1.1", None, 60.2), sign(8, "3.1", None, 70.5), sign(9, "3.24", 40, 75.0), sign(10, "2.5", None, 80.0),
]


def event(i: int, t: float, type_: str, severity: str, code: str | None = None, value: int | None = None,
          detail: dict | None = None, voice: str | None = None, sign_id: str | None = None) -> LessonEvent:
    title = {"sign_explained": "event.sign_explained", "info": "event.speed_not_checked"}.get(type_, f"event.{type_}")
    return LessonEvent(id=f"e{i}", t=t, type=type_, severity=severity, sign_code=code, value=value,
                       title_key=title, detail=detail or {}, clip=(max(0.0, t - 3), min(DURATION, t + 5)),
                       voice_key=voice, thumbnail=f"/media/{LESSON}/thumbs/e{i}.jpg" if sign_id else None,
                       sign_event_id=sign_id)


EVENTS = [
    event(1, 4.2, "sign_explained", "info", "3.24", 60, {"sign_code": "3.24", "value": 60}, "speed_limit_60", "s1"),
    event(2, 12.0, "sign_explained", "info", "5.19.1", None, {"sign_code": "5.19.1"}, "pedestrian_crossing", "s2"),
    event(3, 18.5, "sign_explained", "info", "2.5", None, {"sign_code": "2.5"}, "stop", "s3"),
    event(4, 19.0, "stop_not_made", "violation", "2.5", None, {"min_speed_kmh": 14.0, "window_s": [-2, 6]},
          "stop_not_made", "s3"),
    event(5, 27.3, "speeding", "violation", "3.24", 60,
          {"limit_kmh": 60, "max_speed_kmh": 78.0, "min_speed_kmh": 72.1, "duration_s": 6.5, "err_kmh": 10.8},
          "speeding", "s1"),
    event(6, 33.0, "sign_explained", "info", "1.23", None, {"sign_code": "1.23"}, "children", "s4"),
    event(7, 40.1, "possible_speeding", "possible", "3.24", 60,
          {"limit_kmh": 60, "max_speed_kmh": 66.0, "min_speed_kmh": 64.2, "duration_s": 4.0, "err_kmh": 9.6},
          None, "s1"),
    event(8, 48.0, "sign_explained", "info", "3.18.1", None, {"sign_code": "3.18.1"}, "no_right_turn", "s5"),
    event(9, 55.0, "turn_ban_violation", "violation", "3.18.1", None,
          {"kind": "ban", "ban": "right", "turn_direction": "right", "angle_deg": 84.0}, "turn_ban_violation", "s5"),
    event(10, 60.2, "sign_explained", "info", "4.1.1", None, {"sign_code": "4.1.1"}, "mandatory_straight", "s7"),
    event(11, 70.5, "sign_explained", "info", "3.1", None, {"sign_code": "3.1"}, "no_entry", "s8"),
    event(12, 70.6, "no_entry_possible", "possible", "3.1", None, {}, None, "s8"),
    event(13, 75.0, "sign_explained", "info", "3.24", 40, {"sign_code": "3.24", "value": 40}, "speed_limit_40", "s9"),
    event(14, 80.0, "stop_ok", "complied", "2.5", None, {"min_speed_kmh": 0.0}, None, "s10"),
    event(15, 85.0, "info", "info", None, None, {"from_s": 85.0, "to_s": 95.0}),
    event(16, 86.0, "turn_ban_complied", "complied", "4.1.1", None,
          {"kind": "mandatory", "allowed": ["straight"], "ended_by": "max_distance"}, None, "s7"),
]

TIMELINE = [
    RuleState(t=0.0, speed_limit_kmh=None, limit_source="unknown", limit_sign_event_id=None, active_bans=[], mandatory=[]),
    RuleState(t=4.2, speed_limit_kmh=60, limit_source="sign", limit_sign_event_id="s1", active_bans=[], mandatory=[]),
    RuleState(t=48.0, speed_limit_kmh=60, limit_source="sign", limit_sign_event_id="s1", active_bans=["right"], mandatory=[]),
    RuleState(t=55.0, speed_limit_kmh=None, limit_source="unknown", limit_sign_event_id=None, active_bans=[], mandatory=[]),
    RuleState(t=60.2, speed_limit_kmh=None, limit_source="unknown", limit_sign_event_id=None, active_bans=[], mandatory=["straight"]),
    RuleState(t=75.0, speed_limit_kmh=40, limit_source="sign", limit_sign_event_id="s9", active_bans=[], mandatory=["straight"]),
    RuleState(t=86.0, speed_limit_kmh=40, limit_source="sign", limit_sign_event_id="s9", active_bans=[], mandatory=[]),
]


def build() -> Report:
    violations = sum(e.severity == "violation" for e in EVENTS)
    possible = sum(e.severity == "possible" for e in EVENTS)
    return Report(
        lesson={"id": LESSON, "student": "Demo student", "label": "Synthetic fixture — not produced by the pipeline",
                "created_at": "2026-10-10T17:00:00Z", "duration_s": DURATION, "is_sample": False},
        media={"annotated_video_url": f"/media/{LESSON}/annotated.mp4", "poster_url": f"/media/{LESSON}/poster.jpg"},
        speed_source="video",
        calibration={"scale_k": 1.0, "err_model": {"a": ERR_A, "b": ERR_B}, "camera_height_m": 1.3,
                     "hfov_deg": 70, "calibrated": False},
        summary={"score": max(0, 100 - 10 * violations - 3 * possible),
                 "signs_passed": sum(s.relevant for s in SIGNS), "violations": violations, "possible": possible,
                 "complied": sum(e.severity == "complied" for e in EVENTS)},
        events=EVENTS, timeline=TIMELINE, motion=motion(), signs=SIGNS,
        signs_not_met=["2.4", "2.1", "4.3", "5.20", "1.17"],
        quality_warnings=["warning.speed_not_checked_partial", "warning.camera_uncalibrated"],
        processing={"model": "fixture", "model_version": None, "device": "cpu", "git_sha": "fixture",
                    "seconds_per_stage": {"ingest": 3.1, "analysis": 41.0, "rules": 0.02, "render": 18.4, "report": 0.3},
                    "total_seconds": 62.8, "fps_analysed": 33.0, "peak_vram_mb": None,
                    "note": "Synthetic fixture for frontend development; every number here is made up."},
    )


if __name__ == "__main__":
    out = Path(__file__).with_name("report.sample.json")
    out.write_text(json.dumps(build().model_dump(mode="json"), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")

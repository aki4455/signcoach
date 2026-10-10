"""Assemble report.json (CLAUDE.md §9 `Report`, §10.10).

Collects everything the report page needs: lesson info, media links, events, the rule timeline,
a 2 Hz copy of the motion data, every sign passed, the summary (score is "indicative"), the
signs this student has not met yet, quality warnings and processing statistics (seconds per
stage, frames per second, peak GPU memory, model and code version) — the last ones are our speed
and cost evidence.
"""

from __future__ import annotations

from pipeline.types import LessonEvent, MotionSample, Report, RuleState, SignEvent


def summarize(events: list[LessonEvent], signs: list[SignEvent], weights: dict) -> dict:
    violations = sum(e.severity == "violation" for e in events)
    possible = sum(e.severity == "possible" for e in events)
    return {"score": max(0, 100 - weights["violation"] * violations - weights["possible"] * possible),
            "signs_passed": sum(s.relevant for s in signs), "violations": violations, "possible": possible,
            "complied": sum(e.severity == "complied" for e in events)}


def downsample(motion: list[MotionSample], hz: float = 2.0) -> list[MotionSample]:
    """Keep the sample closest to every 1/hz seconds (the UI graph needs no more)."""
    out, next_t = [], 0.0
    for m in motion:
        if m.t + 1e-6 >= next_t:
            out.append(m)
            next_t = m.t + 1 / hz
    return out


def build(*, lesson: dict, media: dict, speed_source: str, calibration: dict, events: list[LessonEvent],
          timeline: list[RuleState], motion: list[MotionSample], signs: list[SignEvent], signs_not_met: list[str],
          quality_warnings: list[str], processing: dict, score_weights: dict) -> Report:
    unique_warnings = list(dict.fromkeys(quality_warnings))
    return Report(lesson=lesson, media=media, speed_source=speed_source, calibration=calibration,
                  summary=summarize(events, signs, score_weights), events=events, timeline=timeline,
                  motion=downsample(motion), signs=signs, signs_not_met=signs_not_met,
                  quality_warnings=unique_warnings, processing=processing)

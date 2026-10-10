"""Rules engine: relevant signs + motion -> lesson events and the rules in force over time (CLAUDE.md §11.4).

A pure, deterministic function — no I/O, no randomness — so every behaviour is covered by pytest
scenarios. It walks through the lesson in time order and keeps a `RuleState` (speed limit in force,
active turn bans, mandatory directions); a snapshot is stored every time the state changes.

M1 implements the part that needs only signs: explaining each new sign (code + value) once and the
speed-limit timeline. M2 adds speeding, stops, turn bans, no entry and conditional downgrades.
"""

from __future__ import annotations

from pipeline.catalogue import Catalogue
from pipeline.types import LessonEvent, MotionSample, RuleState, SignEvent, TurnEvent


def _default_limit(catalogue: Catalogue) -> tuple[int | None, str]:
    d = catalogue.rules.get("defaults", {})
    inside = d.get("assume_in_settlement", True)
    limit = d.get("speed_limit_kmh", {}).get("in_settlement" if inside else "outside_settlement")
    return (limit, "default") if limit is not None else (None, "unknown")


def run(sign_events: list[SignEvent], motion: list[MotionSample], turns: list[TurnEvent],
        catalogue: Catalogue, settings: dict) -> tuple[list[LessonEvent], list[RuleState]]:
    """Return (lesson events sorted by time, rule-state timeline)."""
    limit, source = _default_limit(catalogue)
    state = RuleState(t=0.0, speed_limit_kmh=limit, limit_source=source, limit_sign_event_id=None,
                      active_bans=[], mandatory=[])
    timeline = [state]
    events: list[LessonEvent] = []
    explained: set[tuple[str, int | None]] = set()

    relevant = sorted((s for s in sign_events if s.relevant and catalogue.rule(s.sign_code) is not None),
                      key=lambda s: s.t_pass)
    for s in relevant:
        if (s.sign_code, s.value) not in explained:
            explained.add((s.sign_code, s.value))
            events.append(LessonEvent(
                id="", t=s.t_pass, type="sign_explained", severity="info", sign_code=s.sign_code, value=s.value,
                title_key="event.sign_explained", detail={"sign_code": s.sign_code, "value": s.value},
                clip=(max(0.0, round(s.t_first - 1.0, 2)), round(s.t_pass + 2.0, 2)),
                voice_key=catalogue.voice_key(s.sign_code, s.value), thumbnail=None, sign_event_id=s.id))

        new = state.model_copy()
        for effect in catalogue.rule(s.sign_code)["effects"]:
            if effect["type"] == "set_speed_limit" and s.value is not None:
                new.speed_limit_kmh, new.limit_source, new.limit_sign_event_id = s.value, "sign", s.id
            elif effect["type"] == "end_speed_limit":
                new.speed_limit_kmh, new.limit_source = _default_limit(catalogue)
                new.limit_sign_event_id = None
        if new.model_dump(exclude={"t"}) != state.model_dump(exclude={"t"}):
            new.t = s.t_pass
            state = new
            timeline.append(state)

    events.sort(key=lambda e: e.t)
    for i, e in enumerate(events, start=1):
        e.id = f"e{i}"
    return events, timeline

"""Rules engine: relevant signs + motion + turns -> lesson events and the rules in force over time (CLAUDE.md §11.4).

A pure, deterministic function — no I/O, no randomness — so every behaviour is covered by the JSON
scenarios in tests/fixtures/rule_scenarios/. It walks through the lesson in time order, keeps the
rules in force (speed limit, turn bans, mandatory directions, inside/outside a built-up area) and
stores a `RuleState` snapshot every time they change. Then it judges speed and stops against them.

What it reports (severity decides colour and score):
- sign_explained   every relevant sign we pass (voice prompt + explanation); it is said when the sign is
                   confirmed while we approach (`announce_at`, as a system in the car could), while its
                   rule takes effect when we pass it. A sign whose code + value was already explained in
                   this lesson is marked `detail.repeat`: the voice says it again, the report lists its
                   explanation only once
- speeding         over the limit even after subtracting the speed error -> violation
- possible_speeding over the limit but within the error, or the speed/limit is uncertain: a default
                   limit we cannot confirm without settlement signs, a sign with a conditions plate,
                   or a camera whose speed estimate was never calibrated (speed.calibrated = false)
- stop_not_made / stop_ok   slowest speed around a stop sign above / below `stop.speed_kmh`;
                   no speed data there -> info "stop not checked", never a violation
- turn_ban_violation / turn_ban_complied   a turn against a no-turn sign or outside a mandatory
                   direction (a left-turn arrow also covers a U-turn, ПДД Annex 1 §4); bans end at
                   our next turn or after ban_scope distance/time (no map: approximation of "the next
                   intersection")
- no_entry_possible a relevant no-entry sign we may have driven into — always "possible": an overhead
                   sign hangs over our own road; a sign at the side closes the side road it stands at,
                   so it counts only if we turned that way around it (`no_entry` settings). Driving
                   straight past it is only explained (no map to tell which road it closes)
Violations caused by a sign with a supplementary plate (conditional) are downgraded to "possible".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.catalogue import Catalogue
from pipeline.types import LessonEvent, MotionSample, RuleState, SignEvent, TurnEvent


def _may_have_entered(sign: SignEvent, turns: list[TurnEvent], cfg: dict) -> bool:
    """Could we have driven into the road this no-entry sign closes?

    An overhead sign hangs over our own road. A sign at the side of the road usually closes the side
    road it stands at (e.g. the exit of a one-way service road): it matters only if we turned that way
    within `turn_window_s` of passing it. Without a map we cannot tell more."""
    if not cfg.get("require_turn_toward_sign", True) or sign.side == "overhead":
        return True
    before, after = cfg.get("turn_window_s", (-3.0, 8.0))
    return any(tr.direction == sign.side and tr.t_end >= sign.t_pass + before and tr.t_start <= sign.t_pass + after
               for tr in turns)


@dataclass
class _Limit:
    value: int | None
    source: str                   # sign | default | unknown
    sign_id: str | None = None
    conditional: bool = False


@dataclass
class _Ban:
    kind: str                     # "ban" (3.18.x, 3.19) | "mandatory" (4.1.x)
    sign: SignEvent
    expires_at: float             # by distance/time (approximation of "next intersection")
    ban: str | None = None        # left | right | uturn
    allowed: list[str] = field(default_factory=list)
    ended_by: str = "max_time"     # what ends it if we never turn: max_distance | max_time

    def forbids(self, direction: str) -> bool:
        if self.kind == "ban":
            return direction == self.ban
        allowed = set(self.allowed)
        if "left" in allowed:     # signs allowing a left turn also allow a U-turn (ПДД, Annex 1, §4)
            allowed.add("uturn")
        return direction not in allowed

    def detail(self) -> dict:
        return {"kind": self.kind, "ban": self.ban} if self.kind == "ban" else {"kind": self.kind, "allowed": self.allowed}


def _event(t, type_, severity, sign: SignEvent | None = None, value=None, detail=None, voice=None,
           title=None, clip=None) -> LessonEvent:
    return LessonEvent(id="", t=round(t, 3), type=type_, severity=severity,
                       sign_code=sign.sign_code if sign else None, value=value if value is not None else
                       (sign.value if sign else None), title_key=title or f"event.{type_}", detail=detail or {},
                       clip=clip or (round(max(0.0, t - 3.0), 2), round(t + 4.0, 2)), voice_key=voice,
                       thumbnail=None, sign_event_id=sign.id if sign else None)


def _expiry(t_start: float, motion: list[MotionSample], scope: dict) -> tuple[float, str]:
    """When a turn ban stops applying if we never turn: after max_distance_m driven or max_time_s."""
    t_time = t_start + scope["max_time_s"]
    dist, prev = 0.0, None
    for m in motion:
        if m.t <= t_start:
            prev = m
            continue
        if prev is not None and prev.speed_kmh is not None:
            dist += prev.speed_kmh / 3.6 * (m.t - prev.t)
            if dist >= scope["max_distance_m"] and m.t < t_time:
                return m.t, "max_distance"
        prev = m
    return t_time, "max_time"


def run(sign_events: list[SignEvent], motion: list[MotionSample], turns: list[TurnEvent],
        catalogue: Catalogue, settings: dict, duration_s: float | None = None,
        announce_at: dict[str, float] | None = None) -> tuple[list[LessonEvent], list[RuleState]]:
    """Return (lesson events sorted by time, rule-state timeline).

    `announce_at`: sign event id -> time to say/explain the sign (default: when we pass it)."""
    announce_at = announce_at or {}
    defaults = catalogue.rules.get("defaults", {})
    inside = defaults.get("assume_in_settlement", True)
    bans_enabled = settings.get("turn_bans_enabled", True)
    ban_scope = settings["ban_scope"]
    motion = sorted(motion, key=lambda m: m.t)
    end_t = duration_s if duration_s is not None else max(
        [m.t for m in motion] + [s.t_pass for s in sign_events] + [tr.t_end for tr in turns] + [0.0])

    def default_limit() -> _Limit:
        v = defaults.get("speed_limit_kmh", {}).get("in_settlement" if inside else "outside_settlement")
        return _Limit(v, "default") if v is not None else _Limit(None, "unknown")

    limit = default_limit()
    active: list[_Ban] = []
    events: list[LessonEvent] = []
    timeline: list[RuleState] = []
    limit_periods: list[tuple[float, _Limit]] = [(0.0, limit)]  # for judging speed afterwards

    def snapshot(t: float) -> None:
        state = RuleState(t=round(t, 3), speed_limit_kmh=limit.value, limit_source=limit.source,
                          limit_sign_event_id=limit.sign_id,
                          active_bans=sorted({b.ban for b in active if b.kind == "ban"}),
                          mandatory=sorted({d for b in active if b.kind == "mandatory" for d in b.allowed}))
        if timeline and state.model_dump(exclude={"t"}) == timeline[-1].model_dump(exclude={"t"}):
            return
        if timeline and timeline[-1].t == state.t:
            timeline[-1] = state
        else:
            timeline.append(state)

    def set_limit(t: float, new: _Limit) -> None:
        nonlocal limit
        limit = new
        limit_periods.append((t, new))

    def complied(b: _Ban, t: float, ended_by: str) -> None:
        if bans_enabled:
            events.append(_event(t, "turn_ban_complied", "complied", b.sign,
                                 detail=b.detail() | {"ended_by": ended_by}))

    def expire_until(t: float) -> None:
        for b in sorted(active, key=lambda b: b.expires_at):
            if b.expires_at <= t:
                active.remove(b)
                complied(b, b.expires_at, b.ended_by)
                snapshot(b.expires_at)

    snapshot(0.0)
    explained: set[tuple[str, int | None]] = set()
    relevant = [s for s in sign_events if s.relevant and catalogue.rule(s.sign_code) is not None]
    happenings = sorted([(s.t_pass, 0, s) for s in relevant] + [(tr.t_start, 1, tr) for tr in turns],
                        key=lambda h: (h[0], h[1]))

    for t, _, item in happenings:
        expire_until(t)
        if isinstance(item, SignEvent):
            s = item
            repeat = (s.sign_code, s.value) in explained
            explained.add((s.sign_code, s.value))
            events.append(_event(min(s.t_pass, announce_at.get(s.id, s.t_pass)), "sign_explained", "info", s,
                                 title="event.sign_explained",
                                 detail={"sign_code": s.sign_code, "value": s.value} | ({"repeat": True} if repeat else {}),
                                 voice=catalogue.voice_key(s.sign_code, s.value),
                                 clip=(round(max(0.0, s.t_first - 1.0), 2), round(s.t_pass + 2.0, 2))))
            # Signs that end the current speed-limit zone (another 3.24, 3.25, 3.31, settlement signs).
            if limit.source == "sign" and f"sign:{s.sign_code}" in _limit_ends(catalogue):
                set_limit(s.t_pass, default_limit())
            for effect in catalogue.rule(s.sign_code)["effects"]:
                kind = effect["type"]
                if kind == "set_speed_limit":
                    # A limit sign whose number could not be read: a new limit applies, we don't know which.
                    set_limit(s.t_pass, _Limit(s.value, "sign", s.id, s.conditional) if s.value is not None
                              else _Limit(None, "unknown", s.id))
                elif kind == "end_speed_limit":
                    set_limit(s.t_pass, default_limit())
                elif kind == "settlement":
                    inside = effect["inside"]
                    if limit.source != "sign":
                        set_limit(s.t_pass, default_limit())
                elif kind in ("add_ban", "mandatory"):
                    t_exp, why = _expiry(s.t_pass, motion, ban_scope)
                    active.append(_Ban("ban" if kind == "add_ban" else "mandatory", s, t_exp, ban=effect.get("ban"),
                                       allowed=list(effect.get("allowed", [])), ended_by=why))
                elif kind == "require_stop":
                    events.append(_judge_stop(s, motion, settings))
                elif kind == "no_entry" and _may_have_entered(s, turns, settings.get("no_entry", {})):
                    events.append(_event(s.t_pass + 0.01, "no_entry_possible", "possible", s,
                                         detail={"reason": "conditional"} if s.conditional else {}))
            snapshot(s.t_pass)
        else:
            tr: TurnEvent = item
            for b in list(active):
                if b.forbids(tr.direction):
                    if bans_enabled:
                        events.append(_event(
                            tr.t_start, "turn_ban_violation", "possible" if b.sign.conditional else "violation",
                            b.sign, detail=b.detail() | {"turn_direction": tr.direction,
                                                         "angle_deg": round(tr.angle_deg, 1)}
                            | ({"reason": "conditional"} if b.sign.conditional else {}),
                            voice="turn_ban_violation" if not b.sign.conditional else None,
                            clip=(round(max(0.0, tr.t_start - 4.0), 2), round(tr.t_end + 2.0, 2))))
                elif ban_scope.get("until_turn", True):
                    complied(b, tr.t_start, "turn")
                if ban_scope.get("until_turn", True):
                    active.remove(b)
            if limit.source == "sign" and "turn" in _limit_ends(catalogue):
                set_limit(tr.t_start, default_limit())
            snapshot(tr.t_start)

    expire_until(end_t)
    events += _judge_speed(motion, limit_periods, settings)
    events.sort(key=lambda e: (e.t, e.type != "sign_explained"))
    for i, e in enumerate(events, start=1):
        e.id = f"e{i}"
    return events, timeline


def _limit_ends(catalogue: Catalogue) -> list[str]:
    """What ends a speed-limit zone (rules.json, sign 3.24 scope)."""
    return ((catalogue.rule("3.24") or {}).get("scope") or {}).get("ends_on", [])


def _judge_stop(s: SignEvent, motion: list[MotionSample], settings: dict) -> LessonEvent:
    w0, w1 = settings["stop"]["window_s"]
    speeds = [m.speed_kmh for m in motion if s.t_pass + w0 <= m.t <= s.t_pass + w1 and m.speed_kmh is not None]
    if not speeds:
        return _event(s.t_pass, "info", "info", s, title="event.stop_not_checked")
    slowest = round(min(speeds), 1)
    if slowest > settings["stop"]["speed_kmh"]:
        return _event(s.t_pass, "stop_not_made", "possible" if s.conditional else "violation", s,
                      detail={"min_speed_kmh": slowest, "window_s": [w0, w1]}
                      | ({"reason": "conditional"} if s.conditional else {}),
                      voice=None if s.conditional else "stop_not_made")
    return _event(s.t_pass, "stop_ok", "complied", s, detail={"min_speed_kmh": slowest})


def _limit_at(periods: list[tuple[float, _Limit]], t: float) -> _Limit:
    current = periods[0][1]
    for start, lim in periods:
        if start <= t:
            current = lim
    return current


def _judge_speed(motion: list[MotionSample], periods: list[tuple[float, _Limit]], settings: dict) -> list[LessonEvent]:
    """Group consecutive over-the-limit samples into speeding / possible_speeding events."""
    tol, min_dur = settings["speeding"]["tolerance_kmh"], settings["speeding"]["min_duration_s"]
    samples = [m for m in motion if m.speed_kmh is not None]
    if len(samples) < 2:
        return []
    dt = sorted(b.t - a.t for a, b in zip(samples, samples[1:]))[len(samples) // 2]  # typical sample spacing
    events, run, run_limit = [], [], None

    def close_run():
        if not run:
            return
        if run[-1].t - run[0].t + dt >= min_dur - 1e-6:
            lim = run_limit
            sure, longest, current = [m.speed_kmh - (m.speed_err_kmh or 0) > lim.value + tol for m in run], 0, []
            for i, ok in enumerate(sure):
                current = current + [run[i]] if ok else []
                if current:
                    longest = max(longest, current[-1].t - current[0].t + dt)
            certain = longest >= min_dur - 1e-6
            reason = ("conditional" if lim.conditional else "default_limit" if lim.source == "default"
                      else None if settings["speed"].get("calibrated", False) else "uncalibrated")
            violation = certain and reason is None
            detail = {"limit_kmh": lim.value, "limit_source": lim.source,
                      "max_speed_kmh": round(max(m.speed_kmh for m in run), 1),
                      "min_speed_kmh": round(min(m.speed_kmh for m in run), 1),
                      "duration_s": round(run[-1].t - run[0].t + dt, 1),
                      "err_kmh": round(max(m.speed_err_kmh or 0 for m in run), 1)}
            if reason:
                detail["reason"] = reason
            events.append(LessonEvent(
                id="", t=round(run[0].t, 3), type="speeding" if violation else "possible_speeding",
                severity="violation" if violation else "possible", sign_code="3.24" if lim.source == "sign" else None,
                value=lim.value, title_key="event.speeding" if violation else "event.possible_speeding", detail=detail,
                clip=(round(max(0.0, run[0].t - 2.0), 2), round(run[-1].t + 2.0, 2)),
                voice_key="speeding" if violation else None, thumbnail=None, sign_event_id=lim.sign_id))

    for m in samples:
        lim = _limit_at(periods, m.t)
        over = lim.value is not None and m.speed_kmh > lim.value + tol
        if over and run and (lim is not run_limit or m.t - run[-1].t > 2.5 * dt):
            close_run()
            run = []
        if over:
            run_limit = lim
            run.append(m)
        elif run:
            close_run()
            run = []
    close_run()
    return events

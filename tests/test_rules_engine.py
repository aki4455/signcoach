"""Rules-engine scenarios (CLAUDE.md §11.4).

Each JSON file in tests/fixtures/rule_scenarios/ is a lesson in miniature:
- "signs":  [{"id", "code", "value", "t", "relevant"?, "conditional"?}]  (t = moment we pass the sign)
- "speed":  [[t_start, t_end, km/h or null], ...]  -> motion sampled every 0.2 s, error = 3 + 0.1*v
- "turns":  [{"t_start", "t_end", "direction", "angle_deg"}]
- "settings" / "defaults": overrides (defaults are fixed here so the tests don't depend on rules.json)
- "expect": {"events": [{"type", "severity", "t", "detail"?}], "limit_at": [[t, km/h], ...]}
Expected events are all events except sign explanations, unless "check_explained" is true.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.settings import get_settings
from pipeline.catalogue import Catalogue
from pipeline.rules_engine import run
from pipeline.types import MotionSample, SignEvent, TurnEvent

SCENARIOS = sorted((Path(__file__).parent / "fixtures" / "rule_scenarios").glob("*.json"))
CATALOGUE = Catalogue.load()


def merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# The scenarios are written for this error model (km/h, a + b*v); calibration may change the one in
# settings.yaml, which must not change what the rules engine is tested for.
SCENARIO_ERR_MODEL = {"speed": {"err_model": {"a": 3.0, "b": 0.10}}}


def build(sc: dict):
    settings = merge(merge(get_settings().cfg, SCENARIO_ERR_MODEL), sc.get("settings", {}))
    err = settings["speed"]["err_model"]
    signs = [SignEvent(id=s["id"], sign_code=s["code"], value=s.get("value"), t_first=s["t"] - 2, t_last=s["t"],
                       t_pass=s["t"], conf=0.9, n_frames=20, side="right", relevant=s.get("relevant", True),
                       relevance_reason="scenario", conditional=s.get("conditional", False), best_frame_idx=0,
                       best_bbox=(0, 0, 1, 1)) for s in sc.get("signs", [])]
    motion = []
    for t0, t1, v in sc.get("speed", []):
        t = t0
        while t < t1 - 1e-9:
            motion.append(MotionSample(t=round(t, 2), speed_kmh=v, speed_err_kmh=None if v is None else err["a"] + err["b"] * v,
                                       source="video" if v is not None else "none", quality=0.8,
                                       yaw_rate_dps=None, heading_deg=None, stopped=v is not None and v < 2))
            t += 0.2
    turns = [TurnEvent(**t) for t in sc.get("turns", [])]
    rules = copy.deepcopy(CATALOGUE.rules)
    rules["defaults"] = merge(rules["defaults"], {"speed_limit_kmh": {"in_settlement": 60, "outside_settlement": 90},
                                                  "assume_in_settlement": True} | sc.get("defaults", {}))
    catalogue = Catalogue(signs=CATALOGUE.signs, rules=rules)
    return signs, motion, turns, catalogue, settings


@pytest.mark.parametrize("path", SCENARIOS, ids=[p.stem for p in SCENARIOS])
def test_scenario(path: Path):
    sc = json.loads(path.read_text(encoding="utf-8"))
    signs, motion, turns, catalogue, settings = build(sc)
    events, timeline = run(signs, motion, turns, catalogue, settings, duration_s=sc.get("duration"))

    got = [e for e in events if sc.get("check_explained") or e.type != "sign_explained"]
    want = sc["expect"]["events"]
    assert [(e.type, e.severity) for e in got] == [(w["type"], w["severity"]) for w in want], \
        [(e.type, e.severity, e.t, e.detail) for e in got]
    for e, w in zip(got, want):
        assert abs(e.t - w["t"]) <= 0.3, (e.type, e.t, w["t"])
        for key, value in w.get("detail", {}).items():
            assert e.detail.get(key) == value, (e.type, key, e.detail)

    for t, expected_limit in sc["expect"].get("limit_at", []):
        state = [s for s in timeline if s.t <= t][-1]
        assert state.speed_limit_kmh == expected_limit, (t, state)


def test_at_least_fifteen_scenarios_cover_every_event_type():
    assert len(SCENARIOS) >= 15
    seen = set()
    for path in SCENARIOS:
        sc = json.loads(path.read_text(encoding="utf-8"))
        seen |= {w["type"] for w in sc["expect"]["events"]}
        seen |= {"sign_explained"} if sc.get("check_explained") else set()
    assert seen >= {"sign_explained", "speeding", "possible_speeding", "stop_not_made", "stop_ok",
                    "turn_ban_violation", "turn_ban_complied", "no_entry_possible", "info"}

"""The sample report fixture must stay valid and must cover every event type (CLAUDE.md §9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

from pipeline.types import EventType, Report, Severity

FIXTURE = Path(__file__).parent / "fixtures" / "report.sample.json"


def test_fixture_validates_against_report_contract():
    Report.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_fixture_covers_every_event_type_and_severity():
    report = Report.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert {e.type for e in report.events} == set(get_args(EventType))
    assert {e.severity for e in report.events} == set(get_args(Severity))


def test_fixture_is_up_to_date_with_its_generator():
    from tests.fixtures.make_report_sample import build

    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == build().model_dump(mode="json")


def test_summary_matches_events():
    report = Report.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    violations = sum(e.severity == "violation" for e in report.events)
    possible = sum(e.severity == "possible" for e in report.events)
    assert report.summary["violations"] == violations
    assert report.summary["possible"] == possible
    assert report.summary["score"] == max(0, 100 - 10 * violations - 3 * possible)

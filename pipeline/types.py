"""Data contracts shared by every pipeline stage, the API and the frontend (CLAUDE.md §9).

These Pydantic models are the common language of the project: the detector produces
`Detection`s, voting turns tracks into `SignEvent`s, ego-motion produces `MotionSample`s and
`TurnEvent`s, the rules engine turns all of that into `LessonEvent`s and `RuleState`s, and
`Report` is what the report page reads (``report.json``).

Units: times are seconds from the start of the normalized video, speeds are km/h, boxes are
(x1, y1, x2, y2) in normalized-video pixels. Sign codes are the Russian GOST numbers used by the
training dataset (e.g. "3.24" = maximum speed limit); the km/h value of a speed sign is `value`.

Changing a model here changes what every other module and the frontend expect, so ask the team
first (CLAUDE.md §3) and keep ``tests/fixtures/report.sample.json`` valid.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in normalized-video pixels

SpeedSource = Literal["gps", "video", "none"]

EventType = Literal[
    "sign_explained",     # first time a sign (code + value) appears in the lesson: explanation + voice prompt
    "speeding",           # over the limit even after subtracting the speed error  -> severity "violation"
    "possible_speeding",  # over the limit, but within the speed error (or the sign was conditional) -> "possible"
    "stop_not_made",      # passed a stop sign without stopping -> "violation" ("possible" if conditional)
    "stop_ok",            # stopped at a stop sign -> "complied"
    "turn_ban_violation", # turned against a ban or against a mandatory-direction sign (detail.kind = "ban" | "mandatory")
    "turn_ban_complied",  # a ban / mandatory direction expired without a turn against it -> "complied"
    "no_entry_possible",  # passed a relevant no-entry sign -> always "possible", never a hard violation
    "info",               # anything else worth showing, e.g. "stop not checked" when speed was unavailable
]
Severity = Literal["info", "violation", "possible", "complied"]


class _Contract(BaseModel):
    # Unknown fields are an error: a typo in a producer should fail loudly, not silently vanish.
    model_config = ConfigDict(extra="forbid")


class Detection(_Contract):
    """One box in one analysed frame, as returned by a SignDetector (with a ByteTrack id)."""

    frame_idx: int
    t: float
    track_id: int | None
    model_class: str              # the detector's own class name, e.g. "3.24.6"
    sign_code: str | None         # catalogue code from the class map; None = not mapped (debug only)
    value: int | None             # e.g. 60 for a 60 km/h speed limit
    conf: float
    bbox: BBox


class SignEvent(_Contract):
    """One physical sign, after track voting and the relevance checks."""

    id: str
    sign_code: str
    value: int | None
    t_first: float
    t_last: float
    t_pass: float                 # last time seen = when the car passes the sign
    conf: float                   # mean confidence of the winning class
    n_frames: int
    side: Literal["right", "left", "overhead"]
    relevant: bool                # does the sign apply to our car?
    relevance_reason: str         # which heuristic decided, in plain words
    conditional: bool             # a supplementary plate was detected under it
    best_frame_idx: int
    best_bbox: BBox


class MotionSample(_Contract):
    """Speed, heading and stop state of our car at one moment."""

    t: float
    speed_kmh: float | None       # None = no trustworthy estimate at this moment
    speed_err_kmh: float | None   # expected error of speed_kmh (from the calibrated error model)
    source: SpeedSource
    quality: float                # 0..1
    yaw_rate_dps: float | None
    heading_deg: float | None
    stopped: bool


class TurnEvent(_Contract):
    """A turn detected from the heading change."""

    t_start: float
    t_end: float
    direction: Literal["left", "right", "uturn"]
    angle_deg: float


class RuleState(_Contract):
    """The rules in force from time `t` on (a snapshot is stored whenever something changes)."""

    t: float
    speed_limit_kmh: int | None
    limit_source: Literal["sign", "default", "unknown"]
    limit_sign_event_id: str | None
    active_bans: list[str]        # e.g. ["right"], ["uturn"]
    mandatory: list[str]          # allowed directions from a 4.1.x sign, e.g. ["straight", "right"]


class LessonEvent(_Contract):
    """One line in the lesson report. Severity alone decides colour and score."""

    id: str
    t: float
    type: EventType
    severity: Severity
    sign_code: str | None
    value: int | None
    title_key: str                # i18n key (config/i18n/en.json)
    detail: dict                  # parameters for the text, e.g. {"min_speed_kmh": 8}
    clip: tuple[float, float]     # suggested playback window (seconds)
    voice_key: str | None         # assets/voice/en/<voice_key>.mp3
    thumbnail: str | None         # URL of a blurred crop
    sign_event_id: str | None


class Report(_Contract):
    """Everything the report page needs for one lesson (written as media/<lesson_id>/report.json)."""

    lesson: dict                  # id, student, label, created_at, duration_s, is_sample
    media: dict                   # annotated_video_url, poster_url
    speed_source: SpeedSource
    calibration: dict             # scale_k, err_model, camera_height_m, hfov_deg, calibrated
    summary: dict                 # score, signs_passed, violations, possible, complied
    events: list[LessonEvent]
    timeline: list[RuleState]
    motion: list[MotionSample]    # downsampled to 2 Hz for the UI overlay/graph
    signs: list[SignEvent]        # every sign after voting (relevant or not) — "every sign the car passed"
    signs_not_met: list[str]      # catalogue codes never seen in this student's lessons
    quality_warnings: list[str]   # i18n keys
    processing: dict              # model, model_version, device, seconds_per_stage, fps, peak_vram_mb, git_sha

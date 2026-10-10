"""Does a detected sign apply to our car? (CLAUDE.md §10.5)

Without a map we use simple, explainable geometry, and every decision is written down in plain
words in `relevance_reason`:

- **Side.** In right-hand traffic signs for us stand on the right or hang overhead. A box whose
  centre is right of the frame centre is "right"; a high, central box is "overhead"; else "left".
- **Facing.** A round or square sign seen at an angle looks narrow; if width/height is well below
  its normal shape ratio it faces a side road, not us.
- **Approaching.** A sign we drive towards grows in the picture; one that does not grow belongs
  to something else (a parallel road, a sign we are moving away from).
- **Conditional.** A supplementary plate (time of day, vehicle type...) directly under the sign
  means the sign may not apply right now; violations from it are reported only as "possible".
"""

from __future__ import annotations

from statistics import median

from pipeline.signs import VotedSign
from pipeline.types import SignEvent

# Normal width/height of each sign shape when seen straight on.
NOMINAL_ASPECT = {"round": 1.0, "square": 1.0, "octagon": 1.0, "diamond": 1.0,
                  "triangle": 1.155, "inverted_triangle": 1.155}


def _wh(box) -> tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def _side(sign: VotedSign, W: int, H: int, r: dict) -> str:
    """Overhead if high and central; otherwise right/left from how the box moves while it grows.

    Driving forward, everything streams outwards from the centre of view: a sign on the right
    drifts right as we approach, one on the left drifts left. That motion is a better cue than the
    position alone (a camera pointing slightly sideways shifts every position). Small motion ->
    fall back to the position of the closest (largest) view.
    """
    box = sign.best.bbox
    cx = (box[0] + box[2]) / 2
    if box[3] < r["overhead_max_bottom"] * H and abs(cx - W / 2) < r["overhead_central_band"] * W / 2:
        return "overhead"
    centres = [(d.bbox[0] + d.bbox[2]) / 2 for d in sign.winning]
    k = max(1, len(centres) // 3)
    dx = median(centres[-k:]) - median(centres[:k])
    if abs(dx) >= r["side_motion_min"] * W:
        return "right" if dx > 0 else "left"
    return "right" if cx > W / 2 + r["side_margin"] * W else "left"


def _growth(sign: VotedSign) -> float:
    """Box area at the end of the track / at the start (medians of the first and last third)."""
    areas = [_wh(d.bbox)[0] * _wh(d.bbox)[1] for d in sign.winning]
    k = max(1, len(areas) // 3)
    start = median(areas[:k])
    return median(areas[-k:]) / start if start > 0 else 1.0


def _has_plate_below(sign: VotedSign, plates: list[VotedSign], max_gap_widths: float) -> bool:
    by_frame = {d.frame_idx: d.bbox for d in sign.winning}
    for plate in plates:
        for d in plate.winning:
            box = by_frame.get(d.frame_idx)
            if box is None:
                continue
            sw = box[2] - box[0]
            overlaps_x = d.bbox[0] < box[2] and d.bbox[2] > box[0]
            gap = d.bbox[1] - box[3]  # plate top minus sign bottom
            if overlaps_x and -0.25 * sw <= gap <= max_gap_widths * sw:
                return True
    return False


def assess(sign: VotedSign, sign_id: str, plates: list[VotedSign], frame_w: int, frame_h: int,
           shape: str, cfg: dict) -> SignEvent:
    """Turn a voted sign into a SignEvent with side, relevance and the reason in words."""
    r = cfg["relevance"]
    best = sign.best
    side = _side(sign, frame_w, frame_h, r)
    reasons, relevant = [], True

    if side == "left" and not r.get("left_side_relevant", False):
        relevant = False
        reasons.append("on the left side of the road")
    else:
        reasons.append("overhead" if side == "overhead" else "on the right side")

    nominal = NOMINAL_ASPECT.get(shape)
    if nominal is not None:
        w, h = _wh(best.bbox)
        if h > 0 and (w / h) / nominal < r["facing_min_aspect"]:
            relevant = False
            reasons.append("seen from the side (faces a side road)")
        else:
            reasons.append("facing us")

    growth = _growth(sign)
    if len(sign.winning) >= 3 and growth < r["approach_min_area_growth"]:
        relevant = False
        reasons.append("not getting closer")
    else:
        reasons.append("getting closer")

    conditional_plates = [p for p in plates if p.sign_code in set(r["conditional_plate_codes"])]
    conditional = _has_plate_below(sign, conditional_plates, r["plate_max_gap_widths"])
    if conditional:
        reasons.append("has a supplementary plate (conditions apply)")

    return SignEvent(
        id=sign_id, sign_code=sign.sign_code, value=sign.value, t_first=round(sign.t_first, 3),
        t_last=round(sign.t_last, 3), t_pass=round(sign.t_last, 3), conf=round(sign.conf, 3),
        n_frames=sign.n_frames, side=side, relevant=relevant, relevance_reason=", ".join(reasons),
        conditional=conditional, best_frame_idx=best.frame_idx, best_bbox=best.bbox)

"""Which areas get pixelated: our own bonnet is not a car to blur, but real cars and plates always are."""

from app.settings import get_settings
from pipeline.privacy import regions_to_hide
from pipeline.vehicles import ObjectBox

CFG = get_settings().cfg["privacy"]
W, H = 1280, 720


def hidden(*objects: ObjectBox) -> list:
    return regions_to_hide(list(objects), 1.0, CFG["plate_band"], (W, H), CFG["own_hood"])


def test_own_bonnet_is_not_blurred():
    assert hidden(ObjectBox("car", (40, 560, 1240, 720), 0.6)) == []


def test_close_car_in_front_keeps_its_plate_band():
    # A car just ahead can touch the bottom edge too, but its box reaches far up the frame.
    assert len(hidden(ObjectBox("car", (300, 300, 1000, 720), 0.9))) == 1


def test_narrow_car_at_the_bottom_corner_keeps_its_plate_band():
    assert len(hidden(ObjectBox("car", (0, 500, 400, 720), 0.8))) == 1


def test_detected_plate_is_hidden_even_on_the_bonnet_area():
    assert hidden(ObjectBox("car", (40, 560, 1240, 720), 0.6), ObjectBox("plate", (600, 650, 700, 690), 0.5)) == [(600, 650, 700, 690)]

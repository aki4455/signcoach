"""Subtitles burned into the video: prompts that overlap in time share one line."""

from pipeline.render import merge_subtitles


def test_overlapping_subtitles_share_one_line():
    subs = [(10.0, 13.0, "Main road"), (10.4, 13.4, "Pedestrian crossing"), (20.0, 23.0, "No stopping")]
    assert merge_subtitles(subs) == [(10.0, 13.4, "Main road · Pedestrian crossing"), (20.0, 23.0, "No stopping")]


def test_same_text_is_not_repeated_and_order_follows_time():
    subs = [(5.0, 8.0, "Give way"), (2.0, 5.5, "Main road"), (4.0, 7.0, "Main road")]
    assert merge_subtitles(subs) == [(2.0, 8.0, "Main road · Give way")]

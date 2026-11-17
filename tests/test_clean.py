"""Removing off-camera and stuck tracking, on tiny synthetic arrays; never touches the match data."""

import numpy as np

from possession import clean

PITCH = (100.0, 50.0)
# Pixels = pitch fractions * 1000: the picture (1920 x 1080) shows x < 192 % and y < 108 %, so
# with a negative offset on x, only x >= 50 m is on camera.
SHIFT = np.array([[1000.0, 0.0, -500.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]])


def test_on_camera_projects_through_the_homography():
    xy = np.array([[[60.0, 25.0], [40.0, 25.0], [np.nan, np.nan]]])
    inside = clean.on_camera(xy, SHIFT[None], np.array([True]), PITCH)
    assert inside.tolist() == [[True, False, False]]
    assert not clean.on_camera(xy, SHIFT[None], np.array([False]), PITCH).any()


def test_a_track_frozen_long_enough_is_still_and_a_moving_one_is_not():
    n = 100
    frozen = np.tile([10.0, 10.0], (n, 1))
    frozen[60:] += np.arange(40)[:, None] * 0.5          # starts walking at frame 60
    walking = np.stack([np.arange(n) * 0.3, np.full(n, 20.0)], axis=1)
    jitter = np.tile([30.0, 30.0], (n, 1)) + 0.1 * np.sin(np.arange(n))[:, None]
    xy = np.stack([frozen, walking, jitter], axis=1)
    out = clean.still(xy, seconds=1, radius=0.3, fps=25)  # windows of 25 frames
    assert out[:60, 0].all() and not out[61:, 0].any()
    assert not out[:, 1].any()
    assert out[:, 2].all()                                # within 0.2 m: still


def test_a_gap_breaks_a_still_window():
    xy = np.tile([10.0, 10.0], (40, 1, 1))
    xy[20] = np.nan
    assert not clean.still(xy, seconds=1, radius=0.3, fps=25).any()   # no 25 tracked frames


def test_clean_removes_and_says_why():
    n = 60
    still_on_camera = np.tile([70.0, 25.0], (n, 1))
    moving_off_camera = np.stack([np.linspace(10, 20, n), np.full(n, 25.0)], axis=1)
    moving_on_camera = np.stack([np.linspace(60, 90, n), np.full(n, 25.0)], axis=1)
    xy = np.stack([still_on_camera, moving_off_camera, moving_on_camera], axis=1)
    xy[5, 2] = np.nan
    speed = np.ones((n, 3))
    out = clean.clean(xy, speed, np.repeat(SHIFT[None], n, axis=0), np.ones(n, bool), PITCH,
                      seconds=1, radius=0.3, fps=25)
    assert (out["removed"][:, 0] == clean.STILL).all()
    assert (out["removed"][:, 1] == clean.OFF_CAMERA).all()
    assert (out["removed"][:, 2] == clean.KEPT).all()
    assert np.isnan(out["track_xy"][:, :2]).all() and np.isnan(out["track_speed"][:, :2]).all()
    np.testing.assert_array_equal(out["track_xy"][:, 2], xy[:, 2])
    assert np.isnan(xy[5, 2, 0]) and not np.isnan(xy[0, 0, 0])   # inputs untouched

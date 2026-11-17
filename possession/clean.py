"""Remove tracking that is not seeing anybody: positions held off camera, and tracks stuck in place.

Two faults of the ATD, both almost only in the keeper slots, confirmed by eye on BAR-ATH:
  - Off camera. The keeper slots go on being tracked when the camera cannot see the keeper (about
    half their frames on BAR-ATH), and there they barely move (median 0.05 m a second, against
    0.5-0.7 m on camera): the tracker holds a last guess. Outfield slots are almost never placed
    outside the picture (140 slot-frames in the whole match).
  - Stuck. A track can freeze where nobody is, sometimes for minutes, also on camera (BAR-ATH: both
    keeper slots in the last minutes). A real player does not stay within STILL_RADIUS for
    STILL_SECONDS: on BAR-ATH no outfield slot does, even in the stoppages where everyone stands
    still (they move 0.5-1.5 m over such stretches).

Removed positions become NaN, so a removed stretch cuts the slot's tracklet in two.

    out = clean(track_xy, track_speed, homography, valid, pitch)
    out["track_xy"], out["track_speed"]   # as the inputs, NaN where removed
    out["removed"]                         # (n, slots) uint8: KEPT, OFF_CAMERA or STILL
"""

import numpy as np

VERSION = 1   # bump whenever clean() returns something different: it is part of the cache key
FPS = 25
VIDEO_SIZE = (1920, 1080)
STILL_SECONDS = 30
STILL_RADIUS = 0.3   # metres: the most a track moves, along x and along y, over STILL_SECONDS

KEPT, OFF_CAMERA, STILL = 0, 1, 2


def on_camera(track_xy, homography, valid, pitch, size=VIDEO_SIZE):
    """(n, slots) bool: the position projects inside the video picture on that frame.

    track_xy is in metres; homography (n, 3, 3) maps pitch fractions to pixels (as the organisers'
    load_homography.project). Untracked positions and frames without a valid matrix are False.
    """
    p = np.asarray(track_xy, dtype=np.float64) / np.asarray(pitch, dtype=np.float64)
    h = np.asarray(homography, dtype=np.float64)[:, None]   # (n, 1, 3, 3) against (n, slots)
    x, y = p[..., 0], p[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = h[..., 2, 0] * x + h[..., 2, 1] * y + h[..., 2, 2]
        u = (h[..., 0, 0] * x + h[..., 0, 1] * y + h[..., 0, 2]) / w
        v = (h[..., 1, 0] * x + h[..., 1, 1] * y + h[..., 1, 2]) / w
        inside = (w > 0) & (u >= 0) & (u < size[0]) & (v >= 0) & (v < size[1])
    return inside & np.asarray(valid, dtype=bool)[:, None] & ~np.isnan(x)


def _window_span(values, window):
    """(n,) max - min of `values` over frames [t - window + 1, t], NaN unless all are tracked."""
    n = len(values)
    out = np.full(n, np.nan)
    if n < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, window)
    with np.errstate(invalid="ignore"):
        out[window - 1:] = view.max(axis=1) - view.min(axis=1)   # NaN anywhere gives NaN
    return out


def still(track_xy, seconds=STILL_SECONDS, radius=STILL_RADIUS, fps=FPS):
    """(n, slots) bool: the frame lies in a window of `seconds` over which the slot is tracked on
    every frame and moves less than `radius` metres along x and along y."""
    xy = np.asarray(track_xy, dtype=np.float64)
    n, slots, _ = xy.shape
    window = int(round(seconds * fps))
    out = np.zeros((n, slots), dtype=bool)
    for j in range(slots):
        span = np.fmax(_window_span(xy[:, j, 0], window), _window_span(xy[:, j, 1], window))
        with np.errstate(invalid="ignore"):
            ends = span < radius                              # window [t - window + 1, t] is still
        # Frame t is covered when some still window ends in [t, t + window - 1].
        covered = np.convolve(ends.astype(np.int32), np.ones(window, dtype=np.int32))[window - 1:]
        out[:, j] = covered > 0
    return out


def clean(track_xy, track_speed, homography, valid, pitch, seconds=STILL_SECONDS,
          radius=STILL_RADIUS, fps=FPS):
    """{track_xy, track_speed, removed}: the tracking with off-camera and stuck positions removed."""
    xy = np.asarray(track_xy)
    tracked = ~np.isnan(xy[..., 0])
    removed = np.full(tracked.shape, KEPT, dtype=np.uint8)
    removed[tracked & ~on_camera(xy, homography, valid, pitch)] = OFF_CAMERA
    removed[tracked & still(xy, seconds, radius, fps)] = STILL
    gone = removed != KEPT
    out_xy = np.array(xy, copy=True)
    out_xy[gone] = np.nan
    out_speed = np.array(track_speed, copy=True)
    out_speed[gone] = np.nan
    return {"track_xy": out_xy, "track_speed": out_speed, "removed": removed}

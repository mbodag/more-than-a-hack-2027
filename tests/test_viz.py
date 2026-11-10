"""The viewer's export on tiny synthetic arrays; never touches the match data.

The page's own decoding and projection (possession/viewer/core.js) are run under Node and compared
with the organisers' project(); those tests are skipped where Node is not installed.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from possession import viz
from starter import load_homography

PITCH = (100.0, 50.0)
N_FRAMES, N_TRACKS, CHUNK = 50, 3, 10
BLOCKS = [(12, 27), (35, 44)]
# The example row of DATA.md: a real tactical-camera matrix.
MATRIX = np.array([[1822.663, -511.2005, 44.48168],
                   [1.698323, 169.2971, 255.7303],
                   [-0.01907188, -0.5269586, 1.0]])

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")


def synthetic():
    rng = np.random.default_rng(3)
    ball = rng.uniform(0, 1, (N_FRAMES, 2)) * PITCH
    ball[15:24] = np.nan      # lost across the start of chunk 2
    ball[:12] = np.nan        # never seen before the first half
    tracks = rng.uniform(-0.05, 1.05, (N_FRAMES, N_TRACKS, 2)) * PITCH
    tracks[23, 1] = np.nan
    homography = MATRIX * (1 + 0.001 * np.arange(N_FRAMES))[:, None, None]
    homography[:, 2, 2] = 1.0
    return ball.astype(np.float32), tracks.astype(np.float32), homography


def write(tmp_path, video="no-such-video.mp4"):
    ball, tracks, homography = synthetic()
    events = viz.tagging_events([{"code": "BUILD UP", "start_frame": 14, "end_frame": 20,
                                  "tags": {"Team": "Home FC", "Side": None}}])
    index = viz.write_viewer(
        str(tmp_path), ball, tracks, np.array(["H", "A", "A"]), ("Home FC", "Away FC"),
        homography, events, blocks=BLOCKS, pitch=PITCH, fps=25,
        video_path=str(tmp_path / video), ground_truth={"first": 12, "labels": "HHAD"},
        chunk_frames=CHUNK, say=lambda message: None)
    return index, ball, tracks, homography


def read_meta(tmp_path):
    text = (tmp_path / "meta.js").read_text()
    return json.loads(text[len("window.VIEWER_META = "):].rstrip().rstrip(";"))


def test_quantise_round_trip_and_missing():
    xy = np.array([[0.0, 1.0], [0.25, 0.75], [-0.4, 1.4], [np.nan, 0.5]])
    steps = viz.quantise(xy)
    assert steps.dtype == np.dtype("<u2")
    assert steps[3].tolist() == [viz.Q_MISSING, viz.Q_MISSING]
    back = viz.dequantise(steps)
    np.testing.assert_allclose(back[:3], xy[:3], atol=viz.Q_SPAN / viz.Q_MAX)
    assert np.isnan(back[3]).all()


def test_only_chunks_touching_a_half_are_written(tmp_path):
    index, _, _, _ = write(tmp_path)
    assert viz.chunk_indices(BLOCKS, CHUNK) == [1, 2, 3, 4]
    assert sorted(os.listdir(tmp_path / "chunks")) == ["c0001.js", "c0002.js", "c0003.js",
                                                        "c0004.js"]
    assert os.path.isfile(index)
    assert all((tmp_path / name).is_file() for name in viz.PAGE_FILES)


def test_frame_i_of_a_chunk_is_frame_i_of_the_match(tmp_path):
    _, ball, tracks, homography = write(tmp_path)
    for name in sorted(os.listdir(tmp_path / "chunks")):
        index, payload = viz.read_chunk_js((tmp_path / "chunks" / name).read_text())
        assert payload["first"] == index * CHUNK and payload["n"] == CHUNK
        xy = viz.dequantise(payload["xy"]) * PITCH
        for k in range(CHUNK):
            frame = payload["first"] + k
            np.testing.assert_allclose(xy[k, 0], ball[frame], atol=0.01)
            np.testing.assert_allclose(xy[k, 1:], tracks[frame], atol=0.01)
            np.testing.assert_allclose(payload["h"][k], homography[frame], rtol=1e-6)


def test_a_chunk_carries_in_the_last_ball_seen_before_it(tmp_path):
    _, ball, _, _ = write(tmp_path)
    _, first = viz.read_chunk_js((tmp_path / "chunks" / "c0001.js").read_text())
    _, second = viz.read_chunk_js((tmp_path / "chunks" / "c0002.js").read_text())
    assert first["carry"] is None  # no ball before frame 10
    frame, x, y = second["carry"]
    assert frame == 14             # lost from 15, and chunk 2 starts at 20
    np.testing.assert_allclose(viz.dequantise(np.array([x, y])) * PITCH, ball[14], atol=0.01)


def test_meta_describes_the_match(tmp_path):
    _, ball, tracks, _ = write(tmp_path)
    meta = read_meta(tmp_path)
    assert meta["n_frames"] == N_FRAMES and meta["blocks"] == [list(b) for b in BLOCKS]
    assert meta["chunks"] == [1, 2, 3, 4] and meta["n_objects"] == 1 + N_TRACKS
    assert meta["track_team"] == "HAA" and meta["teams"] == ["Home FC", "Away FC"]
    assert meta["events"] == [{"s": 14, "e": 20, "code": "BUILD UP", "tags": {"Team": "Home FC"}}]
    assert meta["ground_truth"] == {"first": 12, "labels": "HHAD"}
    assert meta["video"]["uri"].startswith("file://")

    def array(key):
        import base64
        return np.frombuffer(base64.b64decode(meta[key]), dtype=np.uint8)
    np.testing.assert_array_equal(array("ball_present"), ~np.isnan(ball[:, 0]))
    assert array("n_home")[23] == 1 and array("n_away")[23] == 1  # slot 2 is missing on 23
    assert array("n_home")[24] == 1 and array("n_away")[24] == 2


def test_ground_truth_keeps_only_the_labelled_stretch(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text("frame,label\n0,X\n1,X\n2,H\n3,D\n4,A\n5,X\n")
    assert viz.read_ground_truth(str(path)) == {"first": 2, "labels": "HDA"}
    assert viz.read_ground_truth(str(tmp_path / "absent.csv")) is None


def run_node(script, *args):
    done = subprocess.run(["node", "-e", script] + list(args), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_the_page_projects_like_the_organisers_loader(tmp_path):
    """core.js, on a written chunk: positions and pixels for every object of every frame."""
    _, ball, tracks, homography = write(tmp_path)
    script = """
      const [core, metaPath, chunkPath] = process.argv.slice(1);
      const Core = require(core);
      const meta = JSON.parse(require("fs").readFileSync(metaPath, "utf8")
                              .replace("window.VIEWER_META = ", "").replace(/;\\s*$/, ""));
      let chunk = null;
      global.viewerChunk = (index, payload) => { chunk = Core.decodeChunk(payload); };
      require(chunkPath);
      const rows = [];
      for (let frame = chunk.first; frame < chunk.first + chunk.n; frame++) {
        for (let k = 0; k < meta.n_objects; k++) {
          const xy = [0, 0], px = [0, 0];
          if (!Core.point(chunk, meta, frame, k, xy)) { rows.push(null); continue; }
          Core.project(chunk, frame, xy[0], xy[1], px);
          rows.push([xy[0], xy[1], px[0], px[1]]);
        }
      }
      const last = Core.lastBall(chunk, meta, 22);
      console.log(JSON.stringify({rows: rows, last: last}));
    """
    out = run_node(script, os.path.join(viz.PAGE_DIR, "core.js"), str(tmp_path / "meta.js"),
                   str(tmp_path / "chunks" / "c0002.js"))
    rows = iter(out["rows"])
    for frame in range(20, 30):
        objects = np.concatenate([ball[frame][None], tracks[frame]]) / PITCH
        for xy in objects:
            row = next(rows)
            if np.isnan(xy).any():
                assert row is None
                continue
            np.testing.assert_allclose(row[:2], xy, atol=1e-4)
            expected = load_homography.project(homography[frame], xy[0], xy[1])
            np.testing.assert_allclose(row[2:], expected, atol=0.2)  # pixels
    # The ball is lost from frame 15 to 23: on frame 22 the page falls back on the carry.
    assert out["last"]["frame"] == 14
    np.testing.assert_allclose([out["last"]["x"], out["last"]["y"]], ball[14] / PITCH, atol=1e-4)


@needs_node
def test_the_page_reads_a_submission_csv_and_rejects_anything_else():
    script = """
      const Core = require(process.argv[1]);
      const good = Core.parseLabels("frame,label\\n0,H\\n1,A\\n2,D\\n3,X\\n", 4);
      console.log(JSON.stringify({
        good: Array.from(good.codes),
        short: Core.parseLabels("frame,label\\n0,H\\n", 4).error,
        wrong: Core.parseLabels("frame,label\\n0,H\\n1,Q\\n2,D\\n3,X\\n", 4).error,
        clock: Core.matchClock(110, [[10, 60], [100, 160]], 25),
      }));
    """
    out = run_node(script, os.path.join(viz.PAGE_DIR, "core.js"))
    assert out["good"] == [1, 2, 3, 0]
    assert "found 1" in out["short"] and "line 3" in out["wrong"]
    assert out["clock"] == {"half": 2, "seconds": 45 * 60 + 0.4}

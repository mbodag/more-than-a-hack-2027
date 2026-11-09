"""The Match's builders on tiny synthetic files; never touches the match data."""

import numpy as np
import pytest

from possession import match as m

META = """<Root><Teams>
  <Team id="t1"><Name>Home FC</Name></Team>
  <Team id="t2"><Name>Away FC</Name></Team>
</Teams><Players>
  <Player id="1" teamId="t1"/><Player id="2" teamId="t2"/>
</Players></Root>"""


def test_creating_a_match_reads_nothing(tmp_path):
    match = m.Match(data_dir=str(tmp_path / "no-data-here"), cache_dir=str(tmp_path / "cache"),
                    log=lambda message: None)
    assert match.video_path.endswith("BAR-ATH_tactical.mp4")
    assert not (tmp_path / "cache").exists()


def test_read_teams(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_text(META)
    names, slots = m.read_teams(str(path))
    assert names == ("Home FC", "Away FC") and slots.tolist() == ["H", "A"]


def test_parse_atd_frames_and_metres(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("1:0.5,0.5,1.0;NaN,NaN,NaN:0.1,0.2,3.0\n"
                    "2:0.0,1.0,2.0;0.25,0.75,4.0:NaN,NaN,NaN\n")
    out = m.parse_atd(str(path), n_slots=2, n_frames=2, pitch=(100.0, 50.0))
    assert out["ball_xy"].shape == (2, 2) and out["track_xy"].shape == (2, 2, 2)
    np.testing.assert_allclose(out["ball_xy"][0], [10.0, 10.0])
    assert np.isnan(out["ball_xy"][1]).all()
    np.testing.assert_allclose(out["track_xy"][1, 1], [25.0, 37.5])
    np.testing.assert_allclose(out["track_speed"][1], [2.0, 4.0])
    assert out["ball_xy"].dtype == np.float32


def test_parse_atd_refuses_missing_frames(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("1:0,0,0:0,0,0\n3:0,0,0:0,0,0\n")
    with pytest.raises(ValueError, match="frame ids 1..2"):
        m.parse_atd(str(path), n_slots=1, n_frames=2, pitch=(1.0, 1.0))


def test_parse_atd_refuses_wrong_width(tmp_path):
    path = tmp_path / "raw.txt"
    path.write_text("1:0,0,0:0,0,0\n")
    with pytest.raises(ValueError, match="values per line"):
        m.parse_atd(str(path), n_slots=2, n_frames=1, pitch=(1.0, 1.0))


def event(code, start, end, team, kind=None):
    tags = {"Team": team}
    if kind:
        tags["Type"] = kind
    return {"code": code, "start_frame": start, "end_frame": end, "tags": tags}


def test_paint_smart_tagging():
    events = [event("BUILD UP", 1, 3, "Home FC"),
              event("BUILD UP", 2, 4, "Away FC"),
              event("SET PIECES", 0, 2, "N/A", "Throw-in"),
              event("LONG BALL", 4, 99, "Away FC")]
    out = m.paint_smart_tagging(events, ("Home FC", "Away FC"), n_frames=6)
    assert out["codes"].tolist() == ["BUILD UP", "LONG BALL", "SET PIECES: Throw-in"]
    assert out["team"][:, 0].tolist() == ["", "H", "?", "A", "", ""]
    assert out["team"][:, 1].tolist() == ["", "", "", "", "A", "A"]
    assert out["team"][:, 2].tolist() == ["N", "N", "", "", "", ""]


def test_paint_smart_tagging_refuses_unknown_team():
    with pytest.raises(ValueError, match="neither"):
        m.paint_smart_tagging([event("BUILD UP", 0, 1, "Someone")], ("H FC", "A FC"), 2)


def test_every_item_reads_only_committed_files():
    checksums = m.load_checksums()
    for item in m.ITEMS.values():
        for source, asset in item.raw:
            assert asset in checksums[source], (item.name, asset)

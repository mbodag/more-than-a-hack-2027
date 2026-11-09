"""The cache on small synthetic items in a temporary folder; never touches the match data."""

import os

import numpy as np
import pytest

from possession import cache as c

CHECKSUMS = {"tactical": {"atd_raw": "a" * 64, "atd_meta": "b" * 64}}


def make_cache(tmp_path, items, checksums=CHECKSUMS, **kwargs):
    raw = tmp_path / "raw.txt"
    raw.write_text("1 2 3")
    paths = {("tactical", "atd_raw"): str(raw), ("tactical", "atd_meta"): str(raw)}
    kwargs.setdefault("log", lambda message: None)
    return c.Cache(str(tmp_path / "cache"), items, checksums, paths, **kwargs)


class Counter:
    """A build function that counts its calls."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, inputs):
        self.calls += 1
        return self.fn(inputs)


def chain(base_version=1, top_version=1):
    """raw -> base -> top."""
    base = c.Item("base", base_version, Counter(lambda i: np.arange(3)),
                  raw=(("tactical", "atd_raw"),))
    top = c.Item("top", top_version, Counter(lambda i: i.item("base") * 2), deps=("base",))
    other = c.Item("other", 1, Counter(lambda i: {"a": np.zeros(2), "b": np.array(["x"])}),
                   raw=(("tactical", "atd_meta"),))
    return {"base": base, "top": top, "other": other}


# Keys --------------------------------------------------------------------------------------------

def test_key_format(tmp_path):
    key = make_cache(tmp_path, chain()).key("base")
    name, version, fingerprint = key.split("__")
    assert (name, version, len(fingerprint)) == ("base", "v1", c.FINGERPRINT_LENGTH)


def test_keys_are_stable(tmp_path):
    assert make_cache(tmp_path, chain()).key("top") == make_cache(tmp_path, chain()).key("top")


def test_version_bump_changes_downstream_keys(tmp_path):
    before = make_cache(tmp_path, chain())
    after = make_cache(tmp_path, chain(base_version=2))
    assert before.key("base") != after.key("base")
    assert before.key("top") != after.key("top")
    assert before.key("other") == after.key("other")


def test_raw_checksum_change_changes_downstream_keys(tmp_path):
    changed = {"tactical": dict(CHECKSUMS["tactical"], atd_raw="c" * 64)}
    before = make_cache(tmp_path, chain())
    after = make_cache(tmp_path, chain(), checksums=changed)
    assert before.key("base") != after.key("base")
    assert before.key("top") != after.key("top")
    assert before.key("other") == after.key("other")


def test_own_version_bump_leaves_upstream_alone(tmp_path):
    assert (make_cache(tmp_path, chain()).key("base")
            == make_cache(tmp_path, chain(top_version=5)).key("base"))


def test_key_needs_a_committed_checksum(tmp_path):
    with pytest.raises(c.CacheError, match="no committed checksum"):
        make_cache(tmp_path, chain(), checksums={"tactical": {}}).key("base")


# Building and loading ----------------------------------------------------------------------------

def test_built_once_then_loaded(tmp_path):
    items = chain()
    first = make_cache(tmp_path, items)
    assert first.get("top").tolist() == [0, 2, 4]
    assert items["base"].build.calls == 1 and items["top"].build.calls == 1

    second = make_cache(tmp_path, items)
    assert second.get("top").tolist() == [0, 2, 4]
    assert items["base"].build.calls == 1 and items["top"].build.calls == 1


def test_dict_items_round_trip(tmp_path):
    value = make_cache(tmp_path, chain()).get("other")
    assert set(value) == {"a", "b"} and value["b"].tolist() == ["x"]


def test_undeclared_inputs_are_refused(tmp_path):
    items = {"sneaky": c.Item("sneaky", 1, lambda i: i.path("tactical", "atd_raw"))}
    with pytest.raises(c.CacheError, match="does not declare"):
        make_cache(tmp_path, items).get("sneaky")


def test_rebuild_all(tmp_path):
    items = chain()
    make_cache(tmp_path, items).get("top")
    make_cache(tmp_path, items, rebuild=True).get("top")
    assert items["base"].build.calls == 2 and items["top"].build.calls == 2


def test_rebuild_named_only(tmp_path):
    items = chain()
    make_cache(tmp_path, items).get("top")
    make_cache(tmp_path, items, rebuild=["top"]).get("top")
    assert items["base"].build.calls == 1 and items["top"].build.calls == 2


def test_rebuild_unknown_name_lists_items(tmp_path):
    with pytest.raises(c.CacheError, match="no such item; have base, other, top"):
        make_cache(tmp_path, chain(), rebuild=["nope"])


def test_rebuild_announces_name_and_last_duration(tmp_path):
    messages = []
    make_cache(tmp_path, chain()).get("base")
    make_cache(tmp_path, chain(), rebuild=["base"], log=messages.append).get("base")
    assert messages[0].startswith("rebuilding base (last build took ")


def test_last_duration_survives_a_version_bump(tmp_path):
    make_cache(tmp_path, chain()).get("base")
    messages = []
    make_cache(tmp_path, chain(base_version=2), log=messages.append).get("base")
    assert messages[0].startswith("building base (last build took ")


def test_duration_file_next_to_cache_file(tmp_path):
    cache = make_cache(tmp_path, chain())
    cache.get("base")
    assert os.path.isfile(cache.path("base")[:-4] + ".json")
    assert cache.last_duration("base") >= 0


# Atomic writes and backups -----------------------------------------------------------------------

def test_interrupted_build_keeps_old_file(tmp_path):
    items = chain()
    cache = make_cache(tmp_path, items)
    cache.get("base")
    path = cache.path("base")

    def interrupted(inputs):
        raise KeyboardInterrupt

    broken = dict(items, base=c.Item("base", 1, interrupted, raw=(("tactical", "atd_raw"),)))
    with pytest.raises(KeyboardInterrupt):
        make_cache(tmp_path, broken, rebuild=True).get("base")
    assert c.load(path).tolist() == [0, 1, 2]


def test_interrupted_write_keeps_old_file_and_leaves_no_tmp(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, chain())
    cache.get("base")
    path = cache.path("base")

    def half_written(handle, **arrays):
        handle.write(b"PK\x03\x04 not a whole file")
        raise KeyboardInterrupt

    monkeypatch.setattr(c.np, "savez", half_written)
    with pytest.raises(KeyboardInterrupt):
        c.save(path, np.arange(10))
    monkeypatch.undo()

    assert c.load(path).tolist() == [0, 1, 2]
    assert not [f for f in os.listdir(os.path.dirname(path)) if f.endswith(".tmp")]


def test_replaced_file_is_kept_as_prev(tmp_path):
    path = str(tmp_path / "item__v1__00000000.npz")
    c.save(path, np.arange(3))
    c.save(path, np.arange(5))
    assert c.load(path).tolist() == [0, 1, 2, 3, 4]
    assert c.load(path[:-4] + ".prev.npz").tolist() == [0, 1, 2]


def test_rebuild_keeps_prev(tmp_path):
    cache = make_cache(tmp_path, chain())
    cache.get("base")
    make_cache(tmp_path, chain(), rebuild=True).get("base")
    assert os.path.isfile(cache.path("base")[:-4] + ".prev.npz")


def test_leftover_tmp_of_dead_process_removed_on_start(tmp_path):
    folder = tmp_path / "cache"
    folder.mkdir()
    dead = folder / "base__v1__00000000.npz.999999999.tmp"
    nameless = folder / "base__v1__00000000.npz.tmp"
    keep = folder / "base__v1__00000000.npz"
    for f in (dead, nameless, keep):
        f.write_bytes(b"x")
    make_cache(tmp_path, chain())
    assert not dead.exists() and not nameless.exists() and keep.exists()


def test_tmp_of_live_process_left_alone(tmp_path):
    folder = tmp_path / "cache"
    folder.mkdir()
    live = folder / "base__v1__00000000.npz.{}.tmp".format(os.getppid())
    live.write_bytes(b"x")
    make_cache(tmp_path, chain())
    assert live.exists()


def test_nothing_deleted_otherwise(tmp_path):
    cache = make_cache(tmp_path, chain())
    cache.get("top")
    before = set(os.listdir(cache.directory))
    make_cache(tmp_path, chain(base_version=2)).get("top")
    assert before <= set(os.listdir(cache.directory))


# Declarations ------------------------------------------------------------------------------------

def test_bad_item_name_refused(tmp_path):
    with pytest.raises(c.CacheError, match="item name"):
        make_cache(tmp_path, {"a__b": c.Item("a__b", 1, lambda i: np.zeros(1))})


def test_unknown_dependency_refused(tmp_path):
    with pytest.raises(c.CacheError, match="unknown item"):
        make_cache(tmp_path, {"a": c.Item("a", 1, lambda i: np.zeros(1), deps=("b",))})


def test_parse_rebuild():
    assert c.parse_rebuild(None) is None
    assert c.parse_rebuild([]) is True
    assert c.parse_rebuild(["tracking"]) == ["tracking"]

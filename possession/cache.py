"""Build each input item once and keep it on disk (ADR 0002, "The cache").

An item is declared with an `Item`: a name, a version, the raw data files it reads and the other
items it is built from. Its cache key says exactly what produced it:

    outputs/cache/ball_xy__v2__3fa9c2e1.npz
                  item    version  fingerprint

The fingerprint hashes the committed SHA-256 of every raw file the item reads and the full key of
every item it is built from, so a change anywhere upstream changes every key downstream. Computing
a key reads no data.

Files in the cache directory, for one key:

    <key>.npz             the item
    <key>.prev.npz        the file it replaced, kept when a rebuild finishes
    <key>.json            how long the last build took
    <key>.npz.<pid>.tmp   a build being written by process <pid>; swapped in with os.replace

Nothing here deletes a finished file. Leftover .tmp files from processes that are no longer
running are removed when a Cache is created; .tmp files of live processes are left alone, so
parallel sessions can share one cache directory.

numpy and the standard library only.
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

# Characters of the fingerprint kept in the file name. 8 hex = 32 bits, plenty for the handful of
# versions of each item that will ever sit side by side.
FINGERPRINT_LENGTH = 8

# The key under which a single-array item is stored in its .npz; dict items use their own keys.
_ARRAY = "__array__"
_TMP = re.compile(r"\.(\d+)\.tmp$")


class CacheError(Exception):
    """An item is declared wrongly, or asked for under a name that does not exist."""


@dataclass(frozen=True)
class Item:
    """How to build one item.

    build(inputs) returns a numpy array, or a dict of numpy arrays, and may only use what the item
    declares: inputs.path(source, asset) for each entry of `raw`, inputs.item(name) for each entry
    of `deps`. Bump `version` whenever the build code changes what it returns.
    """

    name: str
    version: int
    build: Callable
    raw: Tuple[Tuple[str, str], ...] = ()
    deps: Tuple[str, ...] = ()
    doc: str = field(default="", compare=False)


class Inputs:
    """What one build may read. Anything undeclared is refused, so fingerprints stay honest."""

    def __init__(self, cache, item):
        self._cache = cache
        self._item = item

    def path(self, source, asset):
        if (source, asset) not in self._item.raw:
            raise CacheError("{} reads {}/{} but does not declare it in raw".format(
                self._item.name, source, asset))
        return self._cache.raw_paths[(source, asset)]

    def item(self, name):
        if name not in self._item.deps:
            raise CacheError("{} uses item {} but does not declare it in deps".format(
                self._item.name, name))
        return self._cache.get(name)


class Cache:
    """Items loaded from disk, or built once and saved.

    items       {name: Item}
    checksums   the committed checksums, {source: {asset: sha256}}
    raw_paths   {(source, asset): path} for every raw file an item may read
    rebuild     None (use the cache), True (rebuild every item asked for) or an iterable of names
    log         where messages go; print by default
    """

    def __init__(self, directory, items, checksums, raw_paths, rebuild=None, log=print):
        self.directory = directory
        self.items: Dict[str, Item] = dict(items)
        self.checksums = checksums
        self.raw_paths = raw_paths
        self.log = log
        self._values = {}
        self._keys = {}
        self._building = []

        if rebuild is None or rebuild is False:
            self._rebuild = frozenset()
        elif rebuild is True:
            self._rebuild = None  # every item
        else:
            self._rebuild = frozenset(rebuild)
            unknown = sorted(self._rebuild - set(self.items))
            if unknown:
                raise CacheError("cannot rebuild {}: no such item; have {}".format(
                    ", ".join(unknown), ", ".join(sorted(self.items))))

        self._check_declarations()
        remove_stale_tmp(directory)

    # Keys ---------------------------------------------------------------------------------------

    def key(self, name):
        """`<name>__v<version>__<fingerprint>`. Reads no data."""
        if name not in self._keys:
            item = self._item(name)
            raw = []
            for source, asset in item.raw:
                sha = self.checksums.get(source, {}).get(asset)
                if sha is None:
                    raise CacheError("{} reads {}/{}, which has no committed checksum".format(
                        name, source, asset))
                raw.append([source, asset, sha])
            payload = json.dumps({"raw": sorted(raw),
                                  "deps": sorted(self.key(dep) for dep in item.deps)},
                                 sort_keys=True)
            fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:FINGERPRINT_LENGTH]
            self._keys[name] = "{}__v{}__{}".format(name, item.version, fingerprint)
        return self._keys[name]

    def path(self, name):
        return os.path.join(self.directory, self.key(name) + ".npz")

    # Access -------------------------------------------------------------------------------------

    def get(self, name):
        """The item's value: from memory, from disk, or built now."""
        if name in self._values:
            return self._values[name]
        path = self.path(name)
        if os.path.isfile(path) and not self._wants_rebuild(name):
            value = load(path)
        else:
            value = self._build(name, path)
        self._values[name] = value
        return value

    def _wants_rebuild(self, name):
        return self._rebuild is None or name in self._rebuild

    def _build(self, name, path):
        item = self._item(name)
        if name in self._building:
            raise CacheError("items depend on each other in a loop: {}".format(
                " -> ".join(self._building + [name])))

        last = self.last_duration(name)
        self.log("{} {}{}".format(
            "rebuilding" if os.path.isfile(path) else "building", name,
            " (last build took {})".format(format_duration(last)) if last is not None else ""))

        self._building.append(name)
        try:
            started = time.monotonic()
            value = item.build(Inputs(self, item))
            seconds = time.monotonic() - started
        finally:
            self._building.pop()

        save(path, value)
        _write_json(path[:-len(".npz")] + ".json", {"item": name, "seconds": round(seconds, 3)})
        self.log("built {} in {}".format(name, format_duration(seconds)))
        # Return what was written, so a build and a later load give identical values.
        return load(path)

    def last_duration(self, name):
        """Seconds the most recent build of this item took, under any key, or None."""
        own = self.path(name)[:-len(".npz")] + ".json"
        candidates = [own] if os.path.isfile(own) else []
        if not candidates and os.path.isdir(self.directory):
            prefix = name + "__v"
            candidates = [os.path.join(self.directory, f) for f in os.listdir(self.directory)
                          if f.startswith(prefix) and f.endswith(".json")]
        if not candidates:
            return None
        newest = max(candidates, key=os.path.getmtime)
        try:
            with open(newest) as handle:
                return float(json.load(handle)["seconds"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # Checks -------------------------------------------------------------------------------------

    def _item(self, name):
        try:
            return self.items[name]
        except KeyError:
            raise CacheError("no item {!r}; have {}".format(name, ", ".join(sorted(self.items))))

    def _check_declarations(self):
        for name, item in self.items.items():
            if item.name != name:
                raise CacheError("item registered as {!r} is named {!r}".format(name, item.name))
            if "__" in name or not re.fullmatch(r"[a-z0-9_]+", name):
                raise CacheError("item name {!r}: lower case, digits and single underscores "
                                 "only".format(name))
            for dep in item.deps:
                if dep not in self.items:
                    raise CacheError("{} depends on unknown item {}".format(name, dep))
            for raw in item.raw:
                if raw not in self.raw_paths:
                    raise CacheError("{} reads {}/{}, which has no path".format(name, *raw))


# Files --------------------------------------------------------------------------------------------

def save(path, value):
    """Write value to path atomically, keeping any file it replaces as <key>.prev.npz.

    The build is written to a .tmp file first. Only when it is complete does it replace the old
    file, with os.replace, so the file at `path` is always either the whole old item or the whole
    new one. An interruption at any point leaves the old file where it was.
    """
    arrays = _as_arrays(value)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    tmp = "{}.{}.tmp".format(path, os.getpid())
    try:
        with open(tmp, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.isfile(path):
            _keep_previous(path)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _keep_previous(path):
    """Make <key>.prev.npz a copy of the current file, without ever removing the current file."""
    prev = path[:-len(".npz")] + ".prev.npz"
    staging = "{}.{}.tmp".format(prev, os.getpid())
    try:
        try:
            os.link(path, staging)  # instant, and the old bytes survive the coming os.replace
        except OSError:
            import shutil
            shutil.copy2(path, staging)
        os.replace(staging, prev)
    finally:
        if os.path.exists(staging):
            os.remove(staging)


def load(path):
    """An item written by save(): an array, or a dict of arrays. Fully read into memory."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if set(arrays) == {_ARRAY}:
        return arrays[_ARRAY]
    return arrays


def _as_arrays(value):
    if isinstance(value, np.ndarray):
        return {_ARRAY: value}
    if isinstance(value, dict) and value and all(isinstance(v, np.ndarray) for v in value.values()):
        if _ARRAY in value:
            raise CacheError("{} is a reserved array name".format(_ARRAY))
        return value
    raise CacheError("an item must be a numpy array or a non-empty dict of numpy arrays, "
                     "got {}".format(type(value).__name__))


def remove_stale_tmp(directory):
    """Remove .tmp files left by processes that are no longer running. Returns what was removed."""
    removed = []
    if not os.path.isdir(directory):
        return removed
    for name in os.listdir(directory):
        if not name.endswith(".tmp"):
            continue
        match = _TMP.search(name)
        if match and _alive(int(match.group(1))):
            continue
        try:
            os.remove(os.path.join(directory, name))
            removed.append(name)
        except FileNotFoundError:
            pass
    return removed


def _alive(pid):
    if pid == os.getpid():
        return False  # a leftover of ours: we are not writing it, or save() would be on the stack
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # exists, owned by someone else
    except (OSError, OverflowError, ValueError):
        return False
    return True


def _write_json(path, payload):
    tmp = "{}.{}.tmp".format(path, os.getpid())
    with open(tmp, "w") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def format_duration(seconds):
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return "{:.1f} s".format(seconds)
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return "{} min {} s".format(minutes, rest)
    hours, minutes = divmod(minutes, 60)
    return "{} h {} min".format(hours, minutes)

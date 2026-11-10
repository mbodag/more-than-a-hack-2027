// The viewer's pure functions: decoding what possession/viz.py writes, projecting through the
// homography, parsing a submission CSV. No DOM here, so the tests can run this file under Node.
(function (root) {
  "use strict";

  var LABELS = "XHAD"; // label codes 0..3; NO_LABEL marks frames a series says nothing about
  var NO_LABEL = 255;

  function bytes(base64) {
    var text = atob(base64);
    var out = new Uint8Array(text.length);
    for (var i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
    return out;
  }

  // payload: {first, n, xy, h, carry} as written by viz.chunk_js (little-endian arrays).
  function decodeChunk(payload) {
    return {
      first: payload.first,
      n: payload.n,
      xy: new Uint16Array(bytes(payload.xy).buffer),
      h: new Float32Array(bytes(payload.h).buffer),
      carry: payload.carry,
    };
  }

  function unstep(step, q) {
    return step / q.max * q.span + q.low;
  }

  // Position of object `index` (0 is the ball, 1.. the track slots) on `frame`, as fractions of
  // the pitch, written to out[0], out[1]. False when it is not tracked on that frame.
  function point(chunk, meta, frame, index, out) {
    var at = ((frame - chunk.first) * meta.n_objects + index) * 2;
    var x = chunk.xy[at];
    if (x === meta.q.missing) return false;
    out[0] = unstep(x, meta.q);
    out[1] = unstep(chunk.xy[at + 1], meta.q);
    return true;
  }

  // The last tracked ball at or before `frame`: {frame, x, y}, or null. Looks back through this
  // chunk, then falls back on what the chunk carried in from before its first frame.
  function lastBall(chunk, meta, frame) {
    var out = [0, 0];
    for (var f = frame; f >= chunk.first; f--) {
      if (point(chunk, meta, f, 0, out)) return { frame: f, x: out[0], y: out[1] };
    }
    if (!chunk.carry) return null;
    return { frame: chunk.carry[0], x: unstep(chunk.carry[1], meta.q),
             y: unstep(chunk.carry[2], meta.q) };
  }

  // Pitch fractions -> video pixels with that frame's 3x3 matrix (row-major, as in the CSV).
  // The same arithmetic as starter/load_homography.project. False when there is no matrix or the
  // point is behind the camera.
  function project(chunk, frame, x, y, out) {
    var h = chunk.h, at = (frame - chunk.first) * 9;
    var w = h[at + 6] * x + h[at + 7] * y + h[at + 8];
    if (!(w > 0)) return false;
    out[0] = (h[at] * x + h[at + 1] * y + h[at + 2]) / w;
    out[1] = (h[at + 3] * x + h[at + 4] * y + h[at + 5]) / w;
    return true;
  }

  // A submission CSV as label codes, one per frame: {codes} or {error}.
  function parseLabels(text, nFrames) {
    var codes = new Uint8Array(nFrames);
    var lines = text.split(/\r?\n/);
    var row = 0;
    for (var i = /^frame/i.test(lines[0]) ? 1 : 0; i < lines.length; i++) {
      var line = lines[i];
      if (!line) continue;
      var comma = line.indexOf(",");
      var label = line.slice(comma + 1).trim();
      var code = label.length === 1 ? LABELS.indexOf(label) : -1;
      if (comma < 0 || Number(line.slice(0, comma)) !== row || code < 0) {
        return { error: "line " + (i + 1) + ": expected \"" + row + ",<H|A|D|X>\", found \"" +
                        line.slice(0, 40) + "\"" };
      }
      if (row >= nFrames) return { error: "more than " + nFrames + " rows" };
      codes[row++] = code;
    }
    if (row !== nFrames) return { error: "expected " + nFrames + " rows, found " + row };
    return { codes: codes };
  }

  // {half, seconds} of a frame inside a half, or null between and outside them.
  function matchClock(frame, blocks, fps) {
    for (var i = 0; i < blocks.length; i++) {
      if (frame >= blocks[i][0] && frame <= blocks[i][1]) {
        return { half: i + 1, seconds: (frame - blocks[i][0]) / fps + i * 45 * 60 };
      }
    }
    return null;
  }

  function clockText(seconds) {
    var whole = Math.floor(seconds);
    var s = whole % 60;
    return Math.floor(whole / 60) + ":" + (s < 10 ? "0" : "") + s;
  }

  // The pitch markings as polylines in fractions of the pitch. Straight edges are cut into short
  // pieces so that a projected line survives one end leaving the picture.
  function pitchLines(pitch) {
    var L = pitch[0], W = pitch[1], mid = W / 2, lines = [];
    function add(points) {
      var out = [];
      for (var i = 0; i < points.length - 1; i++) {
        for (var k = 0; k < 8; k++) {
          out.push([(points[i][0] + (points[i + 1][0] - points[i][0]) * k / 8) / L,
                    (points[i][1] + (points[i + 1][1] - points[i][1]) * k / 8) / W]);
        }
      }
      var last = points[points.length - 1];
      out.push([last[0] / L, last[1] / W]);
      lines.push(out);
    }
    function arc(cx, cy, r, from, to) {
      var points = [];
      for (var k = 0; k <= 24; k++) {
        var angle = from + (to - from) * k / 24;
        points.push([(cx + r * Math.cos(angle)) / L, (cy + r * Math.sin(angle)) / W]);
      }
      lines.push(points);
    }
    add([[0, 0], [L, 0], [L, W], [0, W], [0, 0]]);
    add([[L / 2, 0], [L / 2, W]]);
    arc(L / 2, mid, 9.15, 0, 2 * Math.PI);
    var reach = Math.acos((16.5 - 11) / 9.15); // where the penalty arc meets the box
    [[0, 1], [L, -1]].forEach(function (side) {
      var x = side[0], d = side[1];
      add([[x, mid - 20.16], [x + 16.5 * d, mid - 20.16], [x + 16.5 * d, mid + 20.16],
           [x, mid + 20.16]]);
      add([[x, mid - 9.16], [x + 5.5 * d, mid - 9.16], [x + 5.5 * d, mid + 9.16],
           [x, mid + 9.16]]);
      var centre = d > 0 ? 0 : Math.PI;
      arc(x + 11 * d, mid, 9.15, centre - reach, centre + reach);
    });
    return lines;
  }

  var api = {
    LABELS: LABELS, NO_LABEL: NO_LABEL, bytes: bytes, decodeChunk: decodeChunk, point: point,
    lastBall: lastBall, project: project, parseLabels: parseLabels, matchClock: matchClock,
    clockText: clockText, pitchLines: pitchLines,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ViewerCore = api;
})(typeof window !== "undefined" ? window : globalThis);

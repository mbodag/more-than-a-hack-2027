// The match viewer page. One frame number, state.frame, drives everything: the video, the overlay,
// the pitch, the side panel and the timelines. It is always a frame of the tactical video, so it
// is also a submission row. Data comes from meta.js and chunks/ (written by possession/viz.py).
(function () {
  "use strict";

  var M = window.VIEWER_META, Core = window.ViewerCore;
  if (!M) {
    document.body.textContent = "meta.js is missing: generate the viewer with  python -m possession.viz";
    return;
  }
  var $ = function (id) { return document.getElementById(id); };

  var FPS = M.fps, N = M.n_frames, BLOCKS = M.blocks, LAST_BLOCK = BLOCKS[BLOCKS.length - 1];
  var COLOUR = { H: "#3b82f6", A: "#f97316", D: "#9ca3af", X: "#4b5563" };
  var CODE_COLOUR = [COLOUR.X, COLOUR.H, COLOUR.A, COLOUR.D];
  var LABEL_NAME = { H: M.teams[0], A: M.teams[1], D: "Dead ball", X: "Abstain" };
  var STRIP = "#1f2430";   // background of a timeline strip where the match is in play
  var GUTTER = 124;        // room for the row names, left of the timelines
  var MARGIN = 8;
  var HALF_GAP = 10;       // pixels between the halves on the whole-match timeline
  var AXIS = 16;           // height of the clock ticks above the zoomed timeline

  // Whole-match arrays, one value per frame.
  var ballPresent = Core.bytes(M.ball_present);
  var nHome = Core.bytes(M.n_home), nAway = Core.bytes(M.n_away);
  var tagMask = new Uint8Array(N); // Smart Tagging phases covering the frame: 1 home, 2 away, 4 no team
  M.events.forEach(function (event) {
    var team = event.tags.Team;
    var bit = team === M.teams[0] ? 1 : team === M.teams[1] ? 2 : 4;
    for (var f = Math.max(0, event.s); f < Math.min(N, event.e); f++) tagMask[f] |= bit;
  });

  // Label series: the public ground truth (behind its toggle) and every CSV dropped on the page.
  var truth = null;
  if (M.ground_truth) {
    var truthCodes = new Uint8Array(N).fill(Core.NO_LABEL);
    for (var i = 0; i < M.ground_truth.labels.length; i++) {
      truthCodes[M.ground_truth.first + i] = Core.LABELS.indexOf(M.ground_truth.labels[i]);
    }
    truth = { name: "Ground truth", codes: truthCodes, truth: true };
  } else {
    $("optTruth").disabled = true;
  }
  var predictions = [];
  function allSeries() {
    return ($("optTruth").checked && truth ? [truth] : []).concat(predictions);
  }

  var state = { frame: BLOCKS[0][0], playing: false, videoOk: false, timerStart: 0, timerFrame: 0 };
  var video = $("video");
  var lines = Core.pitchLines(M.pitch);
  var tmp = [0, 0], pix = [0, 0];

  // The playable range ----------------------------------------------------------------------------

  // The nearest frame inside a half. From the gap between halves, `direction` picks the side.
  function clampFrame(frame, direction) {
    frame = Math.round(frame);
    if (frame < BLOCKS[0][0]) return BLOCKS[0][0];
    for (var i = 0; i < BLOCKS.length; i++) {
      if (frame <= BLOCKS[i][1]) {
        if (frame >= BLOCKS[i][0]) return frame;
        return direction < 0 ? BLOCKS[i - 1][1] : BLOCKS[i][0];
      }
    }
    return LAST_BLOCK[1];
  }
  function playable(frame) { return clampFrame(frame, 1) === frame; }

  // The clock -------------------------------------------------------------------------------------
  // With a video, the video is the clock: each frame it presents reports its media time. Without
  // one (not downloaded yet, or unreadable) a timer plays the tracking on its own.

  function speed() { return parseFloat($("speed").value); }
  function videoOffset() { return parseInt($("videoOffset").value, 10) || 0; }
  // The middle of the frame, so that rounding in the player cannot land on a neighbour.
  function videoTime(frame) { return (frame + videoOffset() + 0.5) / FPS; }

  function seek(frame, direction) {
    state.frame = clampFrame(frame, direction || 1);
    if (state.videoOk) video.currentTime = videoTime(state.frame);
    else restartTimer();
    render();
  }

  function restartTimer() {
    state.timerStart = performance.now();
    state.timerFrame = state.frame;
  }

  function setPlaying(on) {
    state.playing = on;
    $("play").textContent = on ? "Pause" : "Play";
    if (state.videoOk) {
      if (on) { video.playbackRate = speed(); video.play(); } else video.pause();
    }
    seek(state.frame);
  }

  // Playback has reached `frame`.
  function advanceTo(frame) {
    if (frame === state.frame) return;
    if (playable(frame)) { state.frame = frame; render(); return; }
    if (frame > LAST_BLOCK[1]) { state.frame = LAST_BLOCK[1]; setPlaying(false); return; }
    seek(frame, 1); // ran off the end of a half: carry on from the next one
  }

  var hasFrameCallback = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
  function onVideoFrame(now, info) {
    if (state.playing && state.videoOk && !video.seeking) {
      advanceTo(Math.round(info.mediaTime * FPS) - videoOffset());
    }
    video.requestVideoFrameCallback(onVideoFrame);
  }
  function tick(now) {
    if (state.playing) {
      if (!state.videoOk) {
        advanceTo(state.timerFrame + Math.floor((now - state.timerStart) / 1000 * FPS * speed()));
      } else if (!hasFrameCallback && !video.seeking) {
        advanceTo(Math.floor(video.currentTime * FPS) - videoOffset());
      }
    }
    requestAnimationFrame(tick);
  }

  video.addEventListener("loadedmetadata", function () {
    state.videoOk = true;
    $("videoMessage").textContent = "";
    video.currentTime = videoTime(state.frame);
    if (state.playing) { video.playbackRate = speed(); video.play(); }
    render();
  });
  video.addEventListener("error", function () {
    state.videoOk = false;
    restartTimer();
    $("videoMessage").textContent = "No playable video at " + M.video.path +
      ". Showing the tracking as the camera would see it.";
    render();
  });
  // The browser can pause or resume the video by itself (a hidden tab, a media key): follow it,
  // so the Play button never claims playback that is not happening.
  function followVideo() {
    if (!state.videoOk || state.playing === !video.paused) return;
    state.playing = !video.paused;
    $("play").textContent = state.playing ? "Pause" : "Play";
  }
  video.addEventListener("pause", followVideo);
  video.addEventListener("play", followVideo);
  if (hasFrameCallback) video.requestVideoFrameCallback(onVideoFrame);

  // Chunks ----------------------------------------------------------------------------------------

  var chunks = new Map(), pending = new Set(), available = new Set(M.chunks);
  function chunkOf(frame) { return Math.floor(frame / M.chunk_frames); }

  window.viewerChunk = function (index, payload) {
    chunks.set(index, Core.decodeChunk(payload));
    pending.delete(index);
    var here = chunkOf(state.frame);
    if (chunks.size > 8) {
      var far = null;
      chunks.forEach(function (_, k) {
        if (far === null || Math.abs(k - here) > Math.abs(far - here)) far = k;
      });
      chunks.delete(far);
    }
    if (index === here) render();
  };

  function need(index) {
    if (!available.has(index) || chunks.has(index) || pending.has(index)) return;
    pending.add(index);
    var script = document.createElement("script");
    script.src = "chunks/c" + ("000" + index).slice(-4) + ".js";
    script.onload = function () { script.remove(); };
    script.onerror = function () { pending.delete(index); toast("Could not load " + script.src); };
    document.head.appendChild(script);
  }

  // Drawing helpers -------------------------------------------------------------------------------

  // Size a canvas for the screen's pixel density; drawing is then in CSS pixels.
  function fit(canvas, width, height) {
    var ratio = window.devicePixelRatio || 1;
    var w = Math.round(width * ratio), h = Math.round(height * ratio);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return ctx;
  }

  function disc(ctx, x, y, r) { ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); }

  function esc(text) {
    return String(text).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c];
    });
  }

  function seconds(frames) { return (frames / FPS).toFixed(1) + " s"; }

  // The pitch -------------------------------------------------------------------------------------

  function drawPitch(chunk, frame, ball) {
    var canvas = $("pitch"), box = canvas.parentNode;
    var w = box.clientWidth, h = box.clientHeight;
    if (!w || $("pitchPane").hidden) return;
    var ctx = fit(canvas, w, h);
    var scale = Math.min(w * 0.94 / M.pitch[0], h * 0.9 / M.pitch[1]);
    var pw = M.pitch[0] * scale, ph = M.pitch[1] * scale, ox = (w - pw) / 2, oy = (h - ph) / 2;
    function X(x) { return ox + x * pw; }
    function Y(y) { return oy + y * ph; }

    ctx.fillStyle = "#0f3d24"; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#14532d"; ctx.fillRect(ox, oy, pw, ph);
    ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 1.2; ctx.setLineDash([]);
    lines.forEach(function (line) {
      ctx.beginPath();
      line.forEach(function (p, k) {
        if (k) ctx.lineTo(X(p[0]), Y(p[1])); else ctx.moveTo(X(p[0]), Y(p[1]));
      });
      ctx.stroke();
    });

    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    if (!chunk) {
      ctx.fillStyle = "#fff"; ctx.font = "14px system-ui";
      ctx.fillText("Loading…", w / 2, h / 2);
      return;
    }

    var r = Math.max(4, 1.1 * scale), numbers = $("optNumbers").checked;
    ctx.font = "600 " + Math.max(9, r * 1.2) + "px system-ui";
    for (var slot = 1; slot < M.n_objects; slot++) {
      if (!Core.point(chunk, M, frame, slot, tmp)) continue;
      disc(ctx, X(tmp[0]), Y(tmp[1]), r);
      ctx.fillStyle = COLOUR[M.track_team[slot - 1]]; ctx.fill();
      ctx.strokeStyle = "#000"; ctx.lineWidth = 1; ctx.stroke();
      if (numbers) { ctx.fillStyle = "#fff"; ctx.fillText(slot, X(tmp[0]), Y(tmp[1]) - r * 1.9); }
    }

    var missing = !ball || ball.frame !== frame;
    if (ball) {
      ctx.globalAlpha = missing ? Math.max(0.3, 1 - (frame - ball.frame) / (3 * FPS)) : 1;
      disc(ctx, X(ball.x), Y(ball.y), r * 0.65);
      if (missing) {
        ctx.setLineDash([3, 3]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.fillStyle = "#fff"; ctx.fill();
        ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5; ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    if (missing) {
      ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 4; ctx.strokeRect(2, 2, w - 4, h - 4);
      var text = ball
        ? "Ball not tracked · " + (frame - ball.frame) + " frames (" + seconds(frame - ball.frame) +
          ") · dashed ring: last seen"
        : "Ball not tracked";
      ctx.font = "600 13px system-ui";
      var tw = ctx.measureText(text).width + 16;
      ctx.fillStyle = "#ef4444"; ctx.fillRect((w - tw) / 2, 6, tw, 22);
      ctx.fillStyle = "#fff"; ctx.fillText(text, w / 2, 17);
    }
  }

  // The overlay on the video ----------------------------------------------------------------------

  function drawOverlay(chunk, frame, ball) {
    var canvas = $("overlay"), box = canvas.parentNode;
    var w = box.clientWidth, h = box.clientHeight;
    if (!w || $("videoPane").hidden) return;
    var ctx = fit(canvas, w, h);
    ctx.clearRect(0, 0, w, h);
    if (!chunk) return;
    var s = w / M.video.width; // video pixels -> CSS pixels

    if ($("optLines").checked) {
      ctx.strokeStyle = "rgba(34,211,238,0.8)"; ctx.lineWidth = 1.5; ctx.setLineDash([]);
      lines.forEach(function (line) {
        var drawing = false;
        ctx.beginPath();
        line.forEach(function (p) {
          if (!Core.project(chunk, frame, p[0], p[1], pix)) { drawing = false; return; }
          if (drawing) ctx.lineTo(pix[0] * s, pix[1] * s); else ctx.moveTo(pix[0] * s, pix[1] * s);
          drawing = true;
        });
        ctx.stroke();
      });
    }
    if (!$("optOverlay").checked) return;

    var r = Math.max(5, 15 * s), numbers = $("optNumbers").checked;
    ctx.font = "600 " + Math.max(9, 18 * s) + "px system-ui";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (var slot = 1; slot < M.n_objects; slot++) {
      if (!Core.point(chunk, M, frame, slot, tmp)) continue;
      if (!Core.project(chunk, frame, tmp[0], tmp[1], pix)) continue;
      // A ring at the feet, so the player in the picture stays visible.
      disc(ctx, pix[0] * s, pix[1] * s, r);
      ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.lineWidth = 4; ctx.stroke();
      ctx.strokeStyle = COLOUR[M.track_team[slot - 1]]; ctx.lineWidth = 2; ctx.stroke();
      if (numbers) {
        ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000"; ctx.lineWidth = 3;
        ctx.strokeText(slot, pix[0] * s, pix[1] * s - r * 1.8);
        ctx.fillText(slot, pix[0] * s, pix[1] * s - r * 1.8);
      }
    }
    if (ball && Core.project(chunk, frame, ball.x, ball.y, pix)) {
      var missing = ball.frame !== frame;
      ctx.globalAlpha = missing ? Math.max(0.3, 1 - (frame - ball.frame) / (3 * FPS)) : 1;
      disc(ctx, pix[0] * s, pix[1] * s, Math.max(4, 9 * s));
      ctx.strokeStyle = "rgba(0,0,0,0.8)"; ctx.lineWidth = 4; ctx.stroke();
      if (missing) ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "#fde047"; ctx.lineWidth = 2; ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
  }

  // The timelines ---------------------------------------------------------------------------------
  // Two canvases with the same rows: a window around the current frame, and the whole match (the
  // halves side by side, nothing drawn for pre-match and half-time).

  function rows() {
    var list = [{ kind: "ball", name: "Ball missing", h: 8 },
                { kind: "count", name: "Players tracked", h: 16 },
                { kind: "tags", name: "Smart Tagging", h: 15 }];
    allSeries().forEach(function (series) {
      list.push({ kind: "labels", name: series.name, series: series, h: 10 });
    });
    return list;
  }
  function rowsHeight(list) {
    return list.reduce(function (total, row) { return total + row.h + 4; }, 4);
  }
  function zoomFrames() { return parseInt($("zoom").value, 10) * FPS; }

  // One pixel column of one row, for frames [a, b).
  function paint(ctx, row, x, y, a, b) {
    var f;
    ctx.fillStyle = STRIP; ctx.fillRect(x, y, 1, row.h);
    if (row.kind === "ball") {
      var gone = 0;
      for (f = a; f < b; f++) gone += 1 - ballPresent[f];
      if (gone) {
        ctx.fillStyle = "rgba(239,68,68," + (0.3 + 0.7 * gone / (b - a)) + ")";
        ctx.fillRect(x, y, 1, row.h);
      }
    } else if (row.kind === "count") {
      var mid = (a + b) >> 1, total = nHome[mid] + nAway[mid];
      var bar = Math.min(1, total / 26) * row.h;
      ctx.fillStyle = total > 22 ? "#eab308" : "#a1a1aa"; // more than 22: the tracker duplicated someone
      ctx.fillRect(x, y + row.h - bar, 1, bar);
    } else if (row.kind === "tags") {
      var mask = 0, lane = row.h / 3;
      for (f = a; f < b; f++) mask |= tagMask[f];
      if (mask & 1) { ctx.fillStyle = COLOUR.H; ctx.fillRect(x, y, 1, lane); }
      if (mask & 2) { ctx.fillStyle = COLOUR.A; ctx.fillRect(x, y + lane, 1, lane); }
      if (mask & 4) { ctx.fillStyle = COLOUR.D; ctx.fillRect(x, y + 2 * lane, 1, lane); }
    } else {
      var counts = [0, 0, 0, 0], best = -1, codes = row.series.codes;
      for (f = a; f < b; f++) if (codes[f] !== Core.NO_LABEL) counts[codes[f]]++;
      for (var c = 0; c < 4; c++) if (counts[c] && (best < 0 || counts[c] > counts[best])) best = c;
      if (best < 0) { ctx.fillStyle = "#15171c"; ctx.fillRect(x, y, 1, row.h); }
      else if (best > 0) { ctx.fillStyle = CODE_COLOUR[best]; ctx.fillRect(x, y, 1, row.h); }
      else { ctx.fillStyle = CODE_COLOUR[0]; ctx.fillRect(x, y + row.h / 2 - 1, 1, 2); }
    }
  }

  // range(x) gives the frames [a, b) under pixel column x, or null where nothing is playable.
  function drawStrips(ctx, width, list, top, range) {
    var y = top;
    ctx.font = "11px system-ui"; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    list.forEach(function (row) {
      var name = row.name.length > 19 ? row.name.slice(0, 18) + "…" : row.name;
      ctx.fillStyle = "#8f96a3";
      ctx.fillText(name, GUTTER - 8, y + row.h / 2);
      for (var x = GUTTER; x < width - MARGIN; x++) {
        var frames = range(x);
        if (frames) paint(ctx, row, x, y, frames[0], frames[1]);
      }
      y += row.h + 4;
    });
  }

  function overviewSpans(width) {
    var total = BLOCKS.reduce(function (sum, b) { return sum + b[1] - b[0] + 1; }, 0);
    var per = (width - MARGIN - GUTTER - HALF_GAP * (BLOCKS.length - 1)) / total;
    var x = GUTTER;
    return BLOCKS.map(function (block) {
      var span = { first: block[0], last: block[1], x0: x, per: per,
                   x1: x + (block[1] - block[0] + 1) * per };
      x = span.x1 + HALF_GAP;
      return span;
    });
  }
  function overviewX(spans, frame) {
    frame = clampFrame(frame, 1);
    for (var i = 0; i < spans.length; i++) {
      if (frame <= spans[i].last) return spans[i].x0 + (frame - spans[i].first) * spans[i].per;
    }
    return spans[spans.length - 1].x1;
  }
  function overviewFrame(spans, x) {
    for (var i = 0; i < spans.length; i++) {
      if (x < spans[i].x1 || i === spans.length - 1) {
        return clampFrame(spans[i].first + (x - spans[i].x0) / spans[i].per, 1);
      }
    }
  }

  var overviewCache = null; // the whole-match strips, redrawn only when the size or the rows change

  function drawTimelines(frame) {
    var list = rows();
    var detail = $("detail"), width = detail.clientWidth;
    if (!width) return;

    // The window around the current frame.
    var span = zoomFrames(), inner = width - MARGIN - GUTTER, first = frame - span / 2;
    var ctx = fit(detail, width, AXIS + rowsHeight(list));
    ctx.clearRect(0, 0, width, AXIS + rowsHeight(list));
    drawStrips(ctx, width, list, AXIS, function (x) {
      var a = Math.floor(first + (x - GUTTER) / inner * span);
      var b = Math.max(a + 1, Math.floor(first + (x + 1 - GUTTER) / inner * span));
      return playable(a) && playable(b - 1) ? [a, b] : null;
    });
    var step = { 20: 5, 60: 10, 300: 60 }[$("zoom").value] * FPS;
    ctx.font = "10px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    BLOCKS.forEach(function (block, half) {
      var k = Math.max(0, Math.ceil((first - block[0]) / step));
      for (var f = block[0] + k * step; f <= block[1] && f < first + span; f += step) {
        var x = GUTTER + (f - first) / span * inner;
        ctx.fillStyle = "rgba(255,255,255,0.18)"; ctx.fillRect(x, AXIS, 1, rowsHeight(list) - 4);
        ctx.fillStyle = "#8f96a3";
        if (x > GUTTER + 14 && x < width - MARGIN - 14) {
          ctx.fillText(Core.clockText((f - block[0]) / FPS + half * 45 * 60), x, AXIS / 2);
        }
      }
    });
    ctx.fillStyle = "#fff"; ctx.fillRect(GUTTER + inner / 2, 2, 1.5, AXIS + rowsHeight(list) - 4);
    state.detail = { first: first, span: span, inner: inner };

    // The whole match.
    var overview = $("overview"), height = rowsHeight(list);
    var key = width + "x" + height + "x" + (window.devicePixelRatio || 1);
    var spans = overviewSpans(width);
    if (!overviewCache || overviewCache.key !== key) {
      var canvas = document.createElement("canvas");
      var cacheCtx = fit(canvas, width, height);
      drawStrips(cacheCtx, width, list, 2, function (x) {
        for (var i = 0; i < spans.length; i++) {
          var sp = spans[i];
          if (x >= sp.x0 && x < sp.x1) {
            var a = sp.first + Math.floor((x - sp.x0) / sp.per);
            var b = Math.min(sp.last + 1, sp.first + Math.ceil((x + 1 - sp.x0) / sp.per));
            return b > a ? [a, b] : null;
          }
        }
        return null;
      });
      overviewCache = { key: key, canvas: canvas };
    }
    ctx = fit(overview, width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(overviewCache.canvas, 0, 0, width, height);
    var x0 = overviewX(spans, frame - span / 2), x1 = overviewX(spans, frame + span / 2);
    ctx.fillStyle = "rgba(255,255,255,0.18)"; ctx.fillRect(x0, 0, Math.max(2, x1 - x0), height);
    ctx.fillStyle = "#fff"; ctx.fillRect(overviewX(spans, frame), 0, 1.5, height);
    state.spans = spans;
  }

  // The side panel and the header ------------------------------------------------------------------

  function setHtml(element, html) {
    if (element.lastHtml !== html) { element.innerHTML = html; element.lastHtml = html; }
  }

  function chip(letter, text) {
    return "<span class=\"chip " + (letter || "none") + "\">" + esc(text) + "</span>";
  }

  function updateSide(frame, ball) {
    var series = allSeries();
    setHtml($("labels"), series.length ? series.map(function (s) {
      var code = s.codes[frame], letter = code === Core.NO_LABEL ? null : Core.LABELS[code];
      return "<div class=\"row\">" + chip(letter, letter || "–") +
        "<span class=\"name\" title=\"" + esc(s.name) + "\">" + esc(s.name) + "</span>" +
        "<span class=\"dim\">" + (letter ? esc(LABEL_NAME[letter]) : "not labelled") + "</span>" +
        (s.truth ? "" : "<button class=\"remove\" data-remove=\"" + predictions.indexOf(s) +
                        "\" title=\"Remove\">×</button>") + "</div>";
    }).join("") : "<p class=\"dim\">No prediction loaded.</p>");

    var ballLine = !ball ? "<span class=\"alert\">Ball not tracked</span>"
      : ball.frame === frame ? "Ball tracked"
      : "<span class=\"alert\">Ball not tracked for " + (frame - ball.frame) + " frames (" +
        seconds(frame - ball.frame) + ")</span>";
    setHtml($("tracking"),
      "<div class=\"row\">" + ballLine + "</div>" +
      "<div class=\"row\">" + chip("H", nHome[frame]) + chip("A", nAway[frame]) +
      "<span>" + (nHome[frame] + nAway[frame]) + " players tracked</span></div>");

    var active = M.events.filter(function (event) { return event.s <= frame && frame < event.e; });
    setHtml($("tagging"), active.length ? active.map(function (event) {
      var team = event.tags.Team;
      var letter = team === M.teams[0] ? "H" : team === M.teams[1] ? "A" : null;
      var tags = Object.keys(event.tags).filter(function (group) {
        return group !== "Team" && group !== "Half";
      }).map(function (group) { return esc(group) + ": " + esc(event.tags[group]); });
      tags.push("ends in " + Math.ceil((event.e - frame) / FPS) + " s");
      return "<div class=\"phase\">" + chip(letter, team || "no team") + " " + esc(event.code) +
        "<div class=\"tags\">" + tags.join(" · ") + "</div></div>";
    }).join("") : "<p class=\"dim\">No phase on this frame.</p>");
  }

  function updateHeader(frame) {
    var clock = Core.matchClock(frame, BLOCKS, FPS);
    $("clock").textContent = clock
      ? (clock.half === 1 ? "1st half  " : "2nd half  ") + Core.clockText(clock.seconds) : "–";
    $("frameInfo").textContent = "frame " + frame + " of " + (N - 1);
    if (document.activeElement !== $("frameBox")) $("frameBox").value = frame;
  }

  function render() {
    var frame = state.frame, index = chunkOf(frame);
    need(index); need(index + 1);
    var chunk = chunks.get(index) || null;
    var ball = chunk ? Core.lastBall(chunk, M, frame) : null;
    updateHeader(frame);
    drawPitch(chunk, frame, ball);
    drawOverlay(chunk, frame, ball);
    drawTimelines(frame);
    if (chunk) updateSide(frame, ball);
  }

  // Layout ----------------------------------------------------------------------------------------

  function layout() {
    var list = rows();
    $("detail").style.height = (AXIS + rowsHeight(list)) + "px";
    $("overview").style.height = rowsHeight(list) + "px";
    ["videoPane", "pitchPane"].forEach(function (id) {
      var pane = $(id);
      if (pane.hidden) return;
      var box = pane.querySelector(".box");
      box.style.width = box.style.height = "0px"; // measure the pane, not what the box made of it
      var width = Math.floor(Math.min(pane.clientWidth, pane.clientHeight * 16 / 9));
      box.style.width = width + "px";
      box.style.height = Math.floor(width * 9 / 16) + "px";
    });
    overviewCache = null;
    render();
  }

  function seriesChanged() {
    var select = $("jumpSeries"), keep = select.value;
    var options = [["ball", "Ball lost / found"], ["tags", "Smart Tagging phase"]];
    allSeries().forEach(function (s, k) { options.push(["s" + k, s.name]); });
    select.innerHTML = options.map(function (o) {
      return "<option value=\"" + o[0] + "\">" + esc(o[1]) + "</option>";
    }).join("");
    if (options.some(function (o) { return o[0] === keep; })) select.value = keep;
    layout();
  }

  // Navigation ------------------------------------------------------------------------------------

  function step(delta) {
    if (state.playing) setPlaying(false);
    seek(state.frame + delta, delta);
  }

  // To the nearest frame, in `direction`, where the chosen series changes.
  function jump(direction) {
    var what = $("jumpSeries").value, frame = state.frame, target = null;
    if (what === "tags") {
      M.events.forEach(function (event) {
        var ahead = direction > 0 ? event.s > frame : event.s < frame;
        if (ahead && playable(event.s) &&
            (target === null || (direction > 0 ? event.s < target : event.s > target))) {
          target = event.s;
        }
      });
    } else {
      var values = what === "ball" ? ballPresent : allSeries()[parseInt(what.slice(1), 10)].codes;
      for (var f = frame + direction; f >= BLOCKS[0][0] && f <= LAST_BLOCK[1]; f += direction) {
        if (values[f] !== values[f - 1] && playable(f) && playable(f - 1)) { target = f; break; }
      }
    }
    if (target === null) return;
    if (state.playing) setPlaying(false);
    seek(target, direction);
  }

  document.addEventListener("keydown", function (event) {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName) && event.target.type !== "checkbox") {
      return;
    }
    if (event.key === " ") setPlaying(!state.playing);
    else if (event.key === "ArrowRight") step(event.shiftKey ? FPS : 1);
    else if (event.key === "ArrowLeft") step(event.shiftKey ? -FPS : -1);
    else return;
    event.preventDefault();
  });

  $("play").addEventListener("click", function () { setPlaying(!state.playing); this.blur(); });
  $("half1").addEventListener("click", function () { seek(BLOCKS[0][0]); });
  $("half2").addEventListener("click", function () { seek(LAST_BLOCK[0]); });
  $("prevChange").addEventListener("click", function () { jump(-1); });
  $("nextChange").addEventListener("click", function () { jump(1); });
  $("speed").addEventListener("change", function () {
    video.playbackRate = speed();
    restartTimer();
  });
  $("frameBox").addEventListener("change", function () {
    var frame = parseInt(this.value, 10);
    if (!isNaN(frame)) seek(frame);
    this.blur();
  });
  $("videoOffset").addEventListener("change", function () { seek(state.frame); });
  $("zoom").addEventListener("change", render);
  ["optOverlay", "optNumbers", "optLines"].forEach(function (id) {
    $(id).addEventListener("change", render);
  });
  $("optTruth").addEventListener("change", seriesChanged);

  $("viewButtons").addEventListener("click", function (event) {
    var view = event.target.getAttribute("data-view");
    if (!view) return;
    Array.prototype.forEach.call(this.children, function (button) {
      button.classList.toggle("on", button === event.target);
    });
    $("videoPane").hidden = view === "pitch";
    $("pitchPane").hidden = view === "video";
    layout();
  });

  $("detail").addEventListener("click", function (event) {
    var d = state.detail, x = event.clientX - this.getBoundingClientRect().left;
    if (d && x >= GUTTER) seek(d.first + (x - GUTTER) / d.inner * d.span);
  });
  function scrub(event) {
    var x = event.clientX - $("overview").getBoundingClientRect().left;
    if (state.spans && x >= GUTTER) seek(overviewFrame(state.spans, x));
  }
  $("overview").addEventListener("mousedown", function (event) {
    scrub(event);
    function stop() {
      window.removeEventListener("mousemove", scrub);
      window.removeEventListener("mouseup", stop);
    }
    window.addEventListener("mousemove", scrub);
    window.addEventListener("mouseup", stop);
  });

  // Predictions -----------------------------------------------------------------------------------

  var toastTimer = null;
  function toast(text) {
    $("toast").textContent = text;
    $("toast").hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { $("toast").hidden = true; }, 8000);
  }

  function loadFiles(files) {
    Array.prototype.forEach.call(files, function (file) {
      file.text().then(function (text) {
        var parsed = Core.parseLabels(text, N);
        if (parsed.error) { toast(file.name + " is not a submission CSV: " + parsed.error); return; }
        predictions.push({ name: file.name.replace(/\.csv$/i, ""), codes: parsed.codes });
        seriesChanged();
      });
    });
  }

  $("loadButton").addEventListener("click", function () { $("fileInput").click(); });
  $("fileInput").addEventListener("change", function () { loadFiles(this.files); this.value = ""; });
  $("labels").addEventListener("click", function (event) {
    var index = event.target.getAttribute("data-remove");
    if (index === null) return;
    predictions.splice(parseInt(index, 10), 1);
    seriesChanged();
  });
  document.addEventListener("dragover", function (event) {
    event.preventDefault();
    document.body.classList.add("dropping");
  });
  document.addEventListener("dragleave", function (event) {
    if (!event.relatedTarget) document.body.classList.remove("dropping");
  });
  document.addEventListener("drop", function (event) {
    event.preventDefault();
    document.body.classList.remove("dropping");
    loadFiles(event.dataTransfer.files);
  });

  // Start -----------------------------------------------------------------------------------------

  document.title = $("title").textContent = M.game + " viewer";
  $("frameBox").max = N - 1;
  window.addEventListener("resize", layout);
  seriesChanged();
  video.src = M.video.uri;
  requestAnimationFrame(tick);
})();

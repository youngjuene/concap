/* The console session: seven screens, three controls, one caption band.

   Two halves, split exactly where docs/v2-console/spec-system.md §2 splits the system. Everything
   from a gesture to the skeleton is deterministic and recomputes here, in the
   browser, before the finger lifts: muting a source redraws the crossfader and
   the numbered list from data already in hand, because the shot's regimes for
   every admitted subset arrived with the shot. Only prose is slow, and prose
   is only ever requested by Show caption.

   The controls never touch a number. The server sent bands and registers as
   phrases and regimes as orders with spans; α exists here as one float used to
   carry a selection across an admission change, and it is never drawn, never
   labelled, and never sent as a setting — what is sent is the regime index,
   because within a regime α is unidentifiable (§7).

   State lives in one snapshot object, POSTed with the event stream on a debounce
   and on visibilitychange, so a crash resumes at the interrupted step (§7). */

const AUTOSAVE_MS = 250;
const SOLO_MS = 120; // a press past this is an inspection, not a mute
const GRAINS = ["itemized", "grouped", "scene", "atmospheric"];
const UNNAMED_GRAIN = "atmospheric";
const SNAPSHOT_SCHEMA = "dpo.caption-console-snapshot/v1";
// The one string the page must hold itself: /api/session is gated on the
// participant, so the line for a missing one cannot come from copy.py.
const PARTICIPANT_MISSING = "Open this page with a participant identifier.";
// The root zoom fitScreen sets; rect deltas are in zoomed units, transforms are not.
let pageZoom = 1;

const state = {
  participant: null,
  session: null,
  strings: null,
  cards: null,
  snapshot: null,
  shots: {},      // "clip/shot" -> the /api/shot payload
  authoring: null,
  pending: [],
  timer: null,
  playthrough: false,
};

/* ---- Small DOM helpers ---------------------------------------------------- */

function $(id) {
  return document.getElementById(id);
}

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([key, value]) => {
    if (value === null || value === undefined) return;
    if (key === "text") node.textContent = String(value);
    else if (key === "class") node.className = String(value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, String(value));
  });
  (children || []).forEach((child) => node.appendChild(child));
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function show(id, visible) {
  $(id).hidden = !visible;
}

function button(label, attrs) {
  return el("button", { type: "button", class: "button", text: label, ...(attrs || {}) });
}

/* ---- The wire ------------------------------------------------------------- */

const api = {
  emit(type, payload) {
    state.pending.push({ type, at: Date.now(), ...(payload || {}) });
    api.save();
  },
  save() {
    if (state.timer) clearTimeout(state.timer);
    state.timer = setTimeout(() => api.flush(false), AUTOSAVE_MS);
  },
  flush(keepalive) {
    if (state.timer) clearTimeout(state.timer);
    state.timer = null;
    const events = state.pending;
    state.pending = [];
    const body = JSON.stringify({
      participant: state.participant,
      events,
      snapshot: state.snapshot,
    });
    // A keepalive flush must survive the page going away, so it cannot await.
    if (keepalive && navigator.sendBeacon) {
      navigator.sendBeacon("/api/events", new Blob([body], { type: "application/json" }));
      return Promise.resolve();
    }
    return fetch("/api/events", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
      keepalive: Boolean(keepalive),
    }).catch(() => {
      // Nothing is lost: the events go back on the queue for the next flush.
      state.pending = events.concat(state.pending);
    });
  },
  get(path) {
    return fetch(path).then((response) => {
      if (!response.ok) throw new Error(String(response.status));
      return response.json();
    });
  },
  post(path, payload) {
    return fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ participant: state.participant, ...payload }),
    }).then((response) => {
      if (!response.ok) throw new Error(String(response.status));
      return response.json();
    });
  },
};

/* ---- Snapshot ------------------------------------------------------------- */

function newSnapshot() {
  return {
    schema: SNAPSHOT_SCHEMA,
    participant: state.participant,
    session_id: state.session.session_id,
    config_hash: state.session.config_hash,
    screen: "intro",
    clip_index: 0,
    shot_index: 0,
    // The start state configuration assigned, recorded as a covariate (§11).
    opened: {},
    committed: {},
    revisions: {},
    choices: {},
    probe: "",
    done_at: null,
  };
}

function currentClip() {
  return state.session.clips[state.snapshot.clip_index];
}

function shotKey(clipId, shotId) {
  return clipId + "/" + shotId;
}

function committedFor(clipId, shotId) {
  const clip = state.snapshot.committed[clipId];
  return clip ? clip[shotId] : undefined;
}

/* ---- The authoring model -------------------------------------------------- */

function subsetKey(ids) {
  return ids.slice().sort().join("+");
}

function regimesFor(shotData, admitted) {
  if (!admitted.length) return [];
  return shotData.regimes[subsetKey(admitted)] || [];
}

function regimeAt(regimes, alpha) {
  for (let index = 0; index < regimes.length; index += 1) {
    const [low, high] = regimes[index].span;
    if (alpha >= low && alpha <= high) return index;
  }
  return Math.max(0, regimes.length - 1);
}

// α is kept as the midpoint of the selected regime so that when admission
// changes the count of regimes, the selection carries to the regime whose span
// contains it rather than resetting to an end of the axis.
function carryAlpha(regimes, index) {
  const [low, high] = regimes[index].span;
  return (low + high) / 2;
}

function newAuthoring(clip, shot, opening, committed) {
  const data = state.shots[shotKey(clip.clip_id, shot.shot_id)];
  const stored = committed && committed.settings;
  const admitted = stored
    ? stored.admitted.slice()
    : (opening.admitted || data.sources.map((source) => source.id)).slice();
  const grain = stored ? stored.grain : opening.grain;
  const regimes = regimesFor(data, admitted);
  const alpha = stored && regimes.length ? carryAlpha(regimes, Math.min(stored.regime, regimes.length - 1)) : opening.alpha;
  return {
    clip,
    shot,
    data,
    admitted,
    grain,
    alpha,
    solo: null,
    band: null,   // {caption, key} — the prose currently read
    writing: false,
    error: null,
    revising: false,
  };
}

function currentRegimes(a) {
  return regimesFor(a.data, a.admitted);
}

function currentRegimeIndex(a) {
  const regimes = currentRegimes(a);
  return regimes.length ? regimeAt(regimes, a.alpha) : 0;
}

function currentOrder(a) {
  if (a.grain === UNNAMED_GRAIN) return [];
  const regimes = currentRegimes(a);
  return regimes.length ? regimes[currentRegimeIndex(a)].order : [];
}

function settingsOf(a) {
  if (a.grain === UNNAMED_GRAIN) return { grain: a.grain, admitted: [], regime: 0 };
  return { grain: a.grain, admitted: a.admitted.slice(), regime: currentRegimeIndex(a) };
}

function keyOf(a) {
  const s = settingsOf(a);
  if (s.grain === UNNAMED_GRAIN) return UNNAMED_GRAIN;
  return s.grain + "|" + s.regime + "|" + s.admitted.slice().sort().join(",");
}

// Freshness is a comparison, not a flag: a participant who changes the console
// and changes it back has a fresh caption again without asking for one.
function isFresh(a) {
  return Boolean(a.band) && a.band.key === keyOf(a);
}

function sourceById(a, id) {
  return a.data.sources.find((source) => source.id === id);
}

/* ---- Screens -------------------------------------------------------------- */

function phaseOf(screen) {
  if (screen === "watch1") return 0;
  if (screen === "author" || screen === "author_intro") return 1;
  if (screen === "watch2") return 2;
  if (screen === "check") return 3;
  return -1;
}

function render() {
  const screen = state.snapshot.screen;
  stopLoop();
  renderRail(screen);
  ["stage-block", "band", "console", "skeleton", "actions", "column"].forEach((id) => show(id, false));
  clear($("column"));
  clear($("buttons"));
  clear($("stage-actions"));
  $("criterion").textContent = "";
  const screens = {
    intro: renderIntro,
    watch1: () => renderViewing("first"),
    author_intro: renderAuthorIntro,
    author: renderAuthoring,
    watch2: () => renderViewing("second"),
    check: renderCheck,
    probe: renderProbe,
    done: renderDone,
  };
  (screens[screen] || renderIntro)();
}

function enter(screen) {
  state.snapshot.screen = screen;
  api.emit("screen.enter", { screen });
  render();
}

function renderRail(screen) {
  const rail = clear($("rail"));
  const current = phaseOf(screen);
  state.strings.phase_rail.forEach((label, index) => {
    const marks = index === current ? "phase current" : index < current ? "phase done" : "phase";
    rail.appendChild(el("span", { class: marks, text: label }));
  });
}

function card(key, action) {
  const copy = state.cards[key];
  const node = el("div", { class: "card" }, [
    el("h1", { class: "heading", text: copy.heading }),
    ...copy.body.map((line) => el("p", { class: "body", text: line })),
  ]);
  node.appendChild(el("div", { class: "buttons" }, [action]));
  return node;
}

function renderIntro() {
  show("column", true);
  $("column").appendChild(
    card("intro", button(state.strings.actions.begin, { class: "button primary", onclick: onBegin })),
  );
}

function onBegin() {
  api.emit("session.begin", {});
  const shell = document.documentElement;
  if (shell.requestFullscreen) shell.requestFullscreen().catch(() => undefined);
  enter("watch1");
}

function renderAuthorIntro() {
  show("column", true);
  $("column").appendChild(
    card("author", button(state.strings.actions.start, { class: "button primary", onclick: () => openShot(0) })),
  );
}

/* ---- Viewings (§3.2) ------------------------------------------------------ */

let loopHandle = null;

function stopLoop() {
  if (loopHandle) cancelAnimationFrame(loopHandle);
  loopHandle = null;
  const video = $("video");
  if (video) video.onended = null;
}

function loadMedia(clip) {
  const video = $("video");
  const source = "/media/" + encodeURIComponent(clip.clip_id);
  if (video.getAttribute("data-clip") !== clip.clip_id) {
    video.setAttribute("data-clip", clip.clip_id);
    video.src = source;
  }
  return video;
}

function shotAt(clip, ms) {
  for (let index = 0; index < clip.shots.length; index += 1) {
    const shot = clip.shots[index];
    if (ms >= shot.start_ms && ms < shot.end_ms) return index;
  }
  return clip.shots.length - 1;
}

function renderViewing(which) {
  const clip = currentClip();
  show("stage-block", true);
  show("band", true);
  show("actions", true);
  const video = loadMedia(clip);
  video.loop = false;
  video.controls = false;
  $("placed").hidden = true;
  $("band-text").textContent = "";
  $("band-note").textContent =
    which === "first" ? state.strings.eyebrows.first_viewing : state.strings.eyebrows.second_viewing;
  $("band").className = "band empty";

  if (which === "second") state.playthrough = state.playthrough || false;
  renderViewingActions(which);

  video.onended = () => {
    if (which === "second") state.playthrough = true;
    api.emit("viewing.ended", { clip_id: clip.clip_id, which });
    renderViewingActions(which);
  };
  video.play().catch(() => undefined);
  api.emit("viewing.play", { clip_id: clip.clip_id, which });

  const tick = () => {
    const ms = video.currentTime * 1000;
    const total = clip.shots[clip.shots.length - 1].end_ms;
    $("played").style.width = Math.min(100, (ms / total) * 100) + "%";
    const index = shotAt(clip, ms);
    const shot = clip.shots[index];
    $("shot-label").textContent = "Shot " + (index + 1) + " of " + clip.shots.length;
    const text =
      which === "first"
        ? shot.raw_caption
        : (committedFor(clip.clip_id, shot.shot_id) || {}).caption;
    $("placed-text").textContent = text || "";
    $("placed").hidden = !text;
    loopHandle = requestAnimationFrame(tick);
  };
  loopHandle = requestAnimationFrame(tick);
}

function renderViewingActions(which) {
  const clip = currentClip();
  const video = $("video");
  const buttons = clear($("buttons"));
  const pause = button(video.paused ? state.strings.helpers.replay : state.strings.helpers.pause, {
    onclick: () => {
      if (video.paused) {
        video.play().catch(() => undefined);
        api.emit("viewing.play", { clip_id: clip.clip_id, which });
      } else {
        video.pause();
        api.emit("viewing.pause", { clip_id: clip.clip_id, which });
      }
      renderViewingActions(which);
    },
  });
  buttons.appendChild(pause);

  if (which === "second") {
    // Revise pauses playback and opens the console for the current shot (§3.2).
    buttons.appendChild(
      button(state.strings.actions.revise, {
        onclick: () => {
          video.pause();
          const index = shotAt(clip, video.currentTime * 1000);
          api.emit("revise.open", { clip_id: clip.clip_id, shot_id: clip.shots[index].shot_id });
          openShot(index, true);
        },
      }),
    );
    // Finish stays disabled until one full playthrough (§3.2).
    buttons.appendChild(
      button(state.strings.actions.finish, {
        class: "button primary",
        disabled: state.playthrough ? null : "disabled",
        onclick: () => enter("check"),
      }),
    );
  } else {
    const done = video.ended;
    // The author intro is an instruction card, so it comes once. On a later
    // clip the participant already knows what shaping is and goes straight in.
    const first = state.snapshot.clip_index === 0;
    buttons.appendChild(
      button(state.strings.actions.start, {
        class: "button primary",
        disabled: done ? null : "disabled",
        onclick: () => (first ? enter("author_intro") : openShot(0)),
      }),
    );
  }
}

/* ---- Authoring (§3.2, §4) -------------------------------------------------- */

function openShot(index, revising) {
  const clip = currentClip();
  state.snapshot.shot_index = index;
  const shot = clip.shots[index];
  const key = shotKey(clip.clip_id, shot.shot_id);
  const load = state.shots[key]
    ? Promise.resolve(state.shots[key])
    : api.get("/api/shot/" + encodeURIComponent(clip.clip_id) + "/" + encodeURIComponent(shot.shot_id) +
        "?participant=" + encodeURIComponent(state.participant));
  load.then((data) => {
    state.shots[key] = data;
    const opening = clip.opening || { grain: "itemized", alpha: 0.5, admitted: null };
    state.authoring = newAuthoring(clip, shot, opening, committedFor(clip.clip_id, shot.shot_id));
    state.authoring.revising = Boolean(revising);
    if (!state.snapshot.opened[key]) {
      state.snapshot.opened[key] = { grain: opening.grain, alpha: opening.alpha, admitted: opening.admitted };
    }
    api.emit("shot.open", { clip_id: clip.clip_id, shot_id: shot.shot_id, revising: Boolean(revising) });
    enter("author");
  }).catch(() => {
    // The console is the whole step, so there is nothing to fall back to:
    // say what happened and offer the way forward (§5, the error string).
    notice(state.strings.error);
  });
}

function renderAuthoring() {
  const a = state.authoring;
  if (!a) return;
  show("stage-block", true);
  show("band", true);
  show("console", true);
  show("skeleton", true);
  show("actions", true);
  startShotLoop();
  renderBand();
  renderConsole();
  renderRows();
  renderAuthoringActions();
}

// The current shot loops so the moment stays present while its caption is
// shaped (§3.2).
function startShotLoop() {
  const a = state.authoring;
  const video = loadMedia(a.clip);
  video.loop = false;
  const start = a.shot.start_ms / 1000;
  const end = a.shot.end_ms / 1000;
  if (video.currentTime < start || video.currentTime > end) video.currentTime = start;
  video.play().catch(() => undefined);
  const tick = () => {
    if (video.currentTime >= end) video.currentTime = start;
    const span = a.shot.end_ms - a.shot.start_ms;
    const played = video.currentTime * 1000 - a.shot.start_ms;
    $("played").style.width = Math.max(0, Math.min(100, (played / span) * 100)) + "%";
    loopHandle = requestAnimationFrame(tick);
  };
  $("shot-label").textContent =
    "Shot " + (state.snapshot.shot_index + 1) + " of " + a.clip.shots.length;
  $("placed").hidden = true;
  loopHandle = requestAnimationFrame(tick);
}

function renderBand() {
  const a = state.authoring;
  const band = $("band");
  const text = $("band-text");
  const note = $("band-note");
  if (a.solo && a.solo.caption) {
    band.className = "band auditioning";
    text.textContent = a.solo.caption;
    note.textContent = "";
    return;
  }
  if (a.error) {
    band.className = "band";
    text.textContent = "";
    note.textContent = a.error;
    return;
  }
  if (!a.band) {
    band.className = "band empty";
    text.textContent = "";
    note.textContent = state.strings.helpers.adjust;
    return;
  }
  const fresh = isFresh(a);
  band.className = fresh ? "band" : "band stale";
  text.textContent = a.band.caption;
  // Stale is dimmed as well as labelled, so it reads without colour.
  note.textContent = fresh ? "" : state.strings.helpers.adjust;
}

function renderConsole() {
  const a = state.authoring;
  const unnamed = a.grain === UNNAMED_GRAIN;
  // At the atmospheric detent nothing is named, so these two controls visibly
  // gray and stop responding (§4.3).
  $("sources-control").className = unnamed ? "control sources inert" : "control sources";
  $("balance-control").className = unnamed ? "control balance inert" : "control balance";

  const strip = clear($("strip"));
  a.data.sources.forEach((source) => {
    const muted = a.admitted.indexOf(source.id) < 0;
    const node = el("button", {
      type: "button",
      class: "source" + (muted ? " muted" : "") + (a.solo && a.solo.id === source.id ? " solo" : ""),
      text: source.token,
      "aria-pressed": muted ? "false" : "true",
      "data-id": source.id,
    });
    wireSource(node, source);
    strip.appendChild(node);
  });

  $("pole-eye").textContent = state.strings.poles.eye;
  $("pole-ear").textContent = state.strings.poles.ear;
  const segments = clear($("segments"));
  const regimes = currentRegimes(a);
  const selected = currentRegimeIndex(a);
  regimes.forEach((regime, index) => {
    const width = regime.span[1] - regime.span[0];
    segments.appendChild(
      el("button", {
        type: "button",
        class: "segment",
        // Segment widths vary per shot because the intervals they stand for do
        // (§4.2); a touch minimum in the stylesheet keeps the narrow ones
        // reachable.
        style: "flex-grow:" + Math.max(width, 0.001) * 1000,
        "aria-pressed": index === selected ? "true" : "false",
        "aria-label": "Balance " + (index + 1) + " of " + regimes.length,
        onclick: () => selectRegime(index),
      }),
    );
  });

  const detents = clear($("detents"));
  GRAINS.forEach((grain) => {
    detents.appendChild(
      el("button", {
        type: "button",
        class: "detent",
        text: state.strings.grains[grain],
        "aria-pressed": grain === a.grain ? "true" : "false",
        onclick: () => selectGrain(grain),
      }),
    );
  });
}

function renderRows() {
  const a = state.authoring;
  const rows = clear($("rows"));
  rows.className = a.solo ? "rows soloing" : "rows";
  const order = currentOrder(a);
  if (!order.length) {
    rows.appendChild(
      el("li", { class: "row empty" }, [
        el("span", { class: "eyebrow", text: a.grain === UNNAMED_GRAIN ? "Nothing is named" : "No source admitted" }),
      ]),
    );
    return;
  }
  order.forEach((id) => {
    const source = sourceById(a, id);
    rows.appendChild(
      el("li", { class: "row" + (a.solo && a.solo.id === id ? " solo" : "") }, [
        el("span", { class: "name", text: source.prose }),
        el("span", { class: "phrases eyebrow", text: source.band + " · " + source.register }),
      ]),
    );
  });
}

function renderAuthoringActions() {
  const a = state.authoring;
  const buttons = clear($("buttons"));
  buttons.appendChild(
    button(a.writing ? state.strings.busy.writing : state.strings.actions.show, {
      disabled: a.writing ? "disabled" : null,
      onclick: onShowCaption,
    }),
  );
  // Keep is disabled whenever the displayed prose is stale relative to current
  // settings, so a participant cannot keep a caption they have not read (§4.5).
  buttons.appendChild(
    button(a.revising ? state.strings.actions.revise : state.strings.actions.keep, {
      class: "button primary",
      disabled: isFresh(a) ? null : "disabled",
      onclick: onKeep,
    }),
  );
  $("criterion").textContent = state.strings.criterion;
}

function afterChange() {
  renderBand();
  renderConsole();
  renderRows();
  renderAuthoringActions();
  api.save();
}

function toggleAdmission(id) {
  const a = state.authoring;
  if (a.grain === UNNAMED_GRAIN) return;
  const at = a.admitted.indexOf(id);
  if (at >= 0) {
    if (a.admitted.length === 1) return; // the last source cannot be muted away
    a.admitted.splice(at, 1);
    api.emit("source.mute", { clip_id: a.clip.clip_id, shot_id: a.shot.shot_id, source_id: id });
  } else {
    a.admitted.push(id);
    api.emit("source.restore", { clip_id: a.clip.clip_id, shot_id: a.shot.shot_id, source_id: id });
  }
  // Carry the selection to the regime whose span contains the current α.
  const regimes = currentRegimes(a);
  if (regimes.length) a.alpha = carryAlpha(regimes, regimeAt(regimes, a.alpha));
  afterChange();
}

function selectRegime(index) {
  const a = state.authoring;
  const regimes = currentRegimes(a);
  if (!regimes[index]) return;
  a.alpha = carryAlpha(regimes, index);
  api.emit("balance.select", {
    clip_id: a.clip.clip_id,
    shot_id: a.shot.shot_id,
    regime: index,
    span: regimes[index].span,
    order: regimes[index].order,
  });
  afterChange();
}

function selectGrain(grain) {
  const a = state.authoring;
  if (grain === a.grain) return;
  a.grain = grain;
  api.emit("grain.select", { clip_id: a.clip.clip_id, shot_id: a.shot.shot_id, grain });
  afterChange();
}

// Solo is hold-to-activate and releases on pointer up: a momentary inspection
// gesture that cannot become a resting state (§4.1). A short press is a mute.
function wireSource(node, source) {
  let held = null;
  const release = () => {
    if (held) clearTimeout(held);
    held = null;
    if (state.authoring.solo && state.authoring.solo.id === source.id) {
      state.authoring.solo = null;
      afterChange();
    }
  };
  node.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    node.setPointerCapture(event.pointerId);
    held = setTimeout(() => {
      held = null;
      const a = state.authoring;
      a.solo = { id: source.id, caption: a.data.auditions[source.id] };
      api.emit("source.solo", { clip_id: a.clip.clip_id, shot_id: a.shot.shot_id, source_id: source.id });
      afterChange();
    }, SOLO_MS);
  });
  node.addEventListener("pointerup", () => {
    const wasSolo = Boolean(state.authoring.solo && state.authoring.solo.id === source.id);
    if (held) {
      clearTimeout(held);
      held = null;
      toggleAdmission(source.id);
      return;
    }
    if (wasSolo) release();
  });
  node.addEventListener("pointercancel", release);
  node.addEventListener("pointerleave", release);
  // Keyboard reaches the latching half: solo has no resting state to reach.
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleAdmission(source.id);
    }
  });
}

function onShowCaption() {
  const a = state.authoring;
  const settings = settingsOf(a);
  const key = keyOf(a);
  a.writing = true;
  a.error = null;
  renderAuthoringActions();
  api.emit("caption.request", { clip_id: a.clip.clip_id, shot_id: a.shot.shot_id, settings, key });
  api
    .post("/api/caption", { clip_id: a.clip.clip_id, shot_id: a.shot.shot_id, settings })
    .then((answer) => {
      a.writing = false;
      a.band = { caption: answer.caption, key: answer.key };
      api.emit("caption.written", {
        clip_id: a.clip.clip_id,
        shot_id: a.shot.shot_id,
        key: answer.key,
        caption: answer.caption,
        cached: answer.cached,
        // Provenance for the analysis, never shown: which path wrote it, and
        // any muted source it names anyway.
        writer: answer.writer,
        names_excluded: answer.names_excluded || [],
      });
      afterChange();
    })
    .catch(() => {
      a.writing = false;
      a.error = state.strings.error;
      api.emit("caption.failed", { clip_id: a.clip.clip_id, shot_id: a.shot.shot_id, key });
      afterChange();
    });
}

function onKeep() {
  const a = state.authoring;
  if (!isFresh(a)) return;
  const clipId = a.clip.clip_id;
  const shotId = a.shot.shot_id;
  const settings = settingsOf(a);
  if (!state.snapshot.committed[clipId]) state.snapshot.committed[clipId] = {};
  state.snapshot.committed[clipId][shotId] = {
    caption: a.band.caption,
    key: a.band.key,
    settings,
    revised: a.revising,
  };
  if (a.revising) {
    state.snapshot.revisions[clipId] = (state.snapshot.revisions[clipId] || 0) + 1;
  }
  api.emit("caption.keep", {
    clip_id: clipId,
    shot_id: shotId,
    settings,
    key: a.band.key,
    caption: a.band.caption,
    revision: a.revising,
  });
  if (a.revising) {
    state.authoring = null;
    enter("watch2");
    return;
  }
  const next = state.snapshot.shot_index + 1;
  if (next < a.clip.shots.length) {
    openShot(next);
  } else {
    state.authoring = null;
    state.playthrough = false;
    enter("watch2");
  }
}

/* ---- Check (§3.2) --------------------------------------------------------- */

function renderCheck() {
  const clip = currentClip();
  show("column", true);
  const column = $("column");
  column.appendChild(el("p", { class: "eyebrow", text: state.strings.busy.preparing }));
  api
    .get("/api/check/" + encodeURIComponent(clip.clip_id) + "?participant=" + encodeURIComponent(state.participant))
    .then((answer) => {
      clear(column);
      column.appendChild(el("h1", { class: "heading", text: state.strings.check_heading }));
      answer.shots.forEach((pair, index) => {
        const chosen = (state.snapshot.choices[clip.clip_id] || {})[pair.shot_id];
        const choice = (side) =>
          el(
            "button",
            {
              type: "button",
              class: "choice",
              "aria-pressed": chosen === side ? "true" : "false",
              onclick: () => onChoose(clip.clip_id, pair.shot_id, side),
            },
            [
              el("p", { class: "eyebrow", text: "Shot " + (index + 1) + " · caption " + side.toUpperCase() }),
              el("p", { class: "caption-prose", text: pair[side] }),
            ],
          );
        column.appendChild(el("div", { class: "pair" }, [choice("a"), choice("b")]));
      });
      column.appendChild(
        el("div", { class: "buttons" }, [
          button(state.strings.actions.submit, {
            class: "button primary",
            disabled: checkComplete(clip, answer) ? null : "disabled",
            onclick: onCheckSubmit,
          }),
        ]),
      );
    })
    .catch(() => {
      clear(column);
      column.appendChild(el("p", { class: "body", text: state.strings.error }));
    });
}

// The four phases run per clip. A document with more than one runs them again
// from the first viewing; the probe and the summary come once, at the end.
function onCheckSubmit() {
  const next = state.snapshot.clip_index + 1;
  if (next < state.session.clips.length) {
    state.snapshot.clip_index = next;
    state.snapshot.shot_index = 0;
    state.playthrough = false;
    state.authoring = null;
    enter("watch1");
    return;
  }
  enter("probe");
}

function checkComplete(clip, answer) {
  const chosen = state.snapshot.choices[clip.clip_id] || {};
  return answer.shots.every((pair) => chosen[pair.shot_id]);
}

function onChoose(clipId, shotId, side) {
  if (!state.snapshot.choices[clipId]) state.snapshot.choices[clipId] = {};
  state.snapshot.choices[clipId][shotId] = side;
  api.emit("check.choose", { clip_id: clipId, shot_id: shotId, side });
  render();
}

/* ---- Probe and done (§3.2) ------------------------------------------------ */

function renderProbe() {
  show("column", true);
  const column = $("column");
  const area = el("textarea", {
    class: "probe-field",
    id: "probe",
    "aria-label": state.strings.probe,
  });
  area.value = state.snapshot.probe || "";
  area.addEventListener("input", () => {
    state.snapshot.probe = area.value;
    api.save();
  });
  column.appendChild(
    el("div", { class: "card" }, [
      el("h1", { class: "heading", text: state.strings.probe }),
      area,
      el("div", { class: "buttons" }, [
        button(state.strings.actions.submit, {
          class: "button primary",
          onclick: () => {
            api.emit("probe.submit", { text: state.snapshot.probe });
            enter("done");
          },
        }),
      ]),
    ]),
  );
}

function renderDone() {
  show("column", true);
  const column = $("column");
  const committed = Object.values(state.snapshot.committed).reduce(
    (total, clip) => total + Object.keys(clip).length,
    0,
  );
  const revisions = Object.values(state.snapshot.revisions).reduce((total, count) => total + count, 0);
  const cell = (label, count) =>
    el("div", { class: "cell" }, [
      el("span", { class: "eyebrow", text: label }),
      el("span", { class: "count", text: String(count) }),
    ]);
  column.appendChild(
    el("div", { class: "card" }, [
      el("h1", { class: "heading", text: state.strings.app_title }),
      el("div", { class: "summary" }, [cell("Captions kept", committed), cell("Revisions", revisions)]),
      el("div", { class: "buttons" }, [
        button(state.strings.actions.download, { onclick: onDownload }),
      ]),
    ]),
  );
  if (!state.snapshot.done_at) {
    state.snapshot.done_at = Date.now();
    api.emit("session.done", {});
  }
}

function onDownload() {
  api.emit("log.download", {});
  api.flush(false).then(() => {
    window.location.href = "/api/log?participant=" + encodeURIComponent(state.participant);
  });
}

/* ---- Start ---------------------------------------------------------------- */

function notice(text) {
  show("column", true);
  clear($("column")).appendChild(el("p", { class: "notice body", text }));
}

// The composition is set for the kiosk's 820 tall. A taller screen shows it
// at the same proportions, larger, rather than small in the middle; the
// stylesheet gives the width that scale leaves over to the footage. Never
// below one, so the kiosk itself is untouched.
function fitScreen() {
  pageZoom = Math.max(1, window.innerHeight / 820);
  document.documentElement.style.setProperty("--k", String(pageZoom));
}

function start() {
  fitScreen();
  window.addEventListener("resize", fitScreen);
  const participant = new URLSearchParams(window.location.search).get("participant");
  state.participant = participant;
  fetch("/api/session?participant=" + encodeURIComponent(participant || ""))
    .then((response) => (response.ok ? response.json() : Promise.reject(new Error("no session"))))
    .then((session) => {
      state.session = session;
      state.strings = session.strings;
      state.cards = session.cards;
      return api.get("/api/state?participant=" + encodeURIComponent(participant));
    })
    .then((stored) => {
      state.snapshot = stored.snapshot && stored.snapshot.schema === SNAPSHOT_SCHEMA ? stored.snapshot : newSnapshot();
      // A resume into authoring reopens the shot rather than restoring a
      // half-built console from the snapshot: the shot payload has to be
      // fetched again anyway, and reopening is one path instead of two.
      if (state.snapshot.screen === "author") {
        openShot(state.snapshot.shot_index);
      } else {
        render();
      }
    })
    .catch(() => notice((state.strings || {}).participant_missing || "Open this page with a participant identifier."));

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") api.flush(true);
  });
  window.addEventListener("pagehide", () => api.flush(true));
}

start();

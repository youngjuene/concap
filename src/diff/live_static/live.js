// Live phase 2: the caption changes only at cue boundaries; the previous one stays until a new one is ready.
//
// UI review of 2026-10-19 ("Diff Flow Improved"):
// - phase 1's theme and caption band, so ratings and steering judge the same presentation;
// - each slider has the three stops its axis really has (texture 0–2, context
//   broad/specific/relation); it sends 0, 0.5 or 1, and only when the stop changes;
// - a chip says when a change will show: a countdown to the next cue boundary,
//   then "applied" for 2 s, or "kept" while the new caption is still being written;
// - a hidden tab pauses the video behind phase 1's overlay, so no caption counts as seen unseen;
// - the ratings use phase 1's scale cards and sticky bar.
"use strict";

const POLL_MS = 1000;
const TICK_MS = 1000;
const DEBOUNCE_MS = 300;
const APPLIED_MS = 2000;
const GROUP = 3;
// The session's segments, sized by expected time in the CSS; phase1_pages.js holds the same list.
const SEGMENTS = ["도입", "연습", "첫 번째 시청", "두 번째 시청", "긴 영상", "인터뷰"];
const AXES = ["texture", "context"];
const STOP_LABELS = {
  texture: ["없이", "조금", "많이"],
  context: ["넓은 범주로", "구체적인 음원", "위치와 관계까지"],
};
const CHIP = {
  idle: "조절 막대는 영상을 보는 동안 언제든 움직일 수 있습니다",
  pending: "바뀐 설정은 다음 자막부터 반영됩니다",
  applied: "새 설정이 반영되었습니다",
  waiting: "새 자막을 준비하는 동안 이전 자막이 유지됩니다",
};

let state = null;
let captions = {};
let shown = { cue: -1, token: null, text: "", from: 0 };
let exposures = [];
let debounce = null;
let poller = null;
let ticker = null;
let segments = false;            // the watch follows phase 1, so the session's progress shows
let stops = { texture: 1, context: 1 };
let sent = { texture: 1, context: 1 };
let epoch = 0;                   // bumps when the captions served change meaning: new settings, a new video
let fetchedEpoch = 0;            // the epoch the current captions were fetched in
let awaiting = false;            // a change the server has, not yet on screen
let chip = "idle";
let appliedTimer = null;

const $ = (id) => document.getElementById(id);
const video = () => $("video");

async function call(path, body) {
  const init = body === undefined
    ? { credentials: "same-origin" }
    : { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const response = await fetch(path, init);
  let data = {};
  try { data = await response.json(); } catch (error) { data = {}; }
  return { ok: response.ok, status: response.status, data };
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function progressHeader(segment, fraction) {
  const wrap = element("div", undefined, "session-progress");
  const bars = element("div", undefined, "bars");
  const labels = element("div", undefined, "labels");
  bars.setAttribute("aria-hidden", "true");
  SEGMENTS.forEach((label, index) => {
    const bar = element("span");
    bar.style.setProperty("--fill", `${index < segment ? 100 : index === segment ? Math.round(fraction * 100) : 0}%`);
    const name = element("span", label, index === segment ? "current" : "");
    if (index === segment) name.setAttribute("aria-current", "step");
    bars.appendChild(bar);
    labels.appendChild(name);
  });
  wrap.append(bars, labels);
  return wrap;
}

function show(id) {
  const watching = id === "watch";
  $("shell").hidden = watching;
  $("watch").hidden = !watching;
  for (const section of $("shell").querySelectorAll(":scope > section")) section.hidden = section.id !== id;
  const progress = $("progress");
  progress.textContent = "";
  if (segments && id !== "busy") {
    const at = id === "done" ? { segment: 5, fraction: 0 } : state.progress;
    progress.appendChild(progressHeader(at.segment, at.fraction));
  }
  if (!watching) {
    window.scrollTo(0, 0);
    $(id).querySelector("h1")?.focus();
  }
}

function positionMs() { return Math.round(video().currentTime * 1000); }

function clock(ms) {
  const seconds = Math.floor(ms / 1000);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function cueAt(ms) {
  const cues = state.video.cues;
  let index = 0;
  for (let i = 0; i < cues.length; i += 1) if (cues[i].start_ms <= ms) index = i;
  return index;
}

function nextBoundary(ms) {
  const next = state.video.cues.find((cue) => cue.start_ms > ms);
  return next ? next.start_ms : null;
}

function recordExposure(until) {
  if (shown.token !== null && until > shown.from) {
    exposures.push({ cue: shown.cue, token: shown.token, from_ms: shown.from, to_ms: until });
  }
}

async function flushExposures() {
  if (!exposures.length) return;
  const entries = exposures;
  exposures = [];
  const result = await call("api/exposures", { entries });
  if (!result.ok) exposures = entries.concat(exposures);
}

// -- the change chip -----------------------------------------------------------
function paintChip() {
  const node = $("chip");
  node.dataset.state = chip;
  node.textContent = "";
  if (chip === "pending" || chip === "applied") node.appendChild(element("span", undefined, "dot"));
  node.appendChild(document.createTextNode(CHIP[chip]));
  if (chip === "pending" && state && state.stage === "watch") {
    const ms = positionMs();
    const boundary = nextBoundary(ms);
    if (boundary !== null) {
      const countdown = element("span", `${Math.max(1, Math.ceil((boundary - ms) / 1000))}초`, "countdown");
      countdown.setAttribute("aria-hidden", "true");  // a live region must not read every tick
      node.appendChild(countdown);
    }
  }
}

function setChip(next) {
  clearTimeout(appliedTimer);
  chip = next;
  if (next === "applied") appliedTimer = setTimeout(() => setChip("idle"), APPLIED_MS);
  paintChip();
}

const unsent = () => AXES.some((axis) => stops[axis] !== sent[axis]);

// At a cue boundary: the change shows if the server has it and this cue's caption was written under it.
function boundaryReached(written) {
  if (chip !== "pending" && chip !== "waiting") return;
  if (unsent() || debounce !== null) { paintChip(); return; }
  if (written && fetchedEpoch === epoch) {
    awaiting = false;
    setChip("applied");
  } else setChip("waiting");
}

function paint(ms) {
  const cue = cueAt(ms);
  if (cue === shown.cue) return;
  const first = shown.cue === -1;
  const next = captions[String(cue)];
  recordExposure(ms);
  if (next) {
    shown = { cue, token: next.token, text: next.text, from: ms };
  } else {
    shown = { cue, token: shown.token, text: shown.text, from: ms };  // keep the previous caption
  }
  $("caption").textContent = shown.text;
  if (!first) boundaryReached(Boolean(next));
}

async function poll() {
  const asked = epoch;
  const result = await call("api/captions");
  // A reply that crossed a settings change or a new video may predate it; the next poll is current.
  if (!result.ok || asked !== epoch) return;
  captions = result.data.captions || {};
  fetchedEpoch = asked;
  if (state && state.stage === "ready") {
    const ready = Boolean(captions["0"]);
    $("start").disabled = !ready;
    $("ready-status").textContent = ready ? "준비되었습니다." : "첫 자막을 준비하고 있습니다…";
  }
}

// -- the sliders ---------------------------------------------------------------
function paintStepper(axis) {
  const stop = stops[axis];
  const input = $(axis);
  input.value = String(stop);
  input.setAttribute("aria-valuetext", STOP_LABELS[axis][stop]);
  $(`${axis}-value`).textContent = STOP_LABELS[axis][stop];
  const root = $(`${axis}-stepper`);
  root.querySelector(".stepper-track").style.setProperty("--stop", String(stop));
  root.querySelectorAll(".stepper-stop").forEach((dot, index) => dot.classList.toggle("on", index <= stop));
  root.querySelectorAll(".stepper-labels span").forEach((label, index) => label.classList.toggle("on", index === stop));
}

function sendSettings(origin) {
  clearTimeout(debounce);
  debounce = setTimeout(async () => {
    debounce = null;
    if (!unsent()) return;
    const previous = sent;
    sent = { ...stops };
    const result = await call("api/settings", {
      texture: sent.texture / 2,
      context: sent.context / 2,
      position_ms: positionMs(),
      origin,
    });
    if (!result.ok) { sent = previous; return; }
    epoch += 1;
    awaiting = true;
    // Later cues' captions were written under the old settings, which the server no longer
    // serves: drop them, so a boundary keeps the previous caption until a new one arrives.
    captions = Object.fromEntries(Object.entries(captions).filter(([cue]) => Number(cue) <= shown.cue));
    poll();
  }, DEBOUNCE_MS);
}

function moveStop(axis, stop) {
  if (stop === stops[axis]) return;
  stops[axis] = stop;
  paintStepper(axis);
  if (unsent()) {
    if (chip !== "pending") setChip("pending");
    sendSettings(`slider:${axis}`);
  } else {
    // Back to the stop the server already has before the change was sent.
    clearTimeout(debounce);
    debounce = null;
    if (chip === "pending" && !awaiting) setChip("idle");
  }
}

function sendPlayback() {
  call("api/playback", { position_ms: positionMs(), playing: !video().paused && !video().ended, hidden: document.hidden });
}

// -- the post-watch items ------------------------------------------------------
function ratingItem(item, number, answers, changed) {
  const field = element("fieldset", undefined, "sheet-item");
  field.appendChild(element("legend", `${number}. ${item.text}`));
  const scale = element("div", undefined, "sheet-scale");
  const points = item.max - item.min + 1;
  scale.style.setProperty("--points", String(points));
  for (let value = item.min; value <= item.max; value += 1) {
    const anchor = value === item.min ? item.anchors[0] : value === item.max ? item.anchors[1] : "";
    const label = element("label", undefined, "sheet-option");
    const input = element("input");
    input.type = "radio";
    input.name = item.id;
    input.value = String(value);
    input.required = true;
    input.checked = answers[item.id] === value;
    input.setAttribute("aria-label", `${value}점${anchor ? ` · ${anchor}` : ""}`);
    input.addEventListener("change", () => changed(item, value));
    label.append(input, element("span", String(value), "sheet-score"), element("span", anchor, "sheet-label"));
    scale.appendChild(label);
  }
  field.appendChild(scale);
  return field;
}

async function renderItems() {
  const result = await call("api/items");
  const items = result.data.items || [];
  const form = $("items");
  form.textContent = "";
  const answers = {};
  const fields = new Map();
  const groups = [];
  let group;
  let flagged = false;
  const bar = element("div", undefined, "sheet-submitbar");
  const submit = element("button", "응답 저장 후 계속", "primary");
  submit.type = "submit";
  const count = element("p", "", "sheet-hint");
  count.setAttribute("role", "status");
  const region = element("p", "", "sheet-feedback");
  region.setAttribute("role", "alert");
  const missing = () => items.filter((item) => answers[item.id] === undefined);
  function reveal(item) {
    const field = fields.get(item.id);
    field.classList.add("sheet-invalid");
    field.scrollIntoView({ block: "center" });
    field.querySelector("input").focus();
  }
  function flag(left) {
    region.textContent = "";
    flagged = left.length > 0;
    if (!flagged) return;
    const jump = element("button", `${items.indexOf(left[0]) + 1}번으로 이동`, "link");
    jump.type = "button";
    jump.addEventListener("click", () => reveal(left[0]));
    region.append(element("span", undefined, "dot"), element("span", `응답하지 않은 문항 ${left.length}개`), element("span", "·"), jump);
  }
  function update() {
    const left = missing();
    count.textContent = `${items.length - left.length} / ${items.length} 응답`;
    for (const entry of groups) {
      const done = entry.slice.filter((item) => answers[item.id] !== undefined).length;
      entry.status.textContent = done === entry.slice.length ? "완료" : `${done} / ${entry.slice.length}`;
    }
    if (flagged) flag(left);
  }
  const changed = (item, value) => {
    answers[item.id] = value;
    fields.get(item.id).classList.remove("sheet-invalid");
    update();
  };
  items.forEach((item, index) => {
    if (index % GROUP === 0) {
      const slice = items.slice(index, index + GROUP);
      const status = element("span");
      if (items.length > GROUP) {
        const head = element("div", undefined, "sheet-group-head");
        const range = slice.length > 1 ? `${index + 1}–${index + slice.length}` : `${index + 1}`;
        head.append(element("span", `문항 ${range} / ${items.length}`), status);
        form.appendChild(head);
      }
      group = element("div", undefined, "sheet-group");
      form.appendChild(group);
      groups.push({ slice, status });
    }
    const field = ratingItem(item, index + 1, answers, changed);
    fields.set(item.id, field);
    group.appendChild(field);
  });
  bar.append(submit, count, region);
  form.appendChild(bar);
  form.onsubmit = async (event) => {
    event.preventDefault();
    if (submit.disabled) return;
    const left = missing();
    if (left.length) { flag(left); reveal(left[0]); return; }
    flagged = false;
    submit.disabled = true;
    region.textContent = "응답을 저장하고 있습니다…";
    const saved = await call("api/ratings", { answers });
    if (!saved.ok) {
      region.textContent = saved.data.detail || "응답을 확인해 주세요.";
      submit.disabled = false;
      return;
    }
    state.stage = "done";
    render();
  };
  update();
}

// -- pages ---------------------------------------------------------------------
function render() {
  if (!state) return;
  const where = `2스테이지 · 긴 영상 ${state.video.number} / ${state.video.count}`;
  if (state.stage === "ready") {
    // The previous video's captions and Start state must not carry over.
    epoch += 1;
    captions = {};
    $("start").disabled = true;
    $("ready-status").textContent = "첫 자막을 준비하고 있습니다…";
    $("ready-eyebrow").textContent = where;
    $("ready-lede").textContent = state.video.number > 1
      ? "헤드폰을 착용해 주세요. 조절 막대 설정은 앞 영상에서 이어집니다."
      : "헤드폰을 착용해 주세요.";
    $("watch-eyebrow").textContent = where;
    $("clock").textContent = `${clock(0)} / ${clock(state.video.duration_ms)}`;
    $("elapsed").style.width = "0%";
    for (const axis of AXES) {
      stops[axis] = state.sliders.stops[axis];
      paintStepper(axis);
    }
    sent = { ...stops };
    awaiting = false;
    setChip("idle");
    $("paused").hidden = true;
    shown = { cue: -1, token: null, text: "", from: 0 };
    $("caption").textContent = "";
    video().src = state.video.media;
    video().load();
    show("ready");
  } else if (state.stage === "watch") {
    show("watch");
  } else if (state.stage === "ratings") {
    clearInterval(poller);
    renderItems();
    show("ratings");
  } else if (state.stage === "done") {
    clearInterval(poller);
    show("done");
  }
}

// A hidden tab pauses the watch behind phase 1's overlay; the sliders wait under it too.
function pauseHidden() {
  const ms = positionMs();
  recordExposure(ms);
  shown.from = ms;
  video().pause();
  $("paused").hidden = false;
  flushExposures();
}

async function begin() {
  const result = await call("api/session", {});
  if (result.status === 401 || result.status === 409) {
    // No live session for this browser yet: the entry and phase 1 come first.
    window.location.assign("../");
    return;
  }
  if (result.status === 503) {
    $("busy-message").textContent = result.data.message || result.data.detail || "";
    show("busy");
    return;
  }
  state = result.data;
  segments = Boolean(state.progress);
  poller = setInterval(poll, POLL_MS);
  render();
  poll();
}

function wire() {
  $("start").addEventListener("click", async () => {
    $("start").disabled = true;
    const result = await call("api/start", {});
    if (!result.ok) {
      $("ready-status").textContent = result.data.detail || "첫 자막을 준비하고 있습니다…";
      poll();
      return;
    }
    state = result.data;
    render();
    paint(0);
    await video().play();
    sendPlayback();
  });
  for (const axis of AXES) {
    $(axis).addEventListener("input", () => moveStop(axis, Number($(axis).value)));
    $(`${axis}-stepper`).querySelectorAll(".stepper-labels span").forEach((label, stop) => {
      label.addEventListener("click", () => moveStop(axis, stop));
    });
  }
  video().addEventListener("timeupdate", () => {
    const ms = positionMs();
    paint(ms);
    $("clock").textContent = `${clock(ms)} / ${clock(state.video.duration_ms)}`;
    $("elapsed").style.width = `${Math.min(100, (ms / state.video.duration_ms) * 100)}%`;
    if (chip === "pending") paintChip();
  });
  video().addEventListener("play", () => {
    sendPlayback();
    clearInterval(ticker);
    ticker = setInterval(() => { sendPlayback(); flushExposures(); }, TICK_MS);
  });
  video().addEventListener("pause", () => { sendPlayback(); clearInterval(ticker); });
  video().addEventListener("ended", async () => {
    clearInterval(ticker);
    recordExposure(positionMs());
    shown.token = null;
    await flushExposures();
    const result = await call("api/video-ended", {});
    if (result.ok) {
      state = result.data;
      render();
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state && state.stage === "watch" && !video().paused && !video().ended) pauseHidden();
    else if (!document.hidden && !$("paused").hidden) $("resume").focus();
  });
  $("resume").addEventListener("click", async () => {
    $("paused").hidden = true;
    await video().play();
  });
}

wire();
begin();

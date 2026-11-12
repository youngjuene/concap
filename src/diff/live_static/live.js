// Live phase 2: each cue shows its own caption as the server serves it: under the current settings when
// written, else under earlier ones, else its starting caption. A caption written under the current settings
// replaces the one on screen as soon as it arrives, mid-cue, if each caption gets at least 3 s on screen
// (researcher, 2026-10-21).
//
// UI review of 2026-10-19 ("Diff Flow Improved"):
// - phase 1's theme and caption band, so ratings and steering judge the same presentation;
// - each slider has the three stops its axis really has (texture 0–2, context
//   broad/specific/relation); it sends 0, 0.5 or 1, and only when the stop changes;
// - a chip says a change is being written, then "applied" for 2 s once its caption is on screen,
//   or "waiting" while a cue still shows a caption from earlier settings;
// - a hidden tab pauses the video behind phase 1's overlay, so no caption counts as seen unseen;
// - the ratings use phase 1's scale cards and sticky bar;
// - the perceived-source block (researcher, 2026-11-03) replays cues without captions after the ratings;
// - the last screen says the session is complete and nothing else (researcher, 2026-11-12: participants
//   are remote, so there is no experimenter to call and no next participant on this browser).
"use strict";

const POLL_MS = 500;  // a written caption reaches the screen within half a second
// Researcher, 2026-10-21: a caption stays on screen at least 3 s before a mid-cue swap replaces it, and a
// swap needs 3 s of its cue left, so quick slider moves never make the caption flicker.
const MIN_READ_MS = 3000;
const TICK_MS = 1000;
const DEBOUNCE_MS = 300;
const APPLIED_MS = 2000;
const GROUP = 3;
// The session's segments, sized by expected time in the CSS; phase1_pages.js holds the same list.
const SEGMENTS = ["도입", "연습", "첫 번째 시청", "두 번째 시청", "긴 영상", "완료"];
const AXES = ["texture", "context"];
const STOP_LABELS = {
  texture: ["없이", "조금", "많이"],
  context: ["넓은 범주로", "구체적인 음원", "위치와 관계까지"],
};
const CHIP = {
  idle: "슬라이더는 영상을 보는 동안 언제든 움직일 수 있습니다",
  pending: "새 설정으로 자막을 고쳐 쓰고 있습니다",
  applied: "새 설정이 반영되었습니다",
  waiting: "새 자막을 준비하는 동안 이전 설정의 자막이 표시됩니다",
};

let state = null;
let captions = {};
let shown = { cue: -1, token: null, text: "", from: 0 };
let exposures = [];
let debounce = null;
let poller = null;
let ticker = null;
let segments = false;            // the watch follows phase 1, so the session's progress shows
let restarted = false;           // this page was reloaded in the middle of a video, which starts again
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
}

function setChip(next) {
  clearTimeout(appliedTimer);
  chip = next;
  if (next === "applied") appliedTimer = setTimeout(() => setChip("idle"), APPLIED_MS);
  paintChip();
}

const unsent = () => AXES.some((axis) => stops[axis] !== sent[axis]);

// At a cue boundary: the change shows if the server has it and this cue's caption is current under it.
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
  if (cue === shown.cue) {
    // Mid-cue: a caption written under the current settings replaces this cue's caption when it arrives,
    // once the caption on screen has had MIN_READ_MS, while MIN_READ_MS of the cue is left, and only
    // when no newer slider move is still on its way to the server.
    const now = captions[String(cue)];
    const settled = !unsent() && debounce === null && fetchedEpoch === epoch;
    const room = ms - shown.from >= MIN_READ_MS && state.video.cues[cue].end_ms - ms >= MIN_READ_MS;
    if (now && now.current && settled && room && now.token !== shown.token) {
      recordExposure(ms);
      shown = { cue, token: now.token, text: now.text, from: ms };
      $("caption").textContent = shown.text;
      if (chip === "pending" || chip === "waiting") { awaiting = false; setChip("applied"); }
    }
    return;
  }
  const first = shown.cue === -1;
  const next = captions[String(cue)];
  recordExposure(ms);
  if (next) {
    shown = { cue, token: next.token, text: next.text, from: ms };
  } else {
    shown = { cue, token: shown.token, text: shown.text, from: ms };  // no reply yet: the last caption stays
  }
  $("caption").textContent = shown.text;
  if (!first) boundaryReached(Boolean(next && next.current));
}

async function poll() {
  const asked = epoch;
  const result = await call("api/captions");
  // A reply that crossed a settings change or a new video may predate it; the next poll is current.
  if (!result.ok || asked !== epoch) return;
  captions = result.data.captions || {};
  fetchedEpoch = asked;
  if (state && state.stage === "ready") {
    const ready = Boolean(captions["0"] && captions["0"].settled);
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
    // Keep the cached captions: each cue's own caption under the old settings still beats the
    // previous cue's, and the next poll brings the ones written under the new settings.
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
    state.stage = saved.data.stage;
    render();
  };
  update();
}

// -- the perceived-source block --------------------------------------------------
// Each answer's click plays the next clip, so the browser allows its sound; a refused play falls back to
// the button. A hidden tab stops the clip, and it starts over on return.
const SOURCE_POLL_MS = 50;
const source = { page: null, index: 0, measures: null, clock: null, playing: false };

function sourceStatus(text) { $("source-status").textContent = text; }

function stopClipClock() { clearInterval(source.clock); source.clock = null; source.playing = false; }

function sourceButton(label) {
  stopClipClock();
  const button = $("source-start");
  button.textContent = label;
  button.hidden = false;
  button.disabled = false;
  button.parentElement.hidden = false;
}

async function seekTo(player, ms) {
  if (player.readyState < 1) {
    await new Promise((resolve, reject) => {
      player.addEventListener("loadedmetadata", resolve, { once: true });
      player.addEventListener("error", reject, { once: true });
    });
  }
  await new Promise((resolve, reject) => {
    player.addEventListener("seeked", resolve, { once: true });
    player.addEventListener("error", reject, { once: true });
    player.currentTime = ms / 1000;
  });
}

async function playClip() {
  const trial = source.page.trials[source.index];
  const player = $("source-video");
  const measures = source.measures;
  $("source-start").parentElement.hidden = true;
  $("source-answers").disabled = true;
  sourceStatus("장면을 불러오고 있습니다…");
  const asked = performance.now();
  try {
    const url = new URL(trial.media, window.location.href).href;
    if (player.src !== url) { player.src = url; player.load(); }
    // Started inside the click, muted and out of sight, so the browser allows this element's sound
    // later and nothing of the wrong moment is seen or heard while it seeks.
    player.muted = true;
    player.style.visibility = "hidden";
    const allowed = player.play().catch(() => null);
    await seekTo(player, trial.start_ms);
    await allowed;
    player.pause();
    await seekTo(player, trial.start_ms);
    player.muted = false;
    player.style.visibility = "";
    await player.play();
  } catch (error) {
    player.style.visibility = "";
    sourceStatus("장면을 재생하지 못했습니다. 아래 단추를 눌러 주세요.");
    sourceButton("장면 재생");
    return;
  }
  measures.waited_ms += Math.round(performance.now() - asked);
  sourceStatus("");
  source.playing = true;
  source.clock = setInterval(() => {
    if (player.currentTime * 1000 < trial.end_ms && !player.ended) return;
    stopClipClock();
    player.pause();
    measures.played_ms = Math.round(player.currentTime * 1000) - trial.start_ms;
    $("source-answers").disabled = false;
    $("source-answers").scrollIntoView({ block: "nearest" });
  }, SOURCE_POLL_MS);
}

function showTrial() {
  const trial = source.page.trials[source.index];
  source.measures = { played_ms: 0, waited_ms: 0, stalls: 0, restarts: 0 };
  $("source-eyebrow").textContent = `2스테이지 · 소리가 나는 곳 ${source.index + 1} / ${source.page.trials.length}`;
  // Two trials in a row can name the same sound, so the count sits by the name too.
  $("source-count").textContent = `${source.index + 1} / ${source.page.trials.length}`;
  $("source-sound").textContent = trial.sound;
  $("source-trial").hidden = false;
  $("source-lede").hidden = source.index > 0;
  $("source-answers").disabled = true;
}

async function answerSource(answer, button) {
  $("source-answers").disabled = true;
  sourceStatus("");
  const measures = source.measures;
  const saved = await call("api/source-answer", { index: source.index, answer, ...measures });
  if (!saved.ok) {
    $("source-answers").disabled = false;
    sourceStatus(saved.data.detail || "응답을 저장하지 못했습니다. 다시 눌러 주세요.");
    return;
  }
  if (saved.data.stage === "done") {
    $("source-video").removeAttribute("src");
    state.stage = "done";
    render();
    return;
  }
  source.index = saved.data.next;
  showTrial();
  button.blur();
  playClip();
}

async function renderSource() {
  const result = await call("api/source");
  if (!result.ok) return;
  source.page = result.data;
  source.index = result.data.next;
  $("source-question").textContent = result.data.question;
  const choices = $("source-choices");
  choices.textContent = "";
  for (const answer of result.data.answers) {
    const button = element("button", answer.text);
    button.type = "button";
    button.addEventListener("click", () => answerSource(answer.id, button));
    choices.appendChild(button);
  }
  showTrial();
  sourceButton(source.index > 0 ? "이어서 재생" : "장면 재생");
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
    $("ready-lede").textContent = restarted
      ? "시청이 중단되어 이 영상을 처음부터 다시 봅니다. 헤드폰을 착용해 주세요. 슬라이더 설정은 그대로입니다."
      : state.video.number > 1
        ? "헤드폰을 착용해 주세요. 슬라이더 설정은 앞 영상에서 이어집니다."
        : "헤드폰을 착용해 주세요.";
    restarted = false;
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
  } else if (state.stage === "source") {
    clearInterval(poller);
    show("source");
    renderSource();
  } else if (state.stage === "done") {
    clearInterval(poller);
    show("done");
  }
}

// A hidden tab pauses the watch behind phase 1's overlay; the sliders wait under it too. So does a
// pause from outside the page (a headset button, a media key) or a play() the browser refused: the
// video has no controls, so the overlay's button is the only way on.
const PAUSED = {
  hidden: "화면으로 돌아온 뒤 이어서 시청해 주세요. 멈춘 동안의 자막 설정은 그대로 유지됩니다.",
  stopped: "영상이 멈췄습니다. 아래 단추를 눌러 이어서 시청해 주세요.",
};

function pauseHidden(reason = "hidden") {
  const ms = positionMs();
  recordExposure(ms);
  shown.from = ms;
  video().pause();
  $("paused-text").textContent = PAUSED[reason];
  $("paused").hidden = false;
  $("resume").focus();
  flushExposures();
}

async function playWatch() {
  $("paused").hidden = true;
  try { await video().play(); } catch (error) { pauseHidden("stopped"); }
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
  if (state.stage === "watch") {
    // The page was loaded while a video was playing: a reload, or a browser reopened. The video and
    // its position are gone with the old page, so the server puts this video back on its ready screen.
    const again = await call("api/restart", {});
    if (again.ok) { state = again.data; restarted = true; }
  }
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
    await playWatch();
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
  });
  video().addEventListener("play", () => {
    sendPlayback();
    clearInterval(ticker);
    ticker = setInterval(() => { sendPlayback(); flushExposures(); }, TICK_MS);
  });
  video().addEventListener("pause", () => {
    sendPlayback();
    clearInterval(ticker);
    // The browser also fires "pause" when the video ends; that is the end, not a stop.
    if (state && state.stage === "watch" && !video().ended && $("paused").hidden) pauseHidden("stopped");
  });
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
  $("source-start").addEventListener("click", playClip);
  $("source-video").addEventListener("waiting", () => { if (source.playing) source.measures.stalls += 1; });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && source.playing) {
      $("source-video").pause();
      source.measures.restarts += 1;
      sourceButton("다시 재생");
      sourceStatus("화면을 벗어나 장면이 멈췄습니다. 처음부터 다시 봐 주세요.");
      return;
    }
    if (document.hidden && state && state.stage === "watch" && !video().paused && !video().ended) pauseHidden();
    else if (!document.hidden && !$("paused").hidden) $("resume").focus();
  });
  $("resume").addEventListener("click", playWatch);
}

wire();
begin();

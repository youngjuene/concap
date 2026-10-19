// Live phase 2: the caption changes only at cue boundaries; the previous one stays until a new one is ready.
"use strict";

const POLL_MS = 1000;
const TICK_MS = 1000;
const DEBOUNCE_MS = 300;

let state = null;
let captions = {};
let shown = { cue: -1, token: null, text: "", from: 0 };
let exposures = [];
let debounce = null;
let poller = null;
let ticker = null;
let epoch = 0;  // bumps for each new video, so a reply about the previous one is dropped

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

function show(id) {
  for (const section of document.querySelectorAll("main > section")) section.hidden = section.id !== id;
}

function positionMs() { return Math.round(video().currentTime * 1000); }

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

function paint(ms) {
  const cue = cueAt(ms);
  if (cue === shown.cue) return;
  const next = captions[String(cue)];
  recordExposure(ms);
  if (next) {
    shown = { cue, token: next.token, text: next.text, from: ms };
  } else {
    shown = { cue, token: shown.token, text: shown.text, from: ms };  // keep the previous caption
  }
  $("caption").textContent = shown.text;
}

async function poll() {
  const asked = epoch;
  const result = await call("api/captions");
  // A reply that crossed a new video may hold the previous video's captions; the next poll is current.
  if (!result.ok || asked !== epoch) return;
  captions = result.data.captions || {};
  if (state && state.stage === "ready") {
    const ready = Boolean(captions["0"]);
    $("start").disabled = !ready;
    $("ready-status").textContent = ready ? "준비되었습니다." : "첫 자막을 준비하고 있습니다…";
  }
}

function sendSettings(origin) {
  clearTimeout(debounce);
  debounce = setTimeout(() => {
    call("api/settings", {
      texture: Number($("texture").value),
      context: Number($("context").value),
      position_ms: positionMs(),
      origin,
    });
  }, DEBOUNCE_MS);
}

function sendPlayback() {
  call("api/playback", { position_ms: positionMs(), playing: !video().paused && !video().ended });
}

function render() {
  if (!state) return;
  if (state.stage === "ready") {
    // The previous video's captions and Start state must not carry over.
    epoch += 1;
    captions = {};
    $("start").disabled = true;
    $("ready-status").textContent = "첫 자막을 준비하고 있습니다…";
    show("ready");
    $("video-count").textContent = state.video.count > 1 ? `(${state.video.number}/${state.video.count})` : "";
    $("texture").value = state.sliders.texture;
    $("context").value = state.sliders.context;
    shown = { cue: -1, token: null, text: "", from: 0 };
    $("caption").textContent = "";
    video().src = state.video.media;
    video().load();
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

async function renderItems() {
  const result = await call("api/items");
  const form = $("items");
  form.textContent = "";
  for (const item of result.data.items || []) {
    const block = document.createElement("div");
    block.className = "item";
    const text = document.createElement("p");
    text.textContent = item.text;
    block.appendChild(text);
    const scale = document.createElement("div");
    scale.className = "scale";
    const low = document.createElement("span");
    low.textContent = item.anchors[0];
    scale.appendChild(low);
    for (let value = item.min; value <= item.max; value += 1) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = item.id;
      input.value = String(value);
      label.appendChild(input);
      label.appendChild(document.createTextNode(String(value)));
      scale.appendChild(label);
    }
    const high = document.createElement("span");
    high.textContent = item.anchors[1];
    scale.appendChild(high);
    block.appendChild(scale);
    form.appendChild(block);
  }
}

async function submitRatings() {
  const answers = {};
  for (const input of $("items").querySelectorAll("input[type=radio]:checked")) answers[input.name] = Number(input.value);
  const result = await call("api/ratings", { answers });
  if (!result.ok) {
    $("ratings-status").textContent = result.data.detail || "응답을 확인해 주세요.";
    return;
  }
  state.stage = "done";
  render();
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
  $("texture").addEventListener("input", () => sendSettings("slider:texture"));
  $("context").addEventListener("input", () => sendSettings("slider:context"));
  video().addEventListener("timeupdate", () => paint(positionMs()));
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
  $("submit").addEventListener("click", submitRatings);
}

wire();
begin();

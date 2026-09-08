/* Shared-theme two-stage study. Requests serialize; exposure IDs survive retries. */
"use strict";
const content = document.getElementById("content");
let state, chain = Promise.resolve(), cleanup = () => {};
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const notice = (text = "") => {
  const node = document.getElementById("notice");
  node.textContent = text; node.hidden = !text;
};
function storageKey(kind) { return `caption-study:${state.session_id}:${kind}`; }
function saved(kind, fallback) {
  try { return JSON.parse(localStorage.getItem(storageKey(kind))) ?? fallback; } catch { return fallback; }
}
function save(kind, value) { localStorage.setItem(storageKey(kind), JSON.stringify(value)); }
async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: {"content-type": "application/json", "x-study-request": "1"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) { const error = new Error(result.error || "Request failed"); error.status = response.status; throw error; }
  return result;
}
function send(action, data, redraw = false) {
  const key = crypto.randomUUID();
  data = structuredClone(data);
  const run = async () => {
    const payload = {key, revision: state.revision, data};
    let result;
    try { result = await request(`/api/study/${action}`, payload); }
    catch (error) {
      // A lost response may already have committed: same key safely retrieves its receipt.
      if (!error.status) result = await request(`/api/study/${action}`, payload);
      else if (error.status === 409) {
        const fresh = await request("/api/study/state");
        const stageChanged = fresh.stage !== state.stage;
        state = fresh;
        if (stageChanged) { render(); throw error; }
        payload.revision = state.revision;
        if (action === "playback" || action === "clip-playback") data.sequence = Math.max(data.sequence, state.sequence + 1);
        result = await request(`/api/study/${action}`, payload);
      } else throw error;
    }
    state = result;
    notice();
    if (redraw) render();
    return result;
  };
  const pending = chain.then(run);
  chain = pending.catch(async error => {
    notice(error.message);
    if (error.status === 409) {
      try { state = await request("/api/study/state"); render(); } catch { /* next user retry */ }
    }
  });
  return pending;
}
function button(text, action, primary = false) {
  const node = el("button", text, primary ? "primary" : "");
  node.type = "button";
  node.onclick = async () => {
    node.disabled = true; notice();
    try { await action(); } catch (error) { notice(error.message); }
    finally { node.disabled = false; }
  };
  return node;
}
function heading(title, description) {
  const head = el("div", undefined, "head");
  head.append(el("h1", title));
  if (description) head.append(el("p", description, "lede"));
  content.append(head);
}
function actions(...nodes) { const row = el("div", undefined, "actions"); row.append(...nodes); content.append(row); }
function form(items, submitAction, title, description) {
  heading(title, description);
  const key = `form:${state.stage}`;
  const values = saved(key, state.draft || {});
  const node = el("form");
  node.onsubmit = event => event.preventDefault();
  let timer;
  const changed = () => {
    save(key, values);
    clearTimeout(timer);
    timer = setTimeout(() => send("draft", values).catch(() => {}), 800);
  };
  for (const item of items) {
    const field = el("fieldset", undefined, "study-card");
    field.append(el("legend", item.text));
    if (item.type === "text") {
      const input = el("textarea"); input.maxLength = 2000; input.value = values[item.id] || "";
      input.setAttribute("aria-label", `${item.text} (optional)`);
      input.oninput = () => { values[item.id] = input.value; changed(); };
      field.append(el("p", "Optional"), input);
    } else {
      if (item.type === "rating") field.append(el("p", `${item.low || "Strongly disagree"} (1) → ${item.high || "Strongly agree"} (5)`));
      const choices = item.type === "rating" ? [1, 2, 3, 4, 5, ...(item.na ? ["na"] : [])] : item.options;
      const group = el("div", undefined, item.type === "rating" ? "ratings" : "");
      for (const value of choices) {
        const label = el("label"); const input = el("input");
        input.type = "radio"; input.name = item.id; input.value = String(value);
        input.checked = values[item.id] === value;
        input.onchange = () => { values[item.id] = value; changed(); };
        label.append(input, el("span", value === "na" ? "Not applicable" : String(value))); group.append(label);
      }
      field.append(group);
    }
    node.append(field);
  }
  content.append(node);
  actions(button(submitAction === "preferences" ? "Begin calibration clips" : "Submit your experience", async () => {
    clearTimeout(timer);
    const data = {...values};
    await send(submitAction, data, true); save(key, {});
  }, true));
  cleanup = () => clearTimeout(timer);
}
function player(source) {
  const wrapper = el("div", undefined, "study-player");
  const video = el("video"); video.controls = true; video.playsInline = true; video.preload = "metadata"; video.src = source;
  video.setAttribute("controlslist", "nofullscreen noremoteplayback"); video.disablePictureInPicture = true;
  video.onerror = () => notice("This video could not be played. Reload to retry, or contact the researcher if the problem continues.");
  wrapper.append(video);
  return {wrapper, video};
}
function calibrationClip() {
  const clip = state.clip, resumePosition = state.position_ms;
  heading(state.clip.title, "Watch and listen to this clip. Afterwards, tell us what you noticed.");
  const {wrapper, video} = player(state.clip.video);
  content.append(wrapper);
  let sequence = state.sequence, busy = false, alive = true, initialized = false;
  const tick = async (seek = false) => {
    if (!alive || !initialized || (busy && !seek)) return;
    busy = true;
    sequence = Math.max(sequence, state.sequence) + 1;
    try { await send("clip-playback", {clip_id: clip.id, position_ms: Math.min(clip.duration_ms, video.currentTime * 1000), sequence, playing: !video.paused, hidden: document.hidden, seek}); }
    finally { busy = false; }
  };
  const next = button("Continue to what you noticed", async () => { await tick(); await send("clip-ended", {}, true); }, true);
  next.disabled = true;
  video.onloadedmetadata = () => { initialized = true; if (resumePosition > 0) video.currentTime = resumePosition / 1000; };
  video.onplay = () => tick().catch(() => {});
  video.onpause = () => tick().catch(() => {});
  video.onseeked = () => tick(true).catch(() => {});
  video.onended = () => { tick().catch(() => {}); next.disabled = false; };
  const visibility = () => tick().catch(() => {});
  document.addEventListener("visibilitychange", visibility);
  const timer = setInterval(() => { if (!video.paused) tick().catch(() => {}); }, 750);
  actions(next);
  cleanup = () => { alive = false; clearInterval(timer); document.removeEventListener("visibilitychange", visibility); video.onpause = null; video.pause(); };
}
function observation() {
  heading("What caught your attention?", "Mark what you noticed in the frames, then answer for each sound family.");
  const key = `observation:${state.clip.id}`;
  const values = saved(key, state.draft?.points ? state.draft : {points: [], heard: {}});
  let index = 0, timer;
  const persist = () => { save(key, values); clearTimeout(timer); timer = setTimeout(() => send("draft", values).catch(() => {}), 800); };
  const card = el("div", undefined, "study-card");
  const frame = el("div", undefined, "visual-frame");
  const image = el("img"); image.alt = "Calibration frame. Mark objects you noticed.";
  frame.append(image); card.append(frame);
  const count = el("p");
  const draw = () => {
    image.src = state.clip.frames[index].url;
    frame.querySelectorAll(".visual-point").forEach(n => n.remove());
    for (const point of values.points.filter(p => p.frame === index)) {
      const dot = el("span", undefined, "visual-point"); dot.style.left = `${point.x * 100}%`; dot.style.top = `${point.y * 100}%`; frame.append(dot);
    }
    count.textContent = `${values.points.length} point${values.points.length === 1 ? "" : "s"} marked`;
  };
  const add = (x, y) => { if (values.points.length >= 100) return; values.points.push({frame: index, x, y}); persist(); draw(); };
  frame.onclick = event => { const rect = image.getBoundingClientRect(); add(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))); };
  const moments = el("div", undefined, "moments-list");
  state.clip.frames.forEach((entry, i) => {
    const pick = button(`${(entry.at_ms / 1000).toFixed(1)} s`, () => { index = i; moments.querySelectorAll("button").forEach((n, j) => n.setAttribute("aria-pressed", String(i === j))); draw(); });
    pick.setAttribute("aria-pressed", String(i === 0)); moments.append(pick);
  });
  const keyboard = el("div", undefined, "point-keyboard");
  const coords = ["Horizontal %", "Vertical %"].map(label => {
    const input = el("input"); input.type = "number"; input.min = 0; input.max = 100; input.value = 50;
    const group = el("label", label); group.append(input); keyboard.append(group); return input;
  });
  keyboard.append(button("Add point", () => { if (coords.every(n => n.checkValidity())) add(+coords[0].value / 100, +coords[1].value / 100); }),
                  button("Undo point", () => { values.points.pop(); persist(); draw(); }));
  card.append(moments, keyboard, count); content.append(card); draw();
  const names = {human: "Human sounds", animal: "Animal sounds", natural: "Natural sounds", things: "Sounds of things", music: "Music"};
  for (const family of state.families) {
    const field = el("fieldset", undefined, "study-card"); field.append(el("legend", names[family] || family));
    for (const heard of [true, false]) {
      const label = el("label"), input = el("input"); input.type = "radio"; input.name = family; input.checked = values.heard[family] === heard;
      input.onchange = () => { values.heard[family] = heard; persist(); };
      label.append(input, el("span", heard ? "Heard" : "Did not hear")); field.append(label);
    }
    content.append(field);
  }
  actions(button("Save and continue", async () => { clearTimeout(timer); await send("observation", values, true); save(key, {}); }, true));
  cleanup = () => clearTimeout(timer);
}
function watching() {
  const videoInfo = state.video, resumePosition = state.position_ms;
  heading(videoInfo.title, `Video ${state.video_index + 1} of 3 · Adjust the captions as you watch.`);
  const layout = el("div", undefined, "watch-layout"), {wrapper, video} = player(videoInfo.url);
  const caption = el("div", "", "study-caption"); wrapper.append(caption);
  const panel = el("aside", undefined, "study-card study-controls");
  panel.append(el("h2", "Your caption detail"), el("p", "Move the point or use the sliders. Changes apply at a caption boundary."));
  const pad = el("div", undefined, "detail-pad"); pad.setAttribute("aria-hidden", "true");
  const dot = el("span", undefined, "detail-dot"); pad.append(dot);
  panel.append(el("div", "More acoustic detail ↑", "pad-key"), pad, el("div", "More source and scene detail →", "pad-key"));
  const desired = {...state.axes}, ranges = {}, labels = {};
  let controlTimer, alive = true, jobs = [], activeIndex = -1, applied = null, exposure = null, lastPosition = state.position_ms;
  let polling = false, tickPending = false, sequence = state.sequence, initialized = false;
  const status = el("p", "Your calibrated settings are ready."); status.setAttribute("role", "status");
  const updateControls = () => {
    dot.style.left = `${desired.context * 100}%`; dot.style.top = `${(1 - desired.texture) * 100}%`;
    for (const key of ["texture", "context"]) { ranges[key].value = Math.round(desired[key] * 100); labels[key].textContent = `${key === "texture" ? "Acoustic detail" : "Source and scene detail"}: ${Math.round(desired[key] * 100)}%`; }
  };
  const commit = () => { clearTimeout(controlTimer); status.textContent = "Applying your settings…"; send("settings", {video_id: videoInfo.id, ...desired}).catch(() => {}); };
  for (const key of ["texture", "context"]) {
    const label = el("label"), text = el("span"), input = el("input"); input.type = "range"; input.min = 0; input.max = 100; input.step = 1;
    ranges[key] = input; labels[key] = text; label.append(text, input); panel.append(label);
    input.oninput = () => { desired[key] = +input.value / 100; updateControls(); clearTimeout(controlTimer); controlTimer = setTimeout(commit, 300); };
    input.onchange = commit;
  }
  const move = event => { const box = pad.getBoundingClientRect(); desired.context = Math.round(Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) * 100) / 100; desired.texture = Math.round(Math.max(0, Math.min(1, 1 - (event.clientY - box.top) / box.height)) * 100) / 100; updateControls(); };
  pad.onpointerdown = event => { pad.setPointerCapture(event.pointerId); move(event); };
  pad.onpointermove = event => { if (pad.hasPointerCapture(event.pointerId)) move(event); };
  pad.onpointerup = event => { if (pad.hasPointerCapture(event.pointerId)) { move(event); pad.releasePointerCapture(event.pointerId); commit(); } };
  const presets = el("div", undefined, "presets");
  for (const [name, texture, context] of [["Both brief", 0, 0], ["More texture", 1, 0], ["More context", 0, 1], ["Both detailed", 1, 1]]) presets.append(button(name, () => { Object.assign(desired, {texture, context}); updateControls(); commit(); }));
  const presetMenu = el("details"); presetMenu.append(el("summary", "Detail presets"), presets);
  panel.append(presetMenu, button("Reset to calibration", () => { Object.assign(desired, state.defaults); updateControls(); commit(); }), status);
  layout.append(wrapper, panel); content.append(layout); updateControls();
  const outboxKey = "exposures";
  const interrupted = saved("open-exposure", null);
  if (interrupted) { const queue = saved(outboxKey, []); queue.push({...interrupted, incomplete: true}); save(outboxKey, queue); save("open-exposure", null); }
  function finishExposure() {
    if (!exposure) return;
    exposure.end_ms = Math.max(exposure.start_ms, Math.min(exposure.cue_end, lastPosition));
    const queue = saved(outboxKey, []); queue.push(exposure); save(outboxKey, queue); exposure = null;
    save("open-exposure", null);
  }
  async function flush() {
    const queue = saved(outboxKey, []).filter(e => e.video_id === videoInfo.id);
    if (!queue.length) return;
    const entries = queue.slice(0, 100);
    await send("exposures", {video_id: videoInfo.id, entries});
    const ids = new Set(entries.map(e => e.id)); save(outboxKey, saved(outboxKey, []).filter(e => !ids.has(e.id)));
  }
  function paint() {
    const ms = video.currentTime * 1000;
    if (!video.seeking && ms >= lastPosition && ms - lastPosition < 1000) lastPosition = ms;
    const index = videoInfo.cues.findIndex(cue => cue.start_ms <= ms && ms < cue.end_ms);
    if (index !== activeIndex) {
      finishExposure(); activeIndex = index; applied = null;
      if (index >= 0) {
        const cue = videoInfo.cues[index];
        const job = jobs.find(j => j.cue === index && j.result && j.revision === state.settings_revision);
        applied = job?.result && !job.result.fallback ? {...job.result, job_id: job.id, revision: job.revision, axes: {...state.axes}} : {text: cue.fallback, fallback: true, reason: "not_ready_at_boundary"};
        caption.textContent = applied.text;
        status.textContent = applied.fallback ? "Showing the available caption for this moment." : `Applied: acoustic ${Math.round(state.axes.texture * 100)}%, source/scene ${Math.round(state.axes.context * 100)}%.`;
      } else caption.textContent = "";
    }
    if (!exposure && applied && !video.paused && !video.seeking && !document.hidden) {
      exposure = {id: crypto.randomUUID(), video_id: videoInfo.id, cue: index,
        cue_end: videoInfo.cues[index].end_ms, start_ms: ms, end_ms: ms,
        text: applied.text, fallback: applied.fallback, job_id: applied.job_id || null,
        settings_revision: applied.revision ?? null, axes: applied.fallback ? null : applied.axes};
    }
    if (exposure) { exposure.end_ms = Math.max(exposure.start_ms, Math.min(exposure.cue_end, lastPosition)); save("open-exposure", exposure); }
  }
  async function tick(seek = false) {
    if (!alive || !initialized || (tickPending && !seek)) return;
    tickPending = true;
    sequence = Math.max(sequence, state.sequence) + 1;
    try { await send("playback", {video_id: videoInfo.id, position_ms: Math.min(videoInfo.duration_ms, video.currentTime * 1000),
      sequence, playing: !video.paused, hidden: document.hidden, seek}); }
    finally { tickPending = false; }
  }
  video.onloadedmetadata = () => {
    initialized = true;
    if (resumePosition > 0) video.currentTime = resumePosition / 1000;
    else { tick().catch(() => {}); paint(); }
  };
  video.ontimeupdate = paint;
  video.onplay = () => { tick().catch(() => {}); paint(); };
  video.onpause = () => { finishExposure(); tick().then(flush).catch(() => {}); };
  video.onseeking = () => { finishExposure(); activeIndex = -1; jobs = []; caption.textContent = ""; };
  video.onseeked = () => { lastPosition = video.currentTime * 1000; tick(true).catch(() => {}); paint(); };
  const visibility = () => { if (document.hidden) finishExposure(); tick().then(flush).catch(() => {}); };
  document.addEventListener("visibilitychange", visibility);
  const next = button(state.video_index === 2 ? "Continue to the final survey" : "Finish video", async () => {
    finishExposure(); await tick();
    while (saved(outboxKey, []).some(e => e.video_id === videoInfo.id)) await flush();
    await send("video-ended", {video_id: videoInfo.id}, true);
  }, true);
  next.disabled = true; video.onended = () => { finishExposure(); next.disabled = false; tick().then(flush).catch(() => {}); };
  actions(button("Fullscreen", () => layout.requestFullscreen()), next);
  const timer = setInterval(async () => {
    if (polling || !alive) return; polling = true;
    try {
      const result = await request("/api/study/captions");
      if (alive && result.epoch === state.epoch && result.revision === state.settings_revision) jobs = result.jobs;
      if (!video.paused) await tick();
      await flush();
    } catch (error) { if (alive) notice(`Connection interrupted. Your progress will retry. ${error.message}`); }
    finally { polling = false; }
  }, 1000);
  cleanup = () => { alive = false; clearInterval(timer); clearTimeout(controlTimer); document.removeEventListener("visibilitychange", visibility); finishExposure(); video.onpause = null; video.pause(); };
}
function render() {
  cleanup(); cleanup = () => {}; content.replaceChildren();
  document.querySelector(".study-shell").classList.toggle("is-watching", state.stage === "watch");
  const calibration = ["preferences", "clip", "observe"].includes(state.stage);
  document.getElementById("rail-calibration").setAttribute("aria-current", calibration ? "step" : "false");
  document.getElementById("rail-viewing").setAttribute("aria-current", calibration ? "false" : "step");
  document.getElementById("progress").textContent = calibration ? `Calibration · ${Math.min(state.clip_index + 1, state.calibration_count)} of ${state.calibration_count} clips` : state.stage === "done" ? "Study complete" : `Viewing experience · ${state.completed_videos} of 3 complete`;
  if (state.stage === "preferences") return form(state.items, "preferences", "Make the captions yours", "First, tell us how much detail you prefer. Then watch a few short clips and tell us what you notice.");
  if (state.stage === "clip") return calibrationClip();
  if (state.stage === "observe") return observation();
  if (state.stage === "watch") return watching();
  if (state.stage === "final") return form(state.items, "final-survey", "Your viewing experience", "Thinking about the three videos you just watched, tell us how the captions and controls felt.");
  if (state.stage === "ready") {
    heading("Your calibration is saved", "Next, watch three five-minute videos with captions shaped by your responses. You can adjust the level of detail while watching.");
    actions(button("Start viewing experience", () => send("start-viewing", {}, true), true));
  } else if (state.stage === "break") {
    heading("Take a moment", `You have finished ${state.completed_videos} of 3 videos. Your chosen detail settings will carry into the next video.`);
    actions(button("Continue to the next video", () => send("continue", {}, true), true));
  } else {
    heading("Thank you for taking part", "Your calibration, viewing experience and final responses have been saved. You can close this page.");
  }
}
async function boot(code) {
  try { state = await request("/api/study/session", code ? {code} : {}); render(); }
  catch (error) {
    content.replaceChildren(); heading("Join the study", error.message);
    const label = el("label", "Access code "), input = el("input"); input.type = "password"; input.autocomplete = "off"; label.append(input); content.append(label);
    actions(button("Continue", () => boot(input.value), true));
  }
}
boot();

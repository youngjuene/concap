/* Shared-theme two-stage study. Requests serialize; exposure IDs survive retries. */
"use strict";
const content = document.getElementById("content");
let copy = {};
const studyBase = new URL(".", window.location.href);
const studyURL = path => new URL(path.replace(/^\//, ""), studyBase).pathname;
const t = (text, values = {}) => {
  let result = (state?.language === "ko" ? copy[text] : null) || String(text);
  for (const [key, value] of Object.entries(values)) result = result.replaceAll(`{${key}}`, String(value));
  return result;
};
let state, chain = Promise.resolve(), cleanup = () => {};
let recoveryNotice = "";
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = t(text);
  if (className) node.className = className;
  return node;
};
const notice = (text = "") => {
  const node = document.getElementById("notice");
  const message = text || recoveryNotice;
  node.textContent = t(message); node.hidden = !message;
};
function captionDetailScale(contract, language) {
  const discrete = contract?.version === "five-level/v1";
  const values = discrete ? contract.values : null;
  const labels = discrete ? (contract.labels[language] || contract.labels.en) : null;
  const clamp = value => Math.max(0, Math.min(1, value));
  const toInput = value => Math.round(clamp(value) * (discrete ? 4 : 100)) + (discrete ? 1 : 0);
  const fromInput = value => discrete ? values[Number(value) - 1] : Number(value) / 100;
  const compact = value => discrete ? `${toInput(value)}/5` : `${toInput(value)}%`;
  return {
    discrete, min: discrete ? 1 : 0, max: discrete ? 5 : 100, step: 1,
    labels, toInput, fromInput, compact,
    display: value => discrete ? `${compact(value)} · ${labels[toInput(value) - 1]}` : compact(value),
  };
}
function storageKey(kind) { return `caption-study:${state.session_id}:${kind}`; }
function saved(kind, fallback) {
  try { return JSON.parse(localStorage.getItem(storageKey(kind))) ?? fallback; } catch { return fallback; }
}
function save(kind, value) { localStorage.setItem(storageKey(kind), JSON.stringify(value)); }
function viewingTotal() { return state.viewing_total || Math.max(state.completed_videos || 0, state.video_index + 1, 3); }
function isLastVideo() { return state.video_index + 1 >= viewingTotal(); }
async function request(path, payload) {
  const controller = new AbortController();
  // Public relay requests can take several seconds while media is streaming.
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(studyURL(path), {signal: controller.signal, ...(payload === undefined ? {} : {
      method: "POST", headers: {"content-type": "application/json", "x-study-request": "1"},
      body: JSON.stringify(payload),
    })});
    const result = await response.json();
    if (!response.ok) { const error = new Error(result.error || "Request failed"); error.status = response.status; throw error; }
    return result;
  } finally { clearTimeout(timer); }
}

function pauseForRecovery(video) {
  if (!video.paused || video.ended) {
    // Replay the last unacknowledged interval instead of leaving a gap that
    // would prevent completion after an ordinary interruption.
    const checkpoint = Math.min(video.currentTime, state.position_ms / 1000,
      video.ended ? Math.max(0, video.duration - 0.75) : Infinity);
    video.pause();
    video.currentTime = Math.max(0, checkpoint - 0.25);
    recoveryNotice = "Playback paused after an interruption. Press play to resume from your saved position.";
  }
  notice();
}
function send(action, data, redraw = false, valid = () => true) {
  const key = crypto.randomUUID();
  const input = typeof data === "function" ? data : structuredClone(data);
  const run = async () => {
    if (!valid()) return null;
    const data = typeof input === "function" ? structuredClone(input()) : input;
    const payload = {key, revision: state.revision, data};
    let result;
    try { result = await request(`/api/study/${action}`, payload); }
    catch (error) {
      if (!valid()) return null;
      // A lost response may already have committed: same key safely retrieves its receipt.
      if (!error.status) result = await request(`/api/study/${action}`, payload);
      else if (error.status === 409) {
        const fresh = await request("/api/study/state");
        if (!valid()) return null;
        const stageChanged = fresh.stage !== state.stage;
        state = fresh;
        if (stageChanged) { render(); throw error; }
        payload.revision = state.revision;
        if (action === "playback" || action === "clip-playback") data.sequence = Math.max(data.sequence, state.sequence + 1);
        result = await request(`/api/study/${action}`, payload);
      } else throw error;
    }
    if (!valid()) return null;
    state = result;
    notice();
    if (redraw) render();
    return result;
  };
  const pending = chain.then(run);
  chain = pending.catch(async error => {
    if (!valid()) return;
    notice(error.message);
    if (error.status === 409) {
      try { const fresh = await request("/api/study/state"); if (valid()) { state = fresh; render(); } } catch { /* next user retry */ }
    }
  });
  return pending;
}
function button(text, action, primary = false, available = () => true) {
  const node = el("button", text, primary ? "primary" : "");
  node.type = "button";
  node.onclick = async () => {
    node.disabled = true; notice();
    try { await action(); } catch (error) { notice(recoveryNotice || error.message); }
    finally { node.disabled = !available(); }
  };
  return node;
}
function heading(title, description) {
  const head = el("div", undefined, "head");
  const titleNode = el("h1", title);
  titleNode.tabIndex = -1;
  head.append(titleNode);
  if (description) head.append(el("p", description, "lede"));
  content.append(head);
  titleNode.focus();
}
function actions(...nodes) { const row = el("div", undefined, "actions"); row.append(...nodes); content.append(row); }
function form(items, submitAction, title, description, submitLabel = "Submit your experience") {
  heading(title, description);
  const key = `form:${state.stage}${state.stage === "video-survey" ? `:${state.video_index}` : ""}`;
  const draft = state.stage === "video-survey" && state.draft?.answers ? state.draft.answers : state.draft || {};
  const values = saved(key, draft);
  const node = el("form");
  node.onsubmit = event => event.preventDefault();
  let timer;
  const videoID = state.stage === "video-survey" ? state.video.id : null;
  const changed = () => {
    save(key, values);
    clearTimeout(timer);
    timer = setTimeout(() => send("draft", videoID ? {video_id: videoID, answers: values} : values).catch(() => {}), 800);
  };
  let previousGroup;
  for (const item of items) {
    if (item.group && item.group !== previousGroup) {
      const groupTitle = el("h2");
      groupTitle.textContent = item.group_title || (item.group === "prss" ? t("Overall soundscape experience (PRSS)") : "");
      node.append(groupTitle);
      previousGroup = item.group;
    }
    const field = el("fieldset", undefined, "study-card");
    const itemText = state.instrument_hash ? item.text : t(item.text);
    const legend = el("legend"); legend.textContent = itemText; field.append(legend);
    if (item.type === "text") {
      const input = el("textarea"); input.maxLength = 2000; input.value = values[item.id] || "";
      input.setAttribute("aria-label", t("{question} (optional)", {question: itemText}));
      input.oninput = () => { values[item.id] = input.value; changed(); };
      field.append(el("p", "Optional"), input);
    } else {
      if (item.type === "rating") {
        const [low, high] = item.anchors || [t(item.low || "Strongly disagree"), t(item.high || "Strongly agree")];
        const anchors = el("p"); anchors.textContent = `${low} (1) → ${high} (${item.points || 5})`; field.append(anchors);
      }
      const choices = item.type === "rating"
        ? [...Array.from({length: item.points || 5}, (_, i) => i + 1), ...(item.na ? ["na"] : [])] : item.options;
      const group = el("div", undefined, item.type === "rating" ? "ratings" : "");
      for (const value of choices) {
        const label = el("label"); const input = el("input");
        input.type = "radio"; input.name = item.id; input.value = String(value);
        input.checked = values[item.id] === value;
        input.onchange = () => { values[item.id] = value; changed(); };
        const text = el("span");
        text.textContent = value === "na" ? (item.na_label || t("Not applicable"))
          : item.option_labels?.[item.options?.indexOf(value)] || t(String(value));
        label.append(input, text); group.append(label);
      }
      field.append(group);
    }
    node.append(field);
  }
  content.append(node);
  actions(button(submitAction === "preferences" ? "Begin calibration clips" : submitLabel, async () => {
    clearTimeout(timer);
    const data = videoID ? {video_id: videoID, answers: {...values}} : {...values};
    await send(submitAction, data, true); save(key, {});
  }, true));
  cleanup = () => clearTimeout(timer);
}
function player(source, prepare = false) {
  const wrapper = el("div", undefined, "study-player");
  const video = el("video"); video.controls = !prepare; video.playsInline = true; video.preload = "metadata";
  video.setAttribute("controlslist", "nofullscreen noremoteplayback"); video.disablePictureInPicture = true;
  video.onerror = () => notice("This video could not be played. Reload to retry, or contact the researcher if the problem continues.");
  wrapper.append(video);
  const controller = new AbortController();
  let objectURL;
  if (!prepare) video.src = studyURL(source);
  else {
    const progress = el("p", "Preparing the video… Playback will be available when the download finishes.", "study-preparation");
    progress.setAttribute("role", "status");
    const retry = button("Retry video download", load);
    retry.hidden = true;
    wrapper.append(progress, retry);
    async function load() {
      retry.hidden = true;
      try {
        const sourceURL = new URL(studyURL(source), location.href).href;
        const cacheKey = new URL(sourceURL);
        cacheKey.searchParams.set("study_session", state.session_id);
        let cache;
        try { cache = await window.caches?.open("caption-study-video-v1"); } catch {}
        const cached = await cache?.match(cacheKey.href);
        const response = cached || await fetch(sourceURL, {signal: controller.signal});
        if (!response.ok) throw new Error("Video download failed");
        const total = Number(response.headers.get("content-length"));
        const reader = response.body.getReader(), chunks = [];
        let received = 0;
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          chunks.push(value); received += value.length;
          progress.textContent = total > 0
            ? t("Preparing the video… {percent}%", {percent: Math.floor(received / total * 100)})
            : t("Preparing the video… Playback will be available when the download finishes.");
        }
        const blob = new Blob(chunks, {type: response.headers.get("content-type") || "video/mp4"});
        if (!cached && cache) {
          try { await cache.put(cacheKey.href, new Response(blob, {headers: {
            "content-type": blob.type, "content-length": String(blob.size),
          }})); } catch { /* Storage quota/private mode may require another download on reload. */ }
        }
        if (controller.signal.aborted) return;
        objectURL = URL.createObjectURL(blob);
        video.src = objectURL; video.controls = true;
        progress.textContent = t("Video ready. Press play when you are ready.");
        video.addEventListener("play", () => { progress.hidden = true; }, {once: true});
      } catch (error) {
        if (!controller.signal.aborted) {
          progress.textContent = t("The video download was interrupted. Retry to continue.");
          retry.hidden = false;
        }
      }
    }
    load();
  }
  return {wrapper, video, release: () => {
    controller.abort(); if (objectURL) URL.revokeObjectURL(objectURL);
  }};
}
function calibrationClip() {
  const clip = state.clip, resumePosition = state.position_ms;
  heading(state.clip.title, "Watch and listen to this clip. Afterwards, tell us what you noticed.");
  const {wrapper, video} = player(state.clip.video);
  content.append(wrapper);
  let sequence = state.sequence, busy = false, alive = true, initialized = false;
  let seekGeneration = 0, acknowledgedSeek = 0;
  const tick = async (seek = false, force = false) => {
    if (!alive || !initialized || (video.seeking && !seek) || (busy && !seek && !force)) return;
    const generation = seekGeneration, valid = () => alive && generation === seekGeneration;
    let sentSeek = false;
    busy = true;
    sequence = Math.max(sequence, state.sequence) + 1;
    try {
      const sample = {clip_id: clip.id, position_ms: Math.min(clip.duration_ms, video.currentTime * 1000), sequence, playing: !video.paused, hidden: document.hidden};
      const result = await send("clip-playback", () => {
        sentSeek = generation > acknowledgedSeek;
        return {...sample, seek: sentSeek};
      }, false, valid);
      if (result && valid() && sentSeek) acknowledgedSeek = generation;
    }
    catch (error) { if (valid()) { next.disabled = true; pauseForRecovery(video); } throw error; }
    finally { busy = false; }
  };
  const next = button("Continue to what you noticed", async () => {
    await tick(false, true); await send("clip-ended", {}, true);
  }, true, () => video.ended);
  next.disabled = true;
  video.onloadedmetadata = () => {
    initialized = true;
    if (resumePosition > 0) video.currentTime = resumePosition / 1000;
    else tick(false, true).catch(() => {});
  };
  video.onplay = () => { recoveryNotice = ""; notice(); tick(false, true).catch(() => {}); };
  video.onpause = () => tick(false, true).catch(() => {});
  video.onseeking = () => { seekGeneration++; };
  video.onseeked = () => tick(true).catch(() => {});
  video.onended = async () => {
    next.disabled = true;
    try { await tick(false, true); if (alive && video.ended) next.disabled = false; }
    catch { /* Recovery pauses at the last acknowledged position. */ }
  };
  const visibility = () => { if (document.hidden) pauseForRecovery(video); tick(false, true).catch(() => {}); };
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
  const image = el("img"); image.alt = t("Calibration frame. Mark objects you noticed.");
  frame.append(image); card.append(frame);
  const count = el("p");
  const draw = () => {
    image.src = studyURL(state.clip.frames[index].url);
    frame.querySelectorAll(".visual-point").forEach(n => n.remove());
    for (const point of values.points.filter(p => p.frame === index)) {
      const dot = el("span", undefined, "visual-point"); dot.style.left = `${point.x * 100}%`; dot.style.top = `${point.y * 100}%`; frame.append(dot);
    }
    count.textContent = t("{count} points marked", {count: values.points.length});
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
  const detailScale = captionDetailScale(state.detail_control, state.language);
  heading(videoInfo.title, t("Video {number} of {total} · Adjust the captions as you watch.", {number: state.video_index + 1, total: viewingTotal()}));
  const layout = el("div", undefined, "watch-layout"), {wrapper, video, release} = player(videoInfo.url, true);
  const caption = el("div", "", "study-caption"); wrapper.append(caption);
  const captionLevels = el("p", "No caption applied yet.", "caption-level-status");
  wrapper.append(captionLevels);
  const meter = el("div", undefined, "detail-meter");
  meter.setAttribute("aria-hidden", "true");
  meter.append(el("div", "Selected caption detail", "detail-meter-title"));
  const meterRows = {};
  for (const key of ["texture", "context"]) {
    const row = el("div", undefined, "detail-meter-row");
    const label = el("span", key === "texture" ? "Acoustic detail" : "Source and scene detail");
    const value = el("span", "", "detail-meter-value");
    const track = el("div", undefined, "detail-meter-track"), fill = el("div", undefined, "detail-meter-fill");
    track.append(fill); row.append(label, value, track); meter.append(row);
    meterRows[key] = {value, fill};
  }
  wrapper.append(meter);
  let meterTimer, heldPointer = null;
  const showMeter = () => {
    clearTimeout(meterTimer); meter.classList.add("is-visible");
    if (heldPointer === null) meterTimer = setTimeout(() => meter.classList.remove("is-visible"), 1500);
  };
  const holdMeter = event => { heldPointer = event.pointerId; showMeter(); };
  const releaseMeter = event => {
    if (event.pointerId === heldPointer) { heldPointer = null; showMeter(); }
  };
  const panel = el("aside", undefined, "study-card study-controls");
  panel.append(el("h2", "Your caption detail"), el("p", detailScale.discrete
    ? "Choose a level from 1 to 5 for each kind of detail. Changes apply with a new caption once it is ready."
    : "Use the sliders to adjust caption detail. Changes apply at a caption boundary."));
  const desired = {...state.axes}, ranges = {}, labels = {};
  const settingsKey = axes => JSON.stringify({texture: axes.texture, context: axes.context});
  let acceptedSettingsKey = settingsKey(desired), pendingSettings = null, sendingSettings = false, failedSettings = null;
  let controlTimer, alive = true, jobs = [], activeIndex = -1, applied = null, exposure = null, lastPosition = state.position_ms;
  let polling = false, tickPending = false, sequence = state.sequence, initialized = false;
  let seekGeneration = 0, acknowledgedSeek = 0;
  const status = el("p", "Your calibrated settings are ready."); status.setAttribute("role", "status");
  const updateControls = (reveal = false) => {
    for (const key of ["texture", "context"]) {
      const percent = Math.round(desired[key] * 100);
      ranges[key].value = detailScale.toInput(desired[key]);
      ranges[key].style.setProperty("--detail-level", `${percent}%`);
      ranges[key].setAttribute("aria-valuetext", detailScale.discrete
        ? t("Selected: {level}", {level: detailScale.display(desired[key])}) : t("Selected: {percent}%", {percent}));
      labels[key].textContent = `${t(key === "texture" ? "Acoustic detail" : "Source and scene detail")}: ${detailScale.display(desired[key])}`;
      meterRows[key].value.textContent = detailScale.compact(desired[key]);
      meterRows[key].fill.style.width = `${percent}%`;
    }
    if (reveal) showMeter();
  };
  const retrySettings = button("Retry settings", () => {
    if (!failedSettings) return;
    pendingSettings = failedSettings; failedSettings = null; retrySettings.hidden = true; flushSettings();
  });
  retrySettings.hidden = true;
  async function flushSettings() {
    if (sendingSettings) return;
    sendingSettings = true;
    try {
      while (alive && pendingSettings) {
        const entry = pendingSettings;
        pendingSettings = null;
        if (entry.key === acceptedSettingsKey) continue;
        showMeter(); status.textContent = t("Applying your settings…"); retrySettings.hidden = true;
        try {
          const result = await send("settings", entry.data, false, () => alive);
          if (!result) continue;
          acceptedSettingsKey = settingsKey(result.axes || entry.data); failedSettings = null;
          status.textContent = t("Settings saved. Captions will update at a caption boundary.");
        } catch (error) {
          // The server may have committed the request despite a lost reply.
          // Reconcile the latest selection instead of trusting the old key.
          acceptedSettingsKey = null;
          if (pendingSettings && pendingSettings.key !== entry.key) continue;
          failedSettings = entry; retrySettings.hidden = false;
          status.textContent = t("Settings were not saved. Retry when the connection is stable.");
          break;
        }
      }
    } finally {
      sendingSettings = false;
    }
    if (alive && pendingSettings && (!failedSettings || pendingSettings.key !== failedSettings.key)) flushSettings();
  }
  const commit = (origin = "slider") => {
    clearTimeout(controlTimer);
    const key = settingsKey(desired);
    if (!sendingSettings && key === acceptedSettingsKey && failedSettings?.key !== key) return;
    pendingSettings = {
      key,
      data: {
        video_id: videoInfo.id,
        position_hint_ms: Math.min(videoInfo.duration_ms, video.currentTime * 1000),
        origin,
        ...desired,
      },
    };
    failedSettings = failedSettings?.key === key ? null : failedSettings;
    retrySettings.hidden = true;
    flushSettings();
  };
  for (const key of ["texture", "context"]) {
    const label = el("label"), text = el("span"), input = el("input"); input.type = "range";
    input.min = detailScale.min; input.max = detailScale.max; input.step = detailScale.step;
    ranges[key] = input; labels[key] = text; label.append(text, input); panel.append(label);
    if (detailScale.discrete) {
      const steps = el("span", undefined, "detail-steps"); steps.setAttribute("aria-hidden", "true");
      for (let level = 1; level <= 5; level++) steps.append(el("span", String(level)));
      label.append(steps);
    }
    const endpoints = el("span", undefined, "detail-endpoints"); endpoints.setAttribute("aria-hidden", "true");
    if (detailScale.discrete) {
      const low = el("span"), high = el("span");
      low.textContent = detailScale.labels[0]; high.textContent = detailScale.labels[4];
      endpoints.append(low, high);
    } else endpoints.append(el("span", key === "texture" ? "Brief" : "General"), el("span", key === "texture" ? "Detailed" : "Specific"));
    label.append(endpoints);
    input.onpointerdown = holdMeter;
    input.oninput = () => {
      desired[key] = detailScale.fromInput(input.value); updateControls(true); clearTimeout(controlTimer);
      controlTimer = setTimeout(() => commit(`slider:${key}`), 300);
    };
    input.onchange = () => commit(`slider:${key}`);
  }
  document.addEventListener("pointerup", releaseMeter);
  document.addEventListener("pointercancel", releaseMeter);
  const presets = el("div", undefined, "presets");
  for (const [name, texture, context] of [["Both brief", 0, 0], ["More texture", 1, 0], ["More context", 0, 1], ["Both detailed", 1, 1]]) presets.append(button(name, () => { Object.assign(desired, {texture, context}); updateControls(); commit(`preset:${name}`); }));
  const presetMenu = el("details"); presetMenu.append(el("summary", "Detail presets"), presets);
  panel.append(presetMenu, button("Reset to calibration", () => { Object.assign(desired, state.defaults); updateControls(); commit("reset"); }), status, retrySettings);
  layout.append(wrapper, panel); content.append(layout); updateControls();
  const outboxKey = "exposures";
  let exposureEpisode = crypto.randomUUID();
  const interrupted = saved("open-exposure", null);
  if (interrupted) { const queue = saved(outboxKey, []); queue.push({...interrupted, incomplete: true, closed_by: "interrupted"}); save(outboxKey, queue); save("open-exposure", null); }
  function exposurePosition() {
    const ms = Math.min(videoInfo.duration_ms, video.currentTime * 1000);
    if (!video.seeking && ms >= lastPosition) lastPosition = ms;
    return ms;
  }
  function finishExposure(reason = "replacement", sample = true) {
    if (!exposure) return;
    if (sample) exposurePosition();
    // The DOM retains this text until replacement. The cue's nominal end is
    // provenance, not the actual end of the displayed exposure.
    exposure.end_ms = Math.max(exposure.start_ms, lastPosition);
    exposure.closed_by = reason;
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
    const ms = exposurePosition();
    const index = videoInfo.cues.findIndex(cue => cue.start_ms <= ms && ms < cue.end_ms);
    if (index !== activeIndex) {
      finishExposure("replacement", false); activeIndex = index; applied = null;
      if (index >= 0) {
        const cue = videoInfo.cues[index];
        const job = jobs.find(j => j.cue === index && j.result && j.revision === state.settings_revision);
        applied = job?.result && !job.result.fallback ? {...job.result, job_id: job.id, revision: job.revision, axes: {...state.axes}} : {text: cue.fallback, fallback: true, reason: "not_ready_at_boundary"};
        caption.textContent = applied.text;
        captionLevels.textContent = applied.fallback ? t("Prepared caption · selected levels not applied.")
          : detailScale.discrete ? t("Current caption: acoustic {texture}; source/scene {context}.", {
            texture: detailScale.display(applied.axes.texture), context: detailScale.display(applied.axes.context),
          }) : t("Current caption: acoustic {texture}%, source/scene {context}%.", {
            texture: Math.round(applied.axes.texture * 100), context: Math.round(applied.axes.context * 100),
          });
        status.textContent = applied.fallback ? t("Showing the available caption for this moment.")
          : detailScale.discrete ? t("Applied: acoustic {texture}; source/scene {context}.", {
            texture: detailScale.display(applied.axes.texture), context: detailScale.display(applied.axes.context),
          }) : t("Applied: acoustic {texture}%, source/scene {context}%.", {texture: Math.round(applied.axes.texture * 100), context: Math.round(applied.axes.context * 100)});
      } else { caption.textContent = ""; captionLevels.textContent = t("No caption applied yet."); }
    }
    if (!exposure && applied && !video.paused && !video.seeking && !document.hidden) {
      exposure = {id: crypto.randomUUID(), video_id: videoInfo.id, cue: index,
        timing: "display-v1", episode_id: exposureEpisode,
        cue_end: videoInfo.cues[index].end_ms, start_ms: ms, end_ms: ms,
        text: applied.text, fallback: applied.fallback, job_id: applied.job_id || null,
        settings_revision: applied.revision ?? null, axes: applied.fallback ? null : applied.axes};
    }
    if (exposure) { exposure.end_ms = Math.max(exposure.start_ms, lastPosition); save("open-exposure", exposure); }
  }
  async function tick(seek = false, force = false) {
    if (!alive || !initialized || (video.seeking && !seek) || (tickPending && !seek && !force)) return;
    tickPending = true;
    const generation = seekGeneration, valid = () => alive && generation === seekGeneration;
    let sentSeek = false;
    sequence = Math.max(sequence, state.sequence) + 1;
    try {
      const sample = {video_id: videoInfo.id, position_ms: Math.min(videoInfo.duration_ms, video.currentTime * 1000),
        sequence, playing: !video.paused, hidden: document.hidden};
      const result = await send("playback", () => {
        sentSeek = generation > acknowledgedSeek;
        return {...sample, seek: sentSeek};
      }, false, valid);
      if (result && valid() && sentSeek) acknowledgedSeek = generation;
    }
    catch (error) { if (valid()) { next.disabled = true; pauseForRecovery(video); } throw error; }
    finally { tickPending = false; }
  }
  video.onloadedmetadata = () => {
    initialized = true;
    if (resumePosition > 0) video.currentTime = resumePosition / 1000;
    else { tick().catch(() => {}); paint(); }
  };
  video.ontimeupdate = paint;
  video.onplay = () => { recoveryNotice = ""; notice(); tick(false, true).catch(() => {}); paint(); };
  video.onpause = () => { finishExposure("pause"); tick(false, true).then(flush).catch(() => {}); };
  video.onseeking = () => { seekGeneration++; finishExposure("seek", false); exposureEpisode = crypto.randomUUID(); activeIndex = -1; jobs = []; caption.textContent = ""; captionLevels.textContent = t("No caption applied yet."); };
  video.onseeked = () => { lastPosition = video.currentTime * 1000; tick(true).catch(() => {}); paint(); };
  const visibility = () => {
    if (document.hidden) { finishExposure("hidden"); pauseForRecovery(video); }
    tick(false, true).then(flush).catch(() => { if (alive) pauseForRecovery(video); });
  };
  document.addEventListener("visibilitychange", visibility);
  const pagehide = () => finishExposure("pagehide");
  window.addEventListener("pagehide", pagehide);
  const next = button(state.survey_flow ? "Continue to this video's questions"
    : isLastVideo() ? "Continue to the final survey" : "Finish video", async () => {
    finishExposure();
    try {
      await tick(false, true);
      while (saved(outboxKey, []).some(e => e.video_id === videoInfo.id)) await flush();
    } catch (error) { pauseForRecovery(video); throw error; }
    await send("video-ended", {video_id: videoInfo.id}, true);
  }, true, () => video.ended);
  next.disabled = true;
  video.onended = async () => {
    next.disabled = true; finishExposure("ended");
    try { await tick(false, true); await flush(); if (alive && video.ended) next.disabled = false; }
    catch { if (alive) pauseForRecovery(video); }
  };
  const fullscreen = button("Fullscreen", () => document.fullscreenElement === layout
    ? document.exitFullscreen() : layout.requestFullscreen());
  const fullscreenChanged = () => {
    fullscreen.textContent = t(document.fullscreenElement === layout ? "Exit fullscreen" : "Fullscreen");
  };
  document.addEventListener("fullscreenchange", fullscreenChanged);
  const playbackActions = el("div", undefined, "actions");
  playbackActions.append(fullscreen, next); layout.append(playbackActions);
  // Playback acknowledgements must not wait behind a slow caption fetch.
  const playbackTimer = setInterval(() => {
    if (!video.paused) tick().catch(() => {});
  }, 1000);
  const timer = setInterval(async () => {
    if (polling || !alive || !initialized) return; polling = true;
    try {
      const result = await request("/api/study/captions");
      if (alive && result.epoch === state.epoch && result.revision === state.settings_revision) jobs = result.jobs;
      await flush();
    } catch (error) {
      // A missing caption uses the prepared cue; playback failures have their
      // own pause/recovery path. Caption delivery alone must not rewind video.
      if (alive && !recoveryNotice) notice("Connection interrupted. Your progress will retry.");
    }
    finally { polling = false; }
  }, 1000);
  cleanup = () => {
    alive = false; clearInterval(timer); clearInterval(playbackTimer); clearTimeout(controlTimer);
    clearTimeout(meterTimer);
    document.removeEventListener("pointerup", releaseMeter);
    document.removeEventListener("pointercancel", releaseMeter);
    document.removeEventListener("visibilitychange", visibility);
    document.removeEventListener("fullscreenchange", fullscreenChanged);
    window.removeEventListener("pagehide", pagehide);
    finishExposure("cleanup"); video.onpause = null; video.pause();
    release();
  };
}
function render() {
  cleanup(); cleanup = () => {}; recoveryNotice = ""; content.replaceChildren();
  document.documentElement.lang = state.language;
  document.title = t("Sound captions — your viewing experience");
  document.querySelector(".study-top .eyebrow").textContent = t("SOUND CAPTIONS");
  document.querySelector(".study-rail").setAttribute("aria-label", t("Study progress"));
  content.setAttribute("aria-label", t("Current study step"));
  document.getElementById("rail-calibration").textContent = t("01 · Calibration");
  document.getElementById("rail-viewing").textContent = t("02 · Viewing experience");
  document.querySelector(".study-footer").textContent = t("Sound, attention, and the way you see a scene.");
  document.querySelector(".study-shell").classList.toggle("is-watching", state.stage === "watch");
  const calibration = ["preferences", "clip", "observe"].includes(state.stage);
  document.getElementById("rail-calibration").setAttribute("aria-current", calibration ? "step" : "false");
  document.getElementById("rail-viewing").setAttribute("aria-current", calibration ? "false" : "step");
  document.getElementById("progress").textContent = calibration ? t("Calibration · {number} of {count} clips", {number: Math.min(state.clip_index + 1, state.calibration_count), count: state.calibration_count}) : state.stage === "done" ? t("Study complete") : t("Viewing experience · {count} of {total} complete", {count: state.completed_videos, total: viewingTotal()});
  if (state.stage === "preferences") return form(state.items, "preferences", "Make the captions yours", "First, tell us how much detail you prefer. Then watch a few short clips and tell us what you notice.");
  if (state.stage === "clip") return calibrationClip();
  if (state.stage === "observe") return observation();
  if (state.stage === "watch") return watching();
  if (state.stage === "video-survey") return form(state.items, "video-survey",
    t("Video {number} of {total} · Your experience", {number: state.video_index + 1, total: viewingTotal()}),
    "Answer these four questions about the video you just watched. Overall soundscape questions come after all videos.",
    isLastVideo() ? "Save and continue to overall experience" : "Save this video's answers");
  if (state.stage === "final") return form(state.items, "final-survey", "Your viewing experience",
    state.survey_flow
      ? "Thinking about your overall experience across the longer videos in Chapter 2, answer the soundscape questions (PRSS) and reflect on the caption controls."
      : "Thinking about the videos you just watched, tell us how the captions and controls felt.");
  if (state.stage === "ready") {
    heading("Your calibration is saved", state.survey_flow
      ? "Next, watch the five-minute videos with adjustable captions. Answer four questions after each video, then reflect on your overall experience and soundscapes once at the end."
      : "Next, watch the five-minute videos with captions shaped by your responses. You can adjust the level of detail while watching.");
    actions(button("Start viewing experience", () => send("start-viewing", {}, true), true));
  } else if (state.stage === "break") {
    heading("Take a moment", t("You have finished {count} of {total} videos. Your chosen detail settings will carry into the next video.", {count: state.completed_videos, total: viewingTotal()}));
    actions(button("Continue to the next video", () => send("continue", {}, true), true));
  } else {
    heading("Thank you for taking part", "Your calibration, viewing experience and final responses have been saved. You can close this page.");
    content.append(el("p", t("Receipt: {id}", {id: state.session_id}), "receipt"));
  }
}
async function boot(code) {
  try { copy = await request("/api/study/strings"); state = await request("/api/study/session", code ? {code} : {}); render(); }
  catch (error) {
    content.replaceChildren(); heading("Join the study", error.message);
    const label = el("label", "Access code "), input = el("input"); input.type = "password"; input.autocomplete = "off"; label.append(input); content.append(label);
    actions(button("Continue", () => boot(input.value), true));
  }
}
boot();

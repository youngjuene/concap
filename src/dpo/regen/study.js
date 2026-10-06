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
let recoveryNotice = "", storageNotice = "";
const volatileStorage = new Map();
let activeSessionLock = null, unlockSession = null, pageActive = true;
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = t(text);
  if (className) node.className = className;
  return node;
};
const notice = (text = "") => {
  const node = document.getElementById("notice");
  const message = text || recoveryNotice || (hasVolatileExposure() ? storageNotice : "");
  node.textContent = t(message); node.hidden = !message;
};
function storageKey(kind) { return `caption-study:${state.session_id}:${kind}`; }
function saved(kind, fallback) {
  const key = storageKey(kind);
  if (volatileStorage.has(key)) return structuredClone(volatileStorage.get(key));
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function save(kind, value) {
  const key = storageKey(kind);
  // Retain the current value before touching storage: quota/security failures
  // must not interrupt exposure completion or prevent its API upload.
  volatileStorage.set(key, structuredClone(value));
  try {
    localStorage.setItem(key, JSON.stringify(value));
    volatileStorage.delete(key);
    if (storageNotice) notice();
  } catch {
    // Remove an older durable value if quota permits removal but not a write.
    // Otherwise a later page could recover a stale, already-uploaded exposure.
    try { localStorage.removeItem(key); } catch { /* Keep the in-memory value. */ }
    storageNotice = "Browser storage is unavailable. Keep this tab open until your viewing records are uploaded.";
    notice();
  }
}
function hasVolatileExposure() {
  if (!state) return false;
  return ["exposures", "open-exposure"].some(kind => {
    const value = volatileStorage.get(storageKey(kind));
    return Array.isArray(value) ? value.length > 0 : Boolean(value);
  });
}
function ownsSessionLock() { return pageActive && activeSessionLock === state?.session_id; }
async function acquireSessionLock() {
  if (ownsSessionLock()) return true;
  if (!pageActive || !navigator.locks?.request) return false;
  const sessionID = state.session_id;
  if (activeSessionLock) releaseSessionLock();
  return new Promise(resolve => {
    navigator.locks.request(`caption-study:${sessionID}`, {ifAvailable: true}, async lock => {
      if (!lock || !pageActive) { resolve(false); return; }
      activeSessionLock = sessionID;
      await new Promise(release => { unlockSession = release; resolve(true); });
    }).catch(() => resolve(false));
  });
}
function releaseSessionLock() {
  activeSessionLock = null;
  const release = unlockSession;
  unlockSession = null;
  if (release) release();
}
async function enterSession() {
  if (await acquireSessionLock()) {
    state = await request("/api/study/state");
    render();
    return;
  }
  if (!pageActive) return;
  content.replaceChildren();
  heading(navigator.locks?.request ? "This session is already open" : "Study browser required", navigator.locks?.request
    ? "Continue in the other tab, or close it and try again here. Only one tab can record this session at a time."
    : "This browser cannot safely record the session. Open the study in Chrome using HTTPS or localhost.");
  if (navigator.locks?.request) actions(button("Try this tab again", enterSession, true));
}
function viewingTotal() { return state.viewing_total || Math.max(state.completed_videos || 0, state.video_index + 1, 3); }
function isLastVideo() { return state.video_index + 1 >= viewingTotal(); }
function interactionOnly() { return state.survey_policy === "interaction-only-sheet-v1"; }
async function request(path, payload, timeout = 15000) {
  const controller = new AbortController();
  // Public relay requests can take several seconds while media is streaming.
  const timer = setTimeout(() => controller.abort(), timeout);
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
    if (!ownsSessionLock()) throw new Error("This tab no longer owns the session. Reload to continue.");
    if (!valid()) return null;
    const data = typeof input === "function" ? structuredClone(input()) : input;
    const payload = {key, revision: state.revision, data};
    let result;
    try { result = await request(`/api/study/${action}`, payload); }
    catch (error) {
      if (!ownsSessionLock() || !valid()) return null;
      // A lost response may already have committed: same key safely retrieves its receipt.
      if (!error.status) result = await request(`/api/study/${action}`, payload);
      else if (error.status === 409) {
        const fresh = await request("/api/study/state");
        if (!ownsSessionLock() || !valid()) return null;
        const stageChanged = fresh.stage !== state.stage;
        state = fresh;
        if (stageChanged) { render(); throw error; }
        payload.revision = state.revision;
        if (action === "playback" || action === "clip-playback") data.sequence = Math.max(data.sequence, state.sequence + 1);
        result = await request(`/api/study/${action}`, payload);
      } else throw error;
    }
    if (!ownsSessionLock() || !valid()) return null;
    state = result;
    notice();
    if (redraw) render();
    return result;
  };
  const pending = chain.then(run);
  chain = pending.catch(async error => {
    if (!ownsSessionLock() || !valid()) return;
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
function player(source, prepare = false, deferControls = false) {
  const wrapper = el("div", undefined, "study-player");
  const video = el("video"); video.controls = !prepare; video.playsInline = true; video.preload = "metadata";
  video.setAttribute("controlslist", "nofullscreen noremoteplayback"); video.disablePictureInPicture = true;
  video.onerror = () => notice("This video could not be played. Reload to retry, or contact the researcher if the problem continues.");
  wrapper.append(video);
  const controller = new AbortController();
  let objectURL, enablePlayback = () => {};
  if (!prepare) video.src = studyURL(source);
  else {
    const progress = el("p", "Preparing the video… Playback will be available when the download finishes.", "study-preparation");
    progress.setAttribute("role", "status");
    enablePlayback = () => {
      video.controls = true;
      progress.textContent = t("Video ready. Press play when you are ready.");
    };
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
        video.src = objectURL;
        if (deferControls) progress.textContent = t("Preparing the first caption…");
        else enablePlayback();
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
  return {wrapper, video, enablePlayback, release: () => {
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
  heading(videoInfo.title, t("Video {number} of {total} · Adjust the captions as you watch.", {number: state.video_index + 1, total: viewingTotal()}));
  const layout = el("div", undefined, "watch-layout"), {wrapper, video, release, enablePlayback} = player(videoInfo.url, true, true);
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
  panel.append(el("h2", "Your caption detail"), el("p", "Move the point or use the sliders. Changes apply at a caption boundary."));
  const pad = el("div", undefined, "detail-pad"); pad.setAttribute("aria-hidden", "true");
  const dot = el("span", undefined, "detail-dot"); pad.append(dot);
  const desired = {...state.axes}, ranges = {}, labels = {};
  const settingsKey = axes => JSON.stringify({texture: axes.texture, context: axes.context});
  let acceptedSettingsKey = settingsKey(desired), pendingSettings = null, sendingSettings = false, failedSettings = null;
  let controlTimer, alive = true, jobs = [], activeIndex = -1, applied = null, exposure = null, lastPosition = state.position_ms;
  let polling = false, tickPending = false, sequence = state.sequence, initialized = false;
  let initialCaptionReady = false, initialCaptionPreparing = false, initialFallbackReason = null, hasPlayed = false;
  let seekGeneration = 0, acknowledgedSeek = 0;
  const status = el("p", "Your calibrated settings are ready."); status.setAttribute("role", "status");
  const updateControls = (reveal = false) => {
    dot.style.left = `${desired.context * 100}%`; dot.style.top = `${(1 - desired.texture) * 100}%`;
    for (const key of ["texture", "context"]) {
      const percent = Math.round(desired[key] * 100);
      ranges[key].value = percent;
      ranges[key].style.setProperty("--detail-level", `${percent}%`);
      ranges[key].setAttribute("aria-valuetext", t("Selected: {percent}%", {percent}));
      labels[key].textContent = `${t(key === "texture" ? "Acoustic detail" : "Source and scene detail")}: ${percent}%`;
      meterRows[key].value.textContent = `${percent}%`;
      meterRows[key].fill.style.width = `${percent}%`;
    }
    if (reveal) showMeter();
  };
  const retrySettings = button("Retry settings", () => {
    if (!failedSettings || failedSettings.key !== settingsKey(desired)) {
      failedSettings = null; retrySettings.hidden = true; return;
    }
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
          // The write may have committed even when both replies were lost.
          // Reconcile the newest intent instead of trusting the old baseline.
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
    failedSettings = null;
    retrySettings.hidden = true;
    if (!sendingSettings && key === acceptedSettingsKey) {
      pendingSettings = null;
      status.textContent = t("Settings saved. Captions will update at a caption boundary.");
      return;
    }
    pendingSettings = {
      key,
      data: {
        video_id: videoInfo.id,
        position_hint_ms: Math.min(videoInfo.duration_ms, video.currentTime * 1000),
        origin,
        ...desired,
      },
    };
    flushSettings();
  };
  for (const key of ["texture", "context"]) {
    const label = el("label"), text = el("span"), input = el("input"); input.type = "range"; input.min = 0; input.max = 100; input.step = 1;
    ranges[key] = input; labels[key] = text; label.append(text, input); panel.append(label);
    const endpoints = el("span", undefined, "detail-endpoints"); endpoints.setAttribute("aria-hidden", "true");
    endpoints.append(el("span", key === "texture" ? "Brief" : "General"), el("span", key === "texture" ? "Detailed" : "Specific"));
    label.append(endpoints);
    input.onpointerdown = holdMeter;
    input.oninput = () => {
      desired[key] = +input.value / 100; updateControls(true); clearTimeout(controlTimer);
      controlTimer = setTimeout(() => commit(`slider:${key}`), 300);
    };
    input.onchange = () => commit(`slider:${key}`);
  }
  panel.append(el("div", "More acoustic detail ↑", "pad-key"), pad, el("div", "More source and scene detail →", "pad-key"));
  document.addEventListener("pointerup", releaseMeter);
  document.addEventListener("pointercancel", releaseMeter);
  const move = event => { const box = pad.getBoundingClientRect(); desired.context = Math.round(Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) * 100) / 100; desired.texture = Math.round(Math.max(0, Math.min(1, 1 - (event.clientY - box.top) / box.height)) * 100) / 100; updateControls(true); };
  pad.onpointerdown = event => { holdMeter(event); pad.setPointerCapture(event.pointerId); move(event); };
  pad.onlostpointercapture = releaseMeter;
  pad.onpointermove = event => { if (pad.hasPointerCapture(event.pointerId)) move(event); };
  pad.onpointerup = event => { if (pad.hasPointerCapture(event.pointerId)) { move(event); pad.releasePointerCapture(event.pointerId); commit("pad"); } };
  const presets = el("div", undefined, "presets");
  for (const [name, texture, context] of [["Both brief", 0, 0], ["More texture", 1, 0], ["More context", 0, 1], ["Both detailed", 1, 1]]) presets.append(button(name, () => { Object.assign(desired, {texture, context}); updateControls(); commit(`preset:${name}`); }));
  const presetMenu = el("details"); presetMenu.append(el("summary", "Detail presets"), presets);
  panel.append(presetMenu, button("Reset to calibration", () => { Object.assign(desired, state.defaults); updateControls(); commit("reset"); }), status, retrySettings);
  layout.append(wrapper, panel); content.append(layout); updateControls();
  const outboxKey = "exposures";
  const interrupted = saved("open-exposure", null);
  if (interrupted) {
    const queue = saved(outboxKey, []);
    // Closing an exposure persists the immutable outbox entry before clearing
    // this checkpoint. A reload between those writes must keep that entry.
    if (!queue.some(entry => entry.id === interrupted.id)) {
      queue.push({...interrupted, incomplete: true}); save(outboxKey, queue);
    }
    save("open-exposure", null);
  }
  function finishExposure(position) {
    if (!exposure) return;
    // A seek has already moved currentTime. Use the last observed pre-seek
    // position, never historical played ranges or the new seek destination.
    if (position === undefined) {
      const current = video.currentTime;
      position = !video.seeking && Number.isFinite(current)
        ? Math.min(videoInfo.duration_ms ?? Infinity, current * 1000) : lastPosition;
    }
    exposure.end_ms = Math.max(exposure.start_ms, position);
    if (exposure.end_ms > exposure.start_ms) {
      const queue = saved(outboxKey, []); queue.push(exposure); save(outboxKey, queue);
    }
    exposure = null;
    save("open-exposure", null);
  }
  async function flush() {
    const queue = saved(outboxKey, []).filter(e => e.video_id === videoInfo.id);
    if (!queue.length) return;
    const entries = queue.slice(0, 100);
    await send("exposures", {video_id: videoInfo.id, entries});
    const ids = new Set(entries.map(e => e.id)); save(outboxKey, saved(outboxKey, []).filter(e => !ids.has(e.id)));
  }
  function beginExposure(position) {
    if (exposure || !applied || activeIndex < 0 || video.paused || video.seeking || document.hidden) return;
    const cue = videoInfo.cues[activeIndex];
    exposure = {id: crypto.randomUUID(), video_id: videoInfo.id, cue: activeIndex,
      display_interval_version: 2, cue_start_ms: cue.start_ms, cue_end_ms: cue.end_ms,
      start_ms: position, end_ms: position, text: applied.text, fallback: applied.fallback,
      job_id: applied.job_id || null, settings_revision: applied.revision ?? null,
      axes: applied.fallback ? null : applied.axes,
      ...(applied.fallback ? {fallback_reason: applied.fallback_reason} : {})};
  }
  function paint() {
    if (!initialCaptionReady || video.seeking) return;
    const ms = Math.min(videoInfo.duration_ms, video.currentTime * 1000);
    lastPosition = ms;
    const index = videoInfo.cues.findIndex(cue => cue.start_ms <= ms && ms < cue.end_ms);
    if (index !== activeIndex) {
      // The old text remains visible until this replacement, even if the
      // nominal cue boundary passed between two timeupdate callbacks.
      finishExposure(ms); activeIndex = index; applied = null;
      if (index >= 0) {
        const cue = videoInfo.cues[index];
        const job = jobs.find(j => j.cue === index && j.revision === state.settings_revision);
        applied = job?.result && !job.result.fallback
          ? {...job.result, job_id: job.id, revision: job.revision, axes: {...state.axes}}
          : {text: cue.fallback, fallback: true, job_id: job?.id || null, revision: job?.revision ?? null,
             fallback_reason: job?.result?.fallback ? "job_failed" : initialFallbackReason || "not_ready_at_boundary"};
        initialFallbackReason = null;
        caption.textContent = applied.text;
        captionLevels.textContent = applied.fallback ? t("Prepared caption · selected levels not applied.")
          : t("Current caption: acoustic {texture}%, source/scene {context}%.", {
            texture: Math.round(applied.axes.texture * 100), context: Math.round(applied.axes.context * 100),
          });
        status.textContent = applied.fallback ? t("Showing the available caption for this moment.") : t("Applied: acoustic {texture}%, source/scene {context}%.", {texture: Math.round(applied.axes.texture * 100), context: Math.round(applied.axes.context * 100)});
      } else { caption.textContent = ""; captionLevels.textContent = t("No caption applied yet."); }
    }
    beginExposure(ms);
    if (exposure) { exposure.end_ms = Math.max(exposure.start_ms, ms); save("open-exposure", exposure); }
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
  async function prepareInitialCaption() {
    if (initialCaptionPreparing || initialCaptionReady || !alive) return;
    initialCaptionPreparing = true;
    const deadline = Date.now() + 5000;
    initialFallbackReason = "initial_caption_timeout";
    try {
      while (alive && Date.now() < deadline) {
        const result = await request("/api/study/captions", undefined, Math.max(1, deadline - Date.now()));
        if (!alive) return;
        if (result.epoch === state.epoch && result.revision === state.settings_revision) {
          jobs = result.jobs;
          const ms = video.currentTime * 1000;
          const index = videoInfo.cues.findIndex(cue => cue.start_ms <= ms && ms < cue.end_ms);
          if (jobs.some(job => job.cue === index && job.result)) { initialFallbackReason = null; break; }
        }
        await new Promise(resolve => setTimeout(resolve, 250));
      }
    } catch {
      initialFallbackReason = Date.now() >= deadline ? "initial_caption_timeout" : "caption_fetch_failed";
    } finally {
      initialCaptionPreparing = false;
      if (alive) {
        initialCaptionReady = true; activeIndex = -1; paint(); enablePlayback();
      }
    }
  }
  video.onloadedmetadata = async () => {
    initialized = true;
    if (resumePosition > 0) video.currentTime = resumePosition / 1000;
    else { await tick().catch(() => {}); await prepareInitialCaption(); }
  };
  video.ontimeupdate = paint;
  video.onplay = () => {
    if (!initialCaptionReady) { video.pause(); return; }
    recoveryNotice = ""; notice(); hasPlayed = true;
    // The chosen caption was already visible at the paused position, including
    // the true initial zero before the first delayed play/timeupdate callback.
    beginExposure(lastPosition);
    tick(false, true).catch(() => {}); paint();
  };
  video.onpause = () => {
    if (!video.seeking && Number.isFinite(video.currentTime)) lastPosition = Math.min(videoInfo.duration_ms, video.currentTime * 1000);
    finishExposure(); tick(false, true).then(flush).catch(() => {});
  };
  const beforeSeek = () => { if (!video.seeking) lastPosition = Math.min(videoInfo.duration_ms, video.currentTime * 1000); };
  video.addEventListener("pointerdown", beforeSeek, true);
  video.addEventListener("keydown", beforeSeek, true);
  video.onseeking = () => { seekGeneration++; finishExposure(); activeIndex = -1; jobs = []; caption.textContent = ""; captionLevels.textContent = t("No caption applied yet."); };
  video.onseeked = async () => {
    lastPosition = video.currentTime * 1000;
    await tick(true).catch(() => {});
    if (initialCaptionReady) paint(); else await prepareInitialCaption();
  };
  const visibility = () => {
    if (document.hidden) { finishExposure(); pauseForRecovery(video); }
    tick(false, true).then(flush).catch(() => { if (alive) pauseForRecovery(video); });
  };
  document.addEventListener("visibilitychange", visibility);
  const next = button(interactionOnly() ? (isLastVideo() ? "Continue to the interview" : "Finish video")
    : state.survey_flow ? "Continue to this video's questions"
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
    next.disabled = true; finishExposure();
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
    if (polling || !alive || !initialized || !initialCaptionReady) return; polling = true;
    try {
      const result = await request("/api/study/captions");
      if (alive && result.epoch === state.epoch && result.revision === state.settings_revision) {
        jobs = result.jobs;
        // A bounded initial fallback need not stay locked while the participant
        // has not started playback and a successful caption becomes available.
        if (!hasPlayed && video.paused) { activeIndex = -1; paint(); }
      }
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
    video.removeEventListener("pointerdown", beforeSeek, true);
    video.removeEventListener("keydown", beforeSeek, true);
    finishExposure(); video.onpause = null; video.pause();
    release();
  };
}
function poststudy() {
  const interview = state.stage === "poststudy";
  const details = state.poststudy || {};
  heading(interview ? "Final interview" : "About this study", interview
    ? "Please complete the oral interview with the researcher. You do not need to type your answers here."
    : "Please read the written explanation and discuss it with the researcher.");
  if (interview) {
    for (const prompt of details.interview_prompts || []) {
      const card = el("section", undefined, "study-card");
      card.append(el("h2", prompt.id));
      const wording = el("p"); wording.textContent = prompt.text; card.append(wording);
      content.append(card);
    }
  } else {
    const text = details.debrief?.text;
    if (typeof text !== "string" || !text.trim()) {
      content.append(el("p", "The researcher is preparing the written explanation. Please contact the researcher to finish the study."));
      return;
    }
    const explanation = el("section", undefined, "study-card");
    const body = el("p"); body.textContent = text; body.style.whiteSpace = "pre-wrap";
    explanation.append(body); content.append(explanation);
  }
  const acknowledgements = {};
  const options = interview
    ? [["researcher_confirmed", "I have completed the interview with the researcher."]]
    : [["researcher_confirmed", "The researcher has explained the study to me."],
       ["participant_acknowledged", "I have read the written explanation."]];
  const canContinue = () => options.every(([key]) => acknowledgements[key] === true);
  const next = button(interview ? "Continue to the study explanation" : "Finish participation", () =>
    send(interview ? "interview-complete" : "debrief-complete", acknowledgements, true), true, canContinue);
  next.disabled = true;
  const card = el("fieldset", undefined, "study-card");
  card.append(el("legend", "Completion confirmation"));
  for (const [key, wording] of options) {
    const label = el("label"), input = el("input"); input.type = "checkbox";
    input.onchange = () => { acknowledgements[key] = input.checked; next.disabled = !canContinue(); };
    label.append(input, el("span", wording)); card.append(label);
  }
  content.append(card); actions(next);
}
function render() {
  if (!ownsSessionLock()) return;
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
  if (state.stage === "poststudy" || state.stage === "debrief") return poststudy();
  if (state.stage === "video-survey") return form(state.items, "video-survey",
    t("Video {number} of {total} · Your experience", {number: state.video_index + 1, total: viewingTotal()}),
    "Answer these four questions about the video you just watched. Overall soundscape questions come after all videos.",
    isLastVideo() ? "Save and continue to overall experience" : "Save this video's answers");
  if (state.stage === "final") return form(state.items, "final-survey", "Your viewing experience",
    state.survey_flow
      ? "Thinking about your overall experience across the longer videos in Chapter 2, answer the soundscape questions (PRSS) and reflect on the caption controls."
      : "Thinking about the videos you just watched, tell us how the captions and controls felt.");
  if (state.stage === "ready") {
    heading("Your calibration is saved", interactionOnly()
      ? "Next, watch the longer videos and adjust the captions as you watch. There are no questionnaires in this stage. An oral interview follows all videos."
      : state.survey_flow
      ? "Next, watch the five-minute videos with adjustable captions. Answer four questions after each video, then reflect on your overall experience and soundscapes once at the end."
      : "Next, watch the five-minute videos with captions shaped by your responses. You can adjust the level of detail while watching.");
    actions(button("Start viewing experience", () => send("start-viewing", {}, true), true));
  } else if (state.stage === "break") {
    heading("Take a moment", t("You have finished {count} of {total} videos. Your chosen detail settings will carry into the next video.", {count: state.completed_videos, total: viewingTotal()}));
    actions(button("Continue to the next video", () => send("continue", {}, true), true));
  } else {
    heading("Thank you for taking part", interactionOnly()
      ? "Your responses, viewing records, and interview and explanation confirmations have been saved. You can close this page."
      : "Your calibration, viewing experience and final responses have been saved. You can close this page.");
    content.append(el("p", t("Receipt: {id}", {id: state.session_id}), "receipt"));
  }
}
async function boot(code) {
  try { copy = await request("/api/study/strings"); state = await request("/api/study/session", code ? {code} : {}); await enterSession(); }
  catch (error) {
    content.replaceChildren(); heading("Join the study", error.message);
    const label = el("label", "Access code "), input = el("input"); input.type = "password"; input.autocomplete = "off"; label.append(input); content.append(label);
    actions(button("Continue", () => boot(input.value), true));
  }
}
window.addEventListener("pagehide", () => {
  pageActive = false;
  // Finish the durable outbox entry before another tab can acquire ownership.
  cleanup(); cleanup = () => {};
  releaseSessionLock();
});
window.addEventListener("pageshow", event => {
  pageActive = true;
  if (event.persisted && state) enterSession().catch(error => notice(error.message));
});
window.addEventListener("beforeunload", event => {
  if (hasVolatileExposure()) { event.preventDefault(); event.returnValue = ""; }
});
boot();

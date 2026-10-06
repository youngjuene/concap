/* The workbook questionnaire is a separate, versioned protocol. The server
 * owns its order, item order, response validation, and viewing coverage. */
const API = "/api/questionnaire";
const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
const node = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};
const button = (text, action, primary = false) => {
  const element = node("button", text, primary ? "primary" : "");
  element.type = "button";
  element.addEventListener("click", action);
  return element;
};
const storage = {
  read(key) { try { return JSON.parse(sessionStorage.getItem(key)) || {}; } catch { return {}; } },
  write(key, value) { try { sessionStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; } },
  remove(key) { try { sessionStorage.removeItem(key); } catch { /* Server state remains authoritative. */ } },
};
let shell, current, cleanup = () => {}, generation = 0;

async function request(path, payload) {
  const response = await fetch(`${API}/${path}`, payload === undefined ? {cache: "no-store"} : {
    method: "POST", headers: {"content-type": "application/json", "x-study-request": "1"}, body: JSON.stringify(payload),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(result.error || `요청을 처리하지 못했습니다 (${response.status}).`), {
    status: response.status, detail: result,
  });
  return result;
}

function scope(detail) {
  return `regen.workbook.draft:${JSON.stringify([
    detail.session_id, detail.participant, detail.protocol_version || detail.protocol, detail.instrument_hash,
    detail.practice === true, detail.clip_index, detail.page,
  ])}`;
}

function draftFor(detail) {
  const key = scope(detail);
  let saved = storage.read(key);
  if (!own(saved, "answers")) {
    // Older tabs included the server revision in their draft key. Migrate only
    // this participant/instrument/page and keep the newest complete snapshot.
    const prefix = "regen.workbook.draft:", identity = JSON.parse(key.slice(prefix.length));
    const candidates = [];
    try {
      for (let index = 0; index < sessionStorage.length; index++) {
        const oldKey = sessionStorage.key(index);
        if (!oldKey?.startsWith(prefix)) continue;
        let parts;
        try { parts = JSON.parse(oldKey.slice(prefix.length)); } catch { continue; }
        if (!Array.isArray(parts) || parts.length !== identity.length + 1 ||
            JSON.stringify(parts.slice(0, -1)) !== JSON.stringify(identity) ||
            !Number.isInteger(parts.at(-1))) continue;
        const previous = storage.read(oldKey);
        if (own(previous, "answers")) candidates.push({key: oldKey, revision: parts.at(-1), value: previous});
      }
    } catch { /* A storage policy must not prevent restoration from the server. */ }
    candidates.sort((a, b) => b.revision - a.revision);
    if (candidates.length) {
      saved = candidates[0].value;
      if (storage.write(key, saved)) candidates.forEach(entry => storage.remove(entry.key));
    }
  }
  // A local snapshot includes intentional clearing. Do not fill those blanks
  // with an older server draft; use server answers only for a fresh tab.
  const answers = own(saved, "answers") ? saved.answers : (detail.draft?.answers ?? detail.draft);
  return {...saved, answers: answers && typeof answers === "object" && !Array.isArray(answers) ? answers : {},
    entered_at: saved.entered_at || new Date().toISOString()};
}

async function action(name, data = {}, detail = current) {
  const key = scope(detail);
  const saved = draftFor(detail);
  const body = {page: detail.page, ...data};
  const signature = JSON.stringify({name, body, revision: detail.revision});
  // If the reply is lost, retries keep the same key and original submission.
  const pending = saved.pending?.signature === signature ? saved.pending : {
    signature, payload: {key: crypto.randomUUID(), revision: detail.revision, data: body},
  };
  storage.write(key, {...saved, pending});
  const result = await request(name, pending.payload);
  storage.remove(key);
  return result;
}

function feedback(message, target) {
  const region = target || shell.querySelector(".sheet-feedback");
  if (region) region.textContent = message;
}

async function refresh() { await render(await request("state")); }

async function recover(error, target) {
  if (error.status === 409) {
    try {
      const latest = await request("state"), samePage = scope(latest) === scope(current);
      await render(latest);
      if (samePage) feedback("다른 탭에서 상태가 변경되었습니다. 입력한 응답을 유지했으니 다시 저장해 주세요.");
      return;
    } catch (next) { error = next; }
  }
  feedback(`저장하지 못했습니다. 입력은 이 탭에 유지됩니다. 다시 시도해 주세요.\n${error.message}`, target);
}

function title(detail) {
  const head = node("div", undefined, "head");
  const context = detail.practice ? "연습 · 응답은 분석에서 제외됩니다" : detail.kind === "intro" || detail.kind === "handoff" ? "소리 자막 연구" : Number.isInteger(detail.clip_index)
    ? `1스테이지 · 영상 ${detail.clip_index + 1} / ${detail.clip_count}` : "소리 자막 연구";
  head.append(node("span", context, "eyebrow"));
  const heading = node("h1", detail.title || "소리 자막 연구");
  heading.tabIndex = -1;
  head.append(heading);
  if (detail.instruction) head.append(node("p", detail.instruction, "lede"));
  shell.append(head);
  return heading;
}

function errorRegion() {
  const region = node("p", "", "sheet-feedback");
  region.setAttribute("role", "alert");
  shell.append(region);
  return region;
}

function setup(detail) {
  const panel = node("section", undefined, "sheet-setup");
  panel.append(node("h2", "연구자 확인이 필요합니다"), node("p", "실험에 필요한 설정이 아직 준비되지 않았습니다. 연구자에게 이 화면을 보여 주세요."));
  const list = node("ul");
  for (const missing of detail.setup_missing || []) list.append(node("li", typeof missing === "string" ? missing : missing.message || missing.label || missing.id));
  panel.append(list);
  const retry = button("설정 다시 확인", async () => {
    retry.disabled = true;
    try { await refresh(); } catch (error) { feedback(error.message); retry.disabled = false; }
  });
  panel.append(retry);
  shell.append(panel);
  errorRegion();
}

function itemValid(item, value) {
  if (!item.required && (value === undefined || value === "")) return true;
  if (item.type === "rating") return Number.isInteger(value) && value >= item.min && value <= item.max;
  if (item.type === "choice") return item.options?.some(option => option.value === value);
  if (item.type === "multi") {
    if (!Array.isArray(value) || !value.length || new Set(value).size !== value.length) return false;
    const choices = item.options || [];
    return value.every(entry => choices.some(option => option.value === entry)) &&
      !(value.length > 1 && choices.some(option => option.exclusive && value.includes(option.value)));
  }
  if (item.type === "visual") return Array.isArray(value) && value.length === 1 &&
    typeof value[0].frame_id === "string" && Number.isFinite(value[0].x) && Number.isFinite(value[0].y) &&
    value[0].x >= 0 && value[0].x <= 1 && value[0].y >= 0 && value[0].y <= 1;
  if (item.type === "text") return typeof value === "string" && value.trim().length > 0;
  return false;
}

function optionControl(item, option, value, onChange, rating = false) {
  const label = node("label", undefined, "sheet-option");
  const input = node("input");
  input.type = item.type === "multi" ? "checkbox" : "radio";
  input.name = item.id;
  if (item.type !== "multi") input.required = item.required === true;
  input.value = String(option.value);
  input.checked = Array.isArray(value) ? value.includes(option.value) : value === option.value;
  input.setAttribute("aria-label", rating ? `${option.value}점${option.label ? ` · ${option.label}` : ""}` : option.label);
  input.addEventListener("change", () => onChange(option.value, input.checked));
  label.append(input);
  if (rating) label.append(node("span", String(option.value), "sheet-score"));
  label.append(node("span", option.label || "", "sheet-label"));
  return label;
}

function visualControl(item, field, detail, value, change) {
  const frames = detail.frames || [];
  if (!frames.length) {
    field.append(node("p", "대표 프레임이 준비되지 않았습니다. 연구자에게 문의해 주세요.", "sheet-hint"));
    return;
  }
  const frameID = frame => String(frame.id ?? frame.index);
  let selected = itemValid(item, value) && frames.some(frame => frameID(frame) === value[0].frame_id) ? value[0] : null;
  let active = Math.max(0, frames.findIndex(frame => frameID(frame) === selected?.frame_id));
  let cursor = selected ? {x: selected.x, y: selected.y} : {x: .5, y: .5};
  let keyboard = false;
  const surface = node("div", undefined, "sheet-frame");
  surface.tabIndex = 0;
  surface.setAttribute("role", "application");
  surface.setAttribute("aria-label", "영상의 대표 프레임. 클릭하여 한 곳을 선택하거나 방향키로 위치를 옮긴 뒤 Enter 키로 선택하세요.");
  const picture = node("img");
  picture.draggable = false;
  const mark = node("span", undefined, "sheet-mark");
  mark.setAttribute("aria-hidden", "true");
  surface.append(picture, mark);
  const status = node("p", "", "sheet-hint");
  status.setAttribute("role", "status");
  const picker = node("div", undefined, "sheet-frames");
  picker.setAttribute("role", "group");
  picker.setAttribute("aria-label", "대표 프레임 선택");
  const controls = frames.map((frame, index) => {
    const control = button(`${index + 1}번째 화면 · ${(frame.at_ms / 1000).toFixed(1)}초`, () => {
      active = index; keyboard = false; redraw();
    });
    picker.append(control);
    return control;
  });
  const reset = button("선택 지우기", () => { selected = null; keyboard = false; change(undefined); redraw(); });
  reset.className = "sheet-reset";
  function redraw() {
    const frame = frames[active];
    if (picture.getAttribute("src") !== frame.url) picture.src = frame.url;
    picture.alt = `영상의 ${active + 1}번째 대표 프레임`;
    const here = selected?.frame_id === frameID(frame) ? selected : null;
    const dot = keyboard ? cursor : here;
    mark.hidden = !dot;
    if (dot) { mark.style.left = `${dot.x * 100}%`; mark.style.top = `${dot.y * 100}%`; }
    mark.classList.toggle("sheet-cursor", keyboard);
    controls.forEach((control, index) => control.setAttribute("aria-pressed", String(index === active)));
    status.textContent = selected ? `${frames.findIndex(frame => frameID(frame) === selected.frame_id) + 1}번째 화면에 한 곳을 선택했습니다. 다른 곳을 선택하면 기존 선택이 바뀝니다.` : "한 곳을 선택해 주세요. 방향키로 이동한 뒤 Enter 키로 선택할 수도 있습니다.";
    reset.disabled = !selected;
  }
  function choose(x, y) {
    if (!picture.complete || !picture.naturalWidth) return;
    selected = {frame_id: frameID(frames[active]), x, y};
    cursor = {x, y}; keyboard = false; change([selected]); redraw();
  }
  surface.addEventListener("click", event => {
    const rect = picture.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    choose(Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)));
  });
  surface.addEventListener("keydown", event => {
    const deltas = {ArrowLeft: [-.02, 0], ArrowRight: [.02, 0], ArrowUp: [0, -.02], ArrowDown: [0, .02]};
    if (deltas[event.key]) {
      event.preventDefault(); keyboard = true;
      const [dx, dy] = deltas[event.key];
      cursor = {x: Math.max(0, Math.min(1, cursor.x + dx)), y: Math.max(0, Math.min(1, cursor.y + dy))};
      redraw();
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault(); choose(cursor.x, cursor.y);
    }
  });
  picture.addEventListener("error", () => { status.textContent = "대표 프레임을 불러오지 못했습니다. 네트워크를 확인하고 화면을 새로고침해 주세요."; });
  field.append(surface, picker, status, reset);
  redraw();
}

function renderForm(detail, actionName = "submit") {
  const draft = draftFor(detail), answers = {};
  const items = detail.items || [];
  for (const item of items) {
    const value = draft.answers[item.id];
    const frameExists = item.type !== "visual" || detail.frames?.some(frame => String(frame.id ?? frame.index) === value?.[0]?.frame_id);
    if (own(draft.answers, item.id) && frameExists && itemValid(item, value)) answers[item.id] = value;
  }
  const form = node("form");
  form.noValidate = true;
  const fields = new Map();
  const save = () => storage.write(scope(detail), {...draft, answers});
  const change = (item, value) => {
    if (value === undefined) delete answers[item.id]; else answers[item.id] = value;
    fields.get(item.id)?.classList.remove("sheet-invalid");
    save(); update();
  };
  for (const [index, item] of items.entries()) {
    const field = node("fieldset", undefined, "sheet-item");
    field.dataset.item = item.id;
    fields.set(item.id, field);
    field.append(node("legend", `${index + 1}. ${item.text}`));
    if (item.type === "visual") visualControl(item, field, detail, answers[item.id], value => change(item, value));
    else if (["rating", "choice", "multi"].includes(item.type)) {
      const group = node("div", undefined, item.type === "rating" ? "sheet-scale" : "sheet-choices");
      const options = item.type === "rating" ? Array.from({length: item.max - item.min + 1}, (_, at) => ({value: item.min + at, label: item.labels?.[at] || ""})) : item.options || [];
      for (const option of options) group.append(optionControl(item, option, answers[item.id], (value, checked) => {
        if (item.type !== "multi") { change(item, value); return; }
        let selected = Array.isArray(answers[item.id]) ? [...answers[item.id]] : [];
        if (checked) selected = option.exclusive ? [value] : [...selected.filter(entry => !options.find(candidate => candidate.value === entry)?.exclusive), value];
        else selected = selected.filter(entry => entry !== value);
        change(item, selected);
        group.querySelectorAll("input").forEach((input, at) => { input.checked = selected.includes(options[at].value); });
      }, item.type === "rating"));
      field.append(group);
      if (item.type === "multi") field.append(node("p", "해당하는 것을 모두 선택해 주세요. ‘기억나지 않음’은 다른 선택지와 함께 선택할 수 없습니다.", "sheet-hint"));
      if (!options.length) field.append(node("p", "선택지가 준비되지 않았습니다. 연구자에게 문의해 주세요.", "sheet-hint"));
    } else if (item.type === "text") {
      const input = node("textarea", undefined, "sheet-text");
      input.name = item.id; input.value = answers[item.id] || ""; input.setAttribute("aria-label", item.text);
      input.addEventListener("input", () => change(item, input.value)); field.append(input);
    } else field.append(node("p", "이 응답 형식은 연구자 확인이 필요합니다.", "sheet-hint"));
    form.append(field);
  }
  const bar = node("div", undefined, "sheet-submitbar");
  const submit = node("button", "응답 저장 후 계속", "primary");
  submit.type = "submit";
  const count = node("p", "", "sheet-hint");
  count.setAttribute("role", "status");
  const remaining = () => items.filter(item => !itemValid(item, answers[item.id]));
  function update() { count.textContent = `${items.length - remaining().length} / ${items.length} 응답`; }
  bar.append(submit, count); form.append(bar); shell.append(form);
  const region = errorRegion();
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (submit.disabled) return;
    const missing = remaining();
    if (missing.length) {
      const field = fields.get(missing[0].id);
      field.classList.add("sheet-invalid");
      feedback(`응답하지 않은 문항이 ${missing.length}개 있습니다. 첫 번째 미응답 문항을 확인해 주세요.`, region);
      field.scrollIntoView({block: "center"});
      (field.querySelector("input,textarea,[tabindex='0']") || field).focus();
      return;
    }
    submit.disabled = true;
    fields.forEach(field => { field.disabled = true; });
    feedback("응답을 저장하고 있습니다…", region);
    try {
      await action(actionName, {answers}, detail);
      await refresh();
    } catch (error) {
      await recover(error, region); submit.disabled = false;
      fields.forEach(field => { field.disabled = false; });
    }
  });
  save(); update();
}

function renderIntro(detail) {
  const consentText = detail.consent_text || detail.consent?.text;
  if (!consentText) { setup({...detail, setup_missing: ["연구 동의서 문구가 필요합니다."]}); return; }
  shell.append(node("div", consentText, "sheet-consent"));
  const label = node("label", undefined, "sheet-option sheet-consent-check");
  const input = node("input"); input.type = "checkbox";
  label.append(input, node("span", "안내문과 동의서를 읽었으며 연구 참여에 동의합니다."));
  shell.append(label);
  const next = button("연습 시작", async () => {
    if (!input.checked) { feedback("참여 동의 여부를 확인해 주세요."); input.focus(); return; }
    next.disabled = true;
    try { await action("intro-complete", {consented: true}, detail); await refresh(); }
    catch (error) { await recover(error); next.disabled = false; }
  }, true);
  const actions = node("div", undefined, "actions"); actions.append(next); shell.append(actions); errorRegion();
}

function renderTransition(detail, actionName, label) {
  const next = button(label, async () => {
    next.disabled = true;
    feedback(actionName === "generate" ? "다음 영상을 준비하고 있습니다…" : "저장하고 있습니다…");
    try {
      const result = await action(actionName, {}, detail);
      if (result.next_url && actionName === "handoff") {
        window.location.assign(result.next_url);
      } else await refresh();
    } catch (error) { await recover(error); next.disabled = false; }
  }, true);
  const actions = node("div", undefined, "actions"); actions.append(next); shell.append(actions); errorRegion();
}

async function renderVideo(detail, renderID) {
  const media = detail.media;
  if (!media?.url) { setup({...detail, setup_missing: ["시청할 영상이 준비되지 않았습니다."]}); return; }
  shell.append(node("p", "헤드폰을 착용하고 영상과 자막을 끝까지 시청해 주세요.", "lede"));
  const start = button("영상 준비 중…", () => {}, true); start.disabled = true;
  const actions = node("div", undefined, "actions"); actions.append(start); shell.append(actions);
  const region = errorRegion();
  let blobURL, stage, timer, alive = true, ending = false, chain = Promise.resolve();
  let checkpoint = media.playback || {}, sequence = checkpoint.sequence ?? -1, pendingSample = null;
  const listeners = [];
  const listen = (target, name, callback) => { target.addEventListener(name, callback); listeners.push(() => target.removeEventListener(name, callback)); };
  cleanup = () => {
    alive = false; clearInterval(timer); listeners.forEach(remove => remove());
    const video = stage?.querySelector("video"); if (video) video.pause();
    stage?.remove(); if (blobURL) URL.revokeObjectURL(blobURL);
  };
  try {
    const response = await fetch(media.url);
    if (!response.ok) throw new Error(`영상을 불러오지 못했습니다 (${response.status}).`);
    const blob = await response.blob();
    if (!alive || generation !== renderID) return;
    blobURL = URL.createObjectURL(blob); start.disabled = false; start.textContent = "영상 시청 시작";
  } catch (error) {
    if (!alive) return;
    feedback(error.message, region);
    start.textContent = "영상 다시 불러오기"; start.disabled = false;
    start.onclick = () => refresh().catch(error => feedback(error.message, region));
    return;
  }
  start.onclick = async () => {
    start.disabled = true;
    stage = node("section", undefined, "sheet-viewing");
    const video = node("video"); video.playsInline = true; video.preload = "auto";
    video.disablePictureInPicture = true; video.src = blobURL;
    const caption = node("p", "", "sheet-caption");
    const overlay = node("div", undefined, "sheet-paused"); overlay.hidden = true;
    const pauseTitle = node("h2", "시청이 일시 중지되었습니다");
    const pauseText = node("p", "화면으로 돌아온 뒤 이어서 시청해 주세요.");
    const resume = button("이어서 시청", async () => {
      resume.disabled = true;
      try {
        await stage.requestFullscreen?.().catch(() => {});
        await playFromCheckpoint();
      } catch (error) { pauseText.textContent = error.message; }
      resume.disabled = false;
    }, true);
    overlay.append(pauseTitle, pauseText, resume); stage.append(video, caption, overlay); document.body.append(stage);
    shell.hidden = true;
    const cues = media.captions || [];
    function updateCaption() {
      const ms = video.currentTime * 1000;
      caption.textContent = cues.filter(cue => ms >= cue.start_ms && ms < cue.end_ms).map(cue => cue.text).join("\n");
    }
    const track = video.addTextTrack("captions", "소리 자막", "ko"); track.mode = "hidden";
    for (const cue of cues) track.addCue(new VTTCue(cue.start_ms / 1000, cue.end_ms / 1000, cue.text));
    listen(track, "cuechange", updateCaption); listen(video, "timeupdate", updateCaption);
    function tick(playing = !video.paused) {
      if (!alive) return Promise.resolve();
      const task = chain.then(async () => {
        if (!alive) return;
        const send = async payload => {
          pendingSample = payload;
          let result;
          try { result = await request("playback", payload); }
          catch (error) {
            if (error.status !== 409) throw error;
            // A rejected revision is not an ambiguous lost reply. Replaying
            // that key can never succeed: fetch the accepted position first.
            pendingSample = null;
            const latest = await request("state");
            if (!alive) return false;
            if (scope(latest) !== scope(detail)) {
              if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
              await render(latest);
              return false;
            }
            detail.revision = latest.revision;
            checkpoint = latest.playback || latest.media?.playback || checkpoint;
            sequence = Math.max(sequence, checkpoint.sequence ?? -1);
            video.pause();
            video.currentTime = Math.max(0, checkpoint.position_ms || 0) / 1000;
            error.message = "다른 탭의 진행 상태를 불러왔습니다. 이어서 시청해 주세요.";
            throw error;
          }
          if (!alive) return false;
          if (result.page && scope(result) !== scope(detail)) {
            pendingSample = null;
            if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
            await render(result);
            return false;
          }
          checkpoint = result.playback || result.media?.playback || checkpoint;
          if (result.revision !== undefined) detail.revision = result.revision;
          pendingSample = null;
          return true;
        };
        // A checkpoint can have committed even when its response was lost.
        // Replay that same key before sending the next sequence number.
        if (pendingSample && !await send(pendingSample)) return;
        const sample = {page: detail.page, position_ms: Math.min(media.duration_ms, video.currentTime * 1000), playing, hidden: document.hidden};
        // Hidden playback is never credited. Rewind to the acknowledged
        // position so the remaining interval is actually watched on return.
        if (sample.hidden) sample.position_ms = checkpoint.position_ms || 0;
        if (!await send({key: crypto.randomUUID(), revision: detail.revision, data: {...sample, sequence: ++sequence}})) return;
        if (sample.hidden) video.currentTime = checkpoint.position_ms / 1000;
      });
      chain = task.catch(() => {});
      return task;
    }
    async function playFromCheckpoint() {
      clearInterval(timer);
      await tick(true);
      if (!alive) return;
      await video.play(); overlay.hidden = true;
      // Initial playback and an interrupted start need the same heartbeat.
      timer = setInterval(() => {
        if (!video.paused && alive && !ending) tick().catch(error => pause(`시청 기록을 저장하지 못했습니다. ${error.message}`));
      }, 1000);
    }
    function pause(message) {
      if (ending || !alive) return;
      video.pause(); pauseText.textContent = message;
      overlay.hidden = false; resume.disabled = false; resume.focus();
      tick(false).catch(error => { pauseText.textContent = `시청 기록을 저장하지 못했습니다. ${error.message}`; });
    }
    listen(document, "visibilitychange", () => { if (document.hidden && !video.paused) pause("화면으로 돌아온 뒤 이어서 시청해 주세요."); });
    listen(document, "fullscreenchange", () => { if (!document.fullscreenElement && !video.paused) pause("전체 화면이 종료되었습니다. 이어서 시청해 주세요."); });
    listen(video, "error", () => pause("영상을 재생하지 못했습니다. 연구자에게 문의해 주세요."));
    listen(video, "ended", async () => {
      ending = true; clearInterval(timer);
      try {
        await tick(false);
        if (!alive) return;
        await action("view-ended", {}, detail);
        if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
        await refresh();
      } catch (error) {
        overlay.hidden = false; pauseTitle.textContent = "시청 기록 저장이 필요합니다";
        pauseText.textContent = `시청은 끝났습니다. 아래 버튼으로 기록 저장을 다시 시도해 주세요. ${error.message}`;
        resume.textContent = "시청 기록 저장 다시 시도";
        resume.onclick = null;
        const retry = resume.cloneNode(true); resume.replaceWith(retry);
        retry.addEventListener("click", async () => {
          retry.disabled = true;
          try {
            const latest = await request("state");
            if (latest.page !== detail.page || latest.clip_index !== detail.clip_index) {
              if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
              await render(latest);
              return;
            }
            detail.revision = latest.revision;
            checkpoint = latest.media?.playback || checkpoint;
            sequence = Math.max(sequence, checkpoint.sequence ?? -1);
            pendingSample = null;
            await tick(false);
            if (!alive) return;
            await action("view-ended", {}, detail);
            if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
            await refresh();
          } catch (next) { pauseText.textContent = next.message; retry.disabled = false; }
        });
        retry.focus();
      }
    });
    try {
      await stage.requestFullscreen?.().catch(() => {});
      if (video.readyState < 1) await new Promise((resolve, reject) => {
        video.addEventListener("loadedmetadata", resolve, {once: true});
        video.addEventListener("error", () => reject(new Error("영상 정보를 불러오지 못했습니다.")), {once: true});
      });
      if (checkpoint.position_ms > 0) {
        video.currentTime = Math.min(checkpoint.position_ms / 1000, video.duration);
        if (video.seeking) await new Promise(resolve => video.addEventListener("seeked", resolve, {once: true}));
      }
      await playFromCheckpoint();
    } catch (error) { pause(error.message); }
  };
}

async function render(detail) {
  cleanup(); cleanup = () => {};
  const renderID = ++generation;
  current = detail;
  shell.hidden = false; shell.replaceChildren();
  const heading = title(detail);
  if (detail.setup_missing?.length) setup(detail);
  else if (detail.kind === "intro") renderIntro(detail);
  else if (detail.kind === "video" || detail.kind === "view") await renderVideo(detail, renderID);
  else if (detail.kind === "practice") renderForm(detail, "practice-complete");
  else if (["survey", "visual", "audio", "single", "multi"].includes(detail.kind)) renderForm(detail);
  else if (["generation", "generating", "regenerating"].includes(detail.kind)) renderTransition(detail, "generate", "다음 영상 준비");
  else if (detail.kind === "handoff") renderTransition(detail, "handoff", "2스테이지 시작");
  else if (detail.kind === "done") shell.append(node("p", "응답이 저장되었습니다. 참여해 주셔서 감사합니다.", "lede"));
  else setup({...detail, setup_missing: ["실험 단계 정보를 확인할 수 없습니다. 연구자에게 문의해 주세요."]});
  if (generation === renderID) { window.scrollTo(0, 0); heading.focus(); }
}

export async function boot() {
  document.documentElement.lang = "ko";
  document.title = "소리 자막 연구";
  if (!document.querySelector('link[href="/sheet_questionnaire.css"]')) {
    const stylesheet = node("link"); stylesheet.rel = "stylesheet"; stylesheet.href = "/sheet_questionnaire.css"; document.head.append(stylesheet);
  }
  document.getElementById("screen-viewing")?.setAttribute("hidden", "");
  shell = document.getElementById("shell"); shell.classList.add("sheet-questionnaire"); shell.hidden = false;
  shell.replaceChildren(node("p", "실험 정보를 불러오고 있습니다…", "lede"));
  try {
    let detail;
    try { detail = await request("state"); }
    catch (error) {
      if (![401, 404].includes(error.status)) throw error;
      const code = new URL(window.location.href).searchParams.get("code");
      await request("session", code ? {code} : {}); detail = await request("state");
    }
    await render(detail);
  } catch (error) {
    shell.replaceChildren(node("h1", "실험을 불러오지 못했습니다"), node("p", error.message, "lede"));
    shell.append(button("다시 시도", () => boot(), true));
  }
}

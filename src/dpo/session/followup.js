/* The follow-up page of the caption session (spec 8, Table 9).

   On a later day the participant opens this in the browser of their own phone,
   so nothing here assumes the kiosk: no fullscreen, no event log, and — spec 8
   first line — nothing that shows the skeleton. Three tasks run in the order
   the session document lists them (contract §2 `followup`): recognition once
   per clip, sound only once per excerpt, the check once per shaped clip, then
   a done summary. The page owns three things the server does not:

   - the step sequence and the resumable progress. Answers and the step index
     are saved to localStorage under a key that names the participant, so a
     reload comes back to the START of the interrupted step (contract §9
     "Resume"), never mid-excerpt or mid-loop;
   - the media gestures. A phone will not start sound without a user gesture,
     so Begin — the one press that always precedes any media — primes the
     <audio> and <video> elements, and the check falls back to muted playback
     rather than to a frozen frame when a browser still refuses;
   - the caption on the frame. The check shows the placed caption exactly as
     the kiosk does (spec 8: "exactly as it looks in the kiosk"): the same
     .caption-object.placed from identity.css inside a 16:9 frame, switching
     with the shot's timing (spec 4.4 "appears and clears with the shot's
     timing") from a requestAnimationFrame loop.

   Responses are POSTed once, at the end, in the contract §7 shape. Every
   visible string is defined once in STRINGS (contract §9 "Copy"). */
"use strict";

/* Spec Table 6, verbatim. Entries below the hairline comment are strings
   Table 6 does not carry (the intro card body, the identifier field label,
   the notices, and the done summary); they are written in the interface's
   voice — what happened and the way forward, no apology (spec 5) — and sit
   here so the Korean copy replaces them in one place. */
const STRINGS = {
  title: "Caption session",
  begin: "Begin",
  next: "Next",
  play: "Play",
  showOther: "Show the other caption",
  choose: "I'd rather have this one",
  preparing: "Preparing comparison…",
  recognitionHeading: "Which sounds were in this clip?",
  soundHeading: "Which clip is this sound from?",
  checkHeading: "Which caption would you rather have?",
  captionA: "Caption A",
  captionB: "Caption B",
  // Table 6 "Clip 2 of 6": the clip count pattern, used here as the still's
  // alt text so a clip is named by its position and never by its id.
  clipCount: "Clip {n} of {m}",
  // ---- not in Table 6 ----------------------------------------------------
  introBody:
    "Three short tasks on the clips you watched. Mark the sounds you remember " +
    "in each clip, match each sound excerpt to its clip, then say which caption " +
    "you would rather have. Enter your participant identifier to begin.",
  participantLabel: "Participant identifier",
  noSession: "No kiosk session was found for this identifier. Check the identifier and begin again.",
  loadFailed: "The follow-up could not be loaded. Check the connection and try again.",
  mediaFailed: "This step's media could not be loaded. Check the connection and try again.",
  saveFailed: "Your responses were not saved. Check the connection and try again.",
  doneHeading: "That's everything.",
  doneRecognition: "Clips checked for their sounds: {n}",
  doneSoundOnly: "Excerpts matched to a clip: {n}",
  doneCheck: "Captions compared: {n}",
  doneSaved: "Your responses are saved.",
};

// The kiosk stage the check frame is laid out at (kiosk.css --stage-w). On a
// column at least that wide the frame is the kiosk's picture unscaled, so the
// caption is the identity's 22 on 30 (spec Table 7, spec 8 "exactly as it
// looks in the kiosk"); on a narrower phone the same picture is scaled down,
// never reflowed (docs/v1-session/runbook.md §4).
const KIOSK_STAGE_WIDTH = 640;
// Contract §6: the participant id the server accepts.
const PARTICIPANT_RE = /^[A-Za-z0-9_-]{1,64}$/;
// Contract §7: the responses document.
const RESPONSES_SCHEMA = "dpo.caption-session-followup/v1";
// The browser-owned progress record; the key names the participant so two
// people on one phone never share a record.
const STATE_SCHEMA = "dpo.caption-session-followup-state/v1";
const STORAGE_PREFIX = "dpo.caption-session-followup/v1:";
const LAST_PARTICIPANT_KEY = "dpo.caption-session-followup/v1#last";
// A check clip that fails to load is asked for once more after this pause.
const MEDIA_RETRY_MS = 3000;

function byId(id) {
  return document.getElementById(id);
}

const dom = {
  masthead: byId("masthead"),
  notice: byId("notice"),
  intro: byId("intro"),
  introHeading: byId("intro-heading"),
  introBody: byId("intro-body"),
  participantLabel: byId("participant-label"),
  participant: byId("participant"),
  begin: byId("begin"),
  recognition: byId("recognition"),
  recognitionStill: byId("recognition-still"),
  recognitionHeading: byId("recognition-heading"),
  recognitionList: byId("recognition-list"),
  recognitionNext: byId("recognition-next"),
  sound: byId("sound-only"),
  soundHeading: byId("sound-heading"),
  excerpt: byId("excerpt"),
  play: byId("play"),
  choices: byId("sound-choices"),
  soundNext: byId("sound-next"),
  check: byId("check"),
  checkHeading: byId("check-heading"),
  checkEyebrow: byId("check-eyebrow"),
  stage: byId("stage"),
  clip: byId("clip"),
  placed: byId("placed"),
  placedProse: byId("placed-prose"),
  toggle: byId("toggle"),
  choose: byId("choose"),
  done: byId("done"),
  doneHeading: byId("done-heading"),
  doneSummary: byId("done-summary"),
};

let participant = "";
let payload = null; // GET /api/followup, contract §7
let steps = []; // [{kind, item}...] ending with {kind: "done"}
let state = null; // the persisted progress record
let current = null; // the live in-step state; never persisted (resume restarts the step)

// ---- helpers ----------------------------------------------------------------

function format(template, values) {
  return template.replace(/\{(\w+)\}/g, (_, key) => String(values[key]));
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function stillUrl(clipId) {
  return "/media/still/" + encodeURIComponent(clipId);
}

function excerptUrl(excerptId) {
  return "/media/excerpt/" + encodeURIComponent(excerptId);
}

function clipUrl(clipId) {
  return "/media/" + encodeURIComponent(clipId);
}

/* The participant never sees a clip id: a still is named by the clip's
   position in the session order (Table 6 clip count pattern). */
function clipLabel(clipId) {
  const index = payload.clips.findIndex((clip) => clip.clip_id === clipId);
  return format(STRINGS.clipCount, { n: Math.max(index, 0) + 1, m: payload.clips.length });
}

function notice(text) {
  dom.notice.textContent = text;
  dom.notice.hidden = false;
}

function clearNotice() {
  dom.notice.textContent = "";
  dom.notice.hidden = true;
}

function settle(promise, onFulfilled, onRejected) {
  // Older WebKit returns undefined from play(); treat that as fulfilled.
  if (promise && typeof promise.then === "function") {
    promise.then(onFulfilled, onRejected);
  } else {
    onFulfilled();
  }
}

/* Called inside a click handler: a play() issued during a user gesture lifts
   the phone's gesture requirement for that element, so the excerpt and the
   looping clip can start later from code. The rejection (no source yet) is
   expected and ignored. */
function prime(media) {
  settle(media.play(), () => media.pause(), () => {});
}

// ---- persistence ------------------------------------------------------------

function storageKey(id) {
  return STORAGE_PREFIX + id;
}

function freshState(id) {
  return {
    schema: STATE_SCHEMA,
    participant: id,
    step: 0,
    recognition: {}, // clip_id -> checked sound names
    sound_only: {}, // excerpt_id -> {chosen_clip_id, played}
    check: {}, // clip_id -> {chosen, toggles}
    submitted: false,
  };
}

function loadState(id) {
  try {
    const raw = localStorage.getItem(storageKey(id));
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && parsed.schema === STATE_SCHEMA && parsed.participant === id) return parsed;
    }
  } catch (error) {
    // Storage can be unavailable (private mode); the session still runs.
  }
  return null;
}

function saveState() {
  try {
    localStorage.setItem(storageKey(participant), JSON.stringify(state));
  } catch (error) {
    // As above.
  }
}

function rememberParticipant(id) {
  try {
    localStorage.setItem(LAST_PARTICIPANT_KEY, id);
  } catch (error) {
    // As above.
  }
}

function recallParticipant() {
  const fromUrl = new URLSearchParams(window.location.search).get("participant");
  if (fromUrl) return fromUrl;
  try {
    return localStorage.getItem(LAST_PARTICIPANT_KEY) || "";
  } catch (error) {
    return "";
  }
}

// ---- steps ------------------------------------------------------------------

function buildSteps(document_) {
  const built = [];
  for (const item of document_.recognition) built.push({ kind: "recognition", item });
  for (const item of document_.sound_only) built.push({ kind: "sound", item });
  for (const item of document_.check) built.push({ kind: "check", item });
  built.push({ kind: "done" });
  return built;
}

/* Where to resume: the first step without an answer. The stored step index
   is never trusted against a step list rebuilt from the live document — a
   document whose entries were reordered or changed between sessions would
   land it on a different step, skipping one whose answer then stays absent. */
function answered(step) {
  if (step.kind === "recognition") return step.item.clip_id in state.recognition;
  if (step.kind === "sound") return step.item.excerpt_id in state.sound_only;
  if (step.kind === "check") return step.item.clip_id in state.check;
  return state.submitted;
}

function resumeIndex() {
  const index = steps.findIndex((step) => !answered(step));
  return index < 0 ? steps.length - 1 : index;
}

function hideAll() {
  teardownCheck();
  dom.excerpt.pause();
  for (const screen of [dom.intro, dom.recognition, dom.sound, dom.check, dom.done]) screen.hidden = true;
}

function showStep() {
  hideAll();
  clearNotice();
  window.scrollTo(0, 0);
  if (state.submitted) {
    showDone();
    return;
  }
  const step = steps[state.step];
  if (step.kind === "recognition") showRecognition(step);
  else if (step.kind === "sound") showSound(step);
  else if (step.kind === "check") showCheck(step);
  else if (resumeIndex() < state.step) {
    // "done" reached with an answer missing: the owed POST would be refused,
    // so the first unanswered step is shown instead.
    state.step = resumeIndex();
    saveState();
    showStep();
  } else submit(null); // reached "done" unsubmitted: the POST is what is owed
}

/* Answers are saved before the step advances; the POST happens before the
   done screen is entered, so a reload after a failed save comes back to the
   last step, whose primary re-records and re-POSTs. */
function complete(button) {
  saveState();
  const next = steps[state.step + 1];
  if (next && next.kind === "done") {
    submit(button);
    return;
  }
  state.step += 1;
  saveState();
  showStep();
}

// ---- intro (Table 9: instruction card, participant identifier field, Begin) --

function showIntro() {
  hideAll();
  dom.intro.hidden = false;
  refreshBegin();
}

function refreshBegin() {
  dom.begin.disabled = !PARTICIPANT_RE.test(dom.participant.value.trim());
}

async function begin() {
  const id = dom.participant.value.trim();
  if (!PARTICIPANT_RE.test(id)) return;
  participant = id;
  clearNotice();
  dom.begin.disabled = true;
  prime(dom.excerpt);
  prime(dom.clip);
  let response;
  try {
    response = await fetch("/api/followup?participant=" + encodeURIComponent(participant));
    if (response.status === 409) {
      // Contract §7: no snapshot with kept captions for a check clip.
      notice(STRINGS.noSession);
      refreshBegin();
      return;
    }
    if (!response.ok) throw new Error(String(response.status));
    payload = await response.json();
  } catch (error) {
    notice(STRINGS.loadFailed);
    refreshBegin();
    return;
  }
  rememberParticipant(participant);
  steps = buildSteps(payload);
  state = loadState(participant) || freshState(participant);
  state.step = resumeIndex();
  showStep();
}

// ---- recognition (Table 9: still frame, list of sound names with boxes) -----

function showRecognition(step) {
  const { clip_id: clipId, sounds } = step.item;
  current = { checked: new Set() };
  dom.recognitionStill.src = stillUrl(clipId);
  dom.recognitionStill.alt = clipLabel(clipId);
  dom.recognitionStill.onerror = () => notice(STRINGS.mediaFailed);
  dom.recognitionList.replaceChildren(
    ...sounds.map((sound, index) => {
      // The row is the touch target: a 44 box beside the sound as a
      // bracketed token (spec 8: "in the bracketed register").
      const row = element("button", "check");
      row.type = "button";
      row.setAttribute("role", "checkbox");
      row.setAttribute("aria-checked", "false");
      const box = element("span", "box");
      box.setAttribute("aria-checked", "false");
      row.append(box, element("span", "token", "[" + sound + "]"));
      row.addEventListener("click", () => {
        if (current.checked.has(index)) current.checked.delete(index);
        else current.checked.add(index);
        const checked = current.checked.has(index) ? "true" : "false";
        row.setAttribute("aria-checked", checked);
        // An answered box is filled (spec-identity.md Table 3): identity.css fills
        // `.check .box[aria-checked="true"]`, so the row's state is mirrored
        // onto the box that rule reads.
        box.setAttribute("aria-checked", checked);
        dom.recognition.dataset.state = current.checked.size > 0 ? "some" : "untouched";
      });
      return row;
    }),
  );
  dom.recognition.dataset.state = "untouched";
  dom.recognition.hidden = false;
}

dom.recognitionNext.addEventListener("click", () => {
  // Checking nothing is an answer, so Next is never disabled here; the empty
  // list is recorded like any other.
  const step = steps[state.step];
  state.recognition[step.item.clip_id] = step.item.sounds.filter((_, index) => current.checked.has(index));
  dom.recognition.dataset.state = "complete";
  complete(dom.recognitionNext);
});

// ---- sound only (Table 9: Play, excerpt, row of clip stills) ----------------

function showSound(step) {
  current = { played: 0, chosen: null };
  dom.excerpt.src = excerptUrl(step.item.excerpt_id);
  dom.excerpt.load();
  dom.choices.replaceChildren(
    ...payload.clips.map((clip) => {
      const choice = element("button", "choice");
      choice.type = "button";
      choice.setAttribute("role", "radio");
      choice.setAttribute("aria-checked", "false");
      const still = element("img");
      still.src = stillUrl(clip.clip_id);
      still.alt = clipLabel(clip.clip_id);
      still.draggable = false;
      still.addEventListener("error", () => notice(STRINGS.mediaFailed));
      choice.append(still);
      choice.addEventListener("click", () => {
        current.chosen = clip.clip_id;
        for (const other of dom.choices.children) other.setAttribute("aria-checked", "false");
        choice.setAttribute("aria-checked", "true");
        updateSound();
      });
      return choice;
    }),
  );
  updateSound();
  dom.sound.hidden = false;
}

function updateSound() {
  dom.sound.dataset.state = current.chosen !== null ? "chosen" : current.played > 0 ? "played" : "unplayed";
  // Next waits for the excerpt to have been heard once AND a still chosen.
  dom.soundNext.disabled = !(current.played > 0 && current.chosen !== null);
}

dom.play.addEventListener("click", () => {
  // Play keeps one name (spec 5); pressing it again restarts the excerpt.
  dom.excerpt.pause();
  dom.excerpt.currentTime = 0;
  settle(
    dom.excerpt.play(),
    () => {
      current.played += 1;
      updateSound();
    },
    () => notice(STRINGS.mediaFailed),
  );
});

dom.soundNext.addEventListener("click", () => {
  const step = steps[state.step];
  state.sound_only[step.item.excerpt_id] = { chosen_clip_id: current.chosen, played: current.played };
  complete(dom.soundNext);
});

// ---- check (Table 9: looping clip with placed caption, the toggle, the choice)

/* Fit the kiosk-sized frame to the column (followup.css .stage): unscaled
   whenever the column holds it, scaled down only when it does not. */
function fitStage() {
  const width = dom.stage.clientWidth;
  if (width) dom.stage.style.setProperty("--frame-scale", String(Math.min(1, width / KIOSK_STAGE_WIDTH)));
}

function captionsByShot(entries) {
  const table = {};
  for (const entry of entries) table[entry.shot_id] = entry.caption;
  return table;
}

function showCheck(step) {
  const { clip_id: clipId, captions } = step.item;
  const clip = payload.clips.find((candidate) => candidate.clip_id === clipId);
  current = {
    showing: "A", // which caption set is A is set by configuration (contract §2 followup.check)
    seen: { A: false, B: false },
    toggles: 0,
    prepared: false,
    mutedFallback: false,
    raf: 0,
    lastText: null,
    shots: clip ? clip.shots : [],
    byShot: { A: captionsByShot(captions.A), B: captionsByShot(captions.B) },
  };
  dom.checkEyebrow.textContent = STRINGS.captionA;
  dom.placed.hidden = true;
  dom.placedProse.textContent = "";
  // The busy state lives on the button itself (spec 4.8, spec-identity.md Actions).
  dom.choose.textContent = STRINGS.preparing;
  dom.choose.disabled = true;
  dom.toggle.disabled = true;
  updateCheck();
  dom.clip.src = clipUrl(clipId);
  dom.clip.load();
  dom.check.hidden = false;
  fitStage();
}

function updateCheck() {
  dom.check.dataset.state = current.prepared ? (current.showing === "A" ? "a" : "b") : "preparing";
  const bothSeen = current.seen.A && current.seen.B;
  dom.check.dataset.choice = current.prepared && bothSeen ? "enabled" : "disabled";
  // The choice stays disabled until both captions have been on the frame (spec 8).
  dom.choose.disabled = !(current.prepared && current.seen.A && current.seen.B);
}

function prepareCheck() {
  current.prepared = true;
  dom.choose.textContent = STRINGS.choose;
  dom.toggle.disabled = false;
  // A browser that refuses sound without its own gesture still shows the
  // looping frame; the toggle press restores sound (see unmuteIfNeeded).
  settle(dom.clip.play(), () => {}, () => {
    current.mutedFallback = true;
    dom.clip.muted = true;
    settle(dom.clip.play(), () => {}, () => {});
  });
  current.raf = requestAnimationFrame(loop);
  updateCheck();
}

function unmuteIfNeeded() {
  if (current && current.mutedFallback) {
    current.mutedFallback = false;
    dom.clip.muted = false;
  }
}

/* The placed box shows the current shot's caption from the set on the frame
   and clears when no shot covers the playhead (spec 4.4). The shot is
   resolved as the kiosk's shotAt() resolves it: the last shot holds through
   the media's tail past its end_ms, so the box never blinks at the loop's
   wrap (spec 8: "exactly as it looks in the kiosk"). A set counts as seen
   only when one of its captions has actually been painted. */
function loop() {
  if (!current || !current.prepared) return;
  const at = dom.clip.currentTime * 1000;
  const last = current.shots[current.shots.length - 1];
  const shot =
    current.shots.find((candidate) => at >= candidate.start_ms && at < candidate.end_ms) ||
    (last && at >= last.end_ms ? last : null);
  const text = shot ? current.byShot[current.showing][shot.shot_id] || "" : "";
  if (text !== current.lastText) {
    dom.placedProse.textContent = text;
    dom.placed.hidden = text === "";
    current.lastText = text;
    if (text !== "" && !current.seen[current.showing]) {
      current.seen[current.showing] = true;
      updateCheck();
    }
  }
  current.raf = requestAnimationFrame(loop);
}

function teardownCheck() {
  if (current && current.raf) cancelAnimationFrame(current.raf);
  if (current) current.raf = 0;
  dom.clip.pause();
  if (dom.clip.hasAttribute("src")) {
    dom.clip.removeAttribute("src");
    dom.clip.load();
  }
  dom.clip.muted = false;
}

dom.clip.addEventListener("canplay", () => {
  const step = steps[state ? state.step : -1];
  if (!step || step.kind !== "check" || dom.check.hidden || current.prepared) return;
  prepareCheck();
});

dom.clip.addEventListener("error", () => {
  if (dom.check.hidden) return;
  notice(STRINGS.mediaFailed);
  // Nothing else on the screen can retry while it says "preparing", so the
  // clip is requested once more after a pause; the notice stands until it plays.
  const step = steps[state ? state.step : -1];
  setTimeout(() => {
    if (dom.check.hidden || !current || current.prepared || steps[state.step] !== step) return;
    dom.clip.src = clipUrl(step.item.clip_id);
    dom.clip.load();
  }, MEDIA_RETRY_MS);
});

dom.toggle.addEventListener("click", () => {
  // The single toggle (spec 8): one press, the other caption set on the frame.
  current.showing = current.showing === "A" ? "B" : "A";
  current.toggles += 1;
  dom.checkEyebrow.textContent = current.showing === "A" ? STRINGS.captionA : STRINGS.captionB;
  current.lastText = null; // repaint the box with the other set at once
  unmuteIfNeeded();
  updateCheck();
});

dom.choose.addEventListener("click", () => {
  const step = steps[state.step];
  state.check[step.item.clip_id] = { chosen: current.showing, toggles: current.toggles };
  complete(dom.choose);
});

// ---- responses (contract §7 POST /api/followup/responses) -------------------

function responsesDocument() {
  return {
    schema: RESPONSES_SCHEMA,
    participant,
    recognition: payload.recognition.map((item) => ({
      clip_id: item.clip_id,
      checked: state.recognition[item.clip_id] || [],
    })),
    sound_only: payload.sound_only.map((item) => {
      const answer = state.sound_only[item.excerpt_id] || {};
      return {
        excerpt_id: item.excerpt_id,
        chosen_clip_id: answer.chosen_clip_id || null,
        played: answer.played || 0,
      };
    }),
    check: payload.check.map((item) => {
      const answer = state.check[item.clip_id] || {};
      return { clip_id: item.clip_id, chosen: answer.chosen || null, toggles: answer.toggles || 0 };
    }),
  };
}

async function submit(button) {
  if (button) button.disabled = true;
  let saved = false;
  try {
    const response = await fetch("/api/followup/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(responsesDocument()),
    });
    // 409: the server already holds this participant's responses and keeps
    // the first; this device's copy is superseded, and the session is done.
    saved = response.ok || response.status === 409;
  } catch (error) {
    saved = false;
  }
  if (!saved) {
    if (button) {
      // The step stays; its primary re-records and re-POSTs.
      notice(STRINGS.saveFailed);
      button.disabled = false;
    } else {
      // Resumed straight onto the owed POST: Begin is the way forward.
      showIntro();
      notice(STRINGS.saveFailed);
    }
    return;
  }
  state.submitted = true;
  state.step = steps.length - 1;
  saveState();
  showStep();
}

// ---- done (Table 9: summary) -------------------------------------------------

function showDone() {
  dom.doneHeading.textContent = STRINGS.doneHeading;
  dom.doneSummary.replaceChildren(
    element("li", "body", format(STRINGS.doneRecognition, { n: payload.recognition.length })),
    element("li", "body", format(STRINGS.doneSoundOnly, { n: payload.sound_only.length })),
    element("li", "body", format(STRINGS.doneCheck, { n: payload.check.length })),
    element("li", "body", STRINGS.doneSaved),
  );
  dom.done.hidden = false;
}

// ---- boot -------------------------------------------------------------------

function boot() {
  document.title = STRINGS.title;
  dom.masthead.textContent = STRINGS.title;
  dom.introHeading.textContent = STRINGS.title;
  dom.introBody.textContent = STRINGS.introBody;
  dom.participantLabel.textContent = STRINGS.participantLabel;
  dom.begin.textContent = STRINGS.begin;
  dom.recognitionHeading.textContent = STRINGS.recognitionHeading;
  dom.recognitionNext.textContent = STRINGS.next;
  dom.soundHeading.textContent = STRINGS.soundHeading;
  dom.play.textContent = STRINGS.play;
  dom.soundNext.textContent = STRINGS.next;
  dom.checkHeading.textContent = STRINGS.checkHeading;
  dom.toggle.textContent = STRINGS.showOther;
  dom.choose.textContent = STRINGS.preparing;

  dom.participant.value = recallParticipant();
  dom.participant.addEventListener("input", refreshBegin);
  dom.participant.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !dom.begin.disabled) begin();
  });
  dom.begin.addEventListener("click", begin);
  window.addEventListener("resize", fitStage);
  showIntro();
}

boot();

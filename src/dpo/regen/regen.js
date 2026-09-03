/* The participant's side of the six steps.

   Three things this file is careful about, all of them requirements rather
   than preferences. Section numbers cite docs/v3-regen/spec-behavior.md.

   The step comes from the server. Every screen is rendered from what
   /api/step/<step> returns, and every submit re-reads the step the server
   reports back. The page never decides where it is, so a reload, a second tab
   and a back button all land on the same screen the log says the session is on
   (§9.1). A 409 means the server disagrees, and the page follows the server.

   Nothing measured is ever shown. §4's points get a count and nothing else:
   the page never learns which object a point hit, because matching happens
   once on submit (§4) and telling the participant would turn the task into
   hunting for a mask.

   Play and select are separate controls (§5). They are separate buttons in
   separate grid cells, and neither handler touches the other's state, so a
   participant cannot select a lane by trying to hear it. */

const $ = (id) => document.getElementById(id);

const state = {
  participant: null,
  step: null,
  strings: null,
  scale: null,
  minimumPoints: 3,
  steps: [],
  entered: null,
  points: [],
  frame: 0,
  lanes: new Map(),
  selected: [],
  audio: null,
  playing: null,
};

const SCREENS = [
  "screen-start",
  "screen-survey",
  "screen-visual",
  "screen-auditory",
  "screen-waiting",
  "screen-done",
  "screen-error",
  "screen-viewing",
];

function show(id) {
  for (const screen of SCREENS) $(screen).hidden = screen !== id;
  $("shell").hidden = id === "screen-viewing";
}

function fail(message) {
  $("error-text").textContent = message || state.strings?.error || "Something went wrong.";
  show("screen-error");
}

async function api(path, payload) {
  const options = payload
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) }
    : {};
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    // The server is the authority on the step; a 409 carries the real one.
    if (response.status === 409 && body.step) return render(body.step);
    throw new Error(body.error || `${path} failed with ${response.status}`);
  }
  return body;
}

/* The event stream (§9.5's third file). Batched and best-effort: a dropped
   batch must never block a step, so failures are swallowed here rather than
   surfaced. Anything a measure depends on goes through its own route. */
let pending = [];
let flushing = null;

function note(type, fields) {
  pending.push({ type, at: new Date().toISOString(), ...fields });
  if (!flushing) flushing = setTimeout(flush, 1200);
}

async function flush() {
  flushing = null;
  const events = pending;
  pending = [];
  if (!events.length || !state.participant) return;
  try {
    await fetch("/api/events", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ participant: state.participant, events }),
    });
  } catch {
    /* best effort */
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") flush();
});

function drawRail() {
  const rail = $("rail");
  rail.replaceChildren();
  const at = state.steps.indexOf(state.step);
  state.steps.forEach((_, index) => {
    const span = document.createElement("span");
    if (at >= 0 && index < at) span.className = "done";
    if (index === at) span.className = "at";
    rail.append(span);
  });
}

/* §2 and §7 — the viewing. One start button, then fullscreen and no controls. */
async function renderViewing(step) {
  const detail = await api(`/api/step/${step}?participant=${state.participant}`);
  if (!detail) return;
  const strings = state.strings.view;
  $("start-heading").textContent = state.strings.app_title;
  $("start-headphones").textContent = strings.headphones;
  $("start-ready").textContent = strings.ready;
  const button = $("start-button");
  button.textContent = state.strings.actions.start;
  button.disabled = false;
  show("screen-start");

  button.onclick = async () => {
    button.disabled = true;
    const video = $("video");
    video.src = `/media/video/${detail.segment}`;
    show("screen-viewing");
    try {
      await $("screen-viewing").requestFullscreen();
    } catch {
      /* A browser that refuses fullscreen still plays; the clip is the
         stimulus and losing it to a permissions prompt would be worse. */
    }
    const cues = detail.captions;
    const band = $("cue");
    band.textContent = "";
    let shown = -1;
    const startedAt = new Date().toISOString();
    video.ontimeupdate = () => {
      const ms = video.currentTime * 1000;
      const index = cues.findIndex((cue) => ms >= cue.start_ms && ms < cue.end_ms);
      if (index === shown) return;
      shown = index;
      band.textContent = index === -1 ? "" : cues[index].text;
      if (index !== -1) note("caption.shown", { step, index, text: cues[index].text });
    };
    video.onended = async () => {
      const endedAt = new Date().toISOString();
      if (document.fullscreenElement) await document.exitFullscreen().catch(() => {});
      const result = await api("/api/viewing", {
        participant: state.participant,
        step,
        started_at: startedAt,
        ended_at: endedAt,
      });
      if (result) render(result.step);
    };
    try {
      await video.play();
    } catch (error) {
      fail(`the clip would not start: ${error.message}`);
    }
  };
}

/* §3 and §8 — the surveys, built from the blocks the server sends. */
async function renderSurvey(page) {
  const detail = await api(`/api/step/${page}?participant=${state.participant}`);
  if (!detail) return;
  state.entered = new Date().toISOString();
  const answers = new Map();
  const container = $("survey-blocks");
  container.replaceChildren();
  $("survey-heading").textContent = state.strings.app_title;
  $("survey-instruction").textContent = state.strings.survey.instruction;
  const total = detail.blocks.reduce((sum, block) => sum + block.items.length, 0);

  const submit = $("survey-submit");
  submit.textContent = state.strings.actions.submit;
  const update = () => {
    const left = total - answers.size;
    submit.disabled = left > 0;
    $("survey-remaining").textContent = left ? state.strings.survey.remaining.replace("{count}", left) : "";
  };

  for (const block of detail.blocks) {
    const section = document.createElement("section");
    section.className = "block";
    const heading = document.createElement("h2");
    heading.textContent = block.title;
    const anchors = document.createElement("div");
    anchors.className = "anchors";
    for (const anchor of detail.scale.anchors) {
      const span = document.createElement("span");
      span.className = "eyebrow";
      span.textContent = anchor;
      anchors.append(span);
    }
    section.append(heading, anchors);
    for (const item of block.items) {
      const row = document.createElement("div");
      row.className = "item";
      const text = document.createElement("p");
      text.textContent = item.text;
      const choices = document.createElement("div");
      choices.className = "choices";
      for (let value = 1; value <= detail.scale.points; value += 1) {
        const label = document.createElement("label");
        const input = document.createElement("input");
        input.type = "radio";
        input.name = item.id;
        input.value = String(value);
        input.onchange = () => {
          answers.set(item.id, value);
          note("survey.answered", { page, item: item.id, value });
          update();
        };
        const badge = document.createElement("span");
        badge.textContent = String(value);
        label.append(input, badge);
        choices.append(label);
      }
      row.append(text, choices);
      section.append(row);
    }
    container.append(section);
  }
  update();
  show("screen-survey");

  submit.onclick = async () => {
    submit.disabled = true;
    const result = await api("/api/survey", {
      participant: state.participant,
      page,
      responses: Object.fromEntries(answers),
      entered_at: state.entered,
      submitted_at: new Date().toISOString(),
    });
    if (result) render(result.step);
    else submit.disabled = false;
  };
}

/* §4 — marks on a strip of moments. Scroll to a frame, click to mark it. The
   frame in the middle is the one being marked; its neighbours are smaller and
   softened, so which surface takes a click is never in doubt. Coordinates are
   normalised to the image, so a resize between marking and submitting does not
   move what was meant, and each mark carries the frame it belongs to. */
async function renderVisual() {
  const detail = await api(`/api/step/visual?participant=${state.participant}`);
  if (!detail) return;
  state.points = [];
  state.minimumPoints = detail.minimum;
  state.frame = Math.floor(detail.frames.length / 2);
  const strip = $("strip");
  const copy = state.strings.visual;
  $("visual-heading").textContent = state.strings.app_title;
  $("visual-clear").textContent = state.strings.actions.clear;
  $("visual-next").textContent = state.strings.actions.next;
  strip.setAttribute("aria-label", copy.strip);

  const shells = detail.frames.map((frame) => {
    const shell = document.createElement("div");
    shell.className = "frame";
    shell.dataset.index = String(frame.index);
    const image = document.createElement("img");
    image.src = `/media/frame/${detail.segment}/${frame.index}`;
    image.alt = "";
    image.draggable = false;
    const when = document.createElement("span");
    when.className = "moment";
    when.textContent = copy.moment.replace("{seconds}", (frame.at_ms / 1000).toFixed(1));
    shell.append(image, when);
    // A neighbour is a place to go, not a place to mark: clicking one brings
    // it to the middle instead of dropping a point on a frame the participant
    // cannot see properly.
    shell.onclick = (event) => {
      if (Number(shell.dataset.index) === state.frame) return placeAt(event, shell, image);
      select(Number(shell.dataset.index));
    };
    strip.append(shell);
    return { shell, image, frame };
  });

  // How long the width transition runs, read from the stylesheet so the two
  // cannot drift apart, and zero when the viewer asks for less motion.
  const eased = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const motion = eased
    ? parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--strip-motion")) || 380
    : 0;
  // While this is in the future, the strip is scrolling under its own hand and
  // scroll events are not the participant's.
  let settling = 0;
  let resting = null;

  function middleOf(index) {
    const target = shells[index].shell;
    return target.offsetLeft + target.offsetWidth / 2 - strip.clientWidth / 2;
  }

  function select(index) {
    if (index < 0 || index >= shells.length) return;
    const changed = index !== state.frame;
    state.frame = index;
    for (const { shell } of shells) {
      shell.classList.toggle("at", Number(shell.dataset.index) === index);
    }
    // The chosen frame is held in the middle for the whole of its growth, one
    // frame at a time, rather than being centred once against a width it is
    // about to leave. That is the difference between the strip arriving
    // centred and arriving wherever the old widths put it.
    settling = performance.now() + motion + 60;
    const hold = () => {
      strip.scrollLeft = middleOf(index);
      if (performance.now() < settling) requestAnimationFrame(hold);
    };
    requestAnimationFrame(hold);
    if (changed) note("frame.selected", { frame: index, at_ms: shells[index].frame.at_ms });
    paint();
  }

  // Scrolling is a way of choosing, not just of looking: when the strip comes
  // to rest, whichever frame is nearest the middle becomes the marked one.
  strip.onscroll = () => {
    if (performance.now() < settling) return;
    window.clearTimeout(resting);
    resting = window.setTimeout(() => {
      const middle = strip.scrollLeft + strip.clientWidth / 2;
      let nearest = state.frame;
      let closest = Infinity;
      shells.forEach(({ shell }, index) => {
        const gap = Math.abs(shell.offsetLeft + shell.offsetWidth / 2 - middle);
        if (gap < closest) {
          closest = gap;
          nearest = index;
        }
      });
      if (nearest !== state.frame) select(nearest);
    }, 110);
  };

  const at = (event, image) => {
    const box = image.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)),
      y: Math.min(1, Math.max(0, (event.clientY - box.top) / box.height)),
    };
  };

  function placeAt(event, shell, image) {
    if (event.target.classList.contains("point")) return;
    const point = { frame: state.frame, ...at(event, image) };
    state.points.push(point);
    note("point.placed", { index: state.points.length - 1, ...point });
    paint();
  }

  function paint() {
    for (const { shell } of shells) {
      for (const node of [...shell.querySelectorAll(".point")]) node.remove();
    }
    const here = shells[state.frame].shell;
    state.points.forEach((point, index) => {
      if (point.frame !== state.frame) return;
      const dot = document.createElement("div");
      dot.className = "point";
      dot.style.left = `${point.x * 100}%`;
      dot.style.top = `${point.y * 100}%`;
      dot.textContent = String(index + 1);
      dot.onpointerdown = (event) => beginDrag(event, index, dot, shells[state.frame].image);
      here.append(dot);
    });
    const count = state.points.length;
    $("visual-count").textContent = copy.placed.replace("{count}", count);
    const short = count < state.minimumPoints;
    $("visual-next").disabled = short;
    const floor = state.minimumPoints === 1 ? copy.minimum_one : copy.minimum.replace("{minimum}", state.minimumPoints);
    $("visual-instruction").textContent = short ? `${copy.instruction} ${floor}` : copy.instruction;
  }

  function beginDrag(event, index, dot, image) {
    // A press that never moves is a delete; one that moves is a drag. Deciding
    // on pointerup rather than on pointerdown means neither gesture has to be
    // learned, and a shaky hand does not delete a point it meant to nudge.
    event.preventDefault();
    event.stopPropagation();
    dot.setPointerCapture(event.pointerId);
    dot.classList.add("dragging");
    let moved = false;
    const move = (moveEvent) => {
      moved = true;
      const next = at(moveEvent, image);
      state.points[index] = { ...state.points[index], ...next };
      dot.style.left = `${next.x * 100}%`;
      dot.style.top = `${next.y * 100}%`;
    };
    const up = () => {
      dot.removeEventListener("pointermove", move);
      dot.classList.remove("dragging");
      if (moved) {
        note("point.moved", { index, ...state.points[index] });
      } else {
        const [removed] = state.points.splice(index, 1);
        note("point.removed", { index, ...removed });
      }
      paint();
    };
    dot.addEventListener("pointermove", move);
    dot.addEventListener("pointerup", up, { once: true });
  }

  $("visual-clear").onclick = () => {
    note("points.cleared", { count: state.points.length });
    state.points = [];
    paint();
  };

  $("visual-next").onclick = async () => {
    $("visual-next").disabled = true;
    const result = await api("/api/visual", { participant: state.participant, points: state.points });
    if (result) render(result.step);
    else paint();
  };

  select(state.frame);
  show("screen-visual");
}

/* §5 — the lanes. One stem at a time, levels from the document's gain, and a
   playhead drawn over the envelope while it plays. */
async function renderAuditory() {
  const detail = await api(`/api/step/auditory?participant=${state.participant}`);
  if (!detail) return;
  state.lanes = new Map(detail.stems.map((stem) => [stem.id, { plays: 0, listened_ms: 0 }]));
  state.selected = [];
  state.segment = detail.segment;
  const strings = state.strings.auditory;
  $("auditory-heading").textContent = state.strings.app_title;
  $("auditory-instruction").textContent = strings.instruction;
  $("auditory-note").textContent = strings.none;
  $("auditory-submit").textContent = state.strings.actions.submit;

  const container = $("lanes");
  container.replaceChildren();
  const canvases = new Map();

  for (const stem of detail.stems) {
    const lane = document.createElement("div");
    lane.className = "lane";
    lane.dataset.stem = stem.id;

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = stem.label;
    // The family, where the source vocabulary has one. A label can be a
    // narrower claim than its neighbour, and the colour already groups by
    // family; naming it makes what the colour is doing legible.
    if (stem.parent) {
      const family = document.createElement("small");
      family.textContent = stem.parent;
      label.append(family);
    }

    const canvas = document.createElement("canvas");
    canvas.width = 600;
    canvas.height = 48;
    canvases.set(stem.id, canvas);

    const actions = document.createElement("div");
    actions.className = "lane-actions";
    const play = document.createElement("button");
    play.textContent = strings.play;
    const select = document.createElement("button");
    select.textContent = strings.select;

    play.onclick = () => toggle(stem, play, lane);
    select.onclick = () => {
      const index = state.selected.indexOf(stem.id);
      if (index === -1) state.selected.push(stem.id);
      else state.selected.splice(index, 1);
      const on = state.selected.includes(stem.id);
      lane.classList.toggle("selected", on);
      select.textContent = on ? strings.selected : strings.select;
      note("lane.selected", { stem: stem.id, selected: on, order: state.selected.indexOf(stem.id) });
    };

    actions.append(play, select);
    lane.append(label, canvas, actions);
    container.append(lane);
    drawWave(canvas, stem, 0);
  }

  function drawWave(canvas, stem, progress) {
    const context = canvas.getContext("2d");
    const { width, height } = canvas;
    context.clearRect(0, 0, width, height);
    context.fillStyle = stem.colour;
    const step = width / stem.waveform.length;
    stem.waveform.forEach((sample, index) => {
      const bar = Math.max(1, sample * height);
      context.fillRect(index * step, (height - bar) / 2, Math.max(1, step - 1), bar);
    });
    if (progress > 0) {
      context.fillStyle = "rgba(22, 24, 26, 0.85)";
      context.fillRect(progress * width, 0, 2, height);
    }
  }

  function stop() {
    if (!state.playing) return;
    const { stem, element, since, button } = state.playing;
    element.pause();
    state.lanes.get(stem.id).listened_ms += Date.now() - since;
    button.textContent = strings.play;
    drawWave(canvases.get(stem.id), stem, 0);
    note("lane.stopped", { stem: stem.id, ...state.lanes.get(stem.id) });
    state.playing = null;
  }

  function toggle(stem, button, lane) {
    if (state.playing && state.playing.stem.id === stem.id) return stop();
    // §5: starting another lane stops the previous one. Enforced here rather
    // than by pausing on the element's own play event, so the listening time
    // of the lane being interrupted is closed before the next one opens.
    stop();
    const element = new Audio(`/media/stem/${state.segment}/${stem.id}`);
    // Levels are normalised against the original mix by the gain measured when
    // the stems were cut (§5). WebAudio, not element.volume, because a gain
    // above 1 is a legitimate normalisation and volume clamps at 1.
    if (!state.audio) state.audio = new (window.AudioContext || window.webkitAudioContext)();
    const source = state.audio.createMediaElementSource(element);
    const gain = state.audio.createGain();
    gain.gain.value = stem.gain;
    source.connect(gain).connect(state.audio.destination);
    element.ontimeupdate = () => {
      if (element.duration) drawWave(canvases.get(stem.id), stem, element.currentTime / element.duration);
    };
    element.onended = stop;
    element.play().then(
      () => {
        // The play count is what §5 reports a selection against, so it moves
        // when sound actually starts rather than when the control was pressed.
        const counts = state.lanes.get(stem.id);
        counts.plays += 1;
        state.playing = { stem, element, since: Date.now(), button };
        button.textContent = strings.stop;
        note("lane.played", { stem: stem.id, plays: counts.plays });
      },
      (error) => {
        // A lane that will not play is this lane's problem and not the
        // session's. Ending the run here would lose §6, §7 and §8 to one
        // missing or undecodable file, and the participant has already given
        // two of the study's measures by this point. The lane says it has no
        // sound, its selection control is untouched — they may well have
        // heard the source in the clip — and the log carries the failure so
        // the researcher sees it without the participant losing the session.
        lane.classList.add("unplayable");
        button.textContent = strings.unavailable;
        button.disabled = true;
        note("lane.unplayable", { stem: stem.id, error: String((error && error.message) || error) });
      }
    );
  }

  $("auditory-submit").onclick = async () => {
    stop();
    $("auditory-submit").disabled = true;
    const lanes = Object.fromEntries(state.lanes);
    const result = await api("/api/auditory", {
      participant: state.participant,
      selected: state.selected,
      lanes,
    });
    if (result) render(result.step);
    else $("auditory-submit").disabled = false;
  };

  show("screen-auditory");
}

/* §6 — the wait. One POST, which is idempotent server-side, so a reload here
   returns the track already written rather than starting a second one. */
async function renderWaiting() {
  $("waiting-heading").textContent = state.strings.waiting.heading;
  $("waiting-stay").textContent = state.strings.waiting.stay;
  show("screen-waiting");
  const result = await api("/api/regenerate", { participant: state.participant });
  if (!result) return;
  note("regeneration.finished", { fallback: result.fallback, cached: result.cached });
  const next = await api(`/api/state?participant=${state.participant}`);
  if (next) render(next.step);
}

function renderDone() {
  $("done-heading").textContent = state.strings.done.heading;
  $("done-body").textContent = state.strings.done.body;
  const button = $("done-download");
  button.textContent = state.strings.actions.download;
  button.onclick = () => window.open(`/api/log?participant=${state.participant}`, "_blank");
  show("screen-done");
}

async function render(step) {
  state.step = step;
  drawRail();
  note("step.entered", { step });
  try {
    if (step === "view_prepared" || step === "view_regenerated") return await renderViewing(step);
    if (step === "art" || step === "survey") return await renderSurvey(step);
    if (step === "visual") return await renderVisual();
    if (step === "auditory") return await renderAuditory();
    if (step === "regenerating") return await renderWaiting();
    if (step === "done") return renderDone();
    fail(`unknown step ${step}`);
  } catch (error) {
    fail(error.message);
  }
}

async function boot() {
  try {
    const meta = await fetch("/api/strings").then((response) => response.json());
    state.strings = meta.strings;
    state.scale = meta.scale;
    state.minimumPoints = meta.minimum_points;
    state.steps = meta.steps;
    document.title = state.strings.app_title;
    // The identifier survives a reload; a fresh tab with none enrols anew (§1).
    const stored = window.sessionStorage.getItem("regen.participant");
    const session = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(stored ? { participant: stored } : {}),
    }).then((response) => response.json());
    if (session.error) return fail(session.error);
    state.participant = session.participant;
    window.sessionStorage.setItem("regen.participant", session.participant);
    await render(session.step);
  } catch (error) {
    fail(error.message);
  }
}

boot();

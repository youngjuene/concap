// Automated accessibility and responsive smoke checks for the Regen interfaces.
// Uses installed Playwright only; writes screenshots/results under /tmp by default.
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import net from "node:net";

const { chromium, firefox, webkit } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");

const output = process.env.REGEN_A11Y_OUTPUT || "/tmp/regen-accessibility-qa";
const harnessPath = fileURLToPath(import.meta.url);
const testsRoot = dirname(harnessPath);
const packageRoot = resolve(testsRoot, "..");
const repoRoot = resolve(packageRoot, "../../..");
const legacyBase = process.env.REGEN_A11Y_LEGACY_URL || "http://127.0.0.1:18781";
let studyBase = process.env.REGEN_A11Y_STUDY_URL || "";
const executablePath = process.env.CHROMIUM_EXECUTABLE;
const python = process.env.REGEN_A11Y_PYTHON || "/mnt/hdd/research/2026/concap/.venv/bin/python";
await mkdir(output, { recursive: true });

const errors = [];
const findings = [];
const evidence = {};
const check = (condition, message) => { if (!condition) throw new Error(message); };
const screenshot = (page, name) => page.screenshot({ path: resolve(output, name), fullPage: true });
const materialConsoleError = message => message.type() === "error" && !/Failed to load resource: the server responded with a status of 404/.test(message.text());

function startServer(command, args, readyMatch) {
  const child = spawn(command, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONPATH: [resolve(repoRoot, "src"), process.env.PYTHONPATH].filter(Boolean).join(":"),
    },
  });
  let log = "";
  const ready = new Promise((resolveReady, rejectReady) => {
    const timer = setTimeout(() => rejectReady(new Error(`server did not become ready: ${log.slice(-1200)}`)), 30000);
    const watch = data => {
      log += data.toString();
      if (readyMatch(log)) {
        clearTimeout(timer);
        resolveReady();
      }
    };
    child.stdout.on("data", watch);
    child.stderr.on("data", watch);
    child.on("exit", code => {
      clearTimeout(timer);
      rejectReady(new Error(`server exited with ${code}: ${log.slice(-1200)}`));
    });
  });
  return { child, ready, log: () => log };
}

async function freePort() {
  return new Promise((resolvePort, rejectPort) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close(() => resolvePort(port));
    });
    server.on("error", rejectPort);
  });
}

async function canFetch(url) {
  try {
    const response = await fetch(url);
    return response.ok;
  } catch {
    return false;
  }
}

async function startStudyServer(port) {
  const script = `
import json, tempfile
from pathlib import Path
import uvicorn
from dpo.regen.study_api import build_study_app
from dpo.regen.tests.test_study import make_media
root = Path(tempfile.mkdtemp(prefix="regen-a11y-study-"))
manifest = make_media(root / "media")
manifest["language"] = "en"
manifest["languages"] = ["en", "ko"]
for video in manifest["viewing_videos"]:
    for cue_index, cue in enumerate(video["cues"], start=1):
        cue["fallback"]["en"] = f"Synthetic accessibility fallback caption {cue_index}."
        cue["fallback"]["ko"] = f"합성 접근성 대체 자막 {cue_index} 소리"
path = root / "study.json"
path.write_text(json.dumps(manifest), encoding="utf-8")
print(json.dumps({"root": str(root), "url": "http://127.0.0.1:${port}"}), flush=True)
uvicorn.run(build_study_app(path, root / "media", root / "out"), host="127.0.0.1", port=${port}, log_level="warning")
`;
  return startServer(python, ["-c", script], log => log.includes(`"url": "http://127.0.0.1:${port}"`));
}

async function waitForHTTP(url) {
  const deadline = Date.now() + 30000;
  let last = "";
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
      last = `${response.status}`;
    } catch (error) {
      last = error.message;
    }
    await new Promise(resolveWait => setTimeout(resolveWait, 250));
  }
  throw new Error(`server did not answer ${url}: ${last}`);
}

async function maybeStartServers() {
  const servers = [];
  if (process.env.REGEN_A11Y_LEGACY_URL) {
    await waitForHTTP(legacyBase);
  } else if (await canFetch(legacyBase)) {
    evidence.legacyServer = "reused existing server on http://127.0.0.1:18781";
  } else {
    const legacy = startServer(python, ["src/dpo/regen/tests/preview_readiness.py"], log => log.includes('"url": "http://127.0.0.1:18781"'));
    servers.push(legacy);
    await legacy.ready;
    await waitForHTTP(legacyBase);
    evidence.legacyServer = "started synthetic legacy server on http://127.0.0.1:18781";
  }
  if (!process.env.REGEN_A11Y_STUDY_URL) {
    const port = await freePort();
    studyBase = `http://127.0.0.1:${port}`;
    const study = await startStudyServer(port);
    servers.push(study);
    await study.ready;
    await waitForHTTP(studyBase);
    evidence.studyServer = `started synthetic standalone server on ${studyBase}`;
  } else {
    await waitForHTTP(studyBase);
  }
  return servers;
}

async function allowFastMedia(page, { fakeFullscreen = true } = {}) {
  await page.addInitScript(({ fakeFullscreen }) => {
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: query => ({
        matches: query.includes("prefers-reduced-motion"),
        media: query,
        onchange: null,
        addListener() {},
        removeListener() {},
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() { return false; },
      }),
    });
    const nativePlay = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = async function play() {
      void nativePlay;
      this.dispatchEvent(new Event("play"));
      this.dispatchEvent(new Event("playing"));
      return Promise.resolve();
    };
    if (fakeFullscreen) {
      Element.prototype.requestFullscreen = async function requestFullscreen() {
        Object.defineProperty(document, "fullscreenElement", { configurable: true, value: this });
        document.dispatchEvent(new Event("fullscreenchange"));
      };
      document.exitFullscreen = async () => {
        Object.defineProperty(document, "fullscreenElement", { configurable: true, value: null });
        document.dispatchEvent(new Event("fullscreenchange"));
      };
    }
  }, { fakeFullscreen });
}

async function installStateCapture(page) {
  await page.addInitScript(() => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      if (response.ok) {
        response.clone().json().then(body => {
          if (body.participant) window.__qaParticipant = body.participant;
          if (body.step) window.__qaStep = body.step;
        }).catch(() => {});
      }
      return response;
    };
  });
}

function rgb(value) {
  const match = /rgba?\(([^)]+)\)/.exec(value || "");
  if (!match) return null;
  const parts = match[1].split(",").map(part => Number.parseFloat(part.trim()));
  if (parts.length < 3) return null;
  return { r: parts[0], g: parts[1], b: parts[2], a: parts[3] ?? 1 };
}

function luminance({ r, g, b }) {
  const channel = value => {
    const scaled = value / 255;
    return scaled <= 0.03928 ? scaled / 12.92 : ((scaled + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function ratio(fg, bg) {
  const [lighter, darker] = [luminance(fg), luminance(bg)].sort((a, b) => b - a);
  return (lighter + 0.05) / (darker + 0.05);
}

async function assertNoMissingProgrammaticNames(page, label) {
  const missing = await page.evaluate(() => {
    const visible = node => {
      const style = getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
    };
    const nameFor = node => {
      const id = node.id ? document.querySelector(`label[for="${CSS.escape(node.id)}"]`)?.textContent : "";
      const labelled = node.getAttribute("aria-labelledby")?.split(/\s+/).map(part => document.getElementById(part)?.textContent || "").join(" ");
      const group = node.closest("[role=radiogroup], fieldset")?.querySelector("legend, [id]")?.textContent || "";
      return [
        node.getAttribute("aria-label"),
        labelled,
        id,
        node.closest("label")?.textContent,
        node.textContent,
        group,
        node.getAttribute("title"),
      ].filter(Boolean).join(" ").trim();
    };
    return [...document.querySelectorAll("button, input:not([type=hidden]), textarea, select, [role=application], [role=radiogroup]")]
      .filter(visible)
      .filter(node => !node.disabled)
      .map(node => ({ tag: node.tagName.toLowerCase(), type: node.getAttribute("type"), id: node.id, role: node.getAttribute("role"), name: nameFor(node) }))
      .filter(item => !item.name);
  });
  check(missing.length === 0, `${label} has controls without programmatic names: ${JSON.stringify(missing)}`);
}

async function assertStatusSemantics(page, label) {
  const bad = await page.evaluate(() => [...document.querySelectorAll('[role="status"]')]
    .filter(node => !node.hidden && getComputedStyle(node).display !== "none")
    .map(node => ({ id: node.id, live: node.getAttribute("aria-live"), text: node.textContent.trim() }))
    .filter(node => node.live && !["polite", "assertive", "off"].includes(node.live)));
  check(bad.length === 0, `${label} has invalid status aria-live values: ${JSON.stringify(bad)}`);
}

async function assertFocusOnHeading(page, label) {
  const focusedHeading = await page.evaluate(() => /^H[1-6]$/.test(document.activeElement?.tagName || ""));
  check(focusedHeading, `${label} did not move focus to the step heading`);
}

async function assertNoOverflowAt(page, label, sizes) {
  const failures = [];
  for (const size of sizes) {
    await page.setViewportSize(size);
    await page.waitForTimeout(100);
    const overflow = await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - window.innerWidth));
    if (overflow > 1) failures.push({ size, overflow });
  }
  check(failures.length === 0, `${label} has horizontal overflow: ${JSON.stringify(failures)}`);
}

async function assertContrast(page, label) {
  const failures = await page.evaluate(() => {
    const rgba = value => {
      const match = /rgba?\(([^)]+)\)/.exec(value || "");
      if (!match) return null;
      const parts = match[1].split(",").map(part => Number.parseFloat(part.trim()));
      return { r: parts[0], g: parts[1], b: parts[2], a: parts[3] ?? 1 };
    };
    const rel = color => {
      const channel = value => {
        const scaled = value / 255;
        return scaled <= 0.03928 ? scaled / 12.92 : ((scaled + 0.055) / 1.055) ** 2.4;
      };
      return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
    };
    const cr = (fg, bg) => {
      const values = [rel(fg), rel(bg)].sort((a, b) => b - a);
      return (values[0] + 0.05) / (values[1] + 0.05);
    };
    const backgroundFor = node => {
      let at = node;
      while (at && at !== document.documentElement) {
        const color = rgba(getComputedStyle(at).backgroundColor);
        if (color && color.a > 0.95) return color;
        at = at.parentElement;
      }
      return rgba(getComputedStyle(document.body).backgroundColor) || { r: 255, g: 255, b: 255 };
    };
    const visible = node => {
      const style = getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
    };
    return [...document.querySelectorAll("body *")]
      .filter(visible)
      .filter(node => node.childNodes && [...node.childNodes].some(child => child.nodeType === Node.TEXT_NODE && child.textContent.trim()))
      .map(node => {
        const style = getComputedStyle(node);
        const fg = rgba(style.color);
        const bg = backgroundFor(node);
        const size = Number.parseFloat(style.fontSize);
        const bold = Number.parseInt(style.fontWeight, 10) >= 700;
        const required = size >= 24 || (bold && size >= 18.66) ? 3 : 4.5;
        return { text: node.textContent.trim().slice(0, 80), ratio: fg && bg ? cr(fg, bg) : 99, required };
      })
      .filter(item => item.ratio + 0.01 < item.required)
      .slice(0, 12);
  });
  check(failures.length === 0, `${label} has low contrast text: ${JSON.stringify(failures)}`);
}

async function assertTargetSizes(page, label) {
  const small = await page.evaluate(() => [...document.querySelectorAll("button, input, textarea, select")]
    .filter(node => {
      const style = getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return !node.disabled && style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
    })
    .map(node => {
      const target = node.closest("label") || node;
      const rect = target.getBoundingClientRect();
      return { tag: node.tagName.toLowerCase(), type: node.getAttribute("type"), id: node.id, text: target.textContent.trim(), width: Math.round(rect.width), height: Math.round(rect.height) };
    })
    .filter(node => node.width < 24 || node.height < 24));
  check(small.length === 0, `${label} has controls below 24px hit target: ${JSON.stringify(small)}`);
}

async function assertPageA11y(page, label) {
  await assertNoMissingProgrammaticNames(page, label);
  await assertStatusSemantics(page, label);
  await assertContrast(page, label);
  await assertTargetSizes(page, label);
}

async function finishLegacyViewing(page, expectedStep) {
  await page.getByRole("button", { name: /Start|시작/ }).click();
  await page.locator("#screen-viewing:not([hidden])").waitFor();
  await page.locator("#video").evaluate(video => {
    video.currentTime = video.duration || 10;
    video.dispatchEvent(new Event("timeupdate"));
    video.dispatchEvent(new Event("ended"));
  });
  await page.waitForFunction(step => window.__qaStep === step, expectedStep === "view_prepared" ? "art" : "survey", { timeout: 15000 });
}

async function answerLegacySurvey(page) {
  const names = await page.locator("#survey-blocks input[type=radio]").evaluateAll(inputs => [...new Set(inputs.map(input => input.name))]);
  check(names.length > 0, "legacy survey rendered no radio inputs");
  for (const name of names) await page.locator(`input[name="${name}"][value="4"]`).check();
}

async function runLegacyChecks(browser) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: "reduce" });
  page.on("pageerror", error => errors.push(`legacy pageerror: ${error.message}`));
  page.on("console", message => { if (materialConsoleError(message)) errors.push(`legacy console: ${message.text()}`); });
  await allowFastMedia(page);
  await installStateCapture(page);
  await page.goto(legacyBase);
  await page.getByRole("heading", { name: /Watch|보기/ }).waitFor({ timeout: 15000 });
  await assertFocusOnHeading(page, "legacy start");
  await assertPageA11y(page, "legacy start");
  await assertNoOverflowAt(page, "legacy start", [{ width: 320, height: 700 }, { width: 390, height: 844 }, { width: 1280, height: 900 }]);
  await screenshot(page, "legacy-start.png");

  await finishLegacyViewing(page, "view_prepared");
  await page.locator("#screen-survey:not([hidden])").waitFor();
  await assertFocusOnHeading(page, "legacy ART survey");
  await assertPageA11y(page, "legacy ART survey");
  await assertNoOverflowAt(page, "legacy ART survey", [{ width: 320, height: 700 }, { width: 390, height: 844 }]);
  await answerLegacySurvey(page);
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  await page.waitForFunction(() => window.__qaStep === "visual", null, { timeout: 15000 });

  await page.locator("#screen-visual:not([hidden])").waitFor();
  await assertFocusOnHeading(page, "legacy visual marking");
  await assertPageA11y(page, "legacy visual marking");
  await page.locator("#plate").focus();
  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  check(await page.locator("#plate .point").count() === 2, "legacy visual keyboard did not add two marker points");
  try {
    await page.locator("#plate .drop").first().click({ timeout: 3000 });
  } catch (error) {
    findings.push({
      severity: "High",
      area: "legacy visual marker removal",
      issue: "Marker remove button has an accessible label but is not reachable by a normal pointer click; the parent .point intercepts the event.",
      detail: error.message.split("\n").slice(0, 5).join(" "),
    });
    await page.locator("#plate .drop").first().evaluate(node => node.click());
  }
  check(await page.locator("#plate .point").count() === 1, "legacy visual drop control did not remove one marker point");
  check(await page.evaluate(() => document.activeElement?.id === "plate"), "legacy visual marker removal did not return focus to the plate");
  await page.keyboard.press("Enter");
  while (await page.locator("#plate .point").count() < 3) await page.keyboard.press("Enter");
  await screenshot(page, "legacy-visual-keyboard.png");
  await page.locator("#visual-next").click();
  await page.waitForFunction(() => window.__qaStep === "auditory", null, { timeout: 15000 });

  await page.locator("#screen-auditory:not([hidden])").waitFor();
  await assertFocusOnHeading(page, "legacy auditory");
  await assertPageA11y(page, "legacy auditory");
  const groups = await page.locator('#lanes [role="radiogroup"]').count();
  check(groups === 5, `legacy auditory expected five labelled sound-family groups, saw ${groups}`);
  const names = await page.locator("#lanes input[type=radio]").evaluateAll(inputs => [...new Set(inputs.map(input => input.name))]);
  for (const name of names) await page.locator(`input[name="${name}"]`).first().check();
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  const waitSeen = await page.waitForFunction(() => {
    const waiting = document.querySelector("#screen-waiting:not([hidden])");
    return waiting ? "waiting" : window.__qaStep === "view_regenerated" ? "complete" : "";
  }, null, { timeout: 15000 }).then(value => value.jsonValue());
  if (waitSeen === "waiting") {
    await assertFocusOnHeading(page, "legacy regeneration wait");
    await assertPageA11y(page, "legacy regeneration wait");
    const reducedMotionBar = await page.evaluate(() => {
      const active = document.querySelector(".waiting .bar i.at");
      return active ? getComputedStyle(active, "::after").animationName : "no-active-cell";
    });
    check(reducedMotionBar === "none" || reducedMotionBar === "no-active-cell", `legacy reduced-motion bar still animates: ${reducedMotionBar}`);
    await screenshot(page, "legacy-waiting-reduced-motion.png");
    evidence.legacyWait = "observed rendered regeneration wait with reduced-motion animation disabled";
  } else {
    evidence.legacyWait = "synthetic regeneration completed before the wait screen could be sampled";
  }
  await page.close();
  evidence.legacy = "start, ART survey, visual marking, auditory groups, regeneration wait";
}

async function studyState(page) {
  return page.evaluate(() => fetch("/api/study/state").then(response => response.json()));
}

async function postStudyAction(page, action, data) {
  return page.evaluate(async ({ action, data }) => {
    const state = await fetch("/api/study/state").then(response => response.json());
    const response = await fetch(`/api/study/${action}`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-study-request": "1" },
      body: JSON.stringify({ key: crypto.randomUUID(), revision: state.revision, data }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `${action} failed`);
    return body;
  }, { action, data });
}

async function completeCalibrationForWatch(browser) {
  const setup = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await setup.goto(studyBase);
  await setup.getByRole("heading", { name: "Make the captions yours" }).waitFor({ timeout: 15000 });
  const preferences = await studyState(setup);
  await postStudyAction(setup, "preferences", Object.fromEntries(preferences.items.map((item, index) => [item.id, index === 0 ? 4 : 2])));
  for (let clipIndex = 0; clipIndex < 3; clipIndex += 1) {
    let at = await studyState(setup);
    await postStudyAction(setup, "clip-playback", { clip_id: at.clip.id, position_ms: 0, sequence: at.sequence + 1, playing: true, hidden: false, seek: false });
    await setup.waitForTimeout(Math.max(2200, at.clip.duration_ms + 250));
    at = await studyState(setup);
    await postStudyAction(setup, "clip-playback", { clip_id: at.clip.id, position_ms: at.clip.duration_ms, sequence: at.sequence + 1, playing: false, hidden: false, seek: false });
    await postStudyAction(setup, "clip-ended", {});
    at = await studyState(setup);
    const heard = Object.fromEntries(at.families.map(family => [family, family === "things"]));
    await postStudyAction(setup, "observation", { points: [{ frame: 0, x: 0.5, y: 0.5 }], heard });
  }
  await postStudyAction(setup, "start-viewing", {});
  const watchState = await studyState(setup);
  await setup.close();
  return watchState;
}

async function runStandaloneChecks(browser) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: "reduce" });
  page.on("pageerror", error => errors.push(`study pageerror: ${error.message}`));
  page.on("console", message => { if (materialConsoleError(message)) errors.push(`study console: ${message.text()}`); });
  await allowFastMedia(page);
  await page.goto(studyBase);
  await page.getByRole("heading", { name: "Make the captions yours" }).waitFor({ timeout: 15000 });
  await assertFocusOnHeading(page, "standalone preferences");
  await assertPageA11y(page, "standalone preferences");
  await assertNoOverflowAt(page, "standalone preferences", [{ width: 320, height: 700 }, { width: 390, height: 844 }, { width: 1280, height: 900 }]);
  await screenshot(page, "standalone-preferences.png");

  await page.locator('input[name=texture][value="4"]').check();
  await page.locator('input[name=context][value="2"]').check();
  await page.getByRole("button", { name: "Begin calibration clips" }).click();
  await page.locator("video").waitFor();
  await assertFocusOnHeading(page, "standalone calibration clip");
  await assertPageA11y(page, "standalone calibration clip");
  await page.locator("video").evaluate(video => {
    video.currentTime = video.duration || 2;
    video.dispatchEvent(new Event("timeupdate"));
    video.dispatchEvent(new Event("ended"));
  });
  const clipState = await studyState(page);
  await page.route("**/api/study/session", route => route.fulfill({ json: { ...clipState, stage: "observe", draft: {} } }));
  await page.route("**/api/study/state", route => route.fulfill({ json: { ...clipState, stage: "observe", draft: {} } }));
  await page.route("**/api/study/draft", route => route.fulfill({ json: { ...clipState, stage: "observe", draft: {} } }));
  await page.reload();

  await page.getByRole("heading", { name: "What caught your attention?" }).waitFor();
  await assertFocusOnHeading(page, "standalone observation");
  await assertPageA11y(page, "standalone observation");
  await page.getByRole("button", { name: "Add point", exact: true }).click();
  check(await page.locator(".visual-point").count() === 1, "standalone observation Add point did not create a marker");
  await page.getByRole("button", { name: "Undo point" }).click();
  check(await page.locator(".visual-point").count() === 0, "standalone observation Undo point did not remove marker");
  await page.locator('input[type="number"]').first().fill("25");
  await page.locator('input[type="number"]').nth(1).fill("75");
  await page.getByRole("button", { name: "Add point", exact: true }).click();
  for (const family of ["human", "animal", "things", "music", "natural"]) {
    await page.locator(`input[name=${family}]`).first().check();
  }
  await screenshot(page, "standalone-observation.png");

  const watchState = await completeCalibrationForWatch(browser);
  const watch = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: "reduce" });
  watch.on("pageerror", error => errors.push(`study watch pageerror: ${error.message}`));
  await allowFastMedia(watch, { fakeFullscreen: false });
  Object.assign(watchState, {
    position_ms: 0,
    axes: { texture: 0.5, context: 0.5 },
    defaults: { texture: 0.5, context: 0.5 },
  });
  let settingsPayload;
  await watch.route("**/api/study/session", route => route.fulfill({ json: watchState }));
  await watch.route("**/api/study/state", route => route.fulfill({ json: { ...watchState, axes: settingsPayload ? { texture: settingsPayload.texture, context: settingsPayload.context } : watchState.axes } }));
  await watch.route("**/api/study/settings", async route => {
    settingsPayload = route.request().postDataJSON().data;
    await route.fulfill({ json: { ...watchState, axes: { texture: settingsPayload.texture, context: settingsPayload.context }, revision: watchState.revision + 1 } });
  });
  await watch.route("**/api/study/draft", route => route.fulfill({ json: watchState }));
  await watch.route("**/api/study/playback", route => route.fulfill({ json: watchState }));
  await watch.route("**/api/study/exposures", route => route.fulfill({ json: watchState }));
  await watch.route("**/api/study/captions", route => route.fulfill({ json: { epoch: watchState.epoch, revision: watchState.settings_revision, jobs: [] } }));
  await watch.goto(studyBase);
  await watch.getByRole("heading", { name: "Your caption detail" }).waitFor({ timeout: 15000 });
  await assertFocusOnHeading(watch, "standalone watch");
  await assertPageA11y(watch, "standalone watch");
  await assertNoOverflowAt(watch, "standalone watch", [{ width: 320, height: 700 }, { width: 390, height: 844 }, { width: 1280, height: 900 }]);
  await watch.locator('input[type="range"]').first().focus();
  await watch.keyboard.press("ArrowRight");
  await watch.waitForFunction(() => fetch("/api/study/state").then(r => r.json()).then(s => s.axes.texture > 0.5), null, { timeout: 10000 });
  await watch.getByText("Detail presets", { exact: true }).click();
  await watch.getByRole("button", { name: "Both detailed", exact: true }).click();
  await watch.waitForFunction(() => fetch("/api/study/state").then(r => r.json()).then(s => s.axes.texture === 1 && s.axes.context === 1), null, { timeout: 10000 });
  await screenshot(watch, "standalone-watch-controls.png");
  await watch.getByRole("button", { name: "Fullscreen", exact: true }).click();
  await watch.waitForFunction(() => document.fullscreenElement?.classList.contains("watch-layout"), null, { timeout: 10000 });
  await watch.getByRole("button", { name: "Exit fullscreen", exact: true }).waitFor({ timeout: 10000 });
  await assertNoOverflowAt(watch, "standalone watch fullscreen", [{ width: 1280, height: 900 }]);
  await screenshot(watch, "standalone-watch-fullscreen.png");
  await watch.getByRole("button", { name: "Exit fullscreen", exact: true }).click();
  await watch.waitForFunction(() => !document.fullscreenElement, null, { timeout: 10000 });
  await watch.getByRole("button", { name: "Fullscreen", exact: true }).waitFor({ timeout: 10000 });
  await watch.close();

  const final = await browser.newPage({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });
  final.on("pageerror", error => errors.push(`study final pageerror: ${error.message}`));
  await allowFastMedia(final);
  const surveyState = { ...watchState, stage: "final", completed_videos: 3, video_index: 3, session_id: "a11y-final-fixture", draft: {} };
  await final.route("**/api/study/session", route => route.fulfill({ json: surveyState }));
  await final.route("**/api/study/draft", route => route.fulfill({ json: surveyState }));
  await final.route("**/api/study/final-survey", route => route.fulfill({ json: { ...surveyState, stage: "done" } }));
  await final.goto(studyBase);
  await final.getByRole("heading", { name: "Your viewing experience", exact: true }).waitFor({ timeout: 15000 });
  await assertFocusOnHeading(final, "standalone final survey");
  await assertPageA11y(final, "standalone final survey");
  await assertNoOverflowAt(final, "standalone final survey", [{ width: 320, height: 700 }, { width: 390, height: 844 }]);
  await screenshot(final, "standalone-final-mobile.png");
  await final.close();
  await page.close();
  evidence.standalone = "preferences, calibration clip, observation, watch controls, final survey";
}

async function smokeAlternate(browserType, label) {
  try {
    const browser = await browserType.launch({ executablePath: label === "chromium" ? executablePath : undefined, args: label === "chromium" ? ["--no-sandbox"] : [] });
    const page = await browser.newPage();
    await page.goto(studyBase);
    await page.getByRole("heading").first().waitFor({ timeout: 10000 });
    await browser.close();
    return { label, ok: true };
  } catch (error) {
    return { label, ok: false, error: error.message.split("\n")[0] };
  }
}

let servers = [];
let browser;
let alternateBrowsers = [];
try {
  servers = await maybeStartServers();
  browser = await chromium.launch({ executablePath, args: ["--no-sandbox"] });
  await runLegacyChecks(browser);
  await runStandaloneChecks(browser);
  alternateBrowsers = [];
  alternateBrowsers.push(await smokeAlternate(firefox, "firefox"));
  alternateBrowsers.push(await smokeAlternate(webkit, "webkit"));
  const result = {
    generatedAt: new Date().toISOString(),
    output,
    legacyBase,
    studyBase,
    evidence,
    errors,
    findings,
    alternateBrowsers,
    note: "Accessibility smoke uses synthetic media and mocked transport for direct standalone watch/final stage access; it is not a screen-reader or physical-device pass.",
  };
  await writeFile(resolve(output, "browser-accessibility-result.json"), JSON.stringify(result, null, 2));
  check(errors.length === 0, errors.join("\n"));
  check(findings.length === 0, `accessibility findings: ${JSON.stringify(findings)}`);
  console.log(JSON.stringify(result));
} catch (error) {
  await writeFile(resolve(output, "browser-accessibility-error.json"), JSON.stringify({ error: error.message, stack: error.stack, errors, findings, evidence, alternateBrowsers }, null, 2));
  throw error;
} finally {
  if (browser) await browser.close();
  for (const server of servers) server.child.kill("SIGTERM");
}

// Browser regression for the integrated short-clip and long-video survey hierarchy.
// Starts tests/preview_survey_hierarchy.py unless REGEN_HIERARCHY_URL is supplied.
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const playwrightModule = process.env.PLAYWRIGHT_MODULE || "/tmp/atlas-qa/node_modules/playwright-core/index.mjs";
const loaded = await import(playwrightModule);
const { chromium } = loaded.chromium ? loaded : loaded.default;

const harnessPath = fileURLToPath(import.meta.url);
const testsRoot = dirname(harnessPath);
const packageRoot = resolve(testsRoot, "..");
const repoRoot = resolve(packageRoot, "../../..");
const output = process.env.REGEN_HIERARCHY_OUTPUT || "/tmp/regen-survey-hierarchy";
const base = process.env.REGEN_HIERARCHY_URL || "http://127.0.0.1:18781";
const python = process.env.REGEN_HIERARCHY_PYTHON || resolve(repoRoot, ".venv/bin/python");
const executablePath = process.env.CHROMIUM_EXECUTABLE || "/home/june/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";
await mkdir(output, { recursive: true });

const errors = [];
const evidence = {};
const check = (condition, message) => { if (!condition) throw new Error(message); };

function startServer() {
  const child = spawn(python, ["src/dpo/regen/tests/preview_survey_hierarchy.py"], {
    cwd: repoRoot,
    env: {
      ...process.env,
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONPATH: [resolve(repoRoot, "src"), process.env.PYTHONPATH].filter(Boolean).join(":"),
    },
  });
  let log = "";
  const ready = new Promise((resolveReady, rejectReady) => {
    const timer = setTimeout(() => rejectReady(new Error(`server did not become ready: ${log.slice(-1600)}`)), 45000);
    const watch = data => {
      log += data.toString();
      if (log.includes('"url": "http://127.0.0.1:18781"')) {
        clearTimeout(timer);
        resolveReady();
      }
    };
    child.stdout.on("data", watch);
    child.stderr.on("data", watch);
    child.on("exit", code => {
      clearTimeout(timer);
      rejectReady(new Error(`server exited with ${code}: ${log.slice(-1600)}`));
    });
  });
  return { child, ready, log: () => log };
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

async function allowFastMedia(page) {
  await page.addInitScript(() => {
    const nativePlay = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = async function play() {
      void nativePlay;
      this.dispatchEvent(new Event("play"));
      this.dispatchEvent(new Event("playing"));
      return Promise.resolve();
    };
    Element.prototype.requestFullscreen = async function requestFullscreen() {
      Object.defineProperty(document, "fullscreenElement", { configurable: true, value: this });
      document.dispatchEvent(new Event("fullscreenchange"));
    };
    document.exitFullscreen = async () => {
      Object.defineProperty(document, "fullscreenElement", { configurable: true, value: null });
      document.dispatchEvent(new Event("fullscreenchange"));
    };
  });
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
          if (body.stage) window.__qaStage = body.stage;
          if (Object.prototype.hasOwnProperty.call(body, "survey_flow")) window.__qaSurveyFlow = body.survey_flow;
          if (Number.isInteger(body.clip_index)) window.__qaClipIndex = body.clip_index;
          if (Number.isInteger(body.video_index)) window.__qaVideoIndex = body.video_index;
        }).catch(() => {});
      }
      return response;
    };
  });
}

async function localized(page, source) {
  return page.evaluate(async text => {
    const response = await fetch(new URL("api/study/strings", location.href), { credentials: "same-origin" });
    const copy = await response.json();
    return document.documentElement.lang === "ko" ? (copy[text] || text) : text;
  }, source);
}

async function assertNoOverflow(page, label) {
  const sizes = [{ width: 390, height: 844 }, { width: 1280, height: 900 }];
  const failures = [];
  for (const size of sizes) {
    await page.setViewportSize(size);
    await page.waitForTimeout(75);
    const overflow = await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - window.innerWidth));
    if (overflow > 1) failures.push({ size, overflow });
  }
  check(failures.length === 0, `${label} has horizontal overflow: ${JSON.stringify(failures)}`);
}

async function screenshot(page, name) {
  await page.screenshot({ path: resolve(output, name), fullPage: true });
}

async function captureResponsive(page, name) {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForTimeout(75);
  await screenshot(page, `${name}-desktop.png`);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(75);
  await screenshot(page, `${name}-mobile.png`);
  await page.setViewportSize({ width: 1280, height: 900 });
}

async function finishShortViewing(page, expectedStep) {
  await page.getByRole("button", { name: /Start|시작/ }).click();
  await page.locator("#screen-viewing:not([hidden])").waitFor({ timeout: 15000 });
  const qa = await page.evaluate(async () => {
    const response = await fetch(new URL("qa/complete-short-viewing", location.href), {
      method: "POST",
      headers: { "content-type": "application/json", "x-study-request": "1" },
      credentials: "same-origin",
      body: JSON.stringify({
        participant: window.__qaParticipant,
        step: window.__qaStep,
        clip_index: window.__qaClipIndex,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "QA short completion failed");
    return body;
  });
  check(qa.coverage_ms >= 9900, `short ${qa.step} clip ${qa.clip_index + 1} QA coverage was ${qa.coverage_ms}`);
  await page.locator("#video").evaluate(video => {
    video.currentTime = video.duration || 10;
    video.dispatchEvent(new Event("ended"));
  });
  await page.waitForFunction(step => window.__qaStep === step, expectedStep, { timeout: 20000 });
}

async function surveyDetail(page, pageName) {
  return page.evaluate(async name => {
    const response = await fetch(`/api/step/${name}?participant=${encodeURIComponent(window.__qaParticipant)}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `${name} detail failed`);
    return body;
  }, pageName);
}

function blockIds(detail) {
  return detail.blocks.map(block => block.id);
}

async function checkedRadioCount(page) {
  return page.locator("#survey-blocks input[type=radio]:checked").count();
}

async function assertShortSurveyShape(page, pageName, expectedBlocks, label) {
  await page.locator("#screen-survey:not([hidden])").waitFor({ timeout: 15000 });
  const detail = await surveyDetail(page, pageName);
  check(JSON.stringify(blockIds(detail)) === JSON.stringify(expectedBlocks), `${label} blocks were ${JSON.stringify(blockIds(detail))}`);
  check(!blockIds(detail).includes("prss"), `${label} unexpectedly contains PRSS`);
  const rendered = await page.locator("#survey-blocks h2").evaluateAll(nodes => nodes.map(node => node.textContent.trim()));
  check(rendered.length === expectedBlocks.length, `${label} rendered ${rendered.length} block heading(s)`);
  await assertNoOverflow(page, label);
  return detail;
}

async function answerShortSurvey(page, pageName, expectedBlocks, label, { verifyDraft = false, expectFresh = false } = {}) {
  const detail = await assertShortSurveyShape(page, pageName, expectedBlocks, label);
  if (expectFresh) check(await checkedRadioCount(page) === 0, `${label} inherited answers from another clip`);
  if (verifyDraft) {
    await page.locator("#survey-blocks input[type=radio]").first().check();
    await page.reload();
    await page.locator("#screen-survey:not([hidden])").waitFor({ timeout: 15000 });
    check(await checkedRadioCount(page) === 1, `${label} draft did not survive reload`);
  }
  for (const item of detail.blocks.flatMap(block => block.items)) {
    await page.locator(`input[name="${item.id}"][value="4"]`).check();
  }
  await page.locator("#survey-submit").click();
}

async function completeVisualAndAuditory(page) {
  await page.waitForFunction(() => window.__qaStep === "visual", null, { timeout: 15000 });
  await page.locator("#plate").focus();
  while (await page.locator("#plate .point").count() < 2) await page.keyboard.press("Enter");
  await page.locator("#visual-next").click();
  await page.waitForFunction(() => window.__qaStep === "auditory", null, { timeout: 15000 });
  await page.locator("#screen-auditory:not([hidden])").waitFor({ timeout: 15000 });
  await page.locator("#lanes .family").first().waitFor({ timeout: 15000 });
  const names = await page.locator("#lanes input[type=radio]").evaluateAll(inputs => [...new Set(inputs.map(input => input.name))]);
  check(names.length === 5, `auditory rendered ${names.length} sound-family questions`);
  for (const name of names) await page.locator(`input[name="${name}"]`).first().check();
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  await page.waitForFunction(() => window.__qaStep === "view_regenerated" || Boolean(document.querySelector("#screen-waiting:not([hidden])")), null, { timeout: 15000 });
  await page.waitForFunction(() => window.__qaStep === "view_regenerated", null, { timeout: 30000 });
}

async function answerOverallPRSS(page, locale) {
  await page.waitForFunction(() => window.__qaStep === "overall", null, { timeout: 20000 });
  await page.locator("#screen-survey:not([hidden])").waitFor({ timeout: 15000 });
  const detail = await surveyDetail(page, "overall");
  check(JSON.stringify(blockIds(detail)) === JSON.stringify(["prss"]), `${locale} overall blocks were ${JSON.stringify(blockIds(detail))}`);
  check(detail.scale.points === 7, `${locale} PRSS scale expected 7 points, got ${detail.scale.points}`);
  check(detail.blocks[0].items.length > 0, `${locale} overall PRSS rendered no items`);
  await captureResponsive(page, `${locale}-short-overall-prss`);
  await assertNoOverflow(page, `${locale} short overall PRSS`);
  for (const item of detail.blocks[0].items) await page.locator(`input[name="${item.id}"][value="4"]`).check();
  await page.locator("#survey-submit").click();
}

async function continueToLongPhase(page, locale) {
  await page.waitForURL(/\/viewing\/.+\/$/, { timeout: 20000 });
  await page.getByRole("heading", { name: /Your calibration is saved|사전 응답이 저장되었습니다/ }).waitFor({ timeout: 15000 });
  await page.getByRole("button", { name: /Start viewing experience|영상 시청 시작/ }).click();
  await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
  await assertNoOverflow(page, `${locale} long-video watch`);
}

async function finishLongVideo(page, index, locale) {
  await page.locator("video").waitFor({ timeout: 15000 });
  const qa = await page.evaluate(async () => {
    const response = await fetch(new URL("qa/complete-current-video", location.href), {
      method: "POST",
      headers: { "x-study-request": "1" },
      credentials: "same-origin",
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "QA completion failed");
    return body;
  });
  check(qa.coverage_ms >= 299000, `${locale} video ${index + 1} QA coverage was ${qa.coverage_ms}`);
  await page.locator("video").evaluate(video => {
    video.play();
    const duration = 300000;
    for (let ms = 0; ms <= duration; ms += 5000) {
      video.currentTime = ms / 1000;
      video.dispatchEvent(new Event("timeupdate"));
    }
    Object.defineProperty(video, "ended", { configurable: true, value: true });
    video.dispatchEvent(new Event("ended"));
  });
  await page.getByRole("button", { name: await localized(page, "Continue to this video's questions"), exact: true }).click({ timeout: 20000 });
  await page.waitForFunction(() => window.__qaStage === "video-survey", null, { timeout: 20000 });
  await page.getByRole("heading", { name: /Video .*Your experience|영상 3개 중 .*시청 경험/ }).waitFor({ timeout: 15000 });
  const state = await page.evaluate(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json()));
  check(Boolean(state.survey_flow) || await page.evaluate(() => Boolean(window.__qaSurveyFlow)), `${locale} long phase did not advertise survey_flow`);
  const ids = state.items.map(item => item.id);
  check(JSON.stringify(ids) === JSON.stringify(["accurate", "texture", "context", "personal"]), `${locale} video ${index + 1} survey items were ${JSON.stringify(ids)}`);
  check(!ids.some(id => /^prss|restorative|fascination|being_away|extent|compatibility/.test(id)), `${locale} video ${index + 1} included PRSS-like items`);
  check(await page.locator('input[type="radio"]:checked').count() === 0, `${locale} video ${index + 1} survey inherited answers`);
  if (index === 0) await captureResponsive(page, `${locale}-long-video-survey`);
  await assertNoOverflow(page, `${locale} video ${index + 1} survey`);
  await page.locator('input[type="radio"]').first().check();
  await page.waitForFunction(
    () => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" })
      .then(r => r.json())
      .then(s => Boolean(s.draft?.video_id && s.draft?.answers && Object.keys(s.draft.answers).length === 1)),
    null,
    { timeout: 5000 }
  );
  await page.reload();
  await page.waitForFunction(() => window.__qaStage === "video-survey", null, { timeout: 15000 });
  check(await page.locator('input[type="radio"]:checked').count() === 1, `${locale} video ${index + 1} survey draft did not survive reload`);
  for (const item of state.items) await page.locator(`input[name="${item.id}"][value="4"]`).check();
  const submitName = index === 2 ? "Save and continue to overall experience" : "Save this video's answers";
  await page.getByRole("button", { name: await localized(page, submitName), exact: true }).click();
  if (index < 2) {
    await page.waitForFunction(() => window.__qaStage === "break", null, { timeout: 15000 });
    await page.getByRole("button", { name: await localized(page, "Continue to the next video"), exact: true }).click();
    await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
  } else {
    await page.waitForFunction(() => window.__qaStage === "final", null, { timeout: 15000 });
  }
}

async function submitLongOverall(page, locale) {
  await page.locator("h1").filter({ hasText: /Your viewing experience|시청 경험/ }).waitFor({ timeout: 15000 });
  const state = await page.evaluate(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json()));
  const ids = state.items.map(item => item.id);
  check(ids.includes("control_texture") && ids.includes("timing"), `${locale} final overall controls missing`);
  const prssItems = state.items.filter(item => item.group === "prss" || item.id.startsWith("prss_"));
  check(prssItems.length > 0, `${locale} final overall PRSS items missing`);
  for (const item of prssItems) {
    check(item.points === 7, `${locale} final PRSS item ${item.id} did not expose 7 points`);
    check(await page.locator(`input[name="${item.id}"]`).count() === 7, `${locale} final PRSS item ${item.id} did not render 7 choices`);
  }
  await captureResponsive(page, `${locale}-long-final-overall`);
  await assertNoOverflow(page, `${locale} long final overall`);
  for (const item of state.items) {
    if (item.type === "rating") await page.locator(`input[name="${item.id}"][value="${item.na ? "na" : "4"}"]`).check();
    if (item.type === "choice") await page.locator(`input[name="${item.id}"]`).first().check();
    if (item.type === "text") await page.locator(`textarea[aria-label*="${item.text.slice(0, 12)}"], textarea`).fill(`Hierarchy ${locale} final comments`);
  }
  await page.getByRole("button", { name: /Submit your experience|제출/ }).click();
  await page.getByRole("heading", { name: /Thank you for taking part|완료|참여/ }).waitFor({ timeout: 15000 });
  const exported = await page.evaluate(() => fetch(new URL("api/study/export", location.href), { credentials: "same-origin" }).then(r => r.json()));
  check(exported.state.video_surveys.length === 3, `${locale} export stored ${exported.state.video_surveys.length} video surveys`);
  check(exported.state.final_survey.answers.comments.includes(locale), `${locale} final comments missing from export`);
  return exported;
}

async function runLegacyRailRegression() {
  const browser = await chromium.launch({ executablePath, args: ["--no-sandbox"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(`legacy rail pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error" && !/Failed to load resource/.test(message.text())) {
      errors.push(`legacy rail console: ${message.text()}`);
    }
  });
  await allowFastMedia(page);
  await page.route("**/api/session", route => route.fulfill({
    json: {
      participant: "legacy-rail-fixture",
      assignment: { prepared_segment: "A", regenerated_segment: "B" },
      step: "view_prepared",
      flow_version: "legacy-ab-v1",
      clip_index: 0,
      clip_count: 1,
      session_id: "legacy-rail",
      config_hash: "legacy-config",
      items_provenance: "fixture",
      language: "en",
      languages: ["en", "ko"],
      language_locked: false,
      download: false,
      viewing_enabled: false,
    },
  }));
  await page.route("**/api/step/view_prepared?**", route => route.fulfill({
    json: {
      step: "view_prepared",
      flow_version: "legacy-ab-v1",
      clip_index: 0,
      clip_count: 1,
      segment: "A",
      duration_ms: 10000,
      families: ["human", "animal", "natural", "things", "music"],
    },
  }));
  await page.route("**/api/events", route => route.fulfill({ json: { ok: true } }));
  try {
    await page.goto(base);
    await page.locator("#rail li").first().waitFor({ timeout: 15000 });
    const names = await page.locator("#rail li .name").evaluateAll(nodes => nodes.map(node => node.textContent.trim()));
    check(names.length === 6, `legacy rail expected 6 entries, saw ${names.length}: ${JSON.stringify(names)}`);
    check(names.every(Boolean), `legacy rail contains blank entry: ${JSON.stringify(names)}`);
    check(!names.some(name => /overall|PRSS/i.test(name)), `legacy rail leaked overall/PRSS label: ${JSON.stringify(names)}`);
    await screenshot(page, "legacy-rail-resume.png");
    return { entries: names.length, names };
  } finally {
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
  }
}

async function runLocale(locale) {
  const browser = await chromium.launch({ executablePath, args: ["--no-sandbox"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(`${locale} pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error" && !/Failed to load resource/.test(message.text())) errors.push(`${locale} console: ${message.text()}`);
  });
  await allowFastMedia(page);
  await installStateCapture(page);
  try {
    await page.goto(base);
    await page.getByRole("heading", { name: /Watch|보기/ }).waitFor({ timeout: 15000 });
    if (locale === "ko") await page.getByRole("button", { name: "한국어" }).click();
    for (let clip = 0; clip < 2; clip += 1) {
      await finishShortViewing(page, "art");
      await answerShortSurvey(page, "art", ["art"], `${locale} clip ${clip + 1} original ART`, {
        verifyDraft: clip === 0,
        expectFresh: clip === 1,
      });
      await completeVisualAndAuditory(page);
      await finishShortViewing(page, "survey");
      await answerShortSurvey(page, "survey", ["art", "caption"], `${locale} clip ${clip + 1} updated-caption survey`);
    }
    await answerOverallPRSS(page, locale);
    await continueToLongPhase(page, locale);
    for (let index = 0; index < 3; index += 1) await finishLongVideo(page, index, locale);
    const exported = await submitLongOverall(page, locale);
    await writeFile(resolve(output, `${locale}-hierarchy-export.json`), JSON.stringify(exported, null, 2));
    return {
      locale,
      participant: await page.evaluate(() => window.__qaParticipant),
      videoSurveys: exported.state.video_surveys.length,
      finalStage: exported.state.stage,
      shortFlow: "two clips: original ART, updated ART+caption, then PRSS overall",
    };
  } finally {
    await context.close().catch(() => {});
    await browser.close().catch(() => {});
  }
}

let server;
try {
  if (!process.env.REGEN_HIERARCHY_URL) {
    server = startServer();
    await server.ready;
  }
  await waitForHTTP(base);
  const locales = process.env.REGEN_HIERARCHY_LOCALES
    ? process.env.REGEN_HIERARCHY_LOCALES.split(",").map(locale => locale.trim()).filter(Boolean)
    : ["en", "ko"];
  evidence.journeys = [];
  evidence.legacyRail = await runLegacyRailRegression();
  for (const locale of locales) evidence.journeys.push(await runLocale(locale));
  check(errors.length === 0, errors.join("\n"));
  evidence.generatedAt = new Date().toISOString();
  evidence.output = output;
  evidence.mediaTiming = "Browser media completion is simulated for both 10s short clips and five-minute long videos. Loopback QA endpoints seed only playback coverage checkpoints; viewing, survey, regeneration, draft, instrument, export, and localization APIs remain real.";
  await writeFile(resolve(output, "browser-survey-hierarchy-result.json"), JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify(evidence));
} catch (error) {
  await writeFile(resolve(output, "browser-survey-hierarchy-error.json"), JSON.stringify({ error: error.message, stack: error.stack, errors, evidence }, null, 2));
  throw error;
} finally {
  if (server) server.child.kill("SIGTERM");
}

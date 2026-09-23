// Browser protocol QA for the integrated short phase and linked long phase.
// Short playback timing uses real media time; long-video completion uses a loopback QA coverage helper.
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const testsRoot = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(testsRoot, "..");
const repoRoot = resolve(packageRoot, "../../..");
const output = process.env.REGEN_PROTOCOL_OUTPUT || "/tmp/regen-protocol-qa";
const python = process.env.REGEN_PROTOCOL_PYTHON || resolve(repoRoot, ".venv/bin/python");
const playwrightModule = process.env.PLAYWRIGHT_MODULE || "/tmp/atlas-qa/node_modules/playwright-core/index.mjs";
const chromiumExecutable = process.env.CHROMIUM_EXECUTABLE || "/home/june/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";
const locale = process.env.REGEN_PROTOCOL_LOCALE || "en";
await mkdir(output, { recursive: true });

const loaded = await import(playwrightModule);
const { chromium } = loaded.chromium ? loaded : loaded.default;
const evidence = { checks: [], consoleErrors: [], pageErrors: [] };
const check = (condition, message) => {
  if (!condition) throw new Error(message);
  evidence.checks.push(message);
};

function startServer() {
  const child = spawn(python, ["src/dpo/regen/tests/preview_protocol.py"], {
    cwd: repoRoot,
    env: {
      ...process.env,
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONPATH: [resolve(repoRoot, "src"), process.env.PYTHONPATH].filter(Boolean).join(":"),
    },
  });
  let log = "";
  const ready = new Promise((resolveReady, rejectReady) => {
    const timer = setTimeout(() => rejectReady(new Error(`server did not become ready: ${log.slice(-2000)}`)), 90000);
    const watch = data => {
      log += data.toString();
      for (const line of log.split("\n")) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.url) {
            clearTimeout(timer);
            resolveReady(parsed);
          }
        } catch {}
      }
    };
    child.stdout.on("data", watch);
    child.stderr.on("data", watch);
    child.once("exit", code => {
      clearTimeout(timer);
      rejectReady(new Error(`server exited with ${code}: ${log.slice(-2000)}`));
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
      last = String(response.status);
    } catch (error) {
      last = error.message;
    }
    await new Promise(resolveWait => setTimeout(resolveWait, 250));
  }
  throw new Error(`server did not answer ${url}: ${last}`);
}

async function installCapture(page) {
  await page.addInitScript(() => {
    try {
      if (window.name === "regen-protocol-reset") {
        window.name = "";
        localStorage.clear();
        sessionStorage.clear();
      }
    } catch {}
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      response.clone().json().then(body => {
        if (body.participant) window.__qaParticipant = body.participant;
        if (body.step) window.__qaStep = body.step;
        if (body.protocol_version) window.__qaProtocolVersion = body.protocol_version;
        if (Number.isInteger(body.clip_index)) window.__qaClipIndex = body.clip_index;
        if (Number.isInteger(body.clip_count)) window.__qaClipCount = body.clip_count;
        if (body.event_context) window.__qaEventContext = body.event_context;
      }).catch(() => {});
      return response;
    };
  });
}

async function screenshot(page, name) {
  await page.screenshot({ path: resolve(output, name), fullPage: true });
}

async function participant(page) {
  return page.evaluate(() => window.__qaParticipant);
}

async function protocolLog(page, base) {
  const person = await participant(page);
  return protocolLogFor(base, person);
}

async function protocolLogFor(base, person) {
  return fetch(`${base}/qa/protocol-log/${encodeURIComponent(person)}`).then(response => response.json());
}

async function currentShortStep(page, name) {
  return page.evaluate(async step => {
    const response = await fetch(`/api/step/${step}?participant=${encodeURIComponent(window.__qaParticipant)}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `${step} failed`);
    return body;
  }, name);
}

async function postShort(page, path, payload) {
  return page.evaluate(async ([target, body]) => {
    const response = await fetch(target, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ participant: window.__qaParticipant, ...body }),
    });
    const parsed = await response.json().catch(() => ({}));
    return { ok: response.ok, status: response.status, body: parsed };
  }, [path, payload]);
}

async function assertSkippedViewingBlocked(page) {
  const response = await postShort(page, "/api/viewing", {
    clip_index: 0,
    step: "view_prepared",
    started_at: "2026-09-22T00:00:00Z",
    ended_at: "2026-09-22T00:00:10Z",
  });
  check(response.status === 409, `short viewing skip is rejected with 409, got ${response.status}`);
  check(response.body.playback?.coverage?.length === 0, "short viewing skip returns empty playback coverage");
}

async function startShortPlayback(page) {
  await page.getByRole("button", { name: /Start|시작/ }).click();
  await page.locator("#screen-viewing:not([hidden])").waitFor({ timeout: 15000 });
  await page.locator("#video").evaluate(video => video.play());
}

async function verifyHiddenPauseDuringPendingAck(page) {
  let delayed = 0;
  let released = false;
  await page.route("**/api/playback", async route => {
    delayed += 1;
    await new Promise(resolveDelay => setTimeout(resolveDelay, 1500));
    released = true;
    await route.continue().catch(() => {});
  });
  await page.waitForFunction(() => document.querySelector("#video")?.currentTime >= 1, null, { timeout: 10000 });
  const deadline = Date.now() + 5000;
  while (delayed === 0 && Date.now() < deadline) await page.waitForTimeout(100);
  check(delayed > 0, "hidden-pause probe delayed a playback checkpoint in flight");
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.waitForFunction(() => document.querySelector("#video")?.paused === true, null, { timeout: 5000 });
  await page.locator("#viewing-interrupted:not([hidden])").waitFor({ timeout: 5000 });
  check(released, "hidden-pause probe released delayed playback checkpoint");
  await page.unroute("**/api/playback");
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, value: false });
    document.dispatchEvent(new Event("visibilitychange"));
  });
  await page.locator("#interrupted-resume").click();
  await page.waitForFunction(() => document.querySelector("#video")?.paused === false, null, { timeout: 10000 });
  check(true, "hidden during pending checkpoint pauses, shows resume overlay, and resumes");
}

async function verifyReloadCheckpoint(page) {
  await page.waitForFunction(() => document.querySelector("#video")?.currentTime >= 3, null, { timeout: 15000 });
  await page.waitForTimeout(1400);
  const before = await currentShortStep(page, "view_prepared");
  check(before.playback.position_ms >= 2500, `server acknowledged short checkpoint at ${before.playback.position_ms}ms`);
  await page.reload();
  await page.getByRole("button", { name: /Start|시작/ }).waitFor({ timeout: 15000 });
  await page.getByRole("button", { name: /Start|시작/ }).click();
  await page.locator("#screen-viewing:not([hidden])").waitFor({ timeout: 15000 });
  await page.waitForFunction(ms => document.querySelector("#video")?.currentTime * 1000 >= ms - 750, before.playback.position_ms, { timeout: 10000 });
  const restored = await page.locator("#video").evaluate(video => video.currentTime * 1000);
  check(restored >= before.playback.position_ms - 750, `reload restores short checkpoint near ${before.playback.position_ms}ms`);
  await page.locator("#video").evaluate(video => video.pause());
}

async function resetForFreshSession(page, base, expectedShortCount) {
  await page.evaluate(() => { window.name = "regen-protocol-reset"; });
  await page.goto(base);
  await page.getByRole("heading", { name: /Watch|보기/ }).waitFor({ timeout: 15000 });
  await chooseLocale(page, locale);
  check(await page.evaluate(() => window.__qaProtocolVersion) === 3, "fresh browser session enrols with client_protocol 3");
  check(await page.evaluate(() => window.__qaClipCount) === expectedShortCount, `fresh short UI receives ${expectedShortCount} configured clip(s)`);
}

async function chooseLocale(page, tag) {
  if (tag !== "ko") return;
  const korean = page.getByRole("button", { name: "한국어" });
  if (await korean.isVisible().catch(() => false)) await korean.click();
  await page.waitForFunction(() => document.documentElement.lang === "ko", null, { timeout: 10000 });
}

async function outboxCount(page) {
  return page.evaluate(() => Object.keys(localStorage).filter(key => key.startsWith("regen.events:")).length);
}

async function outboxSnapshot(page) {
  return page.evaluate(() => Object.keys(localStorage)
    .filter(key => key.startsWith("regen.events:"))
    .sort()
    .map(key => {
      let body = null;
      try { body = JSON.parse(localStorage.getItem(key)); }
      catch { body = localStorage.getItem(key); }
      return { key, body };
    }));
}

async function waitShortViewingEnds(page) {
  await page.waitForFunction(() => window.__qaStep === "art", null, { timeout: 25000 });
  const log = await protocolLog(page, new URL(page.url()).origin);
  check(log.viewings.length === 1, `one original short viewing is recorded, saw ${log.viewings.length}`);
  check(log.viewings[0].protocol_version === 3, "short viewing records protocol v3 coverage");
  check(log.viewings[0].coverage?.length > 0, "short viewing stores watched coverage");
}

async function answerShortSurvey(page, pageName) {
  await page.locator("#screen-survey:not([hidden])").waitFor({ timeout: 15000 });
  const detail = await currentShortStep(page, pageName);
  for (const item of detail.blocks.flatMap(block => block.items)) {
    await page.locator(`input[name="${item.id}"][value="4"]`).check();
  }
  await page.locator("#survey-submit").click();
  return detail;
}

async function verifyOutboxRecovery(page, base) {
  await page.waitForFunction(() => window.__qaStep === "visual", null, { timeout: 15000 });
  let failed = false;
  await page.route("**/api/events", route => {
    if (!failed) {
      failed = true;
      return route.fulfill({ status: 500, json: { error: "qa injected event failure" } });
    }
    return route.continue();
  });
  await page.locator("#plate").focus();
  await page.keyboard.press("Enter");
  await page.waitForTimeout(1800);
  const queuedAfterFailure = await page.evaluate(() =>
    Object.keys(localStorage).filter(key => key.startsWith("regen.events:")).length
  );
  check(failed && queuedAfterFailure > 0, "failed event upload leaves a durable browser outbox entry");
  await page.unroute("**/api/events");
  await page.evaluate(() => window.dispatchEvent(new Event("online")));
  await page.waitForFunction(() => fetch(`/qa/protocol-log/${encodeURIComponent(window.__qaParticipant)}`)
    .then(response => response.json())
    .then(log => log.events.some(event => event.type === "point.placed" && event.stage === "visual")),
    null,
    { timeout: 10000 }
  );
  evidence.outboxRemainingAfterRecovery = await page.evaluate(() =>
    Object.keys(localStorage).filter(key => key.startsWith("regen.events:")).length
  );
  const log = await protocolLog(page, base);
  check(log.events.some(event => event.type === "point.placed" && event.stage === "visual"), "recovered outbox uploads the visual point with stage context");
}

async function verifyDuplicateAck(page, base) {
  const context = await page.evaluate(() => window.__qaEventContext);
  const event = {
    ...context,
    event_id: "browser-duplicate-ack",
    type: "caption.shown",
    at: new Date().toISOString(),
    index: 0,
    text: "Duplicate ack probe.",
  };
  const first = await postShort(page, "/api/events", { events: [event] });
  const second = await postShort(page, "/api/events", { events: [event] });
  check(first.body.acknowledged?.includes(event.event_id), "first duplicate-ack probe is acknowledged");
  check(second.body.acknowledged?.includes(event.event_id), "second duplicate-ack probe is acknowledged");
  const log = await protocolLog(page, base);
  const stored = log.events.filter(row => row.event_id === event.event_id);
  check(stored.length === 1, `duplicate event is stored exactly once, saw ${stored.length}`);
}

async function completeVisualAuditory(page) {
  while (await page.locator("#plate .point").count() < 2) {
    await page.locator("#plate").focus();
    await page.keyboard.press("Enter");
  }
  await page.locator("#visual-next").click();
  await page.waitForFunction(() => window.__qaStep === "auditory", null, { timeout: 15000 });
  await page.locator('#lanes input[name^="family-"]').first().waitFor({ timeout: 15000 });
  const names = await page.locator('#lanes input[name^="family-"]').evaluateAll(inputs => [...new Set(inputs.map(input => input.name))]);
  check(names.length === 5, `auditory page renders five sound families, saw ${names.length}`);
  for (const name of names) await page.locator(`input[name="${name}"]`).first().check();
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  await page.waitForFunction(() => window.__qaStep === "view_regenerated", null, { timeout: 30000 });
}

async function completeShortViewingFast(page, step) {
  const detail = await currentShortStep(page, step);
  const context = detail.event_context.context_id;
  for (let sequence = 0; sequence <= 10; sequence += 1) {
    if (sequence > 0) await page.waitForTimeout(1050);
    const position = Math.min(detail.duration_ms, sequence * 1000);
    const tick = await postShort(page, "/api/playback", {
      clip_index: detail.clip_index,
      step,
      context_id: context,
      sequence,
      position_ms: position,
      playing: position < detail.duration_ms,
      hidden: false,
    });
    check(tick.ok, `${step} timed checkpoint ${sequence} accepted: ${tick.status} ${tick.body?.error || ""}`);
  }
  const result = await postShort(page, "/api/viewing", {
    clip_index: detail.clip_index,
    step,
    started_at: "2026-09-22T00:01:00Z",
    ended_at: "2026-09-22T00:01:10Z",
  });
  check(result.ok, `${step} fast completion accepted`);
  await page.evaluate(stepName => { window.__qaStep = stepName; }, result.body.step);
  await page.reload();
  await page.waitForFunction(stepName => window.__qaStep === stepName, result.body.step, { timeout: 15000 });
}

async function finishShortPhase(page, base) {
  const shortParticipant = await participant(page);
  await completeShortViewingFast(page, "view_regenerated");
  await answerShortSurvey(page, "survey");
  await page.waitForFunction(() => window.__qaStep !== "survey", null, { timeout: 15000 });
  const clipCount = await page.evaluate(() => window.__qaClipCount);
  while (await page.evaluate(() => window.__qaStep === "view_prepared")) {
    await completeShortViewingFast(page, "view_prepared");
    await answerShortSurvey(page, "art");
    await page.waitForFunction(() => window.__qaStep === "visual", null, { timeout: 15000 });
    await completeVisualAuditory(page);
    await completeShortViewingFast(page, "view_regenerated");
    await answerShortSurvey(page, "survey");
    await page.waitForFunction(() => window.__qaStep !== "survey", null, { timeout: 15000 });
  }
  await page.waitForFunction(() => window.__qaStep === "overall", null, { timeout: 15000 });
  const overall = await answerShortSurvey(page, "overall");
  check(overall.blocks.map(block => block.id).join(",") === "prss", "short phase ends with one overall PRSS survey");
  await page.waitForURL(/\/viewing\/.+\/$/, { timeout: 30000 });
  const log = await protocolLogFor(base, shortParticipant);
  check(log.viewings.length === clipCount * 2, `short phase stores two viewings per clip for ${clipCount} clip(s)`);
  check(log.responses.length === clipCount * 2 + 1, `short phase stores per-viewing surveys plus one overall survey for ${clipCount} clip(s)`);
}

async function studyAction(page, action, data = {}) {
  return page.evaluate(async ([name, body]) => {
    const state = await fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json());
    const response = await fetch(new URL(`api/study/${name}`, location.href), {
      method: "POST",
      headers: { "content-type": "application/json", "x-study-request": "1" },
      credentials: "same-origin",
      body: JSON.stringify({ revision: state.revision, key: crypto.randomUUID(), data: body }),
    });
    const parsed = await response.json();
    if (!response.ok) throw new Error(parsed.error || `${name} failed`);
    return parsed;
  }, [action, data]);
}

function answersFor(items) {
  return Object.fromEntries(items.map(item => [item.id, item.type === "rating" ? 4 : item.type === "choice" ? item.options[0] : "Browser protocol QA"]));
}

async function verifyLongVariableSurveyFlow(page, expectedLongCount) {
  await page.getByRole("button", { name: /Start viewing experience|영상 시청 시작/ }).click();
  await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
  for (let index = 0; index < expectedLongCount; index += 1) {
    const state = await page.evaluate(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json()));
    check(state.viewing_total === expectedLongCount, `long phase advertises ${expectedLongCount} video(s)`);
    const completed = await page.evaluate(async () => {
      const response = await fetch(new URL("qa/complete-current-video", location.href), {
        method: "POST",
        headers: { "x-study-request": "1" },
        credentials: "same-origin",
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "QA completion failed");
      return body;
    });
    check(completed.coverage_ms >= 299000, `long video ${index + 1} has QA coverage for survey eligibility`);
    await studyAction(page, "video-ended", { video_id: state.video.id });
    await page.waitForFunction(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json()).then(s => s.stage === "video-survey"), null, { timeout: 15000 });
    const surveyState = await page.evaluate(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json()));
    check(surveyState.items.map(item => item.id).join(",") === "accurate,texture,context,personal", `long video ${index + 1} uses per-video survey items`);
    await studyAction(page, "video-survey", { video_id: surveyState.video.id, answers: answersFor(surveyState.items) });
    if (index + 1 < expectedLongCount) await studyAction(page, "continue");
  }
  const finalState = await page.evaluate(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json()));
  check(finalState.stage === "final", "long phase reaches final survey after configured video count");
  await studyAction(page, "final-survey", answersFor(finalState.items));
  await page.waitForFunction(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" }).then(r => r.json()).then(s => s.stage === "done"), null, { timeout: 15000 });
  const exported = await page.evaluate(() => fetch(new URL("api/study/export", location.href), { credentials: "same-origin" }).then(r => r.json()));
  check(exported.state.video_surveys.length === expectedLongCount, `export stores ${expectedLongCount} long video survey(s)`);
  return exported;
}

let server;
let browser;
let page;
try {
  server = startServer();
  const fixture = await server.ready;
  await waitForHTTP(fixture.url);
  evidence.fixture = fixture;
  evidence.locale = locale;
  browser = await chromium.launch({ executablePath: chromiumExecutable, args: ["--no-sandbox"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  page = await context.newPage();
  page.on("pageerror", error => evidence.pageErrors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error" && !/Failed to load resource/.test(message.text())) evidence.consoleErrors.push(message.text());
  });
  await installCapture(page);
  await page.goto(fixture.url);
  await page.getByRole("heading", { name: /Watch|보기/ }).waitFor({ timeout: 15000 });
  await chooseLocale(page, locale);
  check(await page.evaluate(() => window.__qaProtocolVersion) === 3, "browser enrols with client_protocol 3");
  check(await page.evaluate(() => window.__qaClipCount) === fixture.short_count, `short UI receives ${fixture.short_count} configured clip(s)`);
  await assertSkippedViewingBlocked(page);
  await startShortPlayback(page);
  await verifyHiddenPauseDuringPendingAck(page);
  await verifyReloadCheckpoint(page);
  await resetForFreshSession(page, fixture.url, fixture.short_count);
  await startShortPlayback(page);
  await waitShortViewingEnds(page);
  await answerShortSurvey(page, "art");
  await verifyOutboxRecovery(page, fixture.url);
  await verifyDuplicateAck(page, fixture.url);
  await completeVisualAuditory(page);
  await finishShortPhase(page, fixture.url);
  const exported = await verifyLongVariableSurveyFlow(page, fixture.long_count);
  await page.waitForFunction(() =>
    Object.keys(localStorage).filter(key => key.startsWith("regen.events:")).length === 0,
    null,
    { timeout: 10000 }
  ).catch(() => {});
  evidence.finalOutboxPending = await outboxSnapshot(page);
  evidence.finalOutboxRemaining = evidence.finalOutboxPending.length;
  check(evidence.finalOutboxRemaining === 0, `event outbox drains by final handoff, remaining ${evidence.finalOutboxRemaining}`);
  check(evidence.pageErrors.length === 0, `no page errors: ${evidence.pageErrors.join("; ")}`);
  check(evidence.consoleErrors.length === 0, `no unexpected console errors: ${evidence.consoleErrors.join("; ")}`);
  evidence.longExport = {
    stage: exported.state.stage,
    videoSurveys: exported.state.video_surveys.length,
  };
  evidence.mediaTiming = "First short original viewing uses actual 10-second browser playback and server protocol-v3 checkpoints. Later short viewings use direct protocol-v3 checkpoint posts to keep the variable-count journey bounded. Long-video coverage is granted only by the loopback QA endpoint before real survey/final API submissions.";
  evidence.generatedAt = new Date().toISOString();
  await screenshot(page, "protocol-complete.png");
  await writeFile(resolve(output, "browser-protocol-result.json"), JSON.stringify(evidence, null, 2));
  console.log(JSON.stringify(evidence));
  await context.close();
} catch (error) {
  evidence.error = error.message;
  evidence.stack = error.stack;
  if (page) {
    evidence.failureUrl = page.url();
    evidence.failureText = await page.locator("body").innerText().catch(errorText => errorText.message);
    evidence.failureVideo = await page.locator("#video, video").first().evaluate(video => ({
      currentTime: video.currentTime,
      duration: video.duration,
      paused: video.paused,
      ended: video.ended,
      readyState: video.readyState,
      networkState: video.networkState,
      src: video.currentSrc || video.src,
    })).catch(errorVideo => ({ error: errorVideo.message }));
    evidence.failureStep = await page.evaluate(() => ({
      step: window.__qaStep,
      clipIndex: window.__qaClipIndex,
      clipCount: window.__qaClipCount,
      protocol: window.__qaProtocolVersion,
      participant: window.__qaParticipant,
    })).catch(errorStep => ({ error: errorStep.message }));
    evidence.failureOutboxPending = await outboxSnapshot(page).catch(errorOutbox => [{ error: errorOutbox.message }]);
    await screenshot(page, "protocol-failure.png").catch(() => {});
    if (page.url().startsWith("http")) {
      evidence.failureProtocolLog = await protocolLog(page, new URL(page.url()).origin).catch(errorLog => ({ error: errorLog.message }));
    }
  }
  await writeFile(resolve(output, "browser-protocol-error.json"), JSON.stringify(evidence, null, 2));
  throw error;
} finally {
  if (browser) await browser.close().catch(() => {});
  if (server) {
    server.child.kill("SIGTERM");
    await writeFile(resolve(output, "server.log"), server.log());
  }
}

// Full real-media browser QA for regen's survey hierarchy.
// It uses real playback and real API/export writes; no synthetic clock or video completion.
import { spawn, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const playwrightModule = process.env.PLAYWRIGHT_MODULE || "/tmp/atlas-qa/node_modules/playwright-core/index.mjs";
const loaded = await import(
  playwrightModule.endsWith("playwright-core") ? `${playwrightModule}/index.js` : playwrightModule
);
const playwright = loaded.chromium ? loaded : loaded.default;
const { chromium } = playwright;
const output = process.env.REGEN_REAL_OUTPUT || "/tmp/regen-real-study-qa";
const python = process.env.REGEN_QA_PYTHON || resolve(repo, ".venv/bin/python");
const chromiumExecutable = process.env.CHROMIUM_EXECUTABLE || "/home/june/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";
const publicRun = process.env.REGEN_REAL_PUBLIC === "1";
let serverUrl = process.env.REGEN_REAL_URL;
if (serverUrl && process.env.REGEN_REAL_CODE_FILE) {
  const link = new URL(serverUrl);
  link.searchParams.set("code", (await readFile(process.env.REGEN_REAL_CODE_FILE, "utf8")).trim());
  serverUrl = link.href;
}
const qaPrefix = process.env.REGEN_REAL_QA_PREFIX || `qa-public-${Date.now()}`;
await mkdir(output, { recursive: true });

const errors = [];
const checks = [];
const steeringTimings = [];
const check = (condition, message) => {
  if (!condition) throw new Error(message);
  checks.push(message);
};
const stamp = () => new Date().toISOString();
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const progress = fields => console.log(JSON.stringify({ at: stamp(), ...fields }));

let server;
let log = "";
let sourceMetadata;
async function launchServer() {
  if (serverUrl) return { url: serverUrl, root: null, launched: false };
  server = spawn(python, ["-u", "-m", "dpo.regen.tests.preview_real_study"], {
    cwd: repo,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", REGEN_REAL_KEEP: process.env.REGEN_REAL_KEEP || "1" },
  });
  return await new Promise((accept, reject) => {
    const timer = setTimeout(() => reject(new Error(`Real fixture startup timed out: ${log}`)), 120000);
    server.stdout.on("data", data => {
      log += data.toString();
      for (const line of log.split("\n")) {
        try {
          const info = JSON.parse(line);
          if (info.url) {
            clearTimeout(timer);
            accept({ ...info, launched: true });
          }
        } catch {}
      }
    });
    server.stderr.on("data", data => { log += data.toString(); });
    server.once("exit", code => {
      clearTimeout(timer);
      reject(new Error(`Real fixture exited ${code}: ${log}`));
    });
  });
}

async function eventually(fn, label, timeout = 30000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await fn()) return;
    await sleep(250);
  }
  throw new Error(label);
}

async function screenshot(page, name) {
  await page.screenshot({ path: resolve(output, name), fullPage: true });
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function captionAudit(exportJson) {
  const videosByHash = new Map(exportJson.state.viewing.map(video => [sha256(stableJson(video)), video.id]));
  const byVideo = Object.fromEntries(exportJson.state.viewing.map(video => [
    video.id,
    { jobs: 0, nonFallback: 0, fallback: 0, modelNotConfigured: 0, exposures: 0, generatedExposures: 0 },
  ]));
  for (const exposure of exportJson.exposures) {
    if (!byVideo[exposure.video_id]) continue;
    byVideo[exposure.video_id].exposures += 1;
    if (exposure.fallback === false && exposure.job_id) byVideo[exposure.video_id].generatedExposures += 1;
  }
  for (const row of exportJson.jobs) {
    const spec = JSON.parse(row.spec);
    const video = byVideo[videosByHash.get(spec.video_hash)];
    if (!video) continue;
    video.jobs += 1;
    const result = row.result ? JSON.parse(row.result) : null;
    if (!result) continue;
    if (result.reason === "model_not_configured") video.modelNotConfigured += 1;
    if (result.fallback) video.fallback += 1;
    else video.nonFallback += 1;
  }
  return byVideo;
}

function shortClipAudit(exportJson) {
  const rows = exportJson.responses || [];
  const result = {};
  for (const row of rows) {
    const extra = row.extra || row;
    if (!["clip_original", "clip_updated"].includes(extra.scope)) continue;
    const key = `clip-${extra.clip_index}-${extra.scope}`;
    result[key] = {
      page: row.page,
      scope: extra.scope,
      clipIndex: extra.clip_index,
      condition: extra.condition,
      answerCount: Object.keys(row.responses || {}).length,
      clipId: extra.clip_id,
    };
  }
  return result;
}

async function surveyRadios(page, value = 4) {
  const names = await page.locator("#survey-blocks input[type=radio]").evaluateAll(
    inputs => [...new Set(inputs.map(input => input.name))]
  );
  check(names.length > 0, "Legacy survey rendered radio items");
  for (const name of names) await page.locator(`input[name="${name}"][value="${value}"]`).check();
}

async function finishLegacyViewing(page, expectedStep) {
  progress({ event: "short-viewing-start", expectedStep });
  await page.getByRole("button", { name: /Start|시작/ }).click({timeout: publicRun ? 180000 : 30000});
  await page.locator("#screen-viewing:not([hidden])").waitFor();
  await page.locator("#video").evaluate(video => video.play());
  await page.locator("#video").evaluate(video => new Promise(resolve => {
    if (video.ended) resolve();
    else video.addEventListener("ended", resolve, { once: true });
  }), { timeout: 30000 });
  await page.locator("#screen-viewing").waitFor({ state: "hidden", timeout: 15000 });
  await page.waitForFunction(step => window.__qaStep === step, expectedStep, { timeout: 10000 });
  progress({ event: "short-viewing-complete", expectedStep });
}

async function surveyDetail(page, pageName) {
  return page.evaluate(async name => {
    const response = await fetch(`/api/step/${name}?participant=${encodeURIComponent(window.__qaParticipant)}`);
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || `${name} detail failed`);
    return body;
  }, pageName);
}

async function answerShortSurvey(page, pageName, expectedBlocks, label) {
  await page.locator("#screen-survey:not([hidden])").waitFor({ timeout: 60000 });
  const detail = await surveyDetail(page, pageName);
  const blocks = detail.blocks.map(block => block.id);
  check(JSON.stringify(blocks) === JSON.stringify(expectedBlocks), `${label} blocks match hierarchy`);
  check(!blocks.includes("prss"), `${label} does not include per-clip PRSS`);
  await surveyRadios(page);
  await screenshot(page, `${label.replaceAll(/[^a-z0-9]+/gi, "-").toLowerCase()}.png`);
  progress({ event: "short-survey-submit", label, blocks });
  await page.locator("#survey-submit").click();
}

async function answerShortOverall(page, locale) {
  await page.waitForFunction(() => window.__qaStep === "overall", null, { timeout: 30000 });
  await page.locator("#screen-survey:not([hidden])").waitFor({ timeout: 60000 });
  const detail = await surveyDetail(page, "overall");
  const blocks = detail.blocks.map(block => block.id);
  check(JSON.stringify(blocks) === JSON.stringify(["prss"]), `${locale} short overall is PRSS-only`);
  check(detail.scale.points === 7, `${locale} short overall PRSS uses 7-point scale`);
  const names = await page.locator("#survey-blocks input[type=radio]").evaluateAll(
    inputs => [...new Set(inputs.map(input => input.name))]
  );
  check(names.length === detail.blocks[0].items.length, `${locale} short overall rendered all PRSS items`);
  for (const name of names) check(await page.locator(`input[name="${name}"]`).count() === 7, `${locale} ${name} has 7 choices`);
  await screenshot(page, `${locale}-short-overall-prss.png`);
  await surveyRadios(page);
  progress({ event: "short-overall-submit", locale, prssItems: names.length });
  await page.locator("#survey-submit").click();
  await page.waitForURL(/\/viewing\/.+\/$/, { timeout: 120000 });
}

async function completeVisual(page) {
  await page.locator("#screen-visual:not([hidden])").waitFor({timeout: 60000});
  await page.waitForFunction(() => document.querySelector("#plate img")?.naturalWidth > 0);
  await page.locator("#plate").focus();
  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  await page.locator("#visual-next").click();
  await page.locator("#lanes input[type=radio]").first().waitFor({state: "visible", timeout: 60000});
}

async function completeAuditory(page) {
  await page.locator("#lanes input[type=radio]").first().waitFor({state: "visible", timeout: 60000});
  const names = await page.locator("#lanes input[type=radio]").evaluateAll(
    inputs => [...new Set(inputs.map(input => input.name))]
  );
  check(names.length === 5, "Auditory calibration rendered five fixed families");
  for (const name of names) await page.locator(`input[name="${name}"]`).first().check();
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  await page.waitForFunction(() => window.__qaStep === "regenerating", null, { timeout: 10000 });
}

async function continueToViewing(page) {
  await page.getByRole("heading", { name: /Your calibration is saved|사전 응답이 저장되었습니다/ }).waitFor({ timeout: 60000 });
  await page.getByRole("button", { name: /Start viewing experience|영상 시청 시작/ }).click();
  await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
}

async function viewingLabel(page, text) {
  return page.evaluate(async source => {
    const copy = await fetch(new URL("api/study/strings", location.href), { credentials: "same-origin" })
      .then(response => response.json());
    return document.documentElement.lang === "ko" ? (copy[source] || source) : source;
  }, text);
}

async function clickViewingButton(page, text) {
  await page.getByRole("button", { name: await viewingLabel(page, text), exact: true }).click();
}

async function waitState(page, predicateSource, timeout = 15000) {
  await page.waitForFunction(
    source => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" })
      .then(response => response.json())
      .then(state => Function("state", `return (${source})(state);`)(state)),
    predicateSource,
    { timeout }
  );
}

async function applyControls(page, index) {
  if (index === 0) {
    if (await page.locator(".study-controls details").getAttribute("open") === null) {
      await page.getByText(await viewingLabel(page, "Detail presets"), { exact: true }).click();
    }
    await clickViewingButton(page, "Both detailed");
    await waitState(page, "state => state.axes.texture === 1 && state.axes.context === 1");
    await clickViewingButton(page, "Reset to calibration");
    await waitState(page, "state => state.axes.texture === 0.5 && state.axes.context === 0.5");
  } else if (index === 1) {
    const texture = page.locator('input[type="range"]').first();
    await texture.focus();
    await page.keyboard.press("ArrowRight");
    await waitState(page, "state => state.axes.texture > 0.5");
  } else {
    const texture = page.getByRole("slider").first(), context = page.getByRole("slider").nth(1);
    const textureBefore = await texture.inputValue(), contextBefore = await context.inputValue();
    const box = await context.boundingBox();
    check(Boolean(box), "Source/scene slider is present");
    await page.mouse.move(box.x + box.width * 0.7, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.95, box.y + box.height / 2, {steps: 4});
    await page.mouse.up();
    check(await context.inputValue() !== contextBefore, "Pointer drag changes source/scene detail");
    check(await texture.inputValue() === textureBefore, "Source/scene pointer drag leaves acoustic detail unchanged");
    await context.press("End");
    await waitState(page, "state => state.axes.context === 1");
    await clickViewingButton(page, "Fullscreen");
    await page.waitForFunction(() => Boolean(document.fullscreenElement), null, { timeout: 10000 });
    await clickViewingButton(page, "Exit fullscreen");
    await page.waitForFunction(() => !document.fullscreenElement, null, { timeout: 10000 });
  }
}

async function finishRealVideo(page, locale, index) {
  progress({ event: "long-video-start", locale, video: index + 1 });
  await page.locator("video").waitFor();
  await page.waitForFunction(() => document.querySelector("video").readyState >= 2, null, {timeout: 900000});
  const state = await page.evaluate(() => fetch(new URL("api/study/state", location.href), {
    credentials: "same-origin",
  }).then(response => response.json()));
  check(state.video.duration_ms >= 299000 && state.video.duration_ms <= 301000, `${locale} video ${index + 1} is five minutes`);
  if (publicRun) {
    // Observe actual rendering at two control settings, without substituting API
    // responses, playback clocks, caption text, or completion records.
    const playbackStart = await page.locator("video").evaluate(video => video.currentTime);
    await page.locator("video").evaluate(video => { video.play().catch(() => {}); });
    await page.waitForFunction(start => document.querySelector("video").currentTime > start + 2, playbackStart, {timeout: 120000});
    await page.getByText(await viewingLabel(page, "Detail presets"), {exact: true}).click();
    const discrete = state.detail_control?.version === "five-level/v1";
    for (const [preset, percent] of [["Both detailed", 100], ["Both brief", 0]]) {
      const changedAt = Date.now();
      await clickViewingButton(page, preset);
      const expectedLevel = discrete ? 1 + percent / 25 : percent;
      await page.waitForFunction(({expectedLevel, discrete}) => {
        const status = document.querySelector(".caption-level-status")?.textContent || "";
        const levels = [...status.matchAll(discrete ? /(\d+)\/5/g : /(\d+)%/g)].map(match => Number(match[1]));
        return document.querySelector("video").currentTime > 0 &&
          levels.length === 2 && levels.every(value => value === expectedLevel);
      }, {expectedLevel, discrete}, {timeout: 90000});
      check(true, `${locale} video ${index + 1} displayed a generated caption at ${expectedLevel}${discrete ? "/5" : "%"} detail`);
      const timing = {event:"steering-applied",locale,video:index+1,percent,
        ...(discrete ? {level:expectedLevel} : {}),ms:Date.now()-changedAt};
      steeringTimings.push(timing); console.log(JSON.stringify(timing));
    }
    await clickViewingButton(page, "Reset to calibration");
  }
  if (index === 0) {
    await page.locator("video").evaluate(video => video.play());
    await page.waitForFunction(() => document.querySelector("video").currentTime >= 121, null, { timeout: 140000 });
    await waitState(page, "state => state.position_ms >= 121000", 15000);
    const checkpoint = await page.evaluate(() => fetch(new URL("api/study/state", location.href), {
      credentials: "same-origin",
    }).then(response => response.json()).then(state => state.position_ms / 1000));
    await page.reload();
    await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
    await page.waitForFunction(checkpoint => {
      const video = document.querySelector("video");
      return video && video.readyState >= 1 && !video.seeking && Math.abs(video.currentTime - checkpoint) < 3;
    }, checkpoint, { timeout: publicRun ? 900000 : 20000 });
    const restored = await page.locator("video").evaluate(video => video.currentTime);
    check(Math.abs(restored - checkpoint) < 3, `${locale} reload restored acknowledged playback ${checkpoint}s: ${restored}`);
  }
  await applyControls(page, index);
  await page.locator("video").evaluate(video => video.play());
  await page.waitForFunction(() => document.querySelector("video").ended === true, null, { timeout: 330000 });
  await page.getByRole("button", { name: await viewingLabel(page, "Continue to this video's questions"), exact: true }).click({ timeout: 30000 });
  await page.waitForFunction(() => window.__qaStage === "video-survey", null, { timeout: 60000 });
  await page.locator("h1").filter({ hasText: /Video .*Your experience|영상 3개 중 .*시청 경험/ }).waitFor({ timeout: 60000 });
  progress({ event: "long-video-complete", locale, video: index + 1 });
}

async function submitPerVideoSurvey(page, locale, index) {
  const state = await page.evaluate(() => fetch(new URL("api/study/state", location.href), {
    credentials: "same-origin",
  }).then(response => response.json()));
  check(Boolean(state.survey_flow), `${locale} video ${index + 1} advertises per-video survey flow`);
  const ids = state.items.map(item => item.id);
  check(JSON.stringify(ids) === JSON.stringify(["accurate", "texture", "context", "personal"]), `${locale} video ${index + 1} asks four per-video items`);
  check(!state.items.some(item => item.block === "prss" || item.group === "prss"), `${locale} video ${index + 1} has no PRSS`);
  for (const item of state.items) await page.locator(`input[name="${item.id}"][value="4"]`).check();
  await screenshot(page, `${locale}-video-${index + 1}-survey.png`);
  const submitName = index === 2 ? "Save and continue to overall experience" : "Save this video's answers";
  progress({ event: "per-video-survey-submit", locale, video: index + 1, ids });
  await page.getByRole("button", { name: await viewingLabel(page, submitName), exact: true }).click();
  if (index < 2) {
    await page.waitForFunction(() => window.__qaStage === "break", null, { timeout: 60000 });
    await clickViewingButton(page, "Continue to the next video");
    await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 60000 });
  } else {
    await page.waitForFunction(() => window.__qaStage === "final", null, { timeout: 60000 });
    await page.locator("h1").filter({ hasText: /Your viewing experience|시청 경험/ }).waitFor({ timeout: 60000 });
  }
}

async function submitFinal(page, locale) {
  await page.setViewportSize({ width: 390, height: 844 });
  check(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), `${locale} final page fits mobile width`);
  const state = await page.evaluate(() => fetch(new URL("api/study/state", location.href), {
    credentials: "same-origin",
  }).then(response => response.json()));
  const prss = state.items.filter(item => item.block === "prss" || item.group === "prss");
  check(prss.length >= 1, `${locale} final survey includes PRSS`);
  for (const item of prss) {
    check(item.points === 7, `${locale} final ${item.id} exposes 7-point PRSS`);
    check(await page.locator(`input[name="${item.id}"]`).count() === 7, `${locale} final ${item.id} rendered 7 choices`);
  }
  const names = await page.locator('input[type="radio"]').evaluateAll(
    inputs => [...new Set(inputs.map(input => input.name))]
  );
  for (const name of names) {
    const rating = await page.locator(`input[name="${name}"][value="4"]`).count();
    await page.locator(rating ? `input[name="${name}"][value="4"]` : `input[name="${name}"]`).first().check();
  }
  await page.locator("textarea").fill(`Real ${locale} final survey QA ${stamp()}`);
  await page.reload();
  await page.locator("textarea").waitFor();
  check((await page.locator("textarea").inputValue()).includes(locale), `${locale} final survey draft survives reload`);
  await screenshot(page, `${locale}-final-overall.png`);
  progress({ event: "final-overall-submit", locale, prssItems: prss.length });
  await page.getByRole("button", { name: /Submit your experience|제출/ }).click();
  await page.getByRole("heading", { name: /Thank you for taking part|완료|참여/ }).waitFor({ timeout: 15000 });
  check(await page.locator(".receipt").count() === 1, `${locale} final receipt rendered`);
}

async function runLocale(base, locale) {
  const launch = {executablePath: chromiumExecutable, args: ["--no-sandbox"]};
  const options = {viewport: {width:1280,height:900}, acceptDownloads:true};
  let browser, context;
  if (publicRun) {
    const profile = resolve(output, `${locale}-browser`);
    await mkdir(profile, {recursive:true,mode:0o700});
    context = await chromium.launchPersistentContext(profile, {...launch,...options});
    browser = context.browser();
  } else {
    browser = await chromium.launch(launch);
    context = await browser.newContext(options);
  }
  const page = context.pages()[0] || await context.newPage();
  for (const other of context.pages()) if (other !== page) await other.close();
  page.on("pageerror", error => errors.push(`${locale} pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error" && !/Failed to load resource/.test(message.text())) {
      errors.push(`${locale} console: ${message.text()}`);
    }
  });
  await page.addInitScript(({publicRun, participant}) => {
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      if (publicRun && String(args[0]) === "/api/session" && args[1]?.body) {
        const body = JSON.parse(args[1].body);
        if (!body.participant) body.participant = participant;
        args[1].body = JSON.stringify(body);
      }
      const response = await originalFetch(...args);
      if (response.ok && response.headers.get("content-type")?.includes("application/json")) response.clone().json().then(body => {
        if (body.participant) window.__qaParticipant = body.participant;
        if (body.step) window.__qaStep = body.step;
        if (body.stage) window.__qaStage = body.stage;
      }).catch(() => {});
      return response;
    };
  }, {publicRun, participant: `${qaPrefix}-${locale}`});
  const network = [];
  page.on("requestfinished", request => {
    const timing = request.timing();
    network.push({path: new URL(request.url()).pathname, method: request.method(), ms: timing.responseEnd});
  });
  page.on("requestfailed", request => network.push({path: new URL(request.url()).pathname, error: request.failure()?.errorText}));
  const monitor = setInterval(async () => {
    try {
      const video = await page.locator("video:visible").first().evaluate(v => ({seconds:v.currentTime, paused:v.paused, readyState:v.readyState}));
      console.log(JSON.stringify({event:"playback",locale,...video,at:stamp()}));
    } catch {}
  }, 30000);
  try {
    let firstVideo = 0;
    let alreadyDone = false;
    if (publicRun && process.env.REGEN_REAL_RESUME === "1") {
      const resume = JSON.parse(await readFile(resolve(output, `${locale}-resume.json`), "utf8"));
      await page.goto(resume.url);
      const held = await page.evaluate(() => fetch(new URL("api/study/state",location.href)).then(r=>r.json()));
      firstVideo = held.video_index;
      alreadyDone = held.stage === "done";
      if (held.stage === "break") await clickViewingButton(page, "Continue to the next video");
      check(["watch","break","final","done"].includes(held.stage), "Resumed the test browser's own viewing session");
    } else {
    await page.goto(base);
    await page.getByRole("heading", { name: /Watch|보기/ }).waitFor();
    if (locale === "ko") {
      await page.getByRole("button", { name: "한국어" }).click();
      await page.waitForFunction(() => document.documentElement.lang === "ko"
        && document.querySelector("#start-button")?.textContent.trim() === "시작"
        && !document.querySelector("#start-button").disabled);
    }
    for (let clip = 0; clip < 2; clip += 1) {
      await finishLegacyViewing(page, "art");
      await answerShortSurvey(page, "art", ["art"], `${locale} clip ${clip + 1} original ART`);
      await completeVisual(page);
      await completeAuditory(page);
      await page.waitForFunction(() => window.__qaStep === "view_regenerated", null, { timeout: 180000 });
      await page.locator("#start-button").waitFor({ state: "visible", timeout: 30000 });
      await finishLegacyViewing(page, "survey");
      await answerShortSurvey(page, "survey", ["art", "caption"], `${locale} clip ${clip + 1} updated survey`);
      if (clip < 1) {
        await page.waitForFunction(() => window.__qaStep === "view_prepared", null, { timeout: 60000 });
        await page.locator("#start-button").waitFor({ state: "visible", timeout: 30000 });
      }
    }
    await answerShortOverall(page, locale);
    await continueToViewing(page);
    }
    await screenshot(page, `${locale}-ready.png`);
    if (publicRun) await writeFile(resolve(output, `${locale}-resume.json`), JSON.stringify({url:page.url()}), {mode:0o600});
    console.log(JSON.stringify({event:"viewing-ready",locale,at:stamp()}));
    for (let index = firstVideo; index < 3; index += 1) {
      await finishRealVideo(page, locale, index);
      await submitPerVideoSurvey(page, locale, index);
    }
    if (!alreadyDone) await submitFinal(page, locale);
    const exportJson = publicRun ? readPublicExport(await context.cookies()) : await page.evaluate(() => fetch(new URL("api/study/export", location.href), {
      credentials: "same-origin",
    }).then(response => response.json()));
    check(exportJson.state.stage === "done", `${locale} export stage is done`);
    check(exportJson.state.completions.length === 3, `${locale} export records three completions`);
    check(exportJson.state.video_surveys.length === 3, `${locale} export records three per-video surveys`);
    check(exportJson.exposures.length > 0, `${locale} export records caption exposures`);
    check(exportJson.state.calibration_source.language === locale, `${locale} calibration language is frozen in export`);
    check(exportJson.state.final_survey.answers.comments.includes(locale), `${locale} final survey comments exported`);
    const shortAudit = shortClipAudit(exportJson.state.calibration_source);
    check(Object.keys(shortAudit).length === 4, `${locale} export records original and updated surveys for two short clips`);
    const audit = captionAudit(exportJson);
    await writeFile(resolve(output, `${locale}-export.json`), JSON.stringify(exportJson, null, 2));
    for (const [video, counts] of Object.entries(audit)) {
      check(counts.exposures > 0, `${locale} ${video} records real caption exposures`);
      check(counts.generatedExposures > 0, `${locale} ${video} displayed non-fallback generated captions`);
      check(counts.nonFallback > 0, `${locale} ${video} records non-fallback Gemma caption jobs`);
      check(counts.modelNotConfigured === 0, `${locale} ${video} did not fall back because model was missing`);
      check(counts.fallback < counts.jobs, `${locale} ${video} was not all fallback captions`);
      if (publicRun) for (const level of [0, 1]) {
        check(exportJson.exposures.some(e => e.video_id === video && e.fallback === false &&
          e.axes?.texture === level && e.axes?.context === level),
        `${locale} ${video} saved generated-caption exposure at both axes ${level}`);
      }
    }
    await screenshot(page, `${locale}-done.png`);
    return {
      locale,
      participant: await page.evaluate(() => window.__qaParticipant),
      finalStage: exportJson.state.stage,
      completedVideos: exportJson.state.completions.length,
      exposures: exportJson.exposures.length,
      viewingHash: exportJson.state.viewing_hash,
      calibrationLanguage: exportJson.state.calibration_source.language,
      captionAudit: audit,
      shortClipAudit: shortAudit,
      videoSurveys: exportJson.state.video_surveys.length,
    };
  } finally {
    clearInterval(monitor);
    await writeFile(resolve(output, `${locale}-network.json`), JSON.stringify(network, null, 2));
    await screenshot(page, `${locale}-last.png`).catch(() => {});
    await writeFile(resolve(output, `${locale}-last.txt`), await page.locator("body").innerText().catch(() => ""));
    await context.close().catch(() => {});
    await browser?.close();
  }
}

function readPublicExport(cookies) {
  const cookie = cookies.find(value => value.name.startsWith("caption_study_"));
  if (!cookie || !process.env.REGEN_REAL_DATABASE) throw new Error("Public QA requires its browser cookie and a local export database");
  // The credential came from this test browser. It is used only for a local
  // read-only audit and is never printed or retrieved from the database.
  return JSON.parse(execFileSync(python, ["-c", `
import json, sqlite3, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
db.row_factory = sqlite3.Row
token = sys.stdin.read()
state = json.loads(db.execute('SELECT state FROM sessions WHERE id=?', (token,)).fetchone()[0])
events = [dict(r) for r in db.execute('SELECT kind,body,at FROM events WHERE session=?', (token,))]
jobs = [dict(r) for r in db.execute('SELECT * FROM jobs WHERE session=?', (token,))]
for job in jobs: job.pop('session', None)
exposures = [json.loads(r[0]) for r in db.execute('SELECT body FROM exposures WHERE session=?', (token,))]
print(json.dumps(dict(state=state, events=events, jobs=jobs, exposures=exposures)))
`, process.env.REGEN_REAL_DATABASE], {input:cookie.value, encoding:"utf8", maxBuffer:32*1024*1024}));
}

let fixture;
try {
  fixture = await launchServer();
  await eventually(async () => {
    try { return (await fetch(fixture.url)).ok; } catch { return false; }
  }, "Real fixture HTTP readiness", 60000);
  sourceMetadata = publicRun ? {mode:"public-deployment",origin:new URL(fixture.url).origin} :
    await fetch(new URL("/qa/source", fixture.url)).then(response => response.json());
  if (!publicRun) check(sourceMetadata.model_configured === true, "Real QA server is model-configured");
  await writeFile(resolve(output, "source.json"), JSON.stringify(sourceMetadata, null, 2));
  const locales = process.env.REGEN_REAL_LOCALES
    ? process.env.REGEN_REAL_LOCALES.split(",").map(value => value.trim()).filter(Boolean)
    : ["en", "ko"];
  const journeys = await Promise.all(locales.map(locale => runLocale(fixture.url, locale)));
  check(errors.length === 0, errors.join("\n"));
  const result = { generatedAt: stamp(), fixture: {...fixture,url:new URL(fixture.url).origin}, source: sourceMetadata, journeys, checks, steeringTimings, output };
  await writeFile(resolve(output, "browser-real-result.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} catch (error) {
  await writeFile(resolve(output, "browser-real-error.json"), JSON.stringify({
    error: error.message,
    stack: error.stack,
    errors,
    checks,
    serverLog: log,
  }, null, 2));
  throw error;
} finally {
  if (server) {
    server.kill("SIGTERM");
    await writeFile(resolve(output, "server.log"), log);
  }
}

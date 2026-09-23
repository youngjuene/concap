// Fully integrated browser readiness run against tests/preview_readiness.py.
// Uses an already installed Playwright; adds no project dependency.
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
const playwrightModule = process.env.PLAYWRIGHT_MODULE || "playwright-core";
const loaded = await import(playwrightModule.endsWith("playwright-core") ? `${playwrightModule}/index.js` : playwrightModule);
const playwright = loaded.chromium ? loaded : loaded.default;
const { chromium, firefox, webkit } = playwright;

const output = process.env.REGEN_READINESS_OUTPUT || "/tmp/regen-readiness-qa";
const base = process.env.REGEN_READINESS_URL || "http://127.0.0.1:18781";
const executablePath = process.env.CHROMIUM_EXECUTABLE;
const fast = process.env.REGEN_READINESS_FAST === "1";
await mkdir(output, { recursive: true });

const errors = [];
const productFailures = [];
const check = (condition, message) => { if (!condition) throw new Error(message); };
const stamp = () => new Date().toISOString();

async function screenshot(page, name) {
  await page.screenshot({ path: resolve(output, name), fullPage: true });
}

async function allowMedia(page) {
  await page.addInitScript(flag => {
    window.__qaFast = flag;
    if (!window.__qaFast) return;
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
  }, fast);
}

async function finishLegacyViewing(page, step) {
  await page.getByRole("button", { name: /Start|시작/ }).click();
  await page.locator("#screen-viewing:not([hidden])").waitFor();
  if (fast) {
    await page.locator("#video").evaluate(video => {
      video.currentTime = video.duration || 10;
      video.dispatchEvent(new Event("timeupdate"));
      video.dispatchEvent(new Event("ended"));
    });
  } else {
    await page.locator("#video").evaluate(video => new Promise(resolve => {
      if (video.ended) resolve();
      else video.addEventListener("ended", resolve, { once: true });
    }), { timeout: 30000 });
  }
  await page.locator("#screen-viewing").waitFor({ state: "hidden", timeout: 15000 });
  await page.waitForFunction(expected => window.__qaStep === expected, step === "view_prepared" ? "art" : "survey");
}

async function answerVisibleSurvey(page, value = 4) {
  const names = await page.locator("#survey-blocks input[type=radio]").evaluateAll(inputs => [...new Set(inputs.map(input => input.name))]);
  check(names.length > 0, "survey rendered no radio inputs");
  for (const name of names) await page.locator(`input[name="${name}"][value="${value}"]`).check();
}

async function verifyLocalizedScale(page, locale) {
  if (locale !== "ko") return;
  await page.getByText("전혀 그렇지 않다").first().waitFor({ timeout: 10000 });
  await page.getByText("매우 그렇다").first().waitFor({ timeout: 10000 });
}

async function verifySurveyDraft(page, locale) {
  await page.locator("#survey-blocks input[type=radio]").first().check();
  await page.reload();
  await page.locator("#screen-survey:not([hidden])").waitFor();
  check(await page.locator("#survey-blocks input[type=radio]:checked").count() === 1, `${locale} survey draft was not restored`);
}

async function verifyVisualDraft(page, locale) {
  await page.locator("#plate").focus();
  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Enter");
  check(await page.locator("#plate .point").count() >= 2, `${locale} visual marks were not placed`);
  await page.reload();
  await page.locator("#screen-visual:not([hidden])").waitFor();
  check(await page.locator("#plate .point").count() >= 2, `${locale} visual draft was not restored`);
}

async function verifyAuditoryDraft(page, locale) {
  await page.locator('#lanes input[name="family-human"]').first().check();
  await page.reload();
  await page.locator("#screen-auditory:not([hidden])").waitFor();
  check(await page.locator('#lanes input[name="family-human"]:checked').count() === 1, `${locale} auditory draft was not restored`);
}

async function submitSurveyWithRetry(page, locale, pageName) {
  let failed = false;
  await page.route("**/api/survey", async route => {
    if (!failed) {
      failed = true;
      await route.abort("failed");
    } else {
      await route.continue();
    }
  });
  const before = await page.evaluate(() => window.__qaStep);
  const expectedRetry = await page.evaluate(async () => {
    const payload = await fetch("/api/strings").then(r => r.json());
    const lang = document.documentElement.lang || "en";
    return payload.strings[lang].network.retry;
  });
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  await page.waitForFunction(expected => window.__qaStep === expected, before, { timeout: 10000 });
  await page.waitForFunction(
    expected => document.querySelector("#survey-remaining")?.textContent === expected,
    expectedRetry,
    { timeout: 10000 }
  );
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  await page.waitForFunction(expected => window.__qaStep === expected, pageName === "art" ? "visual" : "done", { timeout: 15000 });
  await page.unroute("**/api/survey");
  check(failed, `${locale} did not exercise ${pageName} upload failure`);
}

async function completeVisual(page) {
  while (await page.locator("#plate .point").count() < 2) {
    await page.locator("#plate").focus();
    await page.keyboard.press("Enter");
  }
  await page.locator("#visual-next").click();
  await page.waitForFunction(() => window.__qaStep === "auditory", null, { timeout: 10000 });
}

async function completeAuditory(page) {
  const names = await page.locator("#lanes input[type=radio]").evaluateAll(inputs => [...new Set(inputs.map(input => input.name))]);
  check(names.length === 5, "auditory page did not render five sound families");
  for (const name of names) await page.locator(`input[name="${name}"]`).first().check();
  await page.getByRole("button", { name: /Submit|제출/ }).click();
  await page.waitForFunction(() => window.__qaStep === "regenerating", null, { timeout: 10000 });
}

async function continueToViewing(page) {
  await page.locator("#screen-done:not([hidden])").waitFor({ timeout: 20000 });
  await page.waitForURL(/\/viewing\/.+\/$/, { timeout: 20000 });
  await page.getByRole("heading", { name: /Your calibration is saved|사전 응답이 저장되었습니다/ }).waitFor({ timeout: 15000 });
  await page.getByRole("button", { name: /Start viewing experience|영상 시청 시작/ }).click();
  await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
}

async function waitSettings(page, texture, context) {
  await page.waitForFunction(
    ([t, c]) => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" })
      .then(r => r.json())
      .then(s => s.axes.texture === t && s.axes.context === c),
    [texture, context],
    { timeout: 10000 }
  );
}

async function viewingLabel(page, text) {
  return page.evaluate(async source => {
    const copy = await fetch(new URL("api/study/strings", location.href), { credentials: "same-origin" }).then(r => r.json());
    return document.documentElement.lang === "ko" ? (copy[source] || source) : source;
  }, text);
}

async function clickViewingButton(page, text) {
  await page.getByRole("button", { name: await viewingLabel(page, text), exact: true }).click();
}

async function verifyPlaybackRecovery(page, locale) {
  await page.locator("video").evaluate(video => video.play());
  const background = await page.context().newPage();
  await background.goto("about:blank");
  await background.bringToFront();
  await page.waitForFunction(() => document.hidden === true, null, { timeout: 10000 });
  await page.waitForFunction(() => document.querySelector("video").paused === true, null, { timeout: 10000 });
  await page.bringToFront();
  await page.getByText(await viewingLabel(page, "Playback paused after an interruption. Press play to resume from your saved position.")).waitFor({ timeout: 10000 });
  await background.close();
  let failures = 0;
  await page.route("**/api/study/playback", async route => {
    if (!(await page.locator("video").evaluate(video => video.paused).catch(() => true))) {
      failures += 1;
      await route.abort("failed");
    } else {
      await route.continue();
    }
  });
  await page.locator("video").evaluate(video => video.play());
  await page.waitForFunction(() => document.querySelector("video").paused === true, null, { timeout: 10000 });
  await page.getByText(await viewingLabel(page, "Playback paused after an interruption. Press play to resume from your saved position.")).waitFor({ timeout: 10000 });
  await page.unroute("**/api/study/playback");
  check(failures >= 2, `${locale} outage recovery aborted only ${failures} playback request(s)`);
  await page.locator("video").evaluate(video => video.play());
}

async function applyControls(page, index) {
  if (index === 0) {
    await page.getByText(await viewingLabel(page, "Detail presets"), { exact: true }).click();
    await clickViewingButton(page, "Both detailed");
    await waitSettings(page, 1, 1);
    await clickViewingButton(page, "Reset to calibration");
    await waitSettings(page, 0.5, 0.5);
  } else if (index === 1) {
    const texture = page.locator('input[type="range"]').first();
    await texture.focus();
    await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(600);
    await page.waitForFunction(() => fetch(new URL("api/study/state", location.href), { credentials: "same-origin" })
      .then(r => r.json()).then(s => s.axes.texture > 0.5), null, { timeout: 10000 });
  } else {
    const box = await page.locator(".detail-pad").boundingBox();
    check(Boolean(box), "detail pad missing");
    await page.mouse.move(box.x + box.width * 0.75, box.y + box.height * 0.25);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.9, box.y + box.height * 0.1);
    await page.mouse.up();
    await page.waitForTimeout(600);
    await clickViewingButton(page, "Fullscreen");
    await clickViewingButton(page, "Exit fullscreen");
  }
}

async function finishInteractiveVideo(page, locale, index) {
  await page.locator("video").waitFor();
  if (index === 0) {
    if (fast) {
      await page.locator("video").evaluate(video => { video.currentTime = 121; video.dispatchEvent(new Event("seeked")); });
      await page.waitForTimeout(1200);
    } else {
      await page.locator("video").evaluate(video => video.play());
      await page.waitForFunction(() => document.querySelector("video").currentTime >= 121, null, { timeout: 140000 });
    }
    await page.reload();
    await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
    const restored = await page.locator("video").evaluate(video => video.currentTime);
    check(Math.abs(restored - 121) < 3, `${locale} reload lost viewing position: ${restored}`);
  }
  await applyControls(page, index);
  if (fast) {
    await page.locator("video").evaluate(video => {
      video.play();
      const duration = 300000;
      for (let ms = 0; ms <= duration; ms += 5000) {
        video.currentTime = ms / 1000;
        video.dispatchEvent(new Event("timeupdate"));
      }
      video.dispatchEvent(new Event("pause"));
      video.dispatchEvent(new Event("ended"));
    });
  } else {
    await page.locator("video").evaluate(video => video.play());
    await page.waitForFunction(() => document.querySelector("video").ended === true, null, { timeout: 330000 });
  }
  await page.evaluate(() => document.fullscreenElement ? document.exitFullscreen() : undefined);
  await page.waitForFunction(() => document.fullscreenElement === null, null, { timeout: 10000 }).catch(() => {});
  const nextButton = page.getByRole("button", { name: await viewingLabel(page, index === 2 ? "Continue to the final survey" : "Finish video"), exact: true });
  try {
    await nextButton.click({ timeout: 20000 });
  } catch (error) {
    if (index !== 2) throw error;
    productFailures.push(`${locale} end-of-video continue button was pointer-intercepted: ${error.message.split("\n")[0]}`);
    await screenshot(page, `${locale}-continue-intercepted.png`);
    await nextButton.focus();
    await page.keyboard.press("Enter");
  }
  if (index < 2) {
    await clickViewingButton(page, "Continue to the next video");
    await page.getByRole("heading", { name: /Your caption detail|자막/ }).waitFor({ timeout: 15000 });
  } else {
    await page.getByRole("heading", { name: /Your viewing experience|경험/ }).waitFor({ timeout: 15000 });
  }
}

async function submitFinal(page, locale) {
  await page.setViewportSize({ width: 390, height: 844 });
  check(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), `${locale} mobile viewing/final page overflows`);
  const names = await page.locator('input[type="radio"]').evaluateAll(inputs => [...new Set(inputs.map(input => input.name))]);
  for (const name of names) {
    const preferred = await page.locator(`input[name="${name}"][value="4"]`).count();
    await page.locator(preferred ? `input[name="${name}"][value="4"]` : `input[name="${name}"]`).first().check();
  }
  await page.locator("textarea").fill(`Synthetic ${locale} readiness final comments`);
  await page.reload();
  await page.locator("textarea").waitFor();
  check((await page.locator("textarea").inputValue()).includes(locale), `${locale} final survey draft was not restored`);
  await page.getByRole("button", { name: /Submit your experience|제출/ }).click();
  await page.getByRole("heading", { name: /Thank you for taking part|완료|참여/ }).waitFor({ timeout: 15000 });
  check(await page.locator(".receipt").count() === 1, `${locale} final receipt missing`);
  check(await page.evaluate(() => document.activeElement?.tagName === "H1"), `${locale} final heading did not receive focus`);
}

async function runLocale(locale) {
  const browser = await chromium.launch({ executablePath, args: ["--no-sandbox"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, acceptDownloads: true });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(`${locale} pageerror: ${error.message}`));
  page.on("console", message => {
    if (message.type() === "error" && !/Failed to load resource/.test(message.text())) {
      errors.push(`${locale} console: ${message.text()}`);
    }
  });
  await allowMedia(page);
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

  try {
    await page.goto(base);
    await page.getByRole("heading", { name: /Watch|보기/ }).waitFor();
    if (locale === "ko") await page.getByRole("button", { name: "한국어" }).click();
    await finishLegacyViewing(page, "view_prepared");
    await page.waitForFunction(() => window.__qaStep === "art");
    await verifyLocalizedScale(page, locale);
    await verifySurveyDraft(page, locale);
    await answerVisibleSurvey(page);
    await submitSurveyWithRetry(page, locale, "art");
    await verifyVisualDraft(page, locale);
    await screenshot(page, `${locale}-visual.png`);
    await completeVisual(page);
    await verifyAuditoryDraft(page, locale);
    await completeAuditory(page);
    await page.getByRole("heading", { name: /Second clip|두 번째/ }).waitFor({ timeout: 20000 });
    await finishLegacyViewing(page, "view_regenerated");
    await verifyLocalizedScale(page, locale);
    await answerVisibleSurvey(page);
    await submitSurveyWithRetry(page, locale, "survey");
    await continueToViewing(page);
    await screenshot(page, `${locale}-interactive-ready.png`);
    for (let index = 0; index < 3; index += 1) await finishInteractiveVideo(page, locale, index);
    await submitFinal(page, locale);
    const exportJson = await page.evaluate(() => fetch(new URL("api/study/export", location.href), { credentials: "same-origin" }).then(r => r.json()));
    check(exportJson.state.stage === "done", `${locale} export did not reach done`);
    check(exportJson.state.completions.length === 3, `${locale} export missing video completions`);
    check(exportJson.exposures.length > 0, `${locale} export missing exposures`);
    check(exportJson.state.final_survey.answers.comments.includes(locale), `${locale} final comments missing from export`);
    check(exportJson.state.calibration_source.language === locale, `${locale} frozen calibration language mismatch`);
    check(exportJson.state.profile.calibration_source_hash, `${locale} profile missing calibration source hash`);
    await writeFile(resolve(output, `${locale}-export.json`), JSON.stringify(exportJson, null, 2));
    await screenshot(page, `${locale}-done.png`);
    return {
      locale,
      participant: await page.evaluate(() => window.__qaParticipant),
      completedVideos: exportJson.state.completions.length,
      exposures: exportJson.exposures.length,
      finalStage: exportJson.state.stage,
      calibrationLanguage: exportJson.state.calibration_source.language,
      productFailures,
    };
  } finally {
    await context.close().catch(() => {});
    await browser.close();
  }
}

async function smokeBrowser(browserType, label, launchOptions = {}) {
  const browser = await browserType.launch(launchOptions);
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(`${label} pageerror: ${error.message}`));
  try {
    await page.goto(base);
    await page.getByRole("heading").first().waitFor({ timeout: 10000 });
    await screenshot(page, `${label}-smoke.png`);
    return { label, ok: true };
  } finally {
    await context.close().catch(() => {});
    await browser.close();
  }
}

const results = {};
try {
  const locales = process.env.REGEN_READINESS_LOCALES
    ? process.env.REGEN_READINESS_LOCALES.split(",").map(s => s.trim()).filter(Boolean)
    : ["en", "ko"];
  results.journeys = await Promise.all(locales.map(runLocale));
  results.smoke = [];
  if (process.env.REGEN_READINESS_SMOKE !== "0") {
    try { results.smoke.push(await smokeBrowser(firefox, "firefox")); } catch (error) { results.smoke.push({ label: "firefox", ok: false, error: error.message }); }
    try { results.smoke.push(await smokeBrowser(webkit, "webkit")); } catch (error) { results.smoke.push({ label: "webkit", ok: false, error: error.message }); }
  }
  check(errors.length === 0, errors.join("\n"));
  check(productFailures.length === 0, productFailures.join("\n"));
  results.generatedAt = stamp();
  results.output = output;
  await writeFile(resolve(output, "browser-readiness-result.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
} catch (error) {
  await writeFile(resolve(output, "browser-readiness-error.json"), JSON.stringify({ error: error.message, stack: error.stack, errors }, null, 2));
  throw error;
}

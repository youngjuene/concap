// Uses an already installed Playwright; adds no project dependency.
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const output = process.env.STUDY_QA_OUTPUT || "/tmp/regen-study-qa";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_EXECUTABLE,
  args: ["--no-sandbox"],
});
const errors = [];
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
page.on("pageerror", error => errors.push(error.message));
const url = process.env.STUDY_PREVIEW_URL || "http://127.0.0.1:18780";
const shot = async (name, target = page) => target.screenshot({ path: resolve(output, name), fullPage: true });
const check = (condition, message) => { if (!condition) throw new Error(message); };
try {
  await page.goto(url);
  await page.getByRole("heading", { name: "Make the captions yours" }).waitFor();
  await shot("calibration.png");
  await page.locator('input[name=texture][value="4"]').check();
  await page.locator('input[name=context][value="2"]').check();
  await page.getByRole("button", { name: "Begin calibration clips" }).click();
  for (let i = 0; i < 3; i++) {
    await page.locator("video").waitFor();
    await page.locator("video").evaluate(video => video.play());
    await page.getByRole("button", { name: "Continue to what you noticed" }).click({ timeout: 15000 });
    await page.getByRole("button", { name: "Add point", exact: true }).click();
    for (const family of ["human", "animal", "things", "music", "natural"]) {
      await page.locator(`input[name=${family}]`).first().check();
    }
    if (i === 0) await shot("observations.png");
    await page.getByRole("button", { name: "Save and continue" }).click();
  }
  await page.getByRole("button", { name: "Start viewing experience" }).click();
  await page.getByRole("heading", { name: "Your caption detail" }).waitFor();
  await shot("watching.png");
  await page.getByText("Detail presets", { exact: true }).click();
  await page.getByRole("button", { name: "Both detailed", exact: true }).click();
  await page.waitForTimeout(1200);
  const state = await page.evaluate(() => fetch("/api/study/state").then(r => r.json()));
  check(state.axes.texture === 1 && state.axes.context === 1, "Both axes must update");
  await page.locator("video").evaluate(video => { video.currentTime = 121; });
  await page.waitForTimeout(1200);
  await page.reload();
  await page.getByRole("heading", { name: "Your caption detail" }).waitFor();
  await page.waitForTimeout(1200);
  const restored = await page.locator("video").evaluate(video => video.currentTime);
  check(Math.abs(restored - 121) < 1, `Reload lost the checkpoint: ${restored}`);
  check(!(await page.locator("#notice").isVisible()), "A recovered reload must not leave an error banner");
  await page.setViewportSize({ width: 390, height: 844 });
  await shot("watching-mobile.png");
  check(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)), "Mobile overflow");
  await page.locator('input[type="range"]').first().focus();
  await page.keyboard.press("ArrowLeft");
  await page.waitForTimeout(1000);
  const keyboard = await page.evaluate(() => fetch("/api/study/state").then(r => r.json()));
  check(keyboard.axes.texture === .99, "Keyboard range control did not persist");

  // Endpoint timing/eligibility is exercised by test_study.py. This isolates the
  // final survey's rendering and mixed input types without spending 15 wall minutes.
  const finalPage = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  finalPage.on("pageerror", error => errors.push(error.message));
  const finalState = { ...state, stage: "final", completed_videos: 3, video_index: 3, draft: {},
    session_id: "browser-final-fixture" };
  let submitted;
  await finalPage.route("**/api/study/session", route => route.fulfill({ json: finalState }));
  await finalPage.route("**/api/study/draft", route => route.fulfill({ json: finalState }));
  await finalPage.route("**/api/study/final-survey", async route => {
    submitted = route.request().postDataJSON().data;
    await route.fulfill({ json: { ...finalState, stage: "done" } });
  });
  await finalPage.goto(url);
  await finalPage.getByRole("heading", { name: "Your viewing experience", exact: true }).waitFor();
  for (const item of finalState.items.filter(item => item.type === "rating")) {
    await finalPage.locator(`input[name=${item.id}][value="${item.na ? "na" : "4"}"]`).check();
  }
  await finalPage.locator('input[name=timing][value="Fast enough"]').check();
  await finalPage.locator("textarea").fill("Keep the clear acoustic detail controls.");
  await shot("final-survey.png", finalPage);
  await finalPage.getByRole("button", { name: "Submit your experience" }).click();
  await finalPage.getByRole("heading", { name: "Thank you for taking part" }).waitFor();
  check(submitted.control_texture === "na" && submitted.timing === "Fast enough", "Typed survey values lost");
  check(errors.length === 0, errors.join("\n"));
  const result = { errors, mobileOverflow: false, restoredPosition: restored, keyboardAxis: keyboard.axes.texture,
    calibrationClips: 3, finalSurvey: "typed form submitted (mock transport); full API gating tested separately" };
  await writeFile(resolve(output, "browser-result.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}

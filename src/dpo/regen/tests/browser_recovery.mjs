// Focused interruption tests use real playback and API writes on isolated synthetic media.
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
const output = process.env.REGEN_RECOVERY_OUTPUT || "/tmp/regen-recovery-qa";
await mkdir(output, {recursive: true});
const server = spawn(process.env.REGEN_QA_PYTHON || resolve(repo, ".venv/bin/python"), ["-u", "-c", `
import json, socket, tempfile
from pathlib import Path
import uvicorn
from dpo.regen.study_api import build_study_app
from dpo.regen.tests.test_study import make_media
root = Path(tempfile.mkdtemp(prefix="regen-recovery-"))
manifest = make_media(root / "media")
path = root / "study.json"
path.write_text(json.dumps(manifest))
app = build_study_app(path, root / "media", root / "out")
@app.get("/qa/coverage")
def coverage():
    with app.state.store.connection() as db:
        states = [json.loads(row[0]) for row in db.execute("SELECT state FROM sessions")]
    return [{key: state.get(key) for key in ("stage", "position_ms", "coverage", "last_tick", "epoch")} for state in states]
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
print(json.dumps({"url": f"http://127.0.0.1:{port}", "root": str(root)}), flush=True)
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
`], {cwd: repo, env: {...process.env, PYTHONDONTWRITEBYTECODE: "1"}});
let log = "", browser, page;
const errors = [], checks = [], transportFailures = [];
const check = (condition, description) => { if (!condition) throw new Error(description); checks.push(description); };
const eventually = async (fn, label, timeout = 15000) => {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { if (await fn()) return; await new Promise(r => setTimeout(r, 100)); }
  throw new Error(label);
};
const ready = new Promise((accept, reject) => {
  const timer = setTimeout(() => reject(new Error(`Fixture startup timed out: ${log}`)), 30000);
  server.stdout.on("data", data => {
    log += data.toString();
    for (const line of log.split("\n")) {
      try { const info = JSON.parse(line); if (info.url) { clearTimeout(timer); accept(info); } } catch {}
    }
  });
  server.stderr.on("data", data => { log += data.toString(); });
  server.once("exit", code => { clearTimeout(timer); reject(new Error(`Fixture exited ${code}: ${log}`)); });
});

try {
  const fixture = await ready;
  await eventually(async () => { try { return (await fetch(fixture.url)).ok; } catch { return false; } }, "Fixture HTTP readiness");
  browser = await chromium.launch({executablePath: process.env.CHROMIUM_EXECUTABLE, args: ["--no-sandbox"]});
  page = await browser.newPage();
  page.on("response", async response => {
    if (response.status() >= 400 && response.url().includes("/api/study/")) {
      transportFailures.push({path:new URL(response.url()).pathname,status:response.status(),
        request:response.request().postDataJSON(),response:await response.json().catch(() => ({}))});
    }
  });
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(fixture.url);
  await page.locator('input[name="texture"][value="4"]').check();
  await page.locator('input[name="context"][value="2"]').check();
  await page.getByRole("button", {name: "Begin calibration clips"}).click();
  let endFailures = 0;
  await page.route("**/api/study/clip-playback", route => {
    if (route.request().postDataJSON().data.position_ms >= 1900) { endFailures++; return route.abort("failed"); }
    return route.continue();
  });
  await page.locator("video").evaluate(video => video.play());
  await eventually(async () => endFailures >= 2 && await page.locator("video").evaluate(v => v.paused && !v.ended), "End failure did not rewind and pause");
  check(await page.getByRole("button", {name: "Continue to what you noticed"}).isDisabled(), "Failed calibration end stays gated");
  check((await page.locator("#notice").innerText()).includes("Press play"), "Interrupted playback explains how to resume");
  await page.unroute("**/api/study/clip-playback");
  await page.locator("video").evaluate(video => video.play());
  await page.waitForFunction(() => document.querySelector("video").ended);
  await eventually(() => page.getByRole("button", {name: "Continue to what you noticed"}).isEnabled(), "Calibration end acknowledgment");
  let manualEndFailures = 0;
  await page.route("**/api/study/clip-playback", route => { manualEndFailures++; return route.abort("failed"); });
  await page.getByRole("button", {name: "Continue to what you noticed"}).click();
  await eventually(async () => manualEndFailures >= 2 &&
    await page.locator("video").evaluate(v => v.paused && !v.ended) &&
    await page.getByRole("button", {name: "Continue to what you noticed"}).isDisabled(),
    "Manual calibration finish failure re-enabled the end button");
  check((await page.locator("#notice").innerText()).includes("Press play"), "Manual completion failure keeps the gate and resume guidance");
  await page.unroute("**/api/study/clip-playback");
  await page.locator("video").evaluate(video => video.play());

  for (let clip = 0; clip < 3; clip++) {
    await page.getByRole("button", {name: "Continue to what you noticed"}).click({timeout: 15000});
    await page.getByRole("button", {name: "Add point", exact: true}).click();
    for (const family of ["human", "animal", "natural", "things", "music"]) {
      await page.locator(`input[name="${family}"]`).first().check();
    }
    await page.getByRole("button", {name: "Save and continue"}).click();
    if (clip < 2) await page.locator("video").evaluate(video => video.play());
  }
  check(true, "Recovered calibration completes all three real short clips");
  await page.route("**/study/media/watch/0", async route => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    await route.continue();
  });
  await page.getByRole("button", {name: "Start viewing experience"}).click();
  await page.locator(".study-preparation").waitFor();
  check(await page.locator(".study-preparation").evaluate(node => getComputedStyle(node).color === "rgb(255, 255, 255)"),
    "Video preparation message is readable on the dark player");
  check(await page.locator("video").evaluate(video => !video.controls && video.paused),
    "Playback is gated while the complete video downloads");
  await page.screenshot({path:resolve(output,"preparing-video.png"),fullPage:true});
  await page.waitForFunction(() => document.querySelector("video").readyState >= 2);
  await page.unroute("**/study/media/watch/0");
  await page.getByRole("button", {name: "Fullscreen", exact: true}).click();
  await page.waitForFunction(() => Boolean(document.fullscreenElement));
  check(await page.getByRole("button", {name: "Finish video", exact: true}).evaluate(node =>
    document.fullscreenElement.contains(node)), "Fullscreen contains completion actions");
  await page.getByRole("button", {name: "Exit fullscreen", exact: true}).click();
  await page.waitForFunction(() => !document.fullscreenElement);
  check(await page.evaluate(() => !document.fullscreenElement), "Fullscreen has an explicit working exit");
  await page.locator("video").evaluate(video => video.play());
  await page.waitForFunction(() => document.querySelector("video").currentTime > 3);
  const beforeSeek = (await fetch(`${fixture.url}/qa/coverage`).then(r => r.json()))[0];
  const epochBefore = beforeSeek.epoch;
  let delayedSeek = false;
  await page.route("**/api/study/playback", async route => {
    if (route.request().postDataJSON().data.seek && !delayedSeek) {
      delayedSeek = true;
      await new Promise(resolve => setTimeout(resolve, 2500));
    }
    await route.continue();
  });
  const seekStart = await page.locator("video").evaluate((video, position) => {
    video.currentTime = position;
    return position;
  }, Math.max(0, beforeSeek.position_ms / 1000 - 0.25));
  await page.waitForFunction(start => document.querySelector("video").currentTime > start + 2, seekStart);
  await page.locator("video").evaluate(video => video.pause());
  await eventually(async () => {
    const held = (await fetch(`${fixture.url}/qa/coverage`).then(r => r.json()))[0];
    return held.epoch === epochBefore + 1 && !held.last_tick.playing && held.position_ms >= (seekStart + 2) * 1000 && held.coverage.length === 1;
  }, "Pause queued behind seek acknowledgement duplicated the seek or lost coverage");
  check(delayedSeek, "A pause queued before seek acknowledgement preserves one seek and continuous coverage");
  await page.unroute("**/api/study/playback");
  await page.locator("video").evaluate(video => video.play());
  // A synthetic visibility event exercises the real handler, without changing API responses or clocks.
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", {configurable: true, value: true});
    document.dispatchEvent(new Event("visibilitychange"));
  });
  check(await page.locator("video").evaluate(video => video.paused), "Hidden viewing pauses immediately");
  await page.evaluate(() => { delete document.hidden; document.dispatchEvent(new Event("visibilitychange")); });
  await page.locator("video").evaluate(video => video.play());
  await page.waitForFunction(() => document.querySelector("video").currentTime > 5);

  // A slow caption response must not block playback acknowledgements or force
  // a rewind. This reproduces public-relay latency, without mocking results.
  let delayedCaptions = 0;
  await page.route("**/api/study/captions", async route => {
    delayedCaptions++;
    await new Promise(resolve => setTimeout(resolve, 6500));
    await route.continue().catch(() => {});
  });
  await page.waitForFunction(() => document.querySelector("video").currentTime > 15, null, {timeout: 25000});
  check(delayedCaptions > 0 && await page.locator("video").evaluate(video => !video.paused),
    "Caption requests slower than five seconds do not pause playback");
  await page.unroute("**/api/study/captions");

  let failures = 0, failedSeeks = 0;
  await page.route("**/api/study/playback", route => {
    failures++;
    if (route.request().postDataJSON().data.seek) failedSeeks++;
    return route.abort("failed");
  });
  await eventually(async () => failures >= 2 && failedSeeks >= 2 && await page.locator("video").evaluate(v => v.paused), "Offline playback and its recovery seek did not both fail");
  check((await page.locator("#notice").innerText()).includes("Press play"), "Offline viewing preserves the resume instruction");
  await page.unroute("**/api/study/playback");
  const resumeFrom = await page.locator("video").evaluate(video => video.currentTime);
  await page.locator("video").evaluate(video => video.play());
  await page.waitForFunction(start => document.querySelector("video").currentTime > start + 3, resumeFrom);
  await page.locator("video").evaluate(video => video.pause());
  const pausedAt = await page.locator("video").evaluate(video => video.currentTime * 1000);
  await eventually(async () => {
    const states = await fetch(`${fixture.url}/qa/coverage`).then(r => r.json());
    return states[0].coverage.length === 1 && states[0].coverage[0][0] < 50 &&
      states[0].coverage[0][1] >= pausedAt - 50 && !states[0].last_tick.playing &&
      Math.abs(states[0].position_ms - pausedAt) < 50;
  }, "Recovery left a hole in credited viewing coverage");
  check(true, "Hidden and offline recovery leave continuous credited coverage");
  check(failedSeeks >= 2, "Resume preserves the seek after both original notification and retry were lost");

  // Deliberate seek isolates the end-control failure path; it is not counted as watched coverage.
  await page.locator("video").evaluate(video => { video.currentTime = 299; });
  await page.waitForTimeout(400);
  let watchEndFailures = 0;
  await page.route("**/api/study/playback", route => {
    if (route.request().postDataJSON().data.position_ms >= 299900) { watchEndFailures++; return route.abort("failed"); }
    return route.continue();
  });
  await page.locator("video").evaluate(video => video.play());
  await eventually(async () => watchEndFailures >= 2 && await page.locator("video").evaluate(v => v.paused && !v.ended), "Viewing end failure did not rewind");
  check(await page.getByRole("button", {name: "Finish video", exact: true}).isDisabled(), "Failed viewing end stays gated");
  await page.unroute("**/api/study/playback");
  await page.screenshot({path: resolve(output, "recovered-viewing.png"), fullPage: true});
  check(errors.length === 0, "No unhandled JavaScript errors during interruption recovery");
  const result = {fixture, checks, errors, endFailures, manualEndFailures, failures, watchEndFailures,
    scope: "Real short-clip playback and API writes; injected transport failures/visibility and deliberate end seek; not full long-video completion"};
  await writeFile(resolve(output, "result.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} catch (error) {
  if (page) {
    await page.screenshot({path:resolve(output,"failure.png"),fullPage:true}).catch(() => {});
    await writeFile(resolve(output,"failure.txt"), await page.locator("body").innerText().catch(() => ""));
    await writeFile(resolve(output,"failure-video.json"), JSON.stringify(await page.locator("video").evaluate(v => ({time:v.currentTime,paused:v.paused,ended:v.ended,seeking:v.seeking})).catch(() => ({}))));
  }
  await writeFile(resolve(output, "error.json"), JSON.stringify({error: error.stack, errors, checks, transportFailures, log}, null, 2));
  throw error;
} finally {
  if (browser) await browser.close();
  server.kill("SIGTERM");
  await writeFile(resolve(output, "server.log"), log);
}

// Linked inspector behavior over isolated record fixtures; no live participant data.
import {spawn} from "node:child_process";
import {mkdir, writeFile} from "node:fs/promises";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const output = process.env.DASHBOARD_QA_OUTPUT || "/tmp/concap-dashboard-linked-browser-qa";
await mkdir(output, {recursive: true});
const server = spawn(resolve(repo, ".venv/bin/python"), ["-u", "-c", `
import json,socket,sqlite3,tempfile
from pathlib import Path
import uvicorn
from dpo.regen.dashboard import build_dashboard_app
from dpo.regen.tests.test_dashboard_data import EPOCH,SECRET,make_records
root=Path(tempfile.mkdtemp(prefix='dashboard-linked-browser-'))
legacy,viewing=make_records(root)
with sqlite3.connect(viewing/'study.sqlite3') as db:
    for identifier,video,start,end,text,job in [
        ('overlap-probe','viewing-1',500,1500,'Overlapping caption probe',None),
        ('other-video-probe','viewing-2',0,1000,'Other-video overlap probe',None),
        ('timestamp-probe','viewing-1',30000,31000,'Timestamp-only caption probe','shown-job'),
    ]:
        body={'video_id':video,'start_ms':start,'end_ms':end,'text':text,'job_id':job,'fallback':False}
        db.execute('INSERT INTO exposures VALUES(?,?,?)',(SECRET,identifier,json.dumps(body)))
    for action,video,position,at in [
        ('probe-recorded-near','viewing-1',None,1.5),
        ('probe-recorded-far','viewing-1',None,30),
        ('probe-playback-outside','viewing-1',5001,2.5),
        ('probe-other-video','viewing-2',0,2),
        ('probe-caption-window','viewing-1',30500,30.5),
    ]:
        data={'video_id':video}
        if position is not None: data['position_ms']=position
        db.execute('INSERT INTO events VALUES(?,?,?,?)',
            (SECRET,'mutation',json.dumps({'action':action,'data':data}),EPOCH+at))
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
print(json.dumps({'url':f'http://127.0.0.1:{port}','root':str(root)}),flush=True)
uvicorn.run(build_dashboard_app(legacy,viewing),host='127.0.0.1',port=port,proxy_headers=False,log_level='warning')
`], {cwd: repo, env: {...process.env, PYTHONDONTWRITEBYTECODE: "1"}});

let log = "", browser, page;
const checks = [], errors = [];
const check = (value, label) => {
  if (!value) throw Error(label);
  checks.push(label);
};
const sameScope = (row, selected) => row.phase === selected.phase && row.video_id === selected.video_id;
const positioned = row => typeof row.position_ms === "number" && Number.isFinite(row.position_ms);
const captionOverlap = (row, low, high) => typeof row.start_ms === "number" && typeof row.end_ms === "number"
  && row.end_ms > low && row.start_ms < high;
const playbackBetween = (row, low, high) => positioned(row) && row.position_ms >= low && row.position_ms <= high;
const utc = value => new Date(value).toISOString().slice(0, 19).replace("T", " ") + " UTC";
const loaded = () => page.waitForFunction(() => document.querySelector("#refresh-status")?.textContent.startsWith("Updated"));
const inspectorLoaded = () => page.waitForFunction(() => document.querySelector("#inspector-detail .session-header")
  && !document.querySelector("#inspector-detail").textContent.includes("Refreshing session data"));
const captionRows = () => page.locator('.captions-panel table[aria-label="Caption exposures and generation jobs"] tbody tr').filter({has: page.locator(".caption-moment")});
const eventRows = () => page.locator('.events-panel table[aria-label="Chronological session events"] tbody tr').filter({has: page.locator(".event-moment")});
const screenshot = name => page.screenshot({path: resolve(output, name), fullPage: true});

async function assertCaptions(expected, label) {
  const actual = await captionRows().evaluateAll(rows => rows.map(row => ({
    text: row.textContent,
    start: Number(row.querySelector(".caption-moment").dataset.positionMs),
  })));
  const remaining = [...actual];
  for (const row of expected) {
    const index = remaining.findIndex(candidate => candidate.start === row.start_ms
      && candidate.text.includes(row.text) && candidate.text.includes(`Phase ${row.phase}`)
      && candidate.text.includes(row.video_id));
    if (index < 0) throw Error(`${label}: missing ${row.text} at ${row.start_ms} in ${row.video_id}`);
    remaining.splice(index, 1);
  }
  check(remaining.length === 0, `${label} (${expected.length} records)`);
}

async function assertEvents(expected, label) {
  const remaining = await eventRows().allTextContents();
  for (const row of expected) {
    const index = remaining.findIndex(text => text.includes(row.kind) && text.includes(utc(row.at))
      && text.includes(`Phase ${row.phase}`) && (!row.video_id || text.includes(row.video_id)));
    if (index < 0) throw Error(`${label}: missing ${row.kind} at ${row.at}`);
    remaining.splice(index, 1);
  }
  check(remaining.length === 0, `${label} (${expected.length} records)`);
}

async function clearMoment() {
  await page.locator("#clear-moment").click();
  check(await page.locator(".linked-records .record-focus").count() === 0, "Clearing the moment removes selected-row highlights");
}

try {
  const info = await new Promise((accept, reject) => {
    const timeout = setTimeout(() => reject(Error("Dashboard fixture startup timed out: " + log)), 20000);
    server.stdout.on("data", data => {
      log += data;
      for (const line of log.split("\n")) {
        try {
          const parsed = JSON.parse(line);
          if (parsed.url) { clearTimeout(timeout); accept(parsed); }
        } catch { /* Server logs need not be JSON. */ }
      }
    });
    server.stderr.on("data", data => { log += data; });
    server.once("exit", code => { clearTimeout(timeout); reject(Error(`Fixture exited ${code}: ${log}`)); });
    server.once("error", error => { clearTimeout(timeout); reject(error); });
  });
  const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || "playwright-core");
  browser = await chromium.launch({executablePath: process.env.CHROMIUM_EXECUTABLE, args: ["--no-sandbox"]});
  page = await browser.newPage({viewport: {width: 1920, height: 1080}});
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(info.url);
  await loaded();
  const source = await page.request.get(info.url + "/api/overview").then(response => response.json());
  check(source.metrics.participants === 3 && source.metrics.survey_submissions === 11 && source.metrics.response_values === 14,
    "Linked-record fixture probes preserve participant and submitted-response totals");

  await page.locator('[data-view="responses"]').click();
  await page.locator("#filter-responsePhase").selectOption("2");
  check(await page.locator(".distribution-card").count() === 7,
    "Phase 2 still has seven separate response groups");
  check(await page.locator('.distribution-card[data-item-id="useful"]').count() === 3
    && await page.locator('.distribution-card[data-item-id="na_item"]').count() === 2,
  "Instrument and video boundaries remain separate for repeated survey items");
  check(await page.locator('#view-responses table[aria-label="Submitted survey response values"] tbody tr').count() === 7,
    "All seven Phase 2 answer values remain inspectable");
  const metadata = page.locator(".distribution-card details.metadata-details");
  check(await metadata.count() === 7 && await metadata.evaluateAll(nodes => nodes.every(node => !node.open)),
    "Every response card initially collapses its full metadata");
  const recordedMetadata = metadata.filter({hasText: "new-instrument"}).first();
  check(await recordedMetadata.count() === 1, "Recorded instrument metadata is retained inside its disclosure");
  await recordedMetadata.locator("summary").click();
  check(await recordedMetadata.evaluate(node => node.open)
    && (await recordedMetadata.innerText()).includes("long-video-surveys/v2"),
  "Expanding response metadata reveals the full recorded instrument and flow");
  check((await page.locator('.distribution-card[data-item-id="na_item"]').allTextContents()).some(text => text.includes("N/A"))
    && (await page.locator('.distribution-card[data-item-id="na_item"]').allTextContents()).some(text => text.includes("Missing")),
  "Metadata disclosure preserves separate N/A and missing categories");
  await screenshot("responses-metadata-desktop.png");
  await recordedMetadata.locator("summary").click();

  await page.locator('[data-view="inspector"]').click();
  await page.locator(".participant-option").filter({hasText: "p-new"}).click();
  await inspectorLoaded();
  const participant = source.participants.find(row => row.label === "p-new");
  const detail = await page.request.get(info.url + "/api/participants/" + participant.id).then(response => response.json());
  check(detail.interactions.length === 3 && detail.interactions[0].position_ms === null,
    "Original controls retain missing, zero, and later playback positions");
  check(await page.locator(".inspector-workbench #moment-summary").count() === 1
    && await page.locator(".inspector-workbench .timeline-moment[role=button]").count() > 0,
  "The inspector workbench pairs a selectable control timeline with a moment summary");
  check(await page.locator("details.inspector-appendix").evaluateAll(nodes => nodes.length > 0 && nodes.every(node => !node.open)),
    "Inspector appendices are collapsed initially");
  check(await page.locator('details.inspector-appendix table[aria-label="Submitted survey response values"]').count() === 1
    && await page.locator("details.inspector-appendix .timing-summary").count() === 1,
  "Submitted responses and model timing remain available in inspector appendices");
  await assertCaptions(detail.captions, "Initial inspector shows captions across all phases and videos");
  await assertEvents(detail.events, "Initial inspector shows all recorded events");
  const captionsBox = await page.locator(".linked-records .captions-panel").boundingBox();
  const eventsBox = await page.locator(".linked-records .events-panel").boundingBox();
  check(captionsBox && eventsBox && Math.abs(captionsBox.y - eventsBox.y) <= 2
    && captionsBox.x + captionsBox.width <= eventsBox.x + 2 && captionsBox.width >= 350 && eventsBox.width >= 350,
  "Caption and event panels align side by side at 1920 pixels");
  const timelineBox = await page.locator(".inspector-workbench svg.chart").first().boundingBox();
  check(timelineBox && timelineBox.height <= 245, "Control timeline stays compact above the linked records");

  const zero = detail.interactions.find(row => row.position_ms === 0);
  const scopedCaptions = detail.captions.filter(row => sameScope(row, zero));
  const scopedEvents = detail.events.filter(row => sameScope(row, zero));
  const zeroMarker = page.locator('.timeline-moment[role="button"][data-axis="texture"][data-position-ms="0"]');
  await zeroMarker.focus();
  await zeroMarker.press("Enter");
  check(await page.locator("#filter-inspectorVideo").inputValue() === JSON.stringify([zero.phase, zero.video_id]),
    "Keyboard activation selects the control's phase and video");
  check((await page.locator("#moment-summary").innerText()).includes("0 s"),
    "A recorded zero playback position is shown as a selected moment");
  await assertCaptions(scopedCaptions.filter(row => captionOverlap(row, 0, 5000)),
    "Zero-position control links overlapping captions in its bounded playback window");
  await assertEvents(scopedEvents.filter(row => playbackBetween(row, 0, 5000)),
    "Zero-position control links positioned events through the inclusive five-second boundary");
  check(!(await page.locator(".linked-records").innerText()).includes("Other-video overlap probe")
    && !(await page.locator(".linked-records").innerText()).includes("probe-other-video")
    && !(await page.locator(".linked-records").innerText()).includes("probe-playback-outside"),
  "Moment selection excludes other videos and events just outside its time window");
  await screenshot("linked-control-desktop.png");
  await clearMoment();
  await assertCaptions(scopedCaptions, "Clearing a moment restores all captions for the selected video");
  await assertEvents(scopedEvents, "Clearing a moment restores all events for the selected video");

  await page.locator("#control-time-basis").selectOption("recorded");
  const missing = detail.interactions.find(row => row.position_ms === null);
  await page.locator(`.timeline-moment[role="button"][data-axis="texture"][data-recorded-at="${missing.at}"]`).click();
  check(/playback position not recorded/i.test(await page.locator("#moment-summary").innerText()),
    "Missing playback positions are explicitly described in the moment summary");
  await assertCaptions(scopedCaptions, "A recorded-time moment leaves captions unfiltered within the selected video");
  await assertEvents(scopedEvents.filter(row => Math.abs(Date.parse(row.at) - Date.parse(missing.at)) <= 5000),
    "A control without playback uses only a bounded recorded-time event window");
  check((await page.locator(".events-panel").innerText()).includes("probe-recorded-near")
    && !(await page.locator(".events-panel").innerText()).includes("probe-recorded-far"),
  "Recorded-time linking includes a nearby unpositioned event and excludes a distant one");
  await clearMoment();

  const firstCaption = scopedCaptions.find(row => row.text === "Shown caption" && row.start_ms === 0);
  await captionRows().filter({hasText: "Shown caption"}).filter({has: page.locator('.caption-moment[data-position-ms="0"]')}).locator(".caption-moment").click();
  await assertCaptions(scopedCaptions.filter(row => captionOverlap(row, firstCaption.start_ms, firstCaption.end_ms)),
    "Caption selection links overlapping windows and excludes the next cue at the shared endpoint");
  await assertEvents(scopedEvents.filter(row => playbackBetween(row, firstCaption.start_ms, firstCaption.end_ms)),
    "Caption selection links events by its playback window");
  check(await page.locator(".captions-panel tr.record-focus").count() === 1
    && (await page.locator(".captions-panel tr.record-focus").innerText()).includes("Shown caption"),
  "Only the exact selected caption row is highlighted");
  const selectedSummary = await page.locator("#moment-summary").innerText();
  const refreshDetail = page.waitForResponse(response => response.url().includes(`/api/participants/${participant.id}?`));
  await page.locator("#refresh-button").click();
  await refreshDetail;
  await loaded();
  await inspectorLoaded();
  check(await page.locator("#moment-summary").innerText() === selectedSummary
    && await page.locator(".captions-panel tr.record-focus").count() === 1,
  "Refresh preserves the selected caption moment and exact row highlight");
  await assertCaptions(scopedCaptions.filter(row => captionOverlap(row, firstCaption.start_ms, firstCaption.end_ms)),
    "Refresh preserves the linked caption result set");
  await clearMoment();

  const timestampCaption = scopedCaptions.find(row => row.text === "Timestamp-only caption probe");
  check(timestampCaption.created_at && timestampCaption.start_ms === 30000,
    "Timestamp probe retains an early job-created timestamp and a distant playback window");
  await captionRows().filter({hasText: timestampCaption.text}).locator(".caption-moment").click();
  await assertCaptions([timestampCaption], "A distant caption selects only its overlapping playback exposure");
  await assertEvents(scopedEvents.filter(row => playbackBetween(row, timestampCaption.start_ms, timestampCaption.end_ms)),
    "Caption job creation time is never used to match visible playback events");
  check(!(await page.locator(".events-panel").innerText()).includes("probe-recorded-near"),
    "Events near job creation do not masquerade as events near caption visibility");
  await clearMoment();

  const event = scopedEvents.find(row => row.kind === "probe-playback-outside");
  await eventRows().filter({hasText: event.kind}).locator(".event-moment").click();
  await assertCaptions(scopedCaptions.filter(row => captionOverlap(row, event.position_ms - 5000, event.position_ms + 5000)),
    "Selecting an event links captions around that event's playback position");
  await assertEvents(scopedEvents.filter(row => playbackBetween(row, event.position_ms - 5000, event.position_ms + 5000)),
    "Selecting an event links the same-video playback neighborhood");
  check(await page.locator(".events-panel tr.record-focus").count() === 1
    && (await page.locator(".events-panel tr.record-focus").innerText()).includes(event.kind),
  "Only the exact selected event row is highlighted");

  await page.locator("#filter-inspectorVideo").selectOption(JSON.stringify([2, "viewing-2"]));
  check(await page.locator(".linked-records .record-focus").count() === 0,
    "Changing the phase/video scope clears selected moment highlights");
  await assertCaptions(detail.captions.filter(row => row.phase === 2 && row.video_id === "viewing-2"),
    "Changing video restores its full caption scope");
  await assertEvents(detail.events.filter(row => row.phase === 2 && row.video_id === "viewing-2"),
    "Changing video restores its full event scope");
  await captionRows().filter({hasText: "Other-video overlap probe"}).locator(".caption-moment").click();
  await assertCaptions(detail.captions.filter(row => row.phase === 2 && row.video_id === "viewing-2"),
    "An overlapping caption in another video cannot reintroduce the previous video's captions");
  await page.locator(".participant-option").filter({hasText: "p-old"}).click();
  await inspectorLoaded();
  check(await page.locator(".linked-records .record-focus").count() === 0
    && await page.locator("#filter-inspectorVideo").inputValue() === "",
  "Changing participants clears the selected moment and video scope");

  await page.locator(".participant-option").filter({hasText: "p-new"}).click();
  await inspectorLoaded();
  await page.setViewportSize({width: 1366, height: 768});
  await screenshot("linked-inspector-laptop.png");
  check(!await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),
    "The linked inspector has no page-level horizontal overflow at 1366 pixels");
  check(errors.length === 0, "No unhandled JavaScript errors during linked inspector interactions");
  await writeFile(resolve(output, "result.json"), JSON.stringify({fixture: info, checks, errors, output}, null, 2));
  console.log(JSON.stringify({checks, output}));
} catch (error) {
  await page?.screenshot({path: resolve(output, "failure.png"), fullPage: true}).catch(() => {});
  await writeFile(resolve(output, "error.json"), JSON.stringify({error: error.stack, checks, errors, log}, null, 2));
  throw error;
} finally {
  await browser?.close();
  server.kill("SIGTERM");
}

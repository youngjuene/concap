// Execute the production display/outbox functions; browser media and storage are
// boundary fixtures. This does not seed playback or operate a participant session.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {randomUUID} from 'node:crypto';

const source = fs.readFileSync(new URL('../study.js', import.meta.url), 'utf8');
const scale = source.slice(source.indexOf('function captionDetailScale'), source.indexOf('function storageKey'));
const display = source.slice(source.indexOf('  const outboxKey = "exposures";'),
  source.indexOf('  async function tick(seek = false, force = false)', source.indexOf('function watching()')));
const watching = source.slice(source.indexOf('function watching()'));
const handlers = ['onpause', 'onseeking', 'onseeked'].map(name =>
  watching.match(new RegExp(`  video\\.${name} = [^\\n]+`))[0]).join('\n');
const pagehide = watching.match(/  const pagehide = [^\n]+/)[0];

function fixture(initialStorage = {}) {
  const context = vm.createContext({crypto: {randomUUID}, structuredClone, initialStorage});
  vm.runInContext(`
    ${scale}
    const state = {settings_revision: 0, axes: {texture: .5, context: .5}};
    const detailScale = captionDetailScale({version: 'five-level/v1', values: [0,.25,.5,.75,1],
      labels: {en: ['Very little','A little','A moderate amount','Quite a lot','A lot']}}, 'en');
    const video = {currentTime: 4.9, seeking: false, paused: false};
    const document = {hidden: false};
    const videoInfo = {id: 'fixture', duration_ms: 10000, cues: [
      {start_ms: 0, end_ms: 5000, fallback: 'first caption'},
      {start_ms: 5000, end_ms: 10000, fallback: 'second caption'}
    ]};
    let jobs = [], activeIndex = -1, applied = null, exposure = null, lastPosition = 4900;
    let seekGeneration = 0;
    const caption = {textContent: ''}, captionLevels = {}, status = {};
    const storage = structuredClone(initialStorage);
    const t = (text, values = {}) => Object.entries(values).reduce((result, [key, value]) => result.replaceAll('{' + key + '}', value), text);
    const saved = (key, fallback) => storage[key] ?? fallback;
    const save = (key, value) => { storage[key] = structuredClone(value); };
    const send = async () => {};
    ${display}
    const tick = async () => {};
    ${handlers}
    ${pagehide}
    globalThis.run = code => eval(code);
    globalThis.read = () => structuredClone({storage, exposure, caption, captionLevels});
  `, context);
  return context;
}

// A delayed render leaves the old text visible until 5215 ms. Clipping that
// exposure at its nominal cue end is the production change this catches.
const boundary = fixture();
boundary.run('paint(); video.currentTime = 5.215; paint();');
const transition = boundary.read();
assert.equal(transition.storage.exposures[0].end_ms, 5215);
assert.equal(transition.exposure.start_ms, 5215);
assert.equal(transition.storage.exposures[0].cue_end, 5000);
assert.equal(transition.storage.exposures[0].timing, 'display-v1');
assert.ok(transition.exposure.episode_id);
assert.equal(transition.caption.textContent, 'second caption');

// A transition uses one sampled position. Sampling again while closing the
// previous caption would overlap it with the new caption's earlier start.
const advancingClock = fixture();
advancingClock.run(`paint(); let position = 5.215;
  Object.defineProperty(video, 'currentTime', {get() { const now = position; position += .005; return now; }});
  paint();`);
assert.equal(advancingClock.read().storage.exposures[0].end_ms,
  advancingClock.read().exposure.start_ms);

// Pause must sample the actual pause position, rather than the preceding
// timeupdate; resuming creates a new exposure without counting paused time.
boundary.run('video.currentTime = 5.4; video.paused = true; video.onpause(); paint();');
assert.equal(boundary.read().storage.exposures[1].end_ms, 5400);
assert.equal(boundary.read().exposure, null);
boundary.run('video.paused = false; paint(); video.currentTime = 5.62; paint();');
assert.equal(boundary.read().exposure.start_ms, 5400);

// Seeking must not extend old text through an unplayed jump.
const previousEpisode = boundary.read().exposure.episode_id;
boundary.run('video.currentTime = 8.5; video.seeking = true; video.onseeking();');
assert.equal(boundary.read().storage.exposures[2].end_ms, 5620);
boundary.run('video.seeking = false; video.onseeked();');
assert.equal(boundary.read().exposure.start_ms, 8500);
assert.notEqual(boundary.read().exposure.episode_id, previousEpisode);

// Page exit persists its final sampled endpoint synchronously for reload.
boundary.run('video.currentTime = 8.61; pagehide();');
assert.equal(boundary.read().storage.exposures[3].end_ms, 8610);
assert.equal(boundary.read().storage.exposures[3].closed_by, 'pagehide');

// End-of-video uses the manifest end even when the delivery container has a
// few extra milliseconds, and cannot create an exposure beyond the video.
const ended = fixture();
ended.run('video.currentTime = 9.9; lastPosition = 9900; paint(); video.currentTime = 10.007; video.paused = true; finishExposure("ended");');
assert.equal(ended.read().storage.exposures[0].end_ms, 10000);

// An interrupted page can recover only its last persisted endpoint. It must
// not invent display time up to the next cue boundary or current position.
const interrupted = fixture({'open-exposure': transition.exposure});
assert.equal(interrupted.read().storage.exposures[0].end_ms, 5215);
assert.equal(interrupted.read().storage.exposures[0].incomplete, true);
assert.equal(interrupted.read().storage['open-exposure'], null);

// A saved level must not relabel the old caption before a boundary replacement.
const levels = fixture();
levels.run(`jobs = [{id:'old',cue:0,revision:0,result:{text:'Old generated caption',fallback:false}}]; paint();`);
assert.equal(levels.read().captionLevels.textContent,
  'Current caption: acoustic 3/5 · A moderate amount; source/scene 3/5 · A moderate amount.');
levels.run(`state.axes = {texture:1,context:0}; state.settings_revision = 1;
  jobs = [{id:'new',cue:1,revision:1,result:{text:'New generated caption',fallback:false}}]; paint();`);
assert.ok(levels.read().captionLevels.textContent.includes('acoustic 3/5'));
levels.run('video.currentTime = 5.2; paint();');
assert.equal(levels.read().captionLevels.textContent,
  'Current caption: acoustic 5/5 · A lot; source/scene 1/5 · Very little.');
assert.deepEqual(levels.read().exposure.axes, {texture:1,context:0});
const fallback = fixture(); fallback.run('paint();');
assert.equal(fallback.read().captionLevels.textContent, 'Prepared caption · selected levels not applied.');
console.log('PASS: actual replacement, pause/resume, seek, final boundary, interrupted recovery and five-level applied labels');

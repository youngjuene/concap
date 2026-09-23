// Isolated regression for long-viewing settings coalescing.
// Executes the actual settings queue block from study.js inside a Node VM.
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../study.js", import.meta.url), "utf8");
const start = source.indexOf("  const retrySettings = button");
const end = source.indexOf("  for (const key of [\"texture\", \"context\"])", start);
assert(start > 0 && end > start, "settings coalescing block not found");
const settingsBlock = source.slice(start, end);

const script = `
let state = {axes: {texture: 0.5, context: 0.5}};
const desired = {...state.axes};
const videoInfo = {id: "video-1", duration_ms: 300000};
const video = {currentTime: 45};
const settingsKey = axes => JSON.stringify({texture: axes.texture, context: axes.context});
let acceptedSettingsKey = settingsKey(desired), pendingSettings = null, sendingSettings = false, failedSettings = null;
let controlTimer, alive = true;
const status = {textContent: ""};
const retryButton = {hidden: false};
const sends = [];
const resolvers = [];
function t(text) { return text; }
function showMeter() {}
function button() { return retryButton; }
function clearTimeout() {}
async function send(action, data, redraw, valid) {
  sends.push({action, data: {...data}});
  return await new Promise(resolve => {
    resolvers.push(() => {
      state = {...state, axes: {texture: data.texture, context: data.context}};
      resolve({axes: state.axes});
    });
  });
}
${settingsBlock}
globalThis.setDesired = axes => Object.assign(desired, axes);
globalThis.commitSettings = origin => commit(origin);
globalThis.resolveOne = () => resolvers.shift()?.();
globalThis.snapshot = () => JSON.parse(JSON.stringify({
  sends: sends.map(entry => entry.data),
  state,
  retryHidden: retryButton.hidden,
  status: status.textContent,
}));
`;

const context = vm.createContext({structuredClone, Promise, setTimeout});
vm.runInContext(script, context);
const snap = () => JSON.parse(JSON.stringify(context.snapshot()));

context.setDesired({context: 1});
context.commitSettings("slider:context");
assert.equal(snap().sends.length, 1);
assert.deepEqual(snap().sends[0], {
  video_id: "video-1",
  position_hint_ms: 45000,
  origin: "slider:context",
  texture: 0.5,
  context: 1,
});

context.setDesired({texture: 0.5, context: 0.5});
context.commitSettings("reset");
assert.equal(snap().sends.length, 1, "reversion waits while first request is in flight");

context.resolveOne();
await new Promise(resolve => setTimeout(resolve, 0));
assert.equal(snap().sends.length, 2, "reversion is sent after in-flight acknowledgment");
assert.deepEqual(snap().sends[1], {
  video_id: "video-1",
  position_hint_ms: 45000,
  origin: "reset",
  texture: 0.5,
  context: 0.5,
});

context.resolveOne();
await new Promise(resolve => setTimeout(resolve, 0));
assert.deepEqual(snap().state.axes, {texture: 0.5, context: 0.5});
assert.equal(snap().retryHidden, true);
console.log(JSON.stringify({checks: ["A to B in flight to A ends accepted at A"], sends: snap().sends}));

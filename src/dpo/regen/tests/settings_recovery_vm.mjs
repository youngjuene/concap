// Exercise the production settings queue with committed requests whose replies are lost.
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../study.js", import.meta.url), "utf8");
const start = source.indexOf("  const retrySettings = button");
const end = source.indexOf('  for (const key of ["texture", "context"])', start);
assert(start > 0 && end > start, "production settings queue is present");

function harness() {
  const context = vm.createContext({Promise});
  vm.runInContext(`
    const desired = {texture: 0.5, context: 0.5};
    const videoInfo = {id: "video-1", duration_ms: 300000};
    const video = {currentTime: 45};
    const settingsKey = axes => JSON.stringify({texture: axes.texture, context: axes.context});
    let acceptedSettingsKey = settingsKey(desired), pendingSettings = null, sendingSettings = false, failedSettings = null;
    let controlTimer, alive = true;
    const status = {textContent: ""}, retryButton = {hidden: true};
    let retryAction, server = {...desired};
    const requests = [], replies = [];
    const t = text => text;
    const showMeter = () => {};
    const clearTimeout = () => {};
    function button(label, action) { retryAction = action; return retryButton; }
    async function send(action, data) {
      requests.push({...data});
      return new Promise((resolve, reject) => replies.push({
        acknowledge() { server = {texture: data.texture, context: data.context}; resolve({axes: {...server}}); },
        loseReply() { server = {texture: data.texture, context: data.context}; reject(new Error("both acknowledgment attempts lost")); }
      }));
    }
    ${source.slice(start, end)}
    globalThis.choose = axes => { Object.assign(desired, axes); commit("test-control"); };
    globalThis.acknowledge = () => replies.shift().acknowledge();
    globalThis.loseReply = () => replies.shift().loseReply();
    globalThis.retry = () => retryAction();
    globalThis.snapshot = () => JSON.parse(JSON.stringify({server, desired, requests, retryHidden: retryButton.hidden, status: status.textContent}));
  `, context);
  return context;
}

const A = {texture: 0.5, context: 0.5};
const B = {texture: 1, context: 1};
const snapshot = h => JSON.parse(JSON.stringify(h.snapshot()));
const settle = () => new Promise(resolve => setImmediate(resolve));

test("in-flight final A is sent after B commits but both acknowledgments are lost", async () => {
  const h = harness();
  h.choose(B); h.choose(A); h.loseReply();
  await settle();
  assert.equal(snapshot(h).requests.length, 2, "latest desired A must not be suppressed by a stale acknowledgment");
  h.acknowledge(); await settle();
  assert.deepEqual(snapshot(h).server, A);
  assert.deepEqual(snapshot(h).desired, A);
  assert.equal(snapshot(h).retryHidden, true);
});

test("a later return to A reconciles an ambiguous failed B", async () => {
  const h = harness();
  h.choose(B); h.loseReply(); await settle();
  assert.equal(snapshot(h).retryHidden, false);
  h.choose(A); await settle();
  assert.equal(snapshot(h).requests.length, 2, "A must be resent when server state is uncertain");
  h.acknowledge(); await settle();
  assert.deepEqual(snapshot(h).server, A);
  assert.equal(snapshot(h).retryHidden, true);
});

test("if final A also loses acknowledgment the visible retry sends final A", async () => {
  const h = harness();
  h.choose(B); h.choose(A); h.loseReply(); await settle();
  assert.equal(snapshot(h).requests.length, 2);
  h.loseReply(); await settle();
  assert.equal(snapshot(h).retryHidden, false, "the unresolved latest request needs a visible retry");
  h.retry(); await settle();
  assert.equal(snapshot(h).requests.length, 3);
  h.acknowledge(); await settle();
  assert.deepEqual(snapshot(h).server, A);
  assert.equal(snapshot(h).retryHidden, true);
});

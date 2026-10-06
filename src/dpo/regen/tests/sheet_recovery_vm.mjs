// Execute the shipped recovery functions. HTTP/storage are deterministic doubles.
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../sheet_questionnaire.js", import.meta.url), "utf8");
const prefix = source.slice(0, source.indexOf("function feedback("));
const checkpointSource = source.slice(source.indexOf("    function tick(playing"), source.indexOf("    function pause(message)"));
const checks = [];
const failures = [];
const detail = {session_id: "fixture", participant: "fixture", protocol: "sheet-v1", instrument_hash: "hash", practice: false, clip_index: 0, page: "P4", revision: 8};

function storage(values = new Map()) {
  return {get length() {return values.size;}, key: index => [...values.keys()][index] ?? null,
    getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key)};
}
function client(sessionStorage = storage(), fetch = async () => {throw new Error("unexpected fetch");}) {
  let ids = 0;
  const context = vm.createContext({console, Date, JSON, sessionStorage, fetch, crypto: {randomUUID: () => `request-${++ids}`}});
  vm.runInContext(`${prefix}\nglobalThis.api={scope,draftFor,storage,action};`, context);
  return context.api;
}
async function test(name, fn) {
  try {await fn(); checks.push(name);} catch (error) {failures.push({name, error: error.stack});}
}

await test("same-page revision change preserves zero and multiple-choice draft", () => {
  const api = client();
  api.storage.write(api.scope(detail), {answers: {F1: 0, R2: ["traffic", "bird"]}});
  const result = api.draftFor({...detail, revision: 9});
  assert.equal(result.answers.F1, 0);
  assert.deepEqual(Array.from(result.answers.R2), ["traffic", "bird"]);
});
await test("server draft restores values when this tab has no draft", () => {
  assert.equal(client().draftFor({...detail, draft: {F1: 0}}).answers.F1, 0);
});
await test("an explicitly cleared local draft does not resurrect server answers", () => {
  const api = client(); api.storage.write(api.scope(detail), {answers: {}});
  assert.equal(Object.keys(api.draftFor({...detail, draft: {F1: 6}}).answers).length, 0);
});
await test("old revision-based storage keys migrate without losing pending request", () => {
  const store = storage(), pending = {signature: "old", payload: {key: "pending-uuid"}};
  const identity = [detail.session_id, detail.participant, detail.protocol, detail.instrument_hash, false, 0, "P4"];
  store.setItem(`regen.workbook.draft:${JSON.stringify([...identity, 7])}`, JSON.stringify({answers: {F1: 6}}));
  store.setItem(`regen.workbook.draft:${JSON.stringify([...identity, 8])}`, JSON.stringify({answers: {F1: 0}, pending}));
  const api = client(store), restored = api.draftFor({...detail, revision: 9});
  assert.equal(restored.answers.F1, 0);
  assert.equal(restored.pending.payload.key, "pending-uuid");
  assert.equal(api.storage.read(api.scope(detail)).answers.F1, 0);
});
await test("lost submission reply replays the original key and values after reload", async () => {
  const store = storage(), calls = []; let fail = true;
  const fetch = async (_url, options) => {
    calls.push(JSON.parse(options.body));
    if (fail) throw new TypeError("lost response");
    return {ok: true, json: async () => ({page: "P5", revision: 9})};
  };
  const first = client(store, fetch); first.storage.write(first.scope(detail), {answers: {F1: 0}});
  await assert.rejects(first.action("submit", {answers: {F1: 0}}, detail));
  fail = false; await client(store, fetch).action("submit", {answers: {F1: 0}}, detail);
  assert.deepEqual(calls[0], calls[1]);
});

function playback(request) {
  let id = 0;
  const intervals = new Map(); let timerID = 0;
  const context = vm.createContext({console, crypto: {randomUUID: () => `checkpoint-${++id}`},
    document: {hidden: false, fullscreenElement: null},
    video: {paused: true, currentTime: 1, duration: 10, pause() {this.paused = true;}, async play() {this.paused = false;}},
    media: {duration_ms: 10000}, request,
    setInterval: callback => {intervals.set(++timerID, callback); return timerID;},
    clearInterval: key => intervals.delete(key),
    scope: value => JSON.stringify([value.session_id, value.page, value.clip_index]),
  });
  vm.runInContext(`let alive=true,ending=false,timer,chain=Promise.resolve(),sequence=-1,pendingSample=null,checkpoint={position_ms:0,sequence:-1};
    const overlay={hidden:false};function pause(error){throw new Error(error);}
    const detail={session_id:"fixture",page:"P1",clip_index:0,revision:1};
    const rendered=[];async function render(state){rendered.push(state);alive=false;}
    ${checkpointSource}
    globalThis.api={tick,detail,rendered,playFromCheckpoint:typeof playFromCheckpoint==='function'?playFromCheckpoint:null,snapshot:()=>({alive,pendingSample,sequence,checkpoint})};`, context);
  return {api: context.api, video: context.video, intervals};
}
await test("409 refreshes state and allows a fresh checkpoint instead of retrying rejected key", async () => {
  const calls = []; let conflict = true;
  const p = playback(async (path, payload) => {
    calls.push({path, payload: payload ? structuredClone(payload) : null});
    if (path === "state") return {session_id: "fixture", page: "P1", clip_index: 0, revision: 7, media: {playback: {position_ms: 2000, sequence: 4}}};
    if (conflict) {conflict = false; throw Object.assign(new Error("state conflict"), {status: 409});}
    assert.equal(payload.revision, 7); assert.ok(payload.data.sequence > 4);
    assert.equal(payload.data.position_ms, 2000);
    return {revision: 8, media: {playback: {position_ms: 2000, sequence: payload.data.sequence}}};
  });
  await p.api.tick(true).catch(error => assert.equal(error.status, 409));
  await p.api.tick(true);
  assert.ok(calls.some(call => call.path === "state"));
  const posts = calls.filter(call => call.payload);
  assert.notEqual(posts[0].payload.key, posts[1].payload.key);
  assert.equal(p.api.detail.revision, 8);
});
await test("network response loss still replays the exact checkpoint key before progressing", async () => {
  const calls = []; let first = true;
  const p = playback(async (path, payload) => {
    assert.equal(path, "playback"); calls.push(structuredClone(payload));
    if (first) {first = false; throw new TypeError("lost response");}
    return {revision: 2, media: {playback: {position_ms: payload.data.position_ms, sequence: payload.data.sequence}}};
  });
  await assert.rejects(p.api.tick(true)); await p.api.tick(true);
  assert.deepEqual(calls[0], calls[1]);
  assert.notEqual(calls[1].key, calls[2].key);
});
await test("409 that already moved to another page renders that page and stops old playback", async () => {
  const calls = [];
  const p = playback(async (path, payload) => {
    calls.push({path, payload});
    if (path === "state") return {session_id: "fixture", page: "P2", clip_index: 0, revision: 4};
    throw Object.assign(new Error("advanced"), {status: 409});
  });
  await p.api.tick(true).catch(() => {}); await p.api.tick(true);
  assert.equal(p.api.rendered[0]?.page, "P2");
  assert.equal(calls.filter(call => call.path === "playback").length, 1);
});
await test("resuming after an initial checkpoint conflict restarts periodic checkpoints", async () => {
  let conflict = true, revision = 4; const calls = [];
  const p = playback(async (path, payload) => {
    if (path === "state") return {session_id: "fixture", page: "P1", clip_index: 0, revision, media: {playback: {position_ms: 0, sequence: 1}}};
    calls.push(structuredClone(payload));
    if (conflict) {conflict = false; throw Object.assign(new Error("conflict"), {status: 409});}
    return {revision: ++revision, media: {playback: {position_ms: payload.data.position_ms, sequence: payload.data.sequence}}};
  });
  await assert.rejects(p.api.tick(true));
  assert.equal(typeof p.api.playFromCheckpoint, "function");
  await p.api.playFromCheckpoint();
  assert.equal(p.video.paused, false); assert.equal(p.intervals.size, 1);
  p.video.currentTime = 2; [...p.intervals.values()][0]();
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(calls.at(-1).data.position_ms, 2000);
  await p.api.playFromCheckpoint(); assert.equal(p.intervals.size, 1);
});

console.log(JSON.stringify({checks, failures}, null, 2));
if (failures.length) process.exitCode = 1;

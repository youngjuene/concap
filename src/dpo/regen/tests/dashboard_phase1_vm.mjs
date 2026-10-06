/* CPU-only regressions for the production Phase 1 observational dashboard. */
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {fileURLToPath} from "node:url";

const source = fs.readFileSync(fileURLToPath(new URL("../dashboard.js", import.meta.url)), "utf8");
const start = source.indexOf("  // Phase 1 observational analysis.");
const end = source.indexOf("  function responseFilterValue(", start);
assert.ok(start >= 0 && end > start, "Phase 1 observational dashboard functions are present");

class Element {
  constructor(tag, className = "", text = "") {
    Object.assign(this, {tag, className, text:String(text), children:[], attributes:{}, dataset:{}, listeners:{}});
    this.classList = {add: name => { this.className += ` ${name}`; }};
  }
  append(...children) { this.children.push(...children.filter(value => value != null)); }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  set textContent(value) { this.text = String(value); }
  set innerHTML(_) { throw Error("API data must never be rendered with innerHTML"); }
}
const root = new Element("section"), calls = [], state = {
  view:"phase1", query:"participant & one", language:"ko", status:"phase1", includeQa:false, phase1Clip:"A / 1",
};
const list = value => Array.isArray(value) ? value : [];
const element = (tag, cls = "", text = "") => new Element(tag, cls, text);
const append = (parent, ...children) => { parent.append(...children.flat()); return parent; };
const sandbox = {
  state, pages:{}, phase1:null, phase1Error:"", phase1Loading:false, phase1RequestId:0, phase1LoadedQuery:"",
  URLSearchParams, Node:Element, list, number:value => typeof value === "number" && Number.isFinite(value),
  valueText:value => value == null ? "Missing" : typeof value === "object" ? JSON.stringify(value) : String(value),
  count:value => value == null ? "—" : String(value), timestamp:value => value,
  $:() => root, el:element, append,
  empty:(title, body) => append(element("div", "empty-state", title), element("p", "", body)),
  titleBlock:(title, description, action) => append(element("header", "", title), element("p", "", description), action),
  panel:(title, description, action) => append(element("section", "panel", title), element("p", "", description), action),
  metadata:(entries, label) => append(element("details", "", label), ...entries.map(([key,value]) => element("p", "", `${key}: ${JSON.stringify(value)}`))),
  table:(headers, rows, label) => append(element("table", "", label),
    append(element("tr"), ...headers.map(header => element("th", "", header))),
    ...rows.map(row => append(element("tr"), ...row.map(value => value instanceof Element ? value : element("td", "", value))))),
  paged:rows => rows,
  pager:rows => element("nav", "", `${rows.length} records`),
  participantButton:id => element("button", "", id),
  selectFilter:(label, key, values) => append(element("select", key, label), ...values.map(([value, text]) => {
    const option = element("option", "", text); option.value = value; return option;
  })),
  button:(label, handler) => { const node=element("button", "", label); node.listeners.click=handler; return node; },
  persist:() => {}, renderStatus:() => {}, renderWarnings:() => {},
  fetchJson:url => new Promise((resolve, reject) => calls.push({url, resolve, reject})),
};
vm.createContext(sandbox);
vm.runInContext(source.slice(start, end)+"\nObject.assign(globalThis,{phase1Percent,phase1Delta,phase1Change,phase1Rate,phase1Relation});", sandbox);
const all = node => [node, ...node.children.flatMap(child => child instanceof Element ? all(child) : [])];
const text = node => all(node).map(entry => entry.text).join("\n");

const scope = {clip_id:"clip-1", instrument_hash:"instrument-A", codebook_version:"v1", codebook_hash:"codebook-A",
  av_assigned_condition:"incongruent", description_depth:"shallow", caption_strategy:"opposite", menu_changed:true};
const fixture = {
  schema:"dpo.phase1-analysis/v1", available_clips:["clip-1", "clip-2"], warnings:["Historical mask metrics are unavailable."],
  observations:[{...scope, participant:"<img src=x onerror=alert(1)>", clip_index:0, round:1, status:"partial",
    selected_mask_area_ratio:null, selected_mask_id:null, sound_type:"engine", sound_family:null,
    av_selection_relation:null, offered_options:[{value:"engine", label:"Engine"}], issues:["mask metric missing"]}],
  pairs:[{...scope, participant:"p1", clip_index:0, status:"paired", issues:[],
    selected_mask_area_ratio_1:.1, selected_mask_area_ratio_2:.25, area_delta_pp:15,
    sound_type_1:"engine", sound_type_2:"car", sound_family_1:"things", sound_family_2:"things",
    type_changed:1, family_changed:0, frame_changed:true, order_changed:true,
    av_selection_relation_1:null, av_selection_relation_2:0}],
  summaries:[{...scope,
    type_changed:{total_pairs:3, valid_pairs:2, na_pairs:1, changed_pairs:1, rate:.5},
    family_changed:{total_pairs:3, valid_pairs:1, na_pairs:2, changed_pairs:0, rate:0},
    area_delta_pp:{total_pairs:3, valid_pairs:1, na_pairs:2, mean:15},
    av_selection_relation_1:{total_observations:3, valid_observations:1, na_observations:2, relation_sum:1, rate:1},
    av_selection_relation_2:{total_observations:3, valid_observations:0, na_observations:3, relation_sum:0, rate:null}}],
  selection_rates:[{...scope, kind:"type", round:1, code:"bird", numerator:0, denominator:0, rate:null},
    {...scope, kind:"family", round:2, code:"things", numerator:1, denominator:2, rate:.5}],
  transitions:[{...scope, kind:"type", from_code:"engine", to_code:"car", count:1},
    {...scope, kind:"visual_sound", round:2, from_code:"visual-mask-probe", to_code:"engine", count:2}],
};

assert.equal(sandbox.phase1Percent(null), "NA");
assert.equal(sandbox.phase1Percent(0), "0.0%");
assert.equal(sandbox.phase1Percent(.1), "10.0%");
assert.equal(sandbox.phase1Delta(15), "+15.0 pp");
assert.equal(sandbox.phase1Delta(null), "NA");
assert.equal(sandbox.phase1Change(null), "NA");
assert.equal(sandbox.phase1Change(false), "Unchanged");
assert.equal(sandbox.phase1Change(1), "Changed");
assert.equal(sandbox.phase1Change(0), "Unchanged");
assert.equal(sandbox.phase1Rate(0, 0), "NA");
assert.equal(sandbox.phase1Rate(0, 1), "0.0%");
assert.match(sandbox.phase1Relation(0), /^0 /);
assert.equal(sandbox.phase1Relation(null), "NA");

const query = new URLSearchParams(sandbox.phase1Query());
assert.deepEqual(Object.fromEntries(query), {include_qa:"false", participant:"participant & one", language:"ko", status:"phase1", clip_id:"A / 1"});
sandbox.phase1 = fixture;
assert.deepEqual(JSON.parse(JSON.stringify(sandbox.phase1SummaryRows().map(row=>row.slice(1)))), [
  ["Type change", "50.0%", "3", "2", "1", "1"],
  ["Family change", "0.0%", "3", "1", "2", "0"],
  ["Mask area change", "+15.0 pp", "3", "1", "2", "Not applicable"],
], "Different metrics retain their independently supplied valid and NA denominators");
sandbox.renderPhase1();
const rendered = text(root);
for (const required of ["10.0%", "25.0%", "+15.0 pp", "Type change", "Family change", "Unchanged", "NA",
  "Total pairs", "Valid pairs", "NA pairs", "Eligible responses", "Assigned AV condition", "Selection relation",
  "Total observations", "Valid observations", "NA observations", "선택 교차빈도",
  "instrument-A", "codebook-A", "Historical mask metrics are unavailable.", "<img src=x onerror=alert(1)>"])
  assert(rendered.includes(required), `Rendered output includes ${required}`);
const csv = all(root).filter(node => node.tag === "a");
assert.equal(csv.length, 2);
for (const link of csv) assert.equal(new URL(link.href, "http://localhost").search.slice(1), sandbox.phase1Query());
assert(all(root).some(node => node.tag === "option" && node.value === "clip-2"), "Available clips retain alternatives outside the active clip");
assert(!/accuracy|correctness score/i.test(rendered), "Observed changes are not labelled as accuracy");
const transitionTable=all(root).find(node=>node.tag==="table"&&node.text==="Sound transitions");
assert(!text(transitionTable).includes("visual-mask-probe"), "Within-round visual/sound counts are not shown as sound transitions");
assert(rendered.includes("visual-mask-probe"), "Cross-counts remain available without a relation table");

// A response for an old search must never replace a newer filtered cohort.
state.query = "old";
const older = sandbox.fetchPhase1();
state.query = "new";
const newer = sandbox.fetchPhase1();
assert.equal(calls.length, 2);
calls[1].resolve({...fixture, warnings:["new cohort"]});
await newer;
calls[0].resolve({...fixture, warnings:["old cohort"]});
await older;
assert.equal(sandbox.phase1.warnings[0], "new cohort");
state.query = "unavailable cohort";
const failed = sandbox.fetchPhase1();
assert.equal(sandbox.phase1, null, "A changed filter immediately removes old-cohort results");
calls[2].reject(new Error("Request failed (503)."));
await failed;
assert.equal(sandbox.phase1, null);
assert(text(root).includes("Phase 1 analysis unavailable"));
assert(!text(root).includes("new cohort"), "Failure cannot display a different cohort as current analysis");
console.log("Phase1 dashboard: NA/zero, independent denominators, safe text, cohort exports and stale-response regressions passed.");

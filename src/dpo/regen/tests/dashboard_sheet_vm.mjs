/* Bounded regressions for the production response grouping/filter/chart functions. */
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {fileURLToPath} from "node:url";

const sourcePath = process.env.DASHBOARD_SOURCE || fileURLToPath(new URL("../dashboard.js", import.meta.url));
const source = fs.readFileSync(sourcePath, "utf8");
const start = source.indexOf("  function responseFilterValue(");
const end = source.indexOf("  function responseTable(", start);
assert.ok(start >= 0 && end > start, "response functions are present");
class Element {
  constructor(tag, className = "", text = "") {
    Object.assign(this, {tag, className, text, dataset:{}, style:{}, children:[], attributes:{}});
  }
  append(...nodes) { this.children.push(...nodes.filter(node => node != null)); }
  setAttribute(key, value) { this.attributes[key] = value; }
}
let included = [];
const state = {};
const sandbox = {
  state,
  includedResponses: () => included,
  el: (tag, cls = "", text = "") => new Element(tag, cls, text),
  append: (parent, ...children) => { parent.append(...children.flat()); return parent; },
  metadata: (items) => new Element("metadata", "", JSON.stringify(items)),
  list: value => Array.isArray(value) ? value : [],
  number: value => typeof value === "number" && Number.isFinite(value),
  valueText: value => value == null ? "Missing" : typeof value === "object" ? JSON.stringify(value) : String(value),
  languageName: value => value,
  count: value => String(value),
};
vm.createContext(sandbox);
vm.runInContext(source.slice(start, end), sandbox);
const common = {
  phase:1, scope:"P11", video_id:"short-1", instrument:"frozen", flow_version:"dpo.sheet-questionnaire/v1",
  condition:"second", language:"ko", wording_available:true, item_id:"R2", item_text:"들은 소리를 고르세요.",
  type:"multi", options:["traffic", "bird", "memory_unknown"], option_labels:["차량 소리", "새 소리", "기억나지 않음"],
};
const rowsFor = (sample, values) => values.map((value, index) => ({...sample, participant:`p-${index}`, value}));
const binsFor = (sample, values) => sandbox.distribution({row:sample, rows:rowsFor(sample, values)}).children
  .filter(node => node.className === "distribution-row")
  .map(node => ({label:node.children[0].text, count:Number(node.children[2].text.split(" · ")[0])}));
const tests = [
  ["multi counts membership without click-order categories", () => {
    const bins = binsFor(common, [["traffic", "bird"], ["traffic"], ["bird", "traffic"]]);
    assert.deepEqual(bins, [{label:"차량 소리", count:3}, {label:"새 소리", count:2}, {label:"기억나지 않음", count:0}]);
  }],
  ["exclusive unknown and missing remain distinct", () => {
    const bins = binsFor(common, [["memory_unknown"], null]);
    assert.equal(bins.find(bin => bin.label === "기억나지 않음")?.count, 1);
    assert.equal(bins.find(bin => bin.label === "Missing")?.count, 1);
    assert.ok(!bins.some(bin => bin.label.startsWith("[")));
  }],
  ["zero-to-six ratings retain endpoint labels", () => {
    const bins = binsFor({...common, type:"rating", min:0, max:6, points:7, options:[], labels:["전혀", "", "", "보통", "", "", "매우"]}, [0, 6]);
    assert.equal(bins.length, 7);
    assert.deepEqual(bins.filter(bin => bin.count), [{label:"0 · 전혀", count:1}, {label:"6 · 매우", count:1}]);
  }],
  ["factorial assignment and strategy separate groups consistently", () => {
    const a = {...common, value:["traffic"], caption_strategy:"maintain", stimulus_assignment:{audiovisual_congruence:"congruent", description_depth:"shallow"}};
    const reordered = {...a, stimulus_assignment:{description_depth:"shallow", audiovisual_congruence:"congruent"}};
    const depth = {...a, stimulus_assignment:{...a.stimulus_assignment, description_depth:"deep"}};
    const congruence = {...a, stimulus_assignment:{...a.stimulus_assignment, audiovisual_congruence:"incongruent"}};
    const strategy = {...a, caption_strategy:"opposite"};
    assert.equal(sandbox.responseGroups([a, reordered, depth, congruence, strategy]).length, 4);
    included = [a, depth, congruence, strategy];
    state.responseCongruence = "congruent"; state.responseDepth = "shallow"; state.responseStrategy = "opposite";
    const filtered = sandbox.filteredResponses();
    assert.equal(filtered.length, 1);
    assert.equal(filtered[0].caption_strategy, "opposite");
  }],
];
const results = [];
for (const [name, run] of tests) {
  try { run(); results.push({name, passed:true}); }
  catch (error) { results.push({name, passed:false, error:error.message}); }
}
console.log(JSON.stringify({source:sourcePath, tests:results}, null, 2));
if (results.some(result => !result.passed)) process.exitCode = 1;

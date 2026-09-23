// Execute the production scale decoder, with DOM access only stubbed at startup.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

const source = await readFile(new URL('../study.js', import.meta.url), 'utf8');
const context = vm.createContext({URL, document: {getElementById: () => ({})}, window: {location: {href: 'http://localhost/'}}});
vm.runInContext(source.slice(0, source.indexOf('function storageKey')) + '\nglobalThis.decode = typeof captionDetailScale === "function" ? captionDetailScale : null;', context);
const contract = {version: 'five-level/v1', values: [0, .25, .5, .75, 1], labels: {
  en: ['Very little', 'A little', 'A moderate amount', 'Quite a lot', 'A lot'],
  ko: ['매우 적게', '적게', '보통', '많이', '매우 많이'],
}};
function scale(contractValue = contract, language = 'en') {
  assert.equal(typeof context.decode, 'function', 'production must decode the frozen five-level control contract');
  return context.decode(contractValue, language);
}
test('five ordinal inputs map to exact stored quarter steps and meaningful labels', () => {
  const s = scale();
  assert.deepEqual([s.min, s.max, s.step], [1, 5, 1]);
  for (const [input, axis, label] of [[1,0,'Very little'],[2,.25,'A little'],[3,.5,'A moderate amount'],[4,.75,'Quite a lot'],[5,1,'A lot']]) {
    assert.equal(s.fromInput(input), axis);
    assert.equal(s.toInput(axis), input);
    assert.equal(s.display(axis), `${input}/5 · ${label}`);
  }
});
test('pointer positions snap to five cells, including boundaries and clamped edges', () => {
  const s = scale();
  for (const [input, expected] of [[-.1,0],[.12,0],[.125,.25],[.37,.25],[.375,.5],[.62,.5],[.625,.75],[.87,.75],[.875,1],[1.1,1]]) assert.equal(s.snap(input), expected);
});
test('localized display comes from the frozen contract and does not confuse selected axes', () => {
  const custom = structuredClone(contract);
  custom.labels.ko[1] = '고정된 두 번째 수준';
  const s = scale(custom,'ko');
  assert.equal(s.display(.25), '2/5 · 고정된 두 번째 수준');
  assert.equal(s.display(1), '5/5 · 매우 많이');
  assert.equal(s.compact(.5), '3/5');
});
test('profiles without a control contract preserve legacy percentages and continuous inputs', () => {
  const s = scale(null);
  assert.deepEqual([s.min,s.max,s.step],[0,100,1]);
  assert.equal(s.fromInput(51), .51);
  assert.equal(s.toInput(.77), 77);
  assert.equal(s.snap(.773), .77);
  assert.equal(s.display(.08), '8%');
});

// Run: node --test tests/manual_ocr_frontend.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

test('manual OCR is sent literally, included in cache identity, and cleared for automatic mode', async () => {
  const requests = [];
  const field = { value: ' HTV7 ', addEventListener() {} };
  const asrField = { value: '', addEventListener(event, handler) { this.handler = handler; } };
  const textarea = { value: 'woman cycling', addEventListener() {} };
  const context = vm.createContext({
    document: {
      getElementById: id => id === 'ocr-text-input' ? field : id === 'asr-text-input' ? asrField : { addEventListener() {}, querySelector() { return null; } },
      querySelectorAll: () => [textarea],
    },
    window: {}, console, AbortController, performance,
    debounce: fn => fn,
    toggleLoadingIndicator() {}, updateUIWithSearchResults() {},
    searchCache: new Map(), currentAbortController: null,
    fetch: async (url, options) => {
      requests.push(JSON.parse(options.body));
      return { ok: true, json: async () => ({ kq: [{ id: requests.length }] }) };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../baseline/retrievalBaseline/frontend/src/scripts/query_backend.js'), 'utf8'), context);
  const search = async () => {
    await vm.runInContext('performSearchFromInputs()', context);
    await new Promise(resolve => setImmediate(resolve));
  };
  await search();
  assert.equal(requests[0].ocr_text, 'HTV7');
  field.value = '115';
  await search();
  assert.equal(requests[1].ocr_text, '115');
  field.value = '';
  await search();
  assert.equal(requests[2].ocr_text, '');
  field.value = 'HTV7';
  await search();
  assert.equal(requests.length, 3);
  assert.equal(vm.runInContext('data.kq[0].id', context), 1);
  asrField.value = ' múa lân, lễ hội ';
  let prevented = false;
  asrField.handler({key:'Enter', shiftKey:false, preventDefault() {prevented=true;}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(prevented, true);
  assert.equal(requests.length, 4);
  assert.equal(requests[3].asr_text, 'múa lân, lễ hội');
  assert.equal(requests[3].ocr_text, 'HTV7');
});

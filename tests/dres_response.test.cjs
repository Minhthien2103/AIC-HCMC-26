const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('baseline/retrievalBaseline/frontend/src/scripts/submit_dres.js', 'utf8');
const helper = source.slice(source.indexOf('async function readDresResponse'), source.indexOf('function getDresBaseUrl'));
const scope = {};
vm.createContext(scope);
vm.runInContext(helper, scope);
(async () => {
  for (const [body, expected] of [['{"message":"expired"}', 'expired'], ['Bad Gateway', 'Bad Gateway'], ['<html>404</html>', '<html>404</html>'], ['', undefined]]) {
    let reads = 0;
    const response = {text: async () => { assert.equal(++reads, 1); return body; }};
    const result = await scope.readDresResponse(response);
    assert.equal(result.message, expected);
    assert.equal(reads, 1);
  }
  const config = fs.readFileSync('baseline/retrievalBaseline/frontend/src/scripts/config.js', 'utf8');
  const oldEval = config.match(/getItem\('evaluationID'\) === ("[^"]+")/)[1];
  const oldSession = config.match(/getItem\('contestSessionID'\) === ("[^"]+")/)[1];
  for (const stale of [true, false]) {
    const saved = new Map([['evaluationID', stale ? JSON.parse(oldEval) : 'custom-evaluation'], ['contestSessionID', stale ? JSON.parse(oldSession) : 'custom-session']]);
    const ctx = {window: {}, localStorage: {getItem: k => saved.get(k), setItem: (k,v) => saved.set(k,v)}, console};
    vm.runInNewContext(config, ctx);
    assert.equal(saved.get('evaluationID'), stale ? '675a3430-abaa-4d8c-b650-c8db1246d429' : 'custom-evaluation');
    assert.equal(saved.get('contestSessionID'), stale ? ctx.window.CONFIG.DRES_SESSION_ID : 'custom-session');
  }
  console.log('PASS: 4 response formats read once; stale settings migrated, custom settings preserved. No network calls.');
})().catch(error => { console.error(error); process.exitCode = 1; });

const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture(fetch) {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, {addEventListener() {}, value:''}); return elements.get(id); };
  const updates = [];
  const context = vm.createContext({document:{getElementById:get, querySelectorAll:()=>[]},
    window:{}, console:{error() {}}, AbortController, performance, debounce:fn=>fn,
    searchCache:new Map(), currentAbortController:null, fetch,
    toggleLoadingIndicator() {}, updateUIWithSearchResults:items=>updates.push(items)});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../baseline/retrievalBaseline/frontend/src/scripts/query_backend.js'), 'utf8'), context);
  return {get, updates, run:code=>vm.runInContext(code, context)};
}
const payload = {kq:[{id:1}],query_preparation:[{source:'local_llm', chunks:['Cyclist number 71'], token_counts:[5],token_limit:64,warnings:[]}]};

test('actual English query remains visible on a cache hit', async () => {
  let calls=0;
  const f=fixture(async()=>{calls++; return {ok:true,json:async()=>payload};});
  await f.run("performSearchMulti(['đua xe đạp'])");
  assert.match(f.get('query-diagnostics-text').textContent, /Cyclist number 71/);
  f.get('query-diagnostics-text').textContent='stale';
  await f.run("performSearchMulti(['đua xe đạp'])");
  assert.equal(calls,1);
  assert.match(f.get('query-diagnostics-text').textContent, /Cyclist number 71/);
});

test('search error is shown and previous frames are cleared; failed queries are not cached', async () => {
  let calls=0;
  const f=fixture(async()=>{calls++; return {ok:false,json:async()=>({detail:'Search backend unavailable'})};});
  await f.run("performSearchMulti(['đua xe đạp'])");
  assert.equal(f.get('query-error').hidden,false);
  assert.match(f.get('query-error').textContent,/Search backend unavailable/);
  assert.equal(f.updates.at(-1).length,0);
  await f.run("performSearchMulti(['đua xe đạp'])");
  assert.equal(calls,2);
});

test('late response cannot replace a newer cached result and its query explanation', async () => {
  let finish;
  const f=fixture(async(url,options)=>{
    const q=JSON.parse(options.body).queries[0];
    if(q==='slow') return new Promise(resolve=>{finish=resolve;});
    return {ok:true,json:async()=>payload};
  });
  await f.run("performSearchMulti(['cached'])");
  const slow=f.run("performSearchMulti(['slow'])");
  await f.run("performSearchMulti(['cached'])");
  finish({ok:true,json:async()=>({kq:[{id:99}]})});
  await slow;
  assert.equal(f.updates.at(-1)[0].id,1);
  assert.match(f.get('query-diagnostics-text').textContent,/Cyclist number 71/);
});

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const scripts = path.join(__dirname, '../baseline/retrievalBaseline/frontend/src/scripts');

function fixture(response = {ok: true, status: 200, text: async () => '{"status":"OK"}'}) {
  class Element {
    constructor() {
      this.children = []; this.style = {}; this.dataset = {}; this.value = '';
      this.classList = { contains: () => false, add() {}, remove() {} };
    }
    addEventListener() {}
    setAttribute() {}
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    focus() { this.focused = true; }
  }
  const elements = new Map();
  const element = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const inputs = () => {
    const found = [];
    const walk = item => { if (item.className === 'vqa-input') found.push(item); item.children.forEach(walk); };
    for (const item of elements.values()) walk(item);
    return found;
  };
  const requests = [], messages = [];
  const saved = new Map([['dresHost', 'https://dres.example'], ['evaluationID', 'evaluation-1'], ['contestSessionID', 'secret-session']]);
  const context = vm.createContext({ console: {log() {}, error() {}}, URL, setTimeout: fn => fn(),
    document: { getElementById: element, querySelector: element, querySelectorAll: inputs,
      createElement: () => new Element(), addEventListener() {} },
    window: { location: { href: 'http://localhost:8006/' }, addEventListener() {} },
    socket_share: null, WebSocket: { OPEN: 1 }, alert: message => messages.push(message),
    localStorage: { getItem: key => saved.get(key) || null, setItem: (key, value) => saved.set(key, value) },
    fetch: async (url, options) => {
      requests.push({url, body: JSON.parse(options.body)});
      return response;
    }
  });
  for (const name of ['frame_metadata.js', 'export.js', 'submit_dres.js']) {
    vm.runInContext(fs.readFileSync(path.join(scripts, name), 'utf8'), context, {filename: name});
  }
  context.report = message => messages.push(message);
  vm.runInContext('showTemporaryAlert = report;', context);
  element('global-task-select').value = 'kis-qa';
  vm.runInContext("activeTask = 'kis-qa';", context);
  const run = code => vm.runInContext(code, context);
  const pin = (video, time = 0, keyframe = '001') => {
    context.item = {videoId: video, keyframeName: keyframe, frameId: 999,
      timeSeconds: time, src: `http://localhost:8007/keyframes/${video}/${keyframe}.jpg`};
    run('addImageToExportArea(item)');
  };
  return {context, run, pin, requests, messages, element, inputs, saved};
}

test('KIS collection is optional, scoped to evaluation, and never added to QA text answers', async () => {
  const f = fixture(); f.pin('L01_V001', 1);
  f.element('global-task-select').value = 'kis';
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests[0].body.answerSets[0].answers[0].mediaItemCollectionName, undefined);
  assert.ok(f.messages.some(message => message.includes('Đã nộp thành công')));
  f.run("localStorage.setItem(dresCollectionKey(getDresBaseUrl(), getDresEvaluationId()), 'Official collection')");
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests[1].body.answerSets[0].answers[0].mediaItemCollectionName, 'Official collection');
  f.element('global-task-select').value = 'kis-qa';
  f.run("updateVqaInput(exportedImages[0].key, '7')");
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests[2].body.answerSets[0].answers[0].mediaItemCollectionName, undefined);
  assert.equal(f.run("getDresCollection(getDresBaseUrl(), 'different-evaluation')"), '');
  assert.equal(f.run("getDresCollection('https://other.example', getDresEvaluationId())"), '');
});

test('DRES 500 saves redacted context, displays collection guidance, and does not retry', async () => {
  const f = fixture({ok:false, status:500, text:async()=>JSON.stringify({description:'Collection not found; ?session=secret-session'})});
  f.pin('L01_V001', 1); f.element('global-task-select').value = 'kis';
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests.length, 1);
  const report = f.saved.get('dresLastSubmissionError');
  assert.doesNotMatch(report, /secret-session/);
  const data = JSON.parse(report);
  assert.equal(data.status, 500);
  assert.equal(data.payload.answerSets[0].answers[0].mediaItemName, 'L01_V001');
  assert.equal(data.endpoint, 'https://dres.example/api/v2/submit/evaluation-1');
  assert.ok(f.messages.some(message => message.includes('ban tổ chức')));
  assert.ok(f.messages.every(message => !message.includes('secret-session')));
  assert.equal(f.element('submit-button').disabled, false);
});

test('QA answers remain separate across videos and fullscreen; deletion/reset clear old answers', async () => {
  const f = fixture();
  f.pin('L01_V001', 12.3456); f.pin('L02_V001', 12.3456);
  let fields = f.inputs();
  fields[0].value = '007'; fields[0].oninput();
  fields[1].value = '42'; fields[1].oninput();
  f.run('loadExportContent()');
  assert.deepEqual(f.inputs().map(x => x.value), ['007', '42', '007', '42']);
  // Editing fullscreen updates the pinned input and the same stored answer.
  fields = f.inputs(); fields[2].value = '009'; fields[2].oninput();
  assert.equal(fields[0].value, '009');
  f.run("exportedImages[0].frameInfo = 'WRONG-9999'");
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests[0].body.answerSets[0].answers[0].text, '009-L01_V001-12346');
  f.run('deleteExportImage(0)');
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests[1].body.answerSets[0].answers[0].text, '42-L02_V001-12346');
  f.run('resetExportArea()'); f.pin('L02_V001', 12.3456);
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests.length, 2); // No inherited answer after reset.
});

test('empty/non-numeric answers block the actual submit handler; zero and leading zeroes work', async () => {
  const f = fixture(); f.pin('L01_V001', 0);
  for (const value of ['', ' ', 'red', '1-2', '-1', '1.5']) {
    f.context.answer = value;
    f.run('updateVqaInput(exportedImages[0].key, answer)');
    await f.run('submit_to_dres_v2()');
  }
  assert.equal(f.requests.length, 0);
  f.run("updateVqaInput(exportedImages[0].key, '0')");
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests[0].body.answerSets[0].answers[0].text, '0-L01_V001-0');
});

test('KIS uses full timestamp precision and rejects legacy label-only or missing timestamps', async () => {
  const f = fixture(); f.element('global-task-select').value = 'kis';
  f.pin('L01_V001', 83.56789);
  await f.run('submit_to_dres_v2()');
  assert.deepEqual(f.requests[0].body.answerSets[0].answers[0], {mediaItemName: 'L01_V001', start: 83568, end: 83568});
  for (const value of [null, undefined, '', -1, NaN, Infinity]) {
    f.context.badTime = value;
    f.run('exportedImages[0].timeSeconds = badTime');
    await f.run('submit_to_dres_v2()');
  }
  f.run("exportedImages = [{frameInfo:'L01_V001-83',frameId:83}]");
  await f.run('submit_to_dres_v2()');
  assert.equal(f.requests.length, 1);
});

test('search pins, preview copies and drag/drop preserve metadata; duplicate pins stay unique', () => {
  const f = fixture();
  f.run(`const picture = {src:'http://localhost:8007/keyframes/L01_V001/002.jpg'};
    registerFrameMetadata(picture, frameFromResult({video:'L01_V001', frame_id:'002', original_frame_id:800, time:26.6667}, picture.src));
    pinImage(picture);
    pinImage({src:picture.src});`);
  assert.equal(f.run('exportedImages.length'), 1);
  assert.equal(f.run('exportedImages[0].frameId'), 800);
  let payload;
  f.context.event = { target: {src:'http://localhost:8007/keyframes/L01_V001/002.jpg'}, stopPropagation() {},
    dataTransfer: {setData(type, value) {payload = value;}}, preventDefault() {throw Error('Unexpected rejection');} };
  f.run('drag(event)');
  f.run('resetExportArea()');
  f.context.event = {preventDefault() {}, dataTransfer:{getData:()=>payload}};
  f.run('drop(event)');
  assert.equal(f.run('exportedImages[0].timeSeconds'), 26.6667);
});

test('CSV mapping separates keyframe number, original frame and timestamp; failure clears old timeline', async () => {
  const f = fixture();
  vm.runInContext(fs.readFileSync(path.join(scripts, 'show_videoframe.js'), 'utf8'), f.context);
  f.context.fetch = async () => ({ok:true, text:async ()=>'n,pts_time,fps,frame_idx\n1,0,25,0\n2,12.3456,25,309\n'});
  assert.equal(await f.run("loadFrameListFromCSV('http://localhost:8007/keyframes/L01_V001/', '')"), true);
  assert.equal(f.run("globalSecondList['2']"), 12.3456);
  assert.equal(f.run("globalOriginalFrameList['2']"), 309);
  f.context.fetch = async () => ({ok:false, status:404});
  assert.equal(await f.run("loadFrameListFromCSV('http://localhost:8007/keyframes/L02_V001/', '')"), false);
  assert.equal(f.run('globalFrameList.length'), 0);
  assert.equal(f.run("globalSecondList['2']"), undefined);
});

test('shared frame/answer messages keep identity, allow time zero and never rebroadcast received frames', () => {
  const f = fixture(); const sent = [];
  f.context.socket_share = {readyState:1, send: msg => sent.push(JSON.parse(msg))};
  f.pin('L01_V001', 0);
  assert.equal(sent.length, 1);
  const message = sent[0]; f.run('resetExportArea()');
  f.context.event = {data:JSON.stringify(message)};
  f.run('handleSharedExportMessage(event)');
  assert.equal(f.run('exportedImages.length'), 1);
  assert.equal(sent.length, 1);
  f.context.event = {data:JSON.stringify({type:'vqa_input_update',frameKey:message.frame.key,vqaInput:'001'})};
  f.run('handleSharedExportMessage(event)');
  assert.equal(f.inputs()[0].value, '001');
});

test('timeline pin uses mapping timestamp, never label or frame number', async () => {
  const f = fixture();
  vm.runInContext(fs.readFileSync(path.join(scripts, 'show_videoframe.js'), 'utf8'), f.context);
  f.context.fetch = async () => ({ok:true, text:async ()=>'n,pts_time,frame_idx\n7,3.4567,104\n'});
  await f.run("loadFrameListFromCSV('http://localhost:8007/keyframes/L03_V001/', '')");
  const picture = {dataset:{}, complete:true, setAttribute() {}, addEventListener() {}};
  const label = {};
  const container = f.element('.current-frame-container');
  container.querySelector = selector => selector === 'img' ? picture : label;
  f.context.container = container;
  await f.run("updateFrameContainer(container, '7', 'http://localhost:8007/keyframes/L03_V001/', true)");
  label.textContent = 'WRONG-LABEL';
  f.run('exportCurrentFrame()');
  assert.equal(f.run('exportedImages[0].keyframeName'), '007');
  assert.equal(f.run('exportedImages[0].frameId'), 104);
  assert.equal(f.run('exportedImages[0].timeSeconds'), 3.4567);
  const payload = JSON.parse(f.run("JSON.stringify(buildKisQaBody(exportedImages[0], 'kis-qa', '17'))"));
  assert.equal(payload.answerSets[0].answers[0].text, '17-L03_V001-3457');
});

test('switching timelines discards the late response from the previous video', async () => {
  const f = fixture();
  vm.runInContext(fs.readFileSync(path.join(scripts, 'show_videoframe.js'), 'utf8'), f.context);
  let finishOld;
  f.context.fetch = url => url.includes('L01_V001')
    ? new Promise(resolve => {finishOld = resolve;})
    : Promise.resolve({ok:true, text:async ()=>'n,pts_time,frame_idx\n1,20,600\n'});
  const old = f.run("loadFrameListFromCSV('http://localhost:8007/keyframes/L01_V001/', '')");
  await f.run("loadFrameListFromCSV('http://localhost:8007/keyframes/L02_V001/', '')");
  finishOld({ok:true, text:async ()=>'n,pts_time,frame_idx\n1,10,300\n'});
  assert.equal(await old, false);
  assert.equal(f.run("globalSecondList['1']"), 20);
});

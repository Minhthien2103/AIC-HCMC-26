const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function fixture() {
  const elements = {};
  function element() {
    return { style: {}, listeners: {}, currentTime: 0, readyState: 0, duration: 200,
      addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); },
      pause() {}, play() { this.plays = (this.plays || 0) + 1; return Promise.resolve(); },
      load() { this.readyState = 0; this.loads = (this.loads || 0) + 1; } };
  }
  const context = vm.createContext({ console, window: { CONFIG: { MEDIA_BASE: 'http://localhost:8008' }, addEventListener() {} },
    document: { getElementById: id => elements[id] ||= element(),
      querySelector: id => elements[id] ||= element(), getElementsByClassName: () => [], addEventListener() {} },
    data: { kq: [{ entity: { video: 'WRONG_VIDEO', time: 1 } }] } });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../baseline/retrievalBaseline/frontend/src/scripts/show_video.js'), 'utf8'), context);
  return { context, elements };
}

test('MP4 opens without HLS and seeks only after metadata; repeated previews reuse listeners', async () => {
  const { context, elements } = fixture();
  await vm.runInContext("openVideoPreview('L06_V029', 83.5)", context);
  const video = elements.vid_details;
  assert.equal(video.src, 'http://localhost:8008/videos/L06_V029.mp4');
  assert.equal(video.currentTime, 0);
  video.readyState = 1;
  video.listeners.loadedmetadata[0]();
  assert.equal(video.currentTime, 83.5);
  assert.equal(video.plays, 1);
  await vm.runInContext("openVideoPreview('L13_V016', 42)", context);
  assert.equal(video.listeners.loadedmetadata.length, 1);
  assert.equal(elements.rewindBtn.listeners.click.length, 1);
  video.readyState = 1;
  video.listeners.loadedmetadata[0]();
  assert.equal(video.currentTime, 42);
  elements.rewindBtn.listeners.click[0]();
  assert.equal(video.currentTime, 37);
  elements.forwardBtn.listeners.click[0]();
  assert.equal(video.currentTime, 42);
  video.listeners.error[0]();
  assert.ok(elements['video-status'].textContent.length > 0);
});

test('clicked frame identity is used rather than stale global results', async () => {
  const { context, elements } = fixture();
  await vm.runInContext("showVideo({id: '1', resultEntity: {video: 'L01_V001', time: 19}})", context);
  assert.equal(elements.vid_details.src, 'http://localhost:8008/videos/L01_V001.mp4');
  assert.equal(elements.vid_details.playerInstance.initialTime, 19);
});

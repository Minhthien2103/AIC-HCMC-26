const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('appending a page preserves metadata registered by the card and observes each image', () => {
  const list = {children: [], appendChild(fragment) {this.children.push(...fragment.children);}};
  const observed = [];
  const context = vm.createContext({
    window: {},
    document: {
      addEventListener() {}, getElementById: () => list,
      createDocumentFragment: () => ({children: [], appendChild(child) {this.children.push(child);}}),
    },
    IntersectionObserver: class {observe() {}},
    imageObserver: {observe: img => observed.push(img)},
    registerFrameMetadata() {throw new Error('Metadata must be registered once by the card factory');},
    frameFromResult: () => ({}),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../baseline/retrievalBaseline/frontend/src/scripts/pagnition.js'), 'utf8'), context);
  context.createPagnitionImageDiv = (result, index) => {
    const image = {id: index, frameData: result.entity};
    return {querySelector: () => image};
  };
  const first = {entity: {video: 'L01_V001', time: 0}};
  const next = {entity: {video: 'L01_V002', time: 12.345}};
  context.updatePagnitionRightPanel_list([first], 0);
  context.updatePagnitionRightPanel_list([next], 1);
  assert.equal(list.children.length, 2);
  assert.deepEqual(observed.map(img => img.id), [1, 2]);
  assert.equal(observed[0].frameData, first.entity);
  assert.equal(observed[1].frameData, next.entity);
});

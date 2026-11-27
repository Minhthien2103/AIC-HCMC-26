const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function createRowsFixture(initialMode = null) {
  const storage = new Map();
  if (initialMode) storage.set('rowsSortMode', initialMode);
  const listeners = new Map();
  const select = {
    value: 'rank',
    addEventListener(type, fn) { listeners.set(`select:${type}`, fn); },
  };
  const rowsChildren = [];
  function makeNode(tag) {
    return {
      tagName: tag,
      className: '',
      textContent: '',
      children: [],
      classList: {
        _set: new Set(),
        add(c) { this._set.add(c); },
        contains(c) { return this._set.has(c); },
      },
      appendChild(child) {
        if (child && child.children && child.tagName === 'FRAGMENT') {
          this.children.push(...child.children);
        } else {
          this.children.push(child);
        }
      },
      querySelectorAll() { return []; },
      addEventListener() {},
      set innerHTML(v) { if (v === '') this.children = []; },
    };
  }
  const imagesRows = makeNode('DIV');
  imagesRows.appendChild = child => {
    if (child && child.tagName === 'FRAGMENT') imagesRows.children.push(...child.children);
    else imagesRows.children.push(child);
  };
  const context = vm.createContext({
    window: {},
    localStorage: {
      getItem: k => storage.get(k) ?? null,
      setItem: (k, v) => storage.set(k, String(v)),
    },
    document: {
      addEventListener(type, fn) { listeners.set(`doc:${type}`, fn); },
      getElementById(id) {
        if (id === 'rows-sort-select') return select;
        if (id === 'images-rows') return imagesRows;
        return null;
      },
      querySelectorAll: () => rowsChildren,
      createDocumentFragment: () => makeNode('FRAGMENT'),
      createElement: tag => makeNode(tag.toUpperCase()),
    },
    IntersectionObserver: class { observe() {} unobserve() {} },
    imageObserver: { observe() {}, unobserve() {} },
  });
  vm.runInContext(
    fs.readFileSync(path.join(__dirname, '../baseline/retrievalBaseline/frontend/src/scripts/update_result.js'), 'utf8'),
    context,
  );
  context.createImageDiv = (result, index) => {
    const card = makeNode('DIV');
    card.rankIndex = index;
    card.video = result.entity.video;
    card.time = result.entity.time;
    card.querySelector = () => ({});
    return card;
  };
  listeners.get('doc:DOMContentLoaded')();
  return { context, select, imagesRows, listeners, storage };
}

test('rows view keeps video order by top rank and toggles frame order between rank and time', () => {
  const { context, select, imagesRows, listeners, storage } = createRowsFixture('rank');
  const results = [
    { id: 10, entity: { video: 'L05_V001', time: 45 } },
    { id: 11, entity: { video: 'L05_V001', time: 12 } },
    { id: 12, entity: { video: 'L12_V004', time: 30 } },
    { id: 13, entity: { video: 'L05_V001', time: 20 } },
  ];
  context.window.currentSearchResults = results;
  context.updateRightPanel_rows(results);

  assert.equal(imagesRows.children.length, 2);
  assert.equal(imagesRows.children[0].children[0].textContent, 'L05_V001');
  assert.equal(imagesRows.children[1].children[1].textContent, 'L12_V004');
  // Rank order inside L05_V001: 45s (#1), 12s (#2), 20s (#4)
  const rankCards = imagesRows.children[0].children.slice(1);
  assert.deepEqual(rankCards.map(c => c.time), [45, 12, 20]);
  assert.deepEqual(rankCards.map(c => c.rankIndex), [1, 2, 4]);

  // Switch dropdown to time order
  select.value = 'time';
  listeners.get('select:change')();
  assert.equal(storage.get('rowsSortMode'), 'time');

  const timeCards = imagesRows.children[0].children.slice(1);
  assert.deepEqual(timeCards.map(c => c.time), [12, 20, 45]);
  // Original rankIndex & top-row highlight follow each frame
  assert.deepEqual(timeCards.map(c => c.rankIndex), [2, 4, 1]);
  assert.equal(timeCards[2].classList.contains('top-row-match'), true);
  assert.equal(timeCards[0].classList.contains('top-row-match'), false);
});

import { describe, expect, it } from 'vitest';
import {
  duplicateCell, insertAfter, isMarkdownCell, moveCell, reviveCells, runnableCells, withKind, type Cell,
} from '../src/components/workbench/notebook/cells';
import { cellsToScript } from '../src/lammps/notebookBridge';

const code = (id: string, text: string): Cell => ({ id, text });
const md = (id: string, text: string): Cell => ({ id, text, kind: 'markdown' });

describe('reviveCells', () => {
  it('loads cells saved by the earlier version (no kind) unchanged', () => {
    const old = [{ id: 'c1', text: 'units lj' }, { id: 'c2', text: 'run 10' }];
    const revived = reviveCells(old)!;
    expect(revived).toEqual(old);
    expect(revived.every((c) => !('kind' in c))).toBe(true);
    expect(revived.some(isMarkdownCell)).toBe(false);
  });

  it('keeps markdown and explicit code kinds, and drops unknown kinds', () => {
    const revived = reviveCells([
      { id: 'a', text: '# hi', kind: 'markdown' },
      { id: 'b', text: 'run 1', kind: 'code' },
      { id: 'c', text: 'x', kind: 'html' },
    ])!;
    expect(revived).toEqual([
      { id: 'a', text: '# hi', kind: 'markdown' },
      { id: 'b', text: 'run 1', kind: 'code' },
      { id: 'c', text: 'x' },
    ]);
  });

  it('skips malformed entries and returns null for nothing usable', () => {
    expect(reviveCells([null, 3, { id: 1, text: 'no' }, { id: 'ok', text: '' }])).toEqual([{ id: 'ok', text: '' }]);
    expect(reviveCells([])).toBeNull();
    expect(reviveCells('cells')).toBeNull();
    expect(reviveCells(undefined)).toBeNull();
  });

  it('survives a JSON round trip (the STORAGE_KEY persistence path)', () => {
    const cells: Cell[] = [code('c1', 'units lj'), md('m1', '## Notes\n- a')];
    expect(reviveCells(JSON.parse(JSON.stringify(cells)))).toEqual(cells);
  });
});

describe('cell helpers', () => {
  const cells = [code('a', 'A'), md('b', 'B'), code('c', 'C')];

  it('withKind sets markdown and leaves code cells without a kind field', () => {
    expect(withKind(code('x', 't'), 'markdown')).toEqual({ id: 'x', text: 't', kind: 'markdown' });
    const back = withKind(md('x', 't'), 'code');
    expect('kind' in back).toBe(false);
  });

  it('moves a cell up and down, and ignores out-of-range moves', () => {
    expect(moveCell(cells, 1, -1).map((c) => c.id)).toEqual(['b', 'a', 'c']);
    expect(moveCell(cells, 1, 1).map((c) => c.id)).toEqual(['a', 'c', 'b']);
    expect(moveCell(cells, 0, -1).map((c) => c.id)).toEqual(['a', 'b', 'c']);
    expect(moveCell(cells, 2, 1).map((c) => c.id)).toEqual(['a', 'b', 'c']);
    expect(moveCell(cells, 5, 1).map((c) => c.id)).toEqual(['a', 'b', 'c']);
    // the input is not mutated
    expect(cells.map((c) => c.id)).toEqual(['a', 'b', 'c']);
  });

  it('duplicates a cell right after itself with a new id and the same kind', () => {
    const dup = duplicateCell(cells, 1, 'new');
    expect(dup.map((c) => c.id)).toEqual(['a', 'b', 'new', 'c']);
    expect(dup[2]).toEqual({ id: 'new', text: 'B', kind: 'markdown' });
    expect(duplicateCell(cells, 9, 'z')).toEqual(cells);
  });

  it('inserts after an index, and at the front with -1', () => {
    expect(insertAfter(cells, 0, code('n', 'N')).map((c) => c.id)).toEqual(['a', 'n', 'b', 'c']);
    expect(insertAfter(cells, -1, code('n', 'N')).map((c) => c.id)).toEqual(['n', 'a', 'b', 'c']);
  });

  it('a deleted markdown cell comes back with its kind when reinserted (Undo)', () => {
    const without = cells.filter((c) => c.id !== 'b');
    const restored = insertAfter(without, 0, cells[1]);
    expect(restored).toEqual(cells);
  });

  it('runnableCells keeps code cells only', () => {
    expect(runnableCells(cells).map((c) => c.id)).toEqual(['a', 'c']);
  });
});

describe('markdown cells in the Script Builder hand-over', () => {
  it('cellsToScript skips markdown cells', () => {
    expect(cellsToScript([md('m', '# Title'), code('c', 'run 5')])).toBe('run 5\n');
  });
});

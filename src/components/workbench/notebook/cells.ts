/**
 * The notebook's cell model and its pure helpers (kept free of React so they can be tested).
 * A cell without `kind` is a LAMMPS code cell, so notebooks saved by earlier versions load unchanged.
 */

export interface Cell {
  id: string;
  text: string;
  /** Missing = code. Markdown cells are rendered text and never run. */
  kind?: 'code' | 'markdown';
}

export const isMarkdownCell = (c: Pick<Cell, 'kind'>): boolean => c.kind === 'markdown';

/** Stored cells -> cells; drops anything malformed. A `kind` other than code/markdown is ignored. */
export const reviveCells = (raw: unknown): Cell[] | null => {
  if (!Array.isArray(raw)) return null;
  const cells: Cell[] = [];
  for (const c of raw) {
    if (!c || typeof (c as Cell).id !== 'string' || typeof (c as Cell).text !== 'string') continue;
    const kind = (c as Cell).kind;
    const cell: Cell = { id: (c as Cell).id, text: (c as Cell).text };
    if (kind === 'markdown') cell.kind = 'markdown';
    else if (kind === 'code') cell.kind = 'code';
    cells.push(cell);
  }
  return cells.length ? cells : null;
};

/** The same cell with another kind. Code cells carry no `kind` field, as before. */
export const withKind = (cell: Cell, kind: 'code' | 'markdown'): Cell => {
  const { kind: _old, ...rest } = cell;
  return kind === 'markdown' ? { ...rest, kind } : rest;
};

/** Insert `cell` after index `index` (index -1 puts it first). */
export const insertAfter = (cells: readonly Cell[], index: number, cell: Cell): Cell[] =>
  [...cells.slice(0, index + 1), cell, ...cells.slice(index + 1)];

/** Move the cell at `index` by `delta` places (-1 up, +1 down). Out of range returns the same array. */
export const moveCell = (cells: readonly Cell[], index: number, delta: number): Cell[] => {
  const to = index + delta;
  if (index < 0 || index >= cells.length || to < 0 || to >= cells.length || delta === 0) return [...cells];
  const next = [...cells];
  const [cell] = next.splice(index, 1);
  next.splice(to, 0, cell);
  return next;
};

/** A copy of the cell at `index`, placed right after it with a new id. */
export const duplicateCell = (cells: readonly Cell[], index: number, newId: string): Cell[] => {
  const src = cells[index];
  if (!src) return [...cells];
  return insertAfter(cells, index, { ...src, id: newId });
};

/** Cells that run: every code cell, in order. */
export const runnableCells = (cells: readonly Cell[]): Cell[] => cells.filter((c) => !isMarkdownCell(c));

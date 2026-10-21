import { StyleError } from './force/types';
import type { System } from './system';

/*
 * Per-grid data of computes and fixes, and the references other commands use to read it.
 * docs.lammps.org/Howto_grid.html: "Grids are referenced by commands which output per-grid data ...
 * c_ID:gname:dname or f_ID:gname:dname". docs.lammps.org/dump.html: "These refer to a grid name and data name
 * for a grid that is defined by a compute or fix. Note that a compute or fix can define one or more grids (of
 * different sizes) and one or more data fields for each of those grids." "If c_ID:gname:dname is used as a
 * attribute, then the per-grid vector calculated by the compute is accessed. If c_ID:gname:dname[I] is used, then
 * I must be in the range from 1-M, which will access the Ith column of the per-grid array with M columns
 * calculated by the compute."
 * dump.html: "The ordering of IDs is with the x index varying fastest, then the y index, and the z index varying
 * slowest." Here every data set holds its cells in that order, max(1, ncol) values per cell.
 */

/** What a compute or fix that defines grids offers (pergridFlag set). */
export interface GridSource {
  readonly id: string;
  pergridFlag: boolean;
  /** The index of the grid called name, or -1. */
  gridByName(name: string): number;
  /** Nx Ny Nz of grid igrid (Nz = 1 in 2d). */
  gridSize(igrid: number): [number, number, number];
  /** The index of data set name of grid igrid and its columns (0 = per-grid vector), or null. */
  griddataByName(igrid: number, name: string): { index: number; ncol: number } | null;
  /** The values of data set index, cells in grid-ID order, max(1, ncol) values per cell. */
  griddata(index: number): Float64Array;
}

/** A resolved grid reference: its source, grid, data set and column (0 = the per-grid vector, else 1-based). */
export interface GridRef {
  kind: 'c' | 'f';
  id: string;
  igrid: number;
  idata: number;
  ncol: number;
  /** 0 for the vector form, I for name[I]. */
  index: number;
}

/** c_NAME or f_NAME with an optional [I] (the ArgInfo form); NAME must be ID:gname:dname. */
const REF = /^([cf])_([^[\]\s]+)(?:\[(\d+)\])?$/;

/** Does the word name a compute or fix value (c_... or f_...), which a grid command reads as a grid reference? */
export const isGridRef = (w: string): boolean => /^[cf]_/.test(w);

const sourceOf = (sys: System, kind: 'c' | 'f', id: string): GridSource | null =>
  (kind === 'c' ? sys.computes.find((c) => c.id === id) : sys.fixes.find((f) => f.id === id)) ?? null;

/**
 * Resolves a grid reference for the command `what` (e.g. "Dump grid"), whose output frequency is nevery: a fix's
 * per-grid data is valid on multiples of its per-grid frequency only. As native LAMMPS does
 * (src/utils.cpp check_grid_reference at b332f28), every failure names the reference's part that is wrong.
 */
export const resolveGridRef = (sys: System, what: string, ref: string, nevery: number): GridRef => {
  const m = REF.exec(ref);
  if (!m) throw new StyleError(`${what} grid reference ${ref} is invalid`);
  const kind = m[1] as 'c' | 'f';
  const idx = m[3];
  // utils.cpp parse_grid_id: the name splits at ':' into exactly three non-empty words
  const words = m[2].split(':').filter(Boolean);
  if (words.length !== 3) throw new StyleError(`Grid ID ${m[2]} must be 3 strings separated by 2 ':'characters`);
  const [id, gname, dname] = words;
  const word = kind === 'c' ? 'compute' : 'fix';
  const src = sourceOf(sys, kind, id);
  if (!src) throw new StyleError(`${what} ${word} ID ${id} not found`);
  if (!src.pergridFlag) throw new StyleError(`${what} ${word} ${id} does not compute per-grid info`);
  if (kind === 'f') {
    const freq = (src as unknown as { pergridFreq: number }).pergridFreq;
    if (nevery % freq) throw new StyleError(`${what} fix ${id} not computed at compatible time`);
  }
  const igrid = src.gridByName(gname);
  if (igrid < 0) throw new StyleError(`${what} ${word} ${id} does not recognize grid name ${gname}`);
  const data = src.griddataByName(igrid, dname);
  if (!data) throw new StyleError(`${what} ${word} ${id} does not recognize data name ${dname}`);
  const index = idx === undefined ? 0 : Number(idx);
  if (idx === undefined && data.ncol) throw new StyleError(`${what} ${word} ${id} data ${dname} is not per-grid vector`);
  if (idx !== undefined && data.ncol === 0) throw new StyleError(`${what} ${word} ${id} data ${dname} is not per-grid array`);
  if (idx !== undefined && (index > data.ncol || index < 1)) throw new StyleError(`${what} ${word} ${id} array ${dname} is accessed out-of-range`);
  return { kind, id, igrid, idata: data.index, ncol: data.ncol, index };
};

/**
 * Expands c_ID:gname:dname[*] (and n*, *m, n*m) into one reference per column of the per-grid array, as dump.html
 * allows ("I can include wildcard"); other words are kept.
 */
export const expandGridRefs = (sys: System, words: readonly string[]): string[] => {
  const out: string[] = [];
  for (const w of words) {
    const m = /^([cf])_([^:\s]+):([^:\s]+):([^:[\s]+)\[(\d*)\*(\d*)\](.*)$/.exec(w);
    const src = m ? sourceOf(sys, m[1] as 'c' | 'f', m[2]) : null;
    const igrid = src?.pergridFlag ? src.gridByName(m![3]) : -1;
    if (!m || !src || igrid < 0) {
      out.push(w);
      continue;
    }
    const nmax = src.griddataByName(igrid, m[4])?.ncol ?? 0;
    const lo = m[5] === '' ? 1 : Number(m[5]);
    const hi = m[6] === '' ? nmax : Number(m[6]);
    if (lo < 1 || hi > nmax || lo > hi) throw new StyleError(`Numeric index ${m[5]}*${m[6]} is out of bounds (1-${nmax})`);
    for (let k = lo; k <= hi; k++) out.push(`${m[1]}_${m[2]}:${m[3]}:${m[4]}[${k}]${m[7]}`);
  }
  return out;
};

/** The source of a resolved reference, with its per-grid data current (a compute is invoked once per epoch). */
export const gridSourceOf = (sys: System, ref: GridRef): GridSource => {
  if (ref.kind === 'c') {
    const c = sys.compute(ref.id);
    c.pergridValues();
    return c;
  }
  return sys.fix(ref.id);
};

/** The values of one reference, one per cell in grid-ID order. */
export const gridColumn = (sys: System, ref: GridRef): Float64Array => {
  const src = gridSourceOf(sys, ref);
  const data = src.griddata(ref.idata);
  if (ref.index === 0) return data;
  const ncol = ref.ncol;
  const n = data.length / ncol;
  const out = new Float64Array(n);
  for (let c = 0; c < n; c++) out[c] = data[c * ncol + ref.index - 1];
  return out;
};

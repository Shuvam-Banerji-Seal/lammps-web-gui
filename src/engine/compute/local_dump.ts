import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * dump local (docs.lammps.org/dump.html, section "Attributes used as arguments to the local style"):
 *   "The *index* attribute can be used to generate an index number from 1 to N for each line
 *   written into the dump file, where N is the total number of local datums from all processors"
 *   "The *c_ID* and *c_ID[I]* attributes allow local vectors or arrays calculated by a compute to be
 *   output."
 *   "If *c_ID[I]* is used, then I must be in the range from 1-M, which will print the Ith column of
 *   the local array with M columns calculated by the compute."
 * The dump file layout is in output/dump.ts (writeLocal).
 */

/**
 * The local output an attribute names: a compute (c_ID) or a fix (f_ID, a fix that calculates local values, as
 * fix STORE/LOCAL does). dump.html: "The *c_ID* and *c_ID[I]* attributes allow local vectors or arrays calculated
 * by a compute to be output." The f_ID form is the same for a fix.
 */
export interface LocalOutput {
  localFlag: boolean;
  sizeLocalCols: number;
  localRows: number;
  localValues(): Float64Array;
}

export const localOutputOf = (sys: System, kind: 'c' | 'f', id: string): LocalOutput => (kind === 'c' ? sys.compute(id) : sys.fix(id));

/** Expands the attributes of a dump local (dump ID group local N file attributes) into column names. */
export const localDumpColumns = (sys: System, dumpId: string, words: string[]): string[] => {
  if (!words.length) throw new StyleError(`dump ${dumpId} local needs attributes (index, c_ID or c_ID[I])`);
  const out: string[] = [];
  for (const w of words) {
    if (w === 'index') {
      out.push(w);
      continue;
    }
    const m = /^([cf])_([A-Za-z0-9_]+)(?:\[(\d+|\*)\])?$/.exec(w);
    // As native LAMMPS does (src/dump_local.cpp at b332f28, DumpLocal::parse_fields): an unknown word is an invalid attribute
    if (!m) throw new StyleError(`Invalid attribute ${w} in dump local command`);
    const kind = m[1] as 'c' | 'f';
    const what = kind === 'c' ? 'compute' : 'fix';
    const comp = localOutputOf(sys, kind, m[2]);
    if (!comp.localFlag) throw new StyleError(`dump ${dumpId} local: ${what} ${m[2]} does not calculate local values`);
    const cols = comp.sizeLocalCols;
    if (m[3] === undefined) {
      if (cols !== 0) throw new StyleError(`dump ${dumpId} local: ${what} ${m[2]} does not calculate local vector`);
      out.push(w);
    } else {
      if (cols === 0) throw new StyleError(`dump ${dumpId} local: ${what} ${m[2]} does not calculate local array`);
      if (m[3] === '*') {
        for (let k = 1; k <= cols; k++) out.push(`${kind}_${m[2]}[${k}]`);
      } else {
        const k = Number(m[3]);
        if (k < 1 || k > cols) throw new StyleError(`dump ${dumpId} local: column ${k} of ${what} ${m[2]} is out of range 1..${cols}`);
        out.push(w);
      }
    }
  }
  return out;
};

/** Value source of one local column: the compute's data (null = the index column) and its column (0-based). */
export const localColumnSource = (sys: System, name: string): { data: Float64Array; ncol: number; col: number } | null => {
  if (name === 'index') return null;
  const m = /^([cf])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(name);
  if (!m) throw new StyleError(`dump local: attribute '${name}' is not supported`);
  const comp = localOutputOf(sys, m[1] as 'c' | 'f', m[2]);
  const data = comp.localValues();
  return { data, ncol: Math.max(1, comp.sizeLocalCols), col: m[3] ? Number(m[3]) - 1 : 0 };
};

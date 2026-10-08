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

/** Expands the attributes of a dump local (dump ID group local N file attributes) into column names. */
export const localDumpColumns = (sys: System, dumpId: string, words: string[]): string[] => {
  if (!words.length) throw new StyleError(`dump ${dumpId} local needs attributes (index, c_ID or c_ID[I])`);
  const out: string[] = [];
  for (const w of words) {
    if (w === 'index') {
      out.push(w);
      continue;
    }
    const m = /^c_([A-Za-z0-9_]+)(?:\[(\d+|\*)\])?$/.exec(w);
    if (!m) throw new StyleError(`dump ${dumpId} local: attribute '${w}' is not supported by the browser engine (index and c_ID / c_ID[I] are)`);
    const comp = sys.compute(m[1]);
    if (!comp.localFlag) throw new StyleError(`dump ${dumpId} local: compute ${m[1]} does not calculate local values`);
    const cols = comp.sizeLocalCols;
    if (m[2] === undefined) {
      if (cols !== 0) throw new StyleError(`dump ${dumpId} local: compute ${m[1]} does not calculate local vector`);
      out.push(w);
    } else {
      if (cols === 0) throw new StyleError(`dump ${dumpId} local: compute ${m[1]} does not calculate local array`);
      if (m[2] === '*') {
        for (let k = 1; k <= cols; k++) out.push(`c_${m[1]}[${k}]`);
      } else {
        const k = Number(m[2]);
        if (k < 1 || k > cols) throw new StyleError(`dump ${dumpId} local: column ${k} of compute ${m[1]} is out of range 1..${cols}`);
        out.push(w);
      }
    }
  }
  return out;
};

/** Value source of one local column: the compute's data (null = the index column) and its column (0-based). */
export const localColumnSource = (sys: System, name: string): { data: Float64Array; ncol: number; col: number } | null => {
  if (name === 'index') return null;
  const m = /^c_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(name);
  if (!m) throw new StyleError(`dump local: attribute '${name}' is not supported`);
  const comp = sys.compute(m[1]);
  const data = comp.localValues();
  return { data, ncol: Math.max(1, comp.sizeLocalCols), col: m[2] ? Number(m[2]) - 1 : 0 };
};

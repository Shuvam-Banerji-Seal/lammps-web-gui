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

/**
 * Expands the attributes of a dump local (dump ID group local N file attributes) into column names. As native LAMMPS
 * does (src/dump_local.cpp DumpLocal::DumpLocal and parse_fields at b332f28): a [*] wildcard is expanded as
 * utils::expand_args does (over the columns of a global array, a per-atom array or a local array of an existing compute
 * or fix; otherwise the word is left as it is and is no valid attribute), then each c_ID / f_ID is checked: it exists,
 * computes local info, and is a vector or an array whose column is in range.
 */
export const localDumpColumns = (sys: System, _dumpId: string, words: string[]): string[] => {
  if (!words.length) throw new StyleError('No dump local arguments specified');
  const expanded: string[] = [];
  for (const w of words) {
    const wm = /^([cf])_([A-Za-z0-9_]+)\[(\d*)\*(\d*)\]$/.exec(w);
    if (!wm) { expanded.push(w); continue; }
    const src = wm[1] === 'c' ? sys.computes.find((x) => x.id === wm[2]) : sys.fixes.find((x) => x.id === wm[2]);
    const o = src as unknown as { arrayFlag?: boolean; sizeArrayCols?: number; peratomFlag?: boolean; sizePeratomCols?: number; localFlag?: boolean; sizeLocalCols?: number } | undefined;
    const nmax = !o ? 0 : o.arrayFlag ? o.sizeArrayCols ?? 0 : o.peratomFlag && o.sizePeratomCols ? o.sizePeratomCols : o.localFlag && o.sizeLocalCols ? o.sizeLocalCols : 0;
    if (!nmax) { expanded.push(w); continue; }
    for (let k = wm[3] ? Number(wm[3]) : 1; k <= (wm[4] ? Number(wm[4]) : nmax); k++) expanded.push(`${wm[1]}_${wm[2]}[${k}]`);
  }
  const out: string[] = [];
  for (const w of expanded) {
    if (w === 'index') {
      out.push(w);
      continue;
    }
    const m = /^([cf])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(w);
    if (!m) throw new StyleError(`Invalid attribute ${w} in dump local command`);
    const kind = m[1] as 'c' | 'f';
    const name = m[2];
    const what = kind === 'c' ? 'compute' : 'fix';
    const verb = kind === 'c' ? 'calculate' : 'compute';
    const src = kind === 'c' ? sys.compute(name, `Could not find dump local compute ID ${name}`) : sys.fix(name, `Could not find dump local fix ID ${name}`);
    const comp = src as unknown as LocalOutput;
    if (!comp.localFlag) throw new StyleError(`Dump local ${what} ${name} does not compute local info`);
    const cols = comp.sizeLocalCols;
    const k = m[3] === undefined ? 0 : Number(m[3]);
    if (k === 0 && cols > 0) throw new StyleError(`Dump local ${what} ${name} does not ${verb} local vector`);
    if (k > 0 && cols === 0) throw new StyleError(`Dump local ${what} ${name} does not ${verb} local array`);
    if (k > 0 && k > cols) throw new StyleError(`Dump local ${what} ${name} vector is accessed out-of-range\nFor more information see https://docs.lammps.org/err0020`);
    out.push(w);
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

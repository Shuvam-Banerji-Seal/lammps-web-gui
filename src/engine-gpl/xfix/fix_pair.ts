/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_pair.cpp and src/fix_pair.h at commit b332f28 (class FixPair, fix pair; no contributing-author line in
 * the source). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Fix pair (docs.lammps.org/fix_pair.html): "Store a per-atom quantity computed by a pair style, so that it can be used
 * as a per-atom output (f_ID) of a dump, a compute or a variable." The fix copies the named per-atom fields of the pair
 * style on every Nevery-th step, after the force evaluation (post_force), and the copy is what f_ID reads.
 *
 * Supported here: the EAM family (pair_style eam, eam/alloy, eam/fs), whose native PairEAM::extract_peratom returns
 * the per-atom vectors "rho" (electron density) and "fp" (embedding derivative F'(rho)), both with ncol = 0 (vectors),
 * and the pair styles that serve fields and triggers through FixPairSource (pair_style pace/extrapolation: "gamma" and
 * "corerep" with the triggers "gamma_flag" and "corerep_flag"). A trigger (flag 1) is set to 1 before the force
 * evaluation of a sampled step (pre_force) and back to 0 after the copy (post_force). Every other pair style is refused
 * by name with a StyleError.
 *
 * The EAM pair keeps rho and fp for the owned atoms in its rhoBuf and fpBuf after each force evaluation (eam.ts
 * compute); this fix reads them through their engine field names.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric } from '../../engine/force/util';
import type { System } from '../../engine/system';

/** Pair styles that expose per-atom fields to fix pair, with the fields each one provides (PairEAM::extract_peratom). */
const EAM_STYLES = new Set(['eam', 'eam/alloy', 'eam/fs']);
const EAM_FIELDS = new Set(['rho', 'fp']);

/** The EAM per-atom arrays of the last force evaluation, indexed by owned atom (length >= nlocal once computed). */
interface EAMPeratomArrays {
  rhoBuf?: Float64Array;
  fpBuf?: Float64Array;
}

/**
 * A pair style that serves per-atom fields to fix pair: extract_peratom (undefined for a field it does not know, values
 * null until computed; columns 0 for a vector) and the scalar triggers extract("<field>_flag") returns.
 */
export interface FixPairSource {
  extractPeratom(field: string): { values: Float64Array | null; columns: number } | undefined;
  hasTrigger(name: string): boolean;
  setTrigger(name: string, value: number): boolean;
}

const asSource = (pair: unknown): FixPairSource | null =>
  pair && typeof (pair as FixPairSource).extractPeratom === 'function' ? pair as FixPairSource : null;

function parseInteger(w: string | undefined): number {
  const x = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isInteger(x)) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return x;
}

export class FixPair extends Fix {
  readonly style = 'pair';
  private readonly pairname: string;
  private readonly fieldnames: string[] = [];
  private lasttime = -1;
  private vector: Float64Array<ArrayBuffer> | null = null;
  private array: Float64Array<ArrayBuffer> | null = null;
  private readonly ncols: number;
  /** the trigger names ("<field>_flag") of the fields with flag 1, for a FixPairSource pair style */
  private readonly triggers: string[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError('Illegal fix pair command: missing argument(s)');
    this.nevery = parseInteger(args[0]);
    if (this.nevery < 1) throw new StyleError(`Illegal fix pair every value: ${this.nevery}`);
    this.pairname = args[1];
    if (this.pairname.includes(':')) {
      throw new StyleError(`fix pair: pair sub-style selection ${this.pairname} is not supported (no hybrid sub-styles)`);
    }
    this.checkPairStyle();

    // field/flag pairs; a flag of 1 needs a trigger scalar that no engine pair style extracts (native: "Fix pair pair
    // style cannot extract <field>_flag", which is the native error for a style with no such extract()).
    // As native LAMMPS does (src/fix_pair.cpp FixPair::FixPair at b332f28): all field/flag pairs are read first (each
    // flag with utils::inumeric, 0 or 1), then the trigger and field checks
    const rest = args.slice(2);
    const flags: number[] = [];
    for (let k = 0; k < rest.length; k += 2) {
      if (k + 2 > rest.length) throw new StyleError(`Illegal fix pair ${rest[k]} command: missing argument(s)`);
      const flag = inumeric(rest[k + 1]);
      if (flag !== 0 && flag !== 1) throw new StyleError(`Illegal fix pair ${rest[k]} command flag: ${rest[k + 1]}`);
      flags.push(flag);
    }
    const src = asSource(this.sys.ff.pair);
    let ncols = 0;
    for (let k = 0; k < rest.length; k += 2) {
      const field = rest[k];
      if (src) {
        // native: extract_peratom gives the column count (a field it does not know counts one column and fails at
        // the first copy), then the trigger must be a scalar the style extracts
        const pe = src.extractPeratom(field);
        ncols += pe && pe.columns ? pe.columns : 1;
        if (flags[k / 2] === 1) {
          if (!src.hasTrigger(`${field}_flag`)) throw new StyleError(`Fix pair pair style cannot extract ${field}_flag`);
          this.triggers.push(`${field}_flag`);
        }
        this.fieldnames.push(field);
        continue;
      }
      if (flags[k / 2] === 1) throw new StyleError(`Fix pair pair style cannot extract ${field}_flag`);
      if (!EAM_FIELDS.has(field)) {
        throw new StyleError(`Fix pair cannot extract property ${field} from pair style ${this.pairname}`);
      }
      this.fieldnames.push(field);
      ncols++;
    }
    if (this.fieldnames.length === 0) throw new StyleError('Illegal fix pair command: missing argument(s)');

    // native: a single column is a per-atom vector (size_peratom_cols 0), several columns a per-atom array
    this.ncols = ncols;
    this.peratomFlag = true;
    this.sizePeratomCols = this.ncols === 1 ? 0 : this.ncols;
    this.ensureArrays();
  }

  /** Native FixPair::query_pstyle and init(): the named pair style must be the one in use. */
  private checkPairStyle(): void {
    const pair = this.sys.ff.pair;
    if (!pair || pair.name !== this.pairname) {
      if (pair && pair.name.startsWith('hybrid')) {
        throw new StyleError(`fix pair: pair style ${this.pairname} inside ${pair.name} is not supported`);
      }
      throw new StyleError(`Pair style ${this.pairname} for fix pair not found`);
    }
    if (!EAM_STYLES.has(pair.name) && !asSource(pair)) {
      throw new StyleError(
        `fix pair: pair style ${pair.name} exposes no per-atom fields to fix pair (supported: pair eam, eam/alloy, eam/fs with rho, fp; pace with corerep; pace/extrapolation with gamma, corerep)`,
      );
    }
  }

  init(): void {
    this.checkPairStyle();
  }

  private ensureArrays(): void {
    const n = this.sys.state.n;
    if (this.ncols === 1) {
      if (!this.vector || this.vector.length !== n) {
        this.vector = new Float64Array(n);
        this.vectorAtom = this.vector;
      }
    } else if (!this.array || this.array.length !== n * this.ncols) {
      this.array = new Float64Array(n * this.ncols);
      this.arrayAtom = this.array;
    }
  }

  /** The EAM array for one field, which must hold the owned atoms after a force evaluation. */
  private fieldValues(field: string, nlocal: number): Float64Array {
    const pair = this.sys.ff.pair as unknown as EAMPeratomArrays | null;
    const values = field === 'rho' ? pair?.rhoBuf : pair?.fpBuf;
    // native FixPair::post_force: a null array is an error only when this process holds local atoms
    if (nlocal === 0) return new Float64Array(0);
    if (!values || values.length < nlocal) {
      throw new StyleError(`Fix pair cannot extract property ${field} from pair style ${this.pairname}`);
    }
    return values;
  }

  /** Native FixPair::pre_force: on a step that will be copied, the triggers are set before the force evaluation. */
  preForce(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0 || step === this.lasttime) return;
    const src = asSource(this.sys.ff.pair);
    if (src) for (const t of this.triggers) src.setTrigger(t, 1);
  }

  setupPreForce(): void {
    this.preForce();
  }

  minPreForce(): void {
    this.preForce();
  }

  /** Native FixPair::post_force: on steps that are multiples of Nevery, copy the fields once per step. */
  postForce(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0) return;
    if (step === this.lasttime) return;
    this.lasttime = step;
    this.checkPairStyle();
    this.ensureArrays();
    const nlocal = this.sys.state.n;
    const src = asSource(this.sys.ff.pair);
    if (src) {
      let icol = 0;
      for (const field of this.fieldnames) {
        const pe = src.extractPeratom(field);
        const values = pe?.values ?? null;
        if (!values && nlocal > 0) throw new StyleError(`Fix pair cannot extract property ${field} from pair style`);
        const columns = pe?.columns ?? 0;
        if (columns === 0) {
          for (let i = 0; i < nlocal; i++) {
            if (this.ncols === 1) this.vector![i] = values![i];
            else this.array![i * this.ncols + icol] = values![i];
          }
          icol++;
        } else {
          for (let i = 0; i < nlocal; i++) for (let m = 0; m < columns; m++) this.array![i * this.ncols + icol + m] = values![i * columns + m];
          icol += columns;
        }
      }
      for (const t of this.triggers) src.setTrigger(t, 0);
      return;
    }
    for (let icol = 0; icol < this.ncols; icol++) {
      const src = this.fieldValues(this.fieldnames[icol], nlocal);
      if (this.ncols === 1) {
        for (let i = 0; i < nlocal; i++) this.vector![i] = src[i];
      } else {
        for (let i = 0; i < nlocal; i++) this.array![i * this.ncols + icol] = src[i];
      }
    }
  }

  minPostForce(): void {
    this.postForce();
  }

  minSetup(): void {
    this.postForce();
  }
}

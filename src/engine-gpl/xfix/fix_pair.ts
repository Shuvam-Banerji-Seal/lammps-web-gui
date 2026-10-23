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
 * the per-atom vectors "rho" (electron density) and "fp" (embedding derivative F'(rho)), both with ncol = 0 (vectors).
 * Every other pair style or field is refused by name with a StyleError. Trigger flags (the "<field>_flag" scalar that
 * a pair style extracts with extract(); flag 1) are refused: no engine pair style exposes one.
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
    for (let k = 0; k < rest.length; k += 2) {
      const field = rest[k];
      if (flags[k / 2] === 1) throw new StyleError(`Fix pair pair style cannot extract ${field}_flag`);
      if (!EAM_FIELDS.has(field)) {
        throw new StyleError(`Fix pair cannot extract property ${field} from pair style ${this.pairname}`);
      }
      this.fieldnames.push(field);
    }
    if (this.fieldnames.length === 0) throw new StyleError('Illegal fix pair command: missing argument(s)');

    // native: a single column is a per-atom vector (size_peratom_cols 0), several columns a per-atom array
    this.ncols = this.fieldnames.length;
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
    if (!EAM_STYLES.has(pair.name)) {
      throw new StyleError(
        `fix pair: pair style ${pair.name} exposes no per-atom fields to fix pair (supported: pair eam, eam/alloy, eam/fs with rho, fp)`,
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

  /** Native FixPair::post_force: on steps that are multiples of Nevery, copy the fields once per step. */
  postForce(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0) return;
    if (step === this.lasttime) return;
    this.lasttime = step;
    this.checkPairStyle();
    this.ensureArrays();
    const nlocal = this.sys.state.n;
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

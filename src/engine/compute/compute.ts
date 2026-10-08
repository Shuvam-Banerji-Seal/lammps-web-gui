import type { System } from '../system';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';

/*
 * Base class for computes — docs.lammps.org/compute.html: "Define a
 * diagnostic computation that will be performed on a group of atoms.
 * Quantities calculated by a compute are instantaneous values". A compute
 * produces a global scalar, vector or array and/or per-atom values; thermo
 * output, variables, fixes and dumps reference them as c_ID, c_ID[i],
 * c_ID[i][j]. Per the same page, the results of computes that calculate
 * global quantities can be either "intensive" or "extensive" values;
 * extensive ones are normalized by atom count in thermo output when
 * thermo_modify norm is yes.
 *
 * Results are cached per state epoch (System.epoch), which advances on every
 * timestep and every command that changes the system, so a compute is
 * evaluated at most once per state, as LAMMPS invokes it once per timestep.
 *
 * compute_modify (docs.lammps.org/compute_modify.html): "The extra/dof
 * keyword refers to how many degrees of freedom are subtracted (typically
 * from 3N) ... The default is 2 or 3 for 2d or 3d systems"; "dynamic/dof ...
 * By default, N and their DOF are assumed to be constant."
 */

export abstract class Compute {
  abstract readonly style: string;
  readonly groupBit: number;

  scalarFlag = false;
  vectorFlag = false;
  arrayFlag = false;
  peratomFlag = false;
  sizeVector = 0;
  sizeArrayRows = 0;
  sizeArrayCols = 0;
  /** 0 = per-atom vector, >0 = per-atom array with that many columns. */
  sizePeratomCols = 0;
  /** 1 = extensive (normalized by N with thermo_modify norm yes), 0 = intensive. */
  extscalar = 0;
  extvector = 0;
  /** Per-component extensivity for vectors that mix both, or null. */
  extlist: number[] | null = null;
  /** Is a temperature compute (has dof, may remove a velocity bias). */
  tempFlag = false;
  pressFlag = false;
  /** Needs per-atom energy / virial from the force field on the steps it is invoked. */
  needsEatom = false;
  needsVatom = false;

  scalar = 0;
  vector = new Float64Array(0);
  array = new Float64Array(0);
  vectorAtom = new Float64Array(0);
  arrayAtom = new Float64Array(0);
  private epochScalar = -1;
  private epochVector = -1;
  private epochArray = -1;
  private epochPeratom = -1;

  // temperature computes
  dof = 0;
  extraDof: number;
  /** compute_modify dynamic/dof. */
  dynamicDofFlag = false;
  /**
   * Recount the degrees of freedom at every evaluation: compute_modify dynamic/dof yes, or a
   * dynamic group. Measured with native LAMMPS (black box): compute temp on a dynamic group of
   * 450 and then 404 atoms divides by 3N - 3 of the current count (1347, then 1209).
   */
  get dynamicDof(): boolean { return this.dynamicDofFlag || this.sys.groups.isDynamic(this.groupBit); }

  constructor(protected sys: System, readonly id: string, readonly group: string, protected args: string[]) {
    this.groupBit = sys.groups.bit(group);
    this.extraDof = sys.dimension;
  }

  /** Called at the start of every run and before first use. */
  init(): void {}

  /** Cached evaluation entry points used by thermo, variables, fixes and dumps. */
  scalarValue(): number {
    if (!this.scalarFlag) throw new StyleError(`compute ${this.id} does not calculate a global scalar`);
    if (this.epochScalar !== this.sys.epoch) {
      this.scalar = this.computeScalar();
      this.epochScalar = this.sys.epoch;
    }
    return this.scalar;
  }

  vectorValues(): Float64Array {
    if (!this.vectorFlag) throw new StyleError(`compute ${this.id} does not calculate a global vector`);
    if (this.epochVector !== this.sys.epoch) {
      this.computeVector();
      this.epochVector = this.sys.epoch;
    }
    return this.vector;
  }

  arrayValues(): Float64Array {
    if (!this.arrayFlag) throw new StyleError(`compute ${this.id} does not calculate a global array`);
    if (this.epochArray !== this.sys.epoch) {
      this.computeArray();
      this.epochArray = this.sys.epoch;
    }
    return this.array;
  }

  /** Per-atom values (vector, or array with sizePeratomCols columns). */
  peratomValues(): Float64Array {
    if (!this.peratomFlag) throw new StyleError(`compute ${this.id} does not calculate per-atom values`);
    if (this.epochPeratom !== this.sys.epoch) {
      this.computePeratom();
      this.epochPeratom = this.sys.epoch;
    }
    return this.sizePeratomCols === 0 ? this.vectorAtom : this.arrayAtom;
  }

  /** Forget cached values (the state changed without an epoch bump). */
  invalidate(): void {
    this.epochScalar = this.epochVector = this.epochArray = this.epochPeratom = -1;
  }

  protected computeScalar(): number { throw new StyleError(`compute ${this.id} has no scalar`); }
  protected computeVector(): void { throw new StyleError(`compute ${this.id} has no vector`); }
  protected computeArray(): void { throw new StyleError(`compute ${this.id} has no array`); }
  protected computePeratom(): void { throw new StyleError(`compute ${this.id} has no per-atom values`); }

  /** compute_modify keyword; returns values consumed (0 = unknown keyword). */
  modify(key: string, values: string[]): number {
    if (key === 'extra/dof' || key === 'extra') {
      this.extraDof = parseNum(values[0], 'extra/dof');
      return 1;
    }
    if (key === 'dynamic/dof' || key === 'dynamic') {
      if (values[0] !== 'yes' && values[0] !== 'no') throw new StyleError(`${key} must be yes or no`);
      this.dynamicDofFlag = values[0] === 'yes';
      return 1;
    }
    return 0;
  }

  // ---- velocity bias (temp/partial, temp/region, temp/com, ...): default none
  /** True if this temperature compute removes a velocity bias. */
  hasBias(): boolean { return false; }
  /** Removes the bias from every atom's velocity in the group (for thermostats). */
  removeBiasAll(): void {}
  restoreBiasAll(): void {}
  /** Removes / restores the bias of one atom (index i). */
  removeBias(_i: number): void {}
  restoreBias(_i: number): void {}
  /** Recomputes the bias for the current state before removeBiasAll. */
  computeBias(): void {}

  /** Lines for 'info computes' style listings. */
  describe(): string {
    return `${this.id} ${this.group} ${this.style}`;
  }
}

/** Atoms of a group (indices), for computes that loop over members. */
export const groupCount = (sys: System, bit: number): number => {
  const s = sys.state;
  let n = 0;
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) n++;
  return n;
};

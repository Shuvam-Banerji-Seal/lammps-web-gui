import type { ImageObject } from '../output/image_objects';
import type { System } from '../system';
import { StyleError } from '../force/types';
import { missingArgs, numeric, logical } from '../force/util';

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
  /**
   * Local data (one row per bond, angle, pair, ...): localFlag set, sizeLocalCols 0 = local vector,
   * >0 = local array with that many columns. localValues() returns row-major rows x max(1, cols).
   */
  localFlag = false;
  sizeLocalCols = 0;
  /** Rows of the local data after the last evaluation (valid after localValues()). */
  localRows = 0;
  /** 1 = extensive (normalized by N with thermo_modify norm yes), 0 = intensive. */
  extscalar = 0;
  extvector = 0;
  /** Per-component extensivity for vectors that mix both, or null. */
  extlist: number[] | null = null;
  /** 1 = every element of the global array is extensive (thermo c_ID[I][J] is normalized), 0 = intensive. */
  extarray = 0;
  /** Is a temperature compute (has dof, may remove a velocity bias). */
  tempFlag = false;
  pressFlag = false;
  /** Needs per-atom energy / virial from the force field on the steps it is invoked. */
  needsEatom = false;
  needsVatom = false;
  /** Needs the per-atom centroid virial of the bonded terms (compute centroid/stress/atom). */
  needsCvatom = false;

  /**
   * Per-grid data (grid.ts GridSource): the grids and data sets other commands read as c_ID:gname:dname. A compute
   * that sets pergridFlag overrides the four accessors and computePergrid().
   */
  pergridFlag = false;
  gridByName(_name: string): number { return -1; }
  gridSize(_igrid: number): [number, number, number] { return [0, 0, 0]; }
  griddataByName(_igrid: number, _name: string): { index: number; ncol: number } | null { return null; }
  griddata(_index: number): Float64Array { throw new StyleError(`compute ${this.id} does not compute per-grid info`); }

  scalar = 0;
  vector = new Float64Array(0);
  array = new Float64Array(0);
  vectorAtom = new Float64Array(0);
  arrayAtom = new Float64Array(0);
  private epochScalar = -1;
  private epochVector = -1;
  private epochArray = -1;
  private epochPeratom = -1;
  private epochLocal = -1;
  private epochPergrid = -1;
  private localData: Float64Array<ArrayBuffer> = new Float64Array(0);

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

  /**
   * Set when a run (or another command that initializes the system) has initialized the compute (native
   * Compute::init_flags): before that a variable formula may not read it (Variable::evaluate, variable.cpp at b332f28).
   */
  initialized = false;

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

  /** Local values, row-major (see localFlag); cached per state epoch like the others. */
  localValues(): Float64Array<ArrayBuffer> {
    if (!this.localFlag) throw new StyleError(`compute ${this.id} does not calculate local values`);
    if (this.epochLocal !== this.sys.epoch) {
      this.localData = this.computeLocal();
      this.epochLocal = this.sys.epoch;
    }
    return this.localData;
  }

  /** Brings the per-grid data up to date (computePergrid), cached per state epoch like the other values. */
  pergridValues(): void {
    if (!this.pergridFlag) throw new StyleError(`compute ${this.id} does not compute per-grid info`);
    if (this.epochPergrid !== this.sys.epoch) {
      this.computePergrid();
      this.epochPergrid = this.sys.epoch;
    }
  }

  /** Forget cached values (the state changed without an epoch bump). */
  invalidate(): void {
    this.epochScalar = this.epochVector = this.epochArray = this.epochPeratom = this.epochLocal = this.epochPergrid = -1;
  }

  protected computeScalar(): number { throw new StyleError(`compute ${this.id} has no scalar`); }
  protected computeVector(): void { throw new StyleError(`compute ${this.id} has no vector`); }
  protected computeArray(): void { throw new StyleError(`compute ${this.id} has no array`); }
  protected computePeratom(): void { throw new StyleError(`compute ${this.id} has no per-atom values`); }
  /** Returns the row-major local data and sets localRows. */
  protected computeLocal(): Float64Array<ArrayBuffer> { throw new StyleError(`compute ${this.id} has no local values`); }
  protected computePergrid(): void { throw new StyleError(`compute ${this.id} does not compute per-grid info`); }
  /**
   * The graphics objects of dump image's "compute" keyword (compute_image of native computes; image_objects.ts). Only the
   * computes that provide them (compute hbond/local) override this.
   */
  imageObjects?(): ImageObject[];

  /**
   * compute_modify keyword; returns values consumed (0 = unknown keyword). As native LAMMPS does (src/compute.cpp
   * Compute::modify_params at b332f28): extra/dof and dynamic/dof (compute_modify.html: "keyword = *extra/dof* or
   * *dynamic/dof*"; the older names extra and dynamic are not accepted).
   */
  modify(key: string, values: string[]): number {
    if (key === 'extra/dof') {
      if (values.length < 1) throw missingArgs('compute_modify extra/dof');
      this.extraDof = numeric(values[0]);
      return 1;
    }
    if (key === 'dynamic/dof') {
      if (values.length < 1) throw missingArgs('compute_modify dynamic/dof');
      this.dynamicDofFlag = logical(values[0]);
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
  /**
   * Native dof_remove: the degrees of freedom the bias takes from atom i (i = -1: from every atom alike), which a
   * temperature compute using this one as its bias subtracts. Default none.
   */
  dofRemove(_i: number): number { return 0; }

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

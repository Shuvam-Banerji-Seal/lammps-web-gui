/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_eam.cpp (coeff,
 * coeff_mapped, init_style, init_one, compute bookkeeping, ev_tally) and src/pair.cpp (ev_tally, ev_tally_xyz). Contributing
 * authors named in those sources: Stephen Foiles (SNL), Murray Daw (SNL). Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Shared pieces of the many-body EAM-family styles: the element-to-type mapping of pair_coeff, the masses set from the
 * potential file, the table set-up, and the ghost reverse and forward communication of per-atom arrays.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { buildTables, type EamFile, type EamTables } from './potential';

/** Ghost reverse communication: adds each ghost's values (stride per atom) into its owner (native reverse_comm). */
export const reverseAdd = (owner: Int32Array, nlocal: number, nall: number, a: Float64Array, stride: number): void => {
  for (let k = nlocal; k < nall; k++) {
    const i = owner[k];
    for (let c = 0; c < stride; c++) a[stride * i + c] += a[stride * k + c];
  }
};

/** Ghost forward communication: copies each owner's values (stride per atom) to its ghosts (native forward_comm). */
export const forwardCopy = (owner: Int32Array, nlocal: number, nall: number, a: Float64Array, stride: number): void => {
  for (let k = nlocal; k < nall; k++) {
    const i = owner[k];
    for (let c = 0; c < stride; c++) a[stride * k + c] = a[stride * i + c];
  }
};

/**
 * ev_tally (pair.cpp) without the global virial: the engine's pair virial comes from the fdotr sum over all atoms (native
 * virial_fdotr_compute), so only the energy and the per-atom tallies are made here. fpair is the force over r, del = x_i - x_j.
 */
export const tallyPair = (pc: PairCompute, i: number, j: number, evdwl: number, fpair: number, dx: number, dy: number, dz: number): void => {
  pc.acc.evdwl += evdwl;
  if (pc.eatom) {
    const e = 0.5 * evdwl;
    pc.eatom[i] += e;
    pc.eatom[j] += e;
  }
  if (pc.vatom) {
    const va = pc.vatom;
    const v0 = dx * dx * fpair, v1 = dy * dy * fpair, v2 = dz * dz * fpair;
    const v3 = dx * dy * fpair, v4 = dx * dz * fpair, v5 = dy * dz * fpair;
    for (const k of [i, j]) {
      va[6 * k] += 0.5 * v0; va[6 * k + 1] += 0.5 * v1; va[6 * k + 2] += 0.5 * v2;
      va[6 * k + 3] += 0.5 * v3; va[6 * k + 4] += 0.5 * v4; va[6 * k + 5] += 0.5 * v5;
    }
  }
};

/**
 * Base of the setfl/FS many-body styles: pair_coeff maps the file's elements to the types; the file is read once per name.
 * Subclasses pick the file layout (readFile) and the compute.
 */
export abstract class EamFamilyBase extends Pair {
  manybody = true;
  virialFdotr = true;
  protected map = new Int32Array(0);
  protected file: EamFile | null = null;
  protected tabs: EamTables | null = null;
  protected cutmax = 0;
  protected cutforcesq = 0;
  protected readonly cache = new Map<string, EamFile>();
  /** Per-atom density, embedding derivative and neighbour count of the last compute (native rho, fp, numforce). */
  protected rho = new Float64Array(0);
  protected fp = new Float64Array(0);
  protected numforce = new Int32Array(0);
  protected computed = false;

  /** True for the Finnis-Sinclair density layout (rhor per element pair). */
  protected abstract readonly fsLayout: boolean;

  /** Reads the file of this style's layout. */
  protected abstract readFile(text: string, filename: string, ctx: StyleContext): EamFile;

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.map = new Int32Array(ntypes + 1).fill(-1);
    this.file = null;
    this.tabs = null;
    this.computed = false;
  }

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  /**
   * native PairEAM::coeff_mapped: "pair_coeff * * file El1 ... ElN" with one element (or NULL) per atom type. The file
   * is read before the mapping (so a file error comes first); every mapped I,I type takes the element mass.
   */
  protected mapCoeff(args: string[], ctx: StyleContext, name: string): EamFile {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 3 + this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    const filename = args[2];
    let file = this.cache.get(filename);
    if (!file) {
      file = this.readFile(ctx.readFile(filename), filename, ctx);
      this.cache.set(filename, file);
    }
    this.file = file;
    this.map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let k = 3; k < args.length; k++) {
      const type = k - 2;
      if (args[k] === 'NULL') continue;
      const e = file.names.indexOf(args[k]);
      if (e < 0) throw new StyleError(`No matching element in EAM potential file (${name})`);
      this.map[type] = e;
    }
    for (let i = 1; i <= this.ntypes; i++) {
      if (this.map[i] >= 0) this.setMass(ctx, i, file.mass[this.map[i]]);
    }
    return file;
  }

  /** atom->set_mass (atom.cpp): a value <= 0 is ignored with a warning. */
  protected setMass(ctx: StyleContext, type: number, value: number): void {
    const s = ctx.s;
    if (!s) return;
    if (s.rmass) throw new StyleError('Cannot set per-type atom mass for this atom style');
    if (!(value > 0)) {
      ctx.log(`WARNING: Ignoring invalid mass value ${value} for atom type ${type}`);
      return;
    }
    s.massByType[type] = value;
  }

  /** file2array + array2spline (native init_style): the tables for the mapped types. */
  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.table !== 12) throw new StyleError(`pair_modify table is not supported for pair style ${this.name}`);
    if (!this.file) throw new StyleError('All pair coeffs are not set');
    this.tabs = buildTables(this.file, this.map, this.ntypes, this.fsLayout);
    this.cutmax = this.file.cut;
    this.cutforcesq = this.cutmax * this.cutmax;
  }

  /** native init_one: the global cutoff for mapped pairs; a NULL type is only allowed under hybrid (native Pair::init). */
  initOne(i: number, j: number): number {
    if (!this.file) throw new StyleError('All pair coeffs are not set');
    if (this.map[i] < 0 || this.map[j] < 0) {
      if (i === j && !this.hybridSub) throw new StyleError('All pair coeffs are not set');
      return 0;
    }
    return this.cutmax;
  }

  protected ensureAtomArrays(nall: number): void {
    if (this.rho.length !== nall) {
      this.rho = new Float64Array(nall);
      this.fp = new Float64Array(nall);
    }
  }

  protected get tables(): EamTables {
    if (!this.tabs) throw new StyleError(`pair style ${this.name} is not initialised`);
    return this.tabs;
  }
}

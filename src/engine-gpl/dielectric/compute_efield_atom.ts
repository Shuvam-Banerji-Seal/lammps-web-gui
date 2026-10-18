/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/compute_efield_atom.cpp and compute_efield_atom.h at commit b332f28 (ComputeEfieldAtom:
 * the constructor's keywords, init, setup (the pair and kspace style checks), compute_peratom and the group mask).
 * Contributing author of compute_efield_atom.cpp: Trung Nguyen (Northwestern). DIELECTRIC package authors: Trung Nguyen
 * and Monica Olvera de la Cruz (Northwestern University), and Honghao Li (Northwestern). Copyright Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * compute efield/atom (docs.lammps.org/compute_efield_atom.html): "Define a computation that calculates the electric
 * field at each atom. ... The efield/atom compute is only available for dielectric pair styles and kspace styles
 * (pppm/dielectric and msm/dielectric)." Keywords "pair" and "kspace" select the contributions; with no keyword both
 * are used.
 *
 * Native details kept here: the pair term enters only for atoms with a nonzero charge (q, not the scaled charge), the
 * kspace term enters for every atom, and atoms outside the group give zero. In native setup() the kspace term is always
 * switched on when a kspace style is defined, whatever keyword was given.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { hasCharge } from '../../engine/atoms';

/** Pair styles whose efield the native compute reads (compute_efield_atom.cpp setup()). */
const PAIR_STYLES = new Set([
  'lj/cut/coul/long/dielectric',
  'lj/cut/coul/msm/dielectric',
  'lj/cut/coul/cut/dielectric',
  'lj/cut/coul/debye/dielectric',
  'coul/long/dielectric',
  'coul/cut/dielectric',
]);
/** Kspace styles whose efield the native compute reads (compute_efield_atom.cpp setup()). */
const KSPACE_STYLES = new Set(['pppm/dielectric', 'msm/dielectric']);

export class ComputeEfieldAtom extends Compute {
  readonly style: string = 'efield/atom';
  pairflag = false;
  kspaceflag = false;
  /** Field arrays bound at setup (pair and kspace), null when the term is not used. */
  private efieldPair: Float64Array | null = null;
  private efieldKspace: Float64Array | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
    if (args.length === 0) {
      this.pairflag = true;
      this.kspaceflag = true;
    } else {
      for (const a of args) {
        if (a === 'pair') this.pairflag = true;
        else if (a === 'kspace') this.kspaceflag = true;
        else throw new StyleError('Illegal compute efield/atom command');
      }
    }
  }

  /** compute efield/atom requires atom attribute q (init()). Also re-binds the fields (native setup()). */
  init(): void {
    if (!hasCharge(this.sys.state)) throw new StyleError('compute efield/atom requires atom attribute q');
    this.bind();
  }

  /**
   * Native setup(): the pair style must be one of the dielectric styles that store efield, the kspace style (when one is
   * defined) pppm/dielectric or msm/dielectric, and the kspace term is then on. Checked on every run, as native does.
   * Measured with native LAMMPS: compute efield/atom pair with coul/long/dielectric and pppm/dielectric gives the same
   * field as the default keywords, the kspace term stays on whenever a kspace style is defined.
   */
  private bind(): void {
    const pair = this.sys.ff.pair as { name: string; efield?: Float64Array } | null;
    if (!pair || !PAIR_STYLES.has(pair.name) || !pair.efield) {
      throw new StyleError('Compute efield/atom not supported by pair style');
    }
    this.efieldPair = pair.efield;
    const k = this.sys.ff.kspace as { name: string; efield?: Float64Array } | null;
    if (k) {
      if (!KSPACE_STYLES.has(k.name)) throw new StyleError('Compute efield/atom not supported by kspace style');
      this.efieldKspace = k.efield ?? null;
      this.kspaceflag = true;
    } else {
      this.efieldKspace = null;
      this.kspaceflag = false;
    }
  }

  protected computePeratom(): void {
    this.bind();
    const s = this.sys.state;
    const n = s.n;
    const out = (this.arrayAtom = new Float64Array(3 * n));
    const q = s.q;
    const mask = s.mask;
    const ep = this.efieldPair;
    const ek = this.efieldKspace;
    // the fields are sized by the last force evaluation; a dump before any evaluation has nothing to read
    if (this.pairflag && ep && ep.length < 3 * n) throw new StyleError('compute efield/atom: the pair style has no field yet (run a force evaluation first)');
    if (this.kspaceflag && ek && ek.length < 3 * n) throw new StyleError('compute efield/atom: the kspace style has no field yet (run a force evaluation first)');
    // pair term: atoms with a nonzero charge; kspace term: every atom (native compute_peratom)
    if (this.pairflag && ep) {
      for (let i = 0; i < n; i++) {
        if (q[i] !== 0) for (let j = 0; j < 3; j++) out[3 * i + j] += ep[3 * i + j];
      }
    }
    if (this.kspaceflag && ek) {
      for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) out[3 * i + j] += ek[3 * i + j];
    }
    // zero efield of atoms not in group (after the sum, as native)
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & this.groupBit)) out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = 0;
    }
  }
}

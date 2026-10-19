/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BROWNIAN/fix_propel_self.cpp and
 * fix_propel_self.h (class FixPropelSelf, fix propel/self: a self-propulsion force along the dipole (dipole),
 * along the velocity (velocity), or along a body-frame axis rotated by the quaternion (quat); with the virial
 * contribution of the force, and the unwrapped positions of domain->unmap). Original authors: Sam Cameron
 * (University of Bristol) and Arthur Straube (Zuse Institute Berlin). LAMMPS is Copyright (2003) Sandia Corporation
 * and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { isEllipsoid } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { matvec, quatToMat, type Mat3 } from './mathextra';

/** fix_propel_self.cpp: "static constexpr double TOL = 1.0e-14;" and SMALL for the qvector norm. */
const TOL = 1.0e-14;
const SMALL = 1.0e-14;

type PropelMode = 'dipole' | 'velocity' | 'quat';

export class FixPropelSelf extends Fix {
  readonly style = 'propel/self';
  private readonly mode: PropelMode;
  private readonly magnitude: number;
  private sx = 1.0;
  private sy = 0.0;
  private sz = 0.0;
  /** Per-atom virial (6 per atom, xx yy zz xy xz yz), cleared with this.virial each step. */
  private vAtom = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.virialGlobal = true;
    if (args.length !== 2 && args.length !== 6) throw new StyleError('Incorrect number of fix propel/self arguments');
    if (args[0] === 'velocity') {
      this.mode = 'velocity';
      this.thermoVirial = false;
    } else if (args[0] === 'dipole') {
      this.mode = 'dipole';
      this.thermoVirial = true;
    } else if (args[0] === 'quat') {
      this.mode = 'quat';
      this.thermoVirial = true;
    } else throw new StyleError(`Unknown fix propel/self keyword ${args[0]}`);
    this.magnitude = args[1] === 'inf' ? Infinity : parseNum(args[1], 'magnitude');
    if (args.length === 6) {
      if (this.mode !== 'quat') throw new StyleError("Incorrect number of arguments for 'quat' mode of fix propel/self");
      if (args[2] !== 'qvector') throw new StyleError(`Mismatched fix propel/self keyword ${args[2]}`);
      this.sx = parseNum(args[3], 'qvector');
      this.sy = parseNum(args[4], 'qvector');
      this.sz = parseNum(args[5], 'qvector');
      const snorm = Math.sqrt(this.sx * this.sx + this.sy * this.sy + this.sz * this.sz);
      if (snorm < SMALL) throw new StyleError(`Fix propel/self qvector magnitude ${snorm} is too small`);
      this.sx = this.sx / snorm;
      this.sy = this.sy / snorm;
      this.sz = this.sz / snorm;
    }
  }

  /** FixPropelSelf::init(): the mode's required atom attributes. */
  override init(): void {
    const s = this.sys.state;
    if (this.mode === 'dipole' && !s.mu) throw new StyleError('Fix propel/self with option dipole requires atom attribute mu');
    if (this.mode === 'quat') {
      if (!s.quat || !s.shape) throw new StyleError('Fix propel/self with option quat requires atom style ellipsoid');
      for (let i = 0; i < s.n; i++) {
        if ((s.mask[i] & this.groupBit) && !isEllipsoid(s, i)) throw new StyleError('Fix propel/self with option quat requires extended particles');
      }
    }
  }

  /** FixPropelSelf::post_force: the self-propulsion force is added to the forces of the group. */
  override postForce(): void {
    this.virial.fill(0);
    if (this.thermoVirial) {
      // fix_propel_self.cpp (b332f28): each atom's v_tally(i, vi) adds to vatom[i] as well as to the global virial
      if (this.vAtom.length !== 6 * this.sys.state.n) this.vAtom = new Float64Array(6 * this.sys.state.n);
      else this.vAtom.fill(0);
    }
    if (this.mode === 'dipole') this.postForceDipole();
    else if (this.mode === 'velocity') this.postForceVelocity();
    else this.postForceQuat();
  }

  /** The virial of a force (fx, fy, fz) at the unwrapped position of atom i (v_tally of the source). */
  private readonly unwrapped = [0, 0, 0];

  /**
   * fix_propel_self.cpp (b332f28): v_tally(i, vi) with vi = (fx ux, fy uy, fz uz, fx uy, fx uz, fy uz), u = domain->unmap
   * of x[i]. The global virial and the per-atom virial of atom i both take vi (Fix::v_tally(int i, double *v)).
   */
  private tally(i: number, fx: number, fy: number, fz: number): void {
    if (!this.thermoVirial) return;
    const s = this.sys.state;
    // domain->unmap of the source: the unwrapped position, tilt included for a triclinic box
    const u = this.unwrapped;
    this.sys.geom.unwrap(s.x, s.image, i, u);
    const u0 = u[0], u1 = u[1], u2 = u[2];
    const v0 = fx * u0, v1 = fy * u1, v2 = fz * u2, v3 = fx * u1, v4 = fx * u2, v5 = fy * u2;
    this.virial[0] += v0;
    this.virial[1] += v1;
    this.virial[2] += v2;
    this.virial[3] += v3;
    this.virial[4] += v4;
    this.virial[5] += v5;
    const a = 6 * i;
    this.vAtom[a] += v0;
    this.vAtom[a + 1] += v1;
    this.vAtom[a + 2] += v2;
    this.vAtom[a + 3] += v3;
    this.vAtom[a + 4] += v4;
    this.vAtom[a + 5] += v5;
  }

  /** Per-atom virial of the last post_force, added to out for compute stress/atom (fix_modify virial yes). */
  virialAtom(out: Float64Array): void {
    for (let k = 0; k < this.vAtom.length && k < out.length; k++) out[k] += this.vAtom[k];
  }

  private postForceDipole(): void {
    const s = this.sys.state;
    const { f, mask } = s;
    const mu = s.mu!;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const fx = this.magnitude * mu[4 * i];
      const fy = this.magnitude * mu[4 * i + 1];
      const fz = this.magnitude * mu[4 * i + 2];
      f[3 * i] += fx;
      f[3 * i + 1] += fy;
      f[3 * i + 2] += fz;
      this.tally(i, fx, fy, fz);
    }
  }

  private postForceVelocity(): void {
    const s = this.sys.state;
    const { f, v, mask } = s;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const nv2 = v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2];
      let fnorm = 0.0;
      if (nv2 > TOL) fnorm = this.magnitude / Math.sqrt(nv2);
      const fx = fnorm * v[3 * i];
      const fy = fnorm * v[3 * i + 1];
      const fz = fnorm * v[3 * i + 2];
      f[3 * i] += fx;
      f[3 * i + 1] += fy;
      f[3 * i + 2] += fz;
      this.tally(i, fx, fy, fz);
    }
  }

  private postForceQuat(): void {
    const s = this.sys.state;
    const { f, mask } = s;
    const quat = s.quat!;
    const fAct = [this.sx, this.sy, this.sz];
    const Q: Mat3 = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const q = [0, 0, 0, 0];
    const fRot = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      for (let k = 0; k < 4; k++) q[k] = quat[4 * i + k];
      quatToMat(q, Q);
      matvec(Q, fAct, fRot);
      const fx = this.magnitude * fRot[0];
      const fy = this.magnitude * fRot[1];
      const fz = this.magnitude * fRot[2];
      f[3 * i] += fx;
      f[3 * i + 1] += fy;
      f[3 * i + 2] += fz;
      this.tally(i, fx, fy, fz);
    }
  }
}

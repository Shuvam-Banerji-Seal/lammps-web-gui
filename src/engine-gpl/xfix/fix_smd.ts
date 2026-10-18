/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_smd.cpp and src/EXTRA-FIX/fix_smd.h at commit b332f28 (class FixSMD, fix smd: the
 * constructor, init, post_force, smd_tether, smd_couple, compute_vector; contributing author Axel Kohlmeyer, UPenn).
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Steered molecular dynamics (docs.lammps.org/fix_smd.html): a constant-velocity (cvel K vel) or constant-force
 * (cfor F) pull of the group's centre of mass, tethered to a point (tether) or coupled to a second group (couple).
 * The pulling force is distributed over the group by mass. The global vector is the 7 values of the native
 * compute_vector: the total force on the group (3), the force along the pull direction (or the constant force),
 * the reference distance r_old, the current distance r_now, and the work pmf.
 * The native warning that this fix is unmaintained is printed at construction.
 * Respa is not supported (the core integrator refuses fixes that need the respa levels).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';

/** fix_smd.cpp: SMALL = 0.001 (the distance below which the direction is not updated). */
const SMALL = 0.001;

function numeric(w: string | undefined): number {
  const x = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(x)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return x;
}

export class FixSMD extends Fix {
  readonly style = 'smd';
  vectorFlag = true;
  sizeVector = 7;
  extvector = 1;
  virialGlobal = true;
  private readonly cvel: boolean;
  private kSmd = -1.0;
  private vSmd = -1.0;
  private fSmd = -1.0;
  private readonly tether: boolean;
  private xflag = true;
  private yflag = true;
  private zflag = true;
  private xc = 0.0;
  private yc = 0.0;
  private zc = 0.0;
  private r0 = 0.0;
  private autoX = false;
  private autoY = false;
  private autoZ = false;
  private readonly group2: string = '';
  private group2Bit = 0;
  private xn = 1.0;
  private yn = 1.0;
  private zn = 1.0;
  private pmf = 0.0;
  private rOld = 0.0;
  private rNow = 0.0;
  private masstotal = 0.0;
  private masstotal2 = 0.0;
  private readonly ftotal = [0.0, 0.0, 0.0];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let off = 0;
    if (args[0] === 'cvel') {
      if (args.length < off + 3) throw new StyleError('Illegal fix smd command');
      this.cvel = true;
      this.kSmd = numeric(args[1]);
      this.vSmd = numeric(args[2]); // to be multiplied by update->dt when used
      off += 3;
    } else if (args[0] === 'cfor') {
      if (args.length < off + 2) throw new StyleError('Illegal fix smd command');
      this.cvel = false;
      this.fSmd = numeric(args[1]);
      off += 2;
    } else {
      throw new StyleError('Illegal fix smd command');
    }
    if (args[off] === 'tether') {
      if (args.length < off + 5) throw new StyleError('Illegal fix smd command');
      this.tether = true;
      if (args[off + 1] === 'NULL') this.xflag = false;
      else this.xc = numeric(args[off + 1]);
      if (args[off + 2] === 'NULL') this.yflag = false;
      else this.yc = numeric(args[off + 2]);
      if (args[off + 3] === 'NULL') this.zflag = false;
      else this.zc = numeric(args[off + 3]);
      this.r0 = numeric(args[off + 4]);
      if (this.r0 < 0) throw new StyleError('R0 < 0 for fix smd command');
      off += 5;
    } else if (args[off] === 'couple') {
      if (args.length < off + 6) throw new StyleError('Illegal fix smd command');
      this.tether = false;
      this.group2 = args[off + 1];
      const igroup2 = sys.groups.find(this.group2);
      if (igroup2 === -1) throw new StyleError('Could not find fix smd couple group ID');
      if (this.group2 === group) throw new StyleError('Two groups cannot be the same in fix smd couple');
      this.group2Bit = sys.groups.bit(this.group2);
      if (args[off + 2] === 'NULL') this.xflag = false;
      else if (args[off + 2] === 'auto') this.autoX = true;
      else this.xc = numeric(args[off + 2]);
      if (args[off + 3] === 'NULL') this.yflag = false;
      else if (args[off + 3] === 'auto') this.autoY = true;
      else this.yc = numeric(args[off + 3]);
      if (args[off + 4] === 'NULL') this.zflag = false;
      else if (args[off + 4] === 'auto') this.autoZ = true;
      else this.zc = numeric(args[off + 4]);
      this.r0 = numeric(args[off + 5]);
      if (this.r0 < 0) throw new StyleError('R0 < 0 for fix smd command');
      off += 6;
    } else {
      throw new StyleError('Illegal fix smd command');
    }
    sys.warn('Fix smd is unmaintained. Consider using fix colvars or fix plumed.');
  }

  /** Native Group::mass: the summed mass of the members of a group. */
  private groupMass(bit: number): number {
    const s = this.sys.state;
    let one = 0.0;
    for (const i of nativeOrder(s)) if (s.mask[i] & bit) one += massOf(s, i);
    return one;
  }

  /** Native Group::xcm: the centre of mass of a group from unwrapped positions (cm = sum m x / masstotal). */
  private groupXcm(bit: number, masstotal: number): number[] {
    const s = this.sys.state;
    const cm = [0.0, 0.0, 0.0];
    const u = [0.0, 0.0, 0.0];
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & bit)) continue;
      const massone = massOf(s, i);
      this.sys.geom.unwrap(s.x, s.image, i, u);
      cm[0] += u[0] * massone;
      cm[1] += u[1] * massone;
      cm[2] += u[2] * massone;
    }
    if (masstotal > 0.0) {
      cm[0] /= masstotal;
      cm[1] /= masstotal;
      cm[2] /= masstotal;
    }
    return cm;
  }

  init(): void {
    this.masstotal = this.groupMass(this.groupBit);
    const xcm = this.groupXcm(this.groupBit, this.masstotal);
    let dx: number;
    let dy: number;
    let dz: number;
    if (this.tether) {
      dx = this.xc - xcm[0];
      dy = this.yc - xcm[1];
      dz = this.zc - xcm[2];
    } else {
      this.masstotal2 = this.groupMass(this.group2Bit);
      const xcm2 = this.groupXcm(this.group2Bit, this.masstotal2);
      dx = this.autoX ? xcm2[0] - xcm[0] : this.xc;
      dy = this.autoY ? xcm2[1] - xcm[1] : this.yc;
      dz = this.autoZ ? xcm2[2] - xcm[2] : this.zc;
    }
    if (!this.xflag) dx = 0.0;
    if (!this.yflag) dy = 0.0;
    if (!this.zflag) dz = 0.0;
    this.rOld = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (this.rOld > SMALL) {
      this.xn = dx / this.rOld;
      this.yn = dy / this.rOld;
      this.zn = dz / this.rOld;
    }
  }

  postForce(): void {
    const s = this.sys.state;
    // v_init: the virial is tallied only with fix_modify virial yes
    if (this.thermoVirial) this.virial.fill(0);
    this.ftotal[0] = 0.0;
    this.ftotal[1] = 0.0;
    this.ftotal[2] = 0.0;
    if (this.tether) this.smdTether();
    else this.smdCouple();
    if (this.cvel) this.rOld += this.vSmd * s.dt;
  }

  private smdTether(): void {
    const s = this.sys.state;
    const xcm = this.groupXcm(this.groupBit, this.masstotal);
    const dt = s.dt;
    let dx = xcm[0] - this.xc;
    let dy = xcm[1] - this.yc;
    let dz = xcm[2] - this.zc;
    this.rNow = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (!this.xflag) dx = 0.0;
    if (!this.yflag) dy = 0.0;
    if (!this.zflag) dz = 0.0;
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    let fx: number;
    let fy: number;
    let fz: number;
    if (this.cvel) {
      if (r > SMALL) {
        const dr = r - this.r0 - this.rOld;
        fx = this.kSmd * dx * dr / r;
        fy = this.kSmd * dy * dr / r;
        fz = this.kSmd * dz * dr / r;
        this.pmf += (fx * this.xn + fy * this.yn + fz * this.zn) * this.vSmd * dt;
      } else {
        fx = 0.0;
        fy = 0.0;
        fz = 0.0;
      }
    } else {
      this.rOld = r;
      fx = this.fSmd * dx / r;
      fy = this.fSmd * dy / r;
      fz = this.fSmd * dz / r;
    }
    const { x, f, mask } = s;
    const bit = this.groupBit;
    const u = [0.0, 0.0, 0.0];
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & bit)) continue;
      const massfrac = massOf(s, i) / this.masstotal;
      f[3 * i] -= fx * massfrac;
      f[3 * i + 1] -= fy * massfrac;
      f[3 * i + 2] -= fz * massfrac;
      this.ftotal[0] -= fx * massfrac;
      this.ftotal[1] -= fy * massfrac;
      this.ftotal[2] -= fz * massfrac;
      if (this.thermoVirial) {
        this.sys.geom.unwrap(x, s.image, i, u);
        const v = [
          -fx * massfrac * u[0],
          -fy * massfrac * u[1],
          -fz * massfrac * u[2],
          -fx * massfrac * u[1],
          -fx * massfrac * u[2],
          -fy * massfrac * u[2],
        ];
        for (let k = 0; k < 6; k++) this.virial[k] += v[k];
      }
    }
  }

  private smdCouple(): void {
    const s = this.sys.state;
    const xcm = this.groupXcm(this.groupBit, this.masstotal);
    const xcm2 = this.groupXcm(this.group2Bit, this.masstotal2);
    const dt = s.dt;
    let dx = this.autoX ? xcm2[0] - xcm[0] : this.xn * this.rOld;
    let dy = this.autoY ? xcm2[1] - xcm[1] : this.yn * this.rOld;
    let dz = this.autoZ ? xcm2[2] - xcm[2] : this.zn * this.rOld;
    if (!this.xflag) dx = 0.0;
    if (!this.yflag) dy = 0.0;
    if (!this.zflag) dz = 0.0;
    let r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (r > SMALL) {
      this.xn = dx / r;
      this.yn = dy / r;
      this.zn = dz / r;
    }
    let fx: number;
    let fy: number;
    let fz: number;
    if (this.cvel) {
      dx = xcm2[0] - xcm[0];
      dy = xcm2[1] - xcm[1];
      dz = xcm2[2] - xcm[2];
      this.rNow = Math.sqrt(dx * dx + dy * dy + dz * dz);
      dx -= this.xn * this.rOld;
      dy -= this.yn * this.rOld;
      dz -= this.zn * this.rOld;
      if (!this.xflag) dx = 0.0;
      if (!this.yflag) dy = 0.0;
      if (!this.zflag) dz = 0.0;
      r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const dr = r - this.r0;
      if (r > SMALL) {
        const fsign = this.vSmd < 0.0 ? -1.0 : 1.0;
        fx = this.kSmd * dx * dr / r;
        fy = this.kSmd * dy * dr / r;
        fz = this.kSmd * dz * dr / r;
        this.pmf += (fx * this.xn + fy * this.yn + fz * this.zn) * fsign * this.vSmd * dt;
      } else {
        fx = 0.0;
        fy = 0.0;
        fz = 0.0;
      }
    } else {
      dx = xcm2[0] - xcm[0];
      dy = xcm2[1] - xcm[1];
      dz = xcm2[2] - xcm[2];
      this.rNow = Math.sqrt(dx * dx + dy * dy + dz * dz);
      this.rOld = r;
      fx = this.fSmd * this.xn;
      fy = this.fSmd * this.yn;
      fz = this.fSmd * this.zn;
    }
    const { f, mask } = s;
    for (const i of nativeOrder(s)) {
      if (mask[i] & this.groupBit) {
        const massfrac = massOf(s, i) / this.masstotal;
        f[3 * i] += fx * massfrac;
        f[3 * i + 1] += fy * massfrac;
        f[3 * i + 2] += fz * massfrac;
        this.ftotal[0] += fx * massfrac;
        this.ftotal[1] += fy * massfrac;
        this.ftotal[2] += fz * massfrac;
      }
      if (mask[i] & this.group2Bit) {
        const massfrac = massOf(s, i) / this.masstotal2;
        f[3 * i] -= fx * massfrac;
        f[3 * i + 1] -= fy * massfrac;
        f[3 * i + 2] -= fz * massfrac;
      }
    }
  }

  /** compute_vector(n), 0-based: the 7 global values of the native compute_vector. */
  computeVector(n: number): number {
    if (n < 0 || n >= 7) throw new StyleError(`fix smd: vector index ${n + 1} out of range`);
    if (n < 3) return this.ftotal[n];
    if (n === 3) return this.cvel ? this.ftotal[0] * this.xn + this.ftotal[1] * this.yn + this.ftotal[2] * this.zn : this.fSmd;
    if (n === 4) return this.rOld;
    if (n === 5) return this.rNow;
    return this.pmf;
  }
}

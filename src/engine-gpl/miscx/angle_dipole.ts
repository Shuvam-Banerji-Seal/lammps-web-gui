/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/DIPOLE/angle_dipole.cpp and
 * angle_dipole.h (the LAMMPS developers; Copyright (2003) Sandia Corporation, GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * angle_style dipole (docs.lammps.org/angle_dipole.html): restrains the orientation of a point dipole. Atoms of an angle
 * are the dipole (atom 1), the reference atom it points to (atom 2) and a dummy atom (atom 3) that takes no force:
 *   E = K (cos gamma - cos gamma0)^2,   cos gamma = mu . r / (|mu| |r|),  r = x_ref - x_dip
 * The torque on the dipole is written into atom->torque, and the force that goes with it onto the two atoms. It needs
 * atom attributes mu and torque (atom_style hybrid sphere dipole: the dipole style alone has no torque).
 * Coefficients: "K gamma0" with gamma0 in degrees (angle_coeff N K gamma0).
 */
import { SimpleBonded, atomIndex, delta } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { cosFma as cos } from '../libm/sincos_fma';
import { evTallyAngle } from '../xmolecule/ev';

/** angle_dipole.cpp: SMALL = 1.0e-100. */
const SMALL = 1.0e-100;

export class AngleDipole extends SimpleBonded {
  readonly name = 'dipole';
  readonly kind = 'angle' as const;
  readonly paramNames = ['k', 'gamma0'];

  /** angle_dipole.cpp coeff(): narg must be 3 ("Incorrect args for angle coefficients"). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for angle coefficients');
    this.params.setRange(args[0], this.parse(args.slice(1)));
  }

  /** gamma0 is given in degrees and stored in radians: gamma0 = gamma0_one / 180.0 * MY_PI (angle_dipole.cpp). */
  protected parse(args: string[]): number[] {
    const k = parseNum(args[0], 'k');
    const g = parseNum(args[1], 'gamma0');
    return [k, g / 180.0 * Math.PI];
  }

  /** angle_dipole.cpp init_style(): the atoms need mu and torque. */
  init(ctx?: StyleContext): void {
    const s = ctx?.s;
    if (!s || !s.mu || !s.torque) throw new StyleError('Angle style dipole requires atom attributes mu and torque');
    super.init(ctx);
  }

  /** The equilibrium angle in degrees (fix shake convention). */
  equilibrium(type: number): number {
    return this.params.p('gamma0')[type] / Math.PI * 180.0;
  }

  /** angle_dipole.cpp equilibrium_angle(): gamma0 in radians. */
  equilibriumRadians(type: number): number {
    return this.params.p('gamma0')[type];
  }

  compute(bc: BondedCompute): void {
    const s = bc.s;
    const angles = s.topo.angles;
    const mu = s.mu!;
    const torque = s.torque!;
    const K = this.params.p('k'), gamma0 = this.params.p('gamma0');
    const f = bc.f;
    const d = [0, 0, 0];
    const zero = [0, 0, 0];
    for (let n = 0; n < angles.n; n++) {
      const type = angles.type[n];
      const iDip = atomIndex(bc, angles.atoms[3 * n], 'angle');
      const iRef = atomIndex(bc, angles.atoms[3 * n + 1], 'angle');
      const iDummy = atomIndex(bc, angles.atoms[3 * n + 2], 'angle');
      // separation from the dipole to the reference atom (the copy nearest to the dipole)
      delta(bc, iDip, iRef, d);
      const delx = d[0], dely = d[1], delz = d[2];
      const r = Math.sqrt(delx * delx + dely * dely + delz * delz);
      if (r < SMALL) continue;
      const mx = mu[4 * iDip], my = mu[4 * iDip + 1], mz = mu[4 * iDip + 2];
      const rmu = r * mu[4 * iDip + 3];
      const cosGamma = (mx * delx + my * dely + mz * delz) / rmu;
      const deltaGamma = cosGamma - cos(gamma0[type]);
      const kdg = K[type] * deltaGamma;
      const eangle = kdg * deltaGamma;
      const tangle = 2.0 * kdg / rmu;
      const delTx = tangle * (dely * mz - delz * my);
      const delTy = tangle * (delz * mx - delx * mz);
      const delTz = tangle * (delx * my - dely * mx);
      torque[3 * iDip] += delTx;
      torque[3 * iDip + 1] += delTy;
      torque[3 * iDip + 2] += delTz;
      // force direction: r x (-T); magnitude |T| / r
      const fx = dely * delTz - delz * delTy;
      const fy = delz * delTx - delx * delTz;
      const fz = delx * delTy - dely * delTx;
      const fmod = Math.sqrt(delTx * delTx + delTy * delTy + delTz * delTz) / r;
      const len = Math.sqrt(fx * fx + fy * fy + fz * fz);
      if (len < SMALL) continue;
      const fmodLen = fmod / len;
      const fi = [fx * fmodLen, fy * fmodLen, fz * fmodLen];
      const fj = [-fi[0], -fi[1], -fi[2]];
      f[3 * iDip] += fj[0]; f[3 * iDip + 1] += fj[1]; f[3 * iDip + 2] += fj[2];
      f[3 * iRef] += fi[0]; f[3 * iRef + 1] += fi[1]; f[3 * iRef + 2] += fi[2];
      // virial = rij . fi = 0 (fj = -fi and the dummy atom has no force)
      evTallyAngle(bc, iRef, iDip, iDummy, eangle, fj, fi, zero, zero);
    }
  }
}

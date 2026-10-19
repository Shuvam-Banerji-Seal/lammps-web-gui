/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/YAFF/angle_cross.cpp and src/YAFF/angle_cross.h at commit b332f28 (YAFF package; contributing author
 * Steven Vandenbrande). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * angle_style cross — docs.lammps.org/angle_cross.html:
 *   "E = K_{SS} (r_{12}-r_{12,0})(r_{32}-r_{32,0}) + K_{BS0}(r_{12}-r_{12,0})(\theta-\theta_0)
 *       + K_{BS1}(r_{32}-r_{32,0})(\theta-\theta_0)"
 *   Coefficients: "K_{SS} (energy/distance^2)", "K_{BS0} (energy/distance)", "K_{BS1} (energy/distance)",
 *   "r_{12,0} (distance)", "r_{32,0} (distance)", "\theta_0 (degrees)"; "\theta_0 is specified in degrees, but
 *   LAMMPS converts it to radians internally; hence the K_{BS0} and K_{BS1} are effectively energy/distance per
 *   radian."
 * The forces are the native expressions of angle_cross.cpp (not an analytic gradient of a different form): the
 * bond-stretch term pushes along each bond with the other bond's stretch, and the bond-angle term uses the
 * 1/sin(theta) factor with native's clamp s >= 0.001. Atom 2 (the vertex) takes -(f1 + f3).
 * Geometry: every atom is placed relative to the vertex (delta from the owner), as the other angle styles do.
 */
import { SimpleBonded, atomIndex, delta } from '../../engine/force/bonded_util';
import { fmtCoeff, parseNum } from '../../engine/force/util';
import { StyleError, type BondedCompute } from '../../engine/force/types';
import { acos as libmAcos } from '../libm/acos';
import { tallyAngle } from '../../engine/force/angle/harmonic';

// LAMMPS: static constexpr double SMALL = 0.001 (angle_cross.cpp)
const SMALL = 0.001;

export class AngleCross extends SimpleBonded {
  readonly name = 'cross';
  readonly kind = 'angle' as const;
  readonly paramNames = ['Kss', 'Kbs0', 'Kbs1', 'r00', 'r01', 'theta0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 6) throw new StyleError('angle_coeff cross needs Kss Kbs0 Kbs1 r00 r01 theta0');
    const v = args.map((w, k) => parseNum(w, this.paramNames[k]));
    v[5] = (v[5] * Math.PI) / 180;   // theta0 in degrees -> radians (angle_cross.cpp coeff)
    return v;
  }

  dataCoeffs(): string[] {
    const P = (n: string) => this.params.p(n);
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff(P('Kss')[t])} ${fmtCoeff(P('Kbs0')[t])} ${fmtCoeff(P('Kbs1')[t])} ${fmtCoeff(P('r00')[t])} ${fmtCoeff(P('r01')[t])} ${fmtCoeff((P('theta0')[t] * 180) / Math.PI)}`);
    }
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

  compute(bc: BondedCompute): void {
    const P = (n: string) => this.params.p(n);
    const kss = P('Kss'), kbs0 = P('Kbs0'), kbs1 = P('Kbs1'), r00 = P('r00'), r01 = P('r01'), th0 = P('theta0');
    const A = bc.s.topo.angles;
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      delta(bc, j, i, d1); // r_i - r_j (bond 1, as native delx1)
      delta(bc, j, k, d2); // r_k - r_j (bond 2, as native delx2)
      const rsq1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
      const rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
      const r1 = Math.sqrt(rsq1), r2 = Math.sqrt(rsq2);
      let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
      if (c > 1) c = 1;
      if (c < -1) c = -1;
      let s = Math.sqrt(1 - c * c);
      if (s < SMALL) s = SMALL;
      s = 1 / s;
      const dr1 = r1 - r00[t], dr2 = r2 - r01[t];
      const tk1 = kss[t] * dr1, tk2 = kss[t] * dr2;
      // bond-bond term forces (native f1, f3 before the bond-angle part)
      const f1 = [0, 0, 0], f3 = [0, 0, 0];
      for (let q = 0; q < 3; q++) {
        f1[q] = (-d1[q] * tk2) / r1;
        f3[q] = (-d2[q] * tk1) / r2;
      }
      let eang = kss[t] * dr1 * dr2;
      // bond-angle term
      const dtheta = libmAcos(c) - th0[t];
      const aa1 = s * dr1 * kbs0[t];
      const aa2 = s * dr2 * kbs1[t];
      let aa11 = (aa1 * c) / rsq1;
      const aa12 = -aa1 / (r1 * r2);
      const aa21 = (aa2 * c) / rsq1;
      const aa22 = -aa2 / (r1 * r2);
      const vx11 = [0, 0, 0], vx12 = [0, 0, 0];
      for (let q = 0; q < 3; q++) {
        vx11[q] = aa11 * d1[q] + aa12 * d2[q];
        vx12[q] = aa21 * d1[q] + aa22 * d2[q];
      }
      aa11 = (aa1 * c) / rsq2;
      const aa21b = (aa2 * c) / rsq2;
      const vx21 = [0, 0, 0], vx22 = [0, 0, 0];
      for (let q = 0; q < 3; q++) {
        vx21[q] = aa11 * d2[q] + aa12 * d1[q];
        vx22[q] = aa21b * d2[q] + aa22 * d1[q];
      }
      const b1 = (kbs0[t] * dtheta) / r1;
      const b2 = (kbs1[t] * dtheta) / r2;
      for (let q = 0; q < 3; q++) {
        f1[q] -= vx11[q] + b1 * d1[q] + vx12[q];
        f3[q] -= vx21[q] + b2 * d2[q] + vx22[q];
      }
      eang += kbs0[t] * dr1 * dtheta + kbs1[t] * dr2 * dtheta;
      e += eang;
      // the forces and native ev_tally with delx1, delx2, f1, f3: the virial, a third of the energy and of the virial to
      // each atom, the centroid virial (angle.cpp)
      tallyAngle(bc, i, j, k, eang, f1, f3, d1, d2);
    }
    bc.acc.eangle += e;
  }
}

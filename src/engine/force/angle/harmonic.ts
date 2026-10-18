import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { StyleError, addCentroid, type BondedCompute } from '../types';
import { libm } from '../../libm';

/*
 * angle_style harmonic — docs.lammps.org/angle_harmonic.html:
 *   E = K (theta - theta0)^2  "Note that the usual 1/2 factor is included in K."
 *   coefficients "K (energy)", "theta0 (degrees)"; "theta0 is specified in
 *   degrees, but LAMMPS converts it to radians internally; hence K is
 *   effectively energy per radian^2."
 * theta is the angle at the middle atom j of the triplet i-j-k.
 */

export class AngleHarmonic extends SimpleBonded {
  readonly name = 'harmonic';
  readonly kind = 'angle' as const;
  readonly paramNames = ['K', 'theta0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('angle_coeff harmonic needs K theta0');
    // As native LAMMPS does (src/MOLECULE/angle_harmonic.cpp at b332f28): theta0[i] = DEG2RAD * theta0_one
    return [parseNum(args[0], 'K'), (Math.PI / 180) * parseNum(args[1], 'theta0')];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    // As native LAMMPS does (src/MOLECULE/angle_harmonic.cpp at b332f28): write_data: RAD2DEG * theta0[i]
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((180 / Math.PI) * th[t])}`);
    return out;
  }

  // As native LAMMPS does (src/angle_write.cpp at b332f28, double theta0 = equilibrium_angle(atype) * RAD2DEG
  equilibrium(type: number): number { return (180 / Math.PI) * this.params.p('theta0')[type]; }
  equilibriumRadians(type: number): number { return this.params.p('theta0')[type]; }

  compute(bc: BondedCompute): void {
    const A = bc.s.topo.angles;
    const K = this.params.p('K'), th0 = this.params.p('theta0');
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    let e = 0;
    for (let a = 0; a < A.n; a++) {
      const i = atomIndex(bc, A.atoms[3 * a], 'angle');
      const j = atomIndex(bc, A.atoms[3 * a + 1], 'angle');
      const k = atomIndex(bc, A.atoms[3 * a + 2], 'angle');
      const t = A.type[a];
      angleTerm(bc, i, j, k, d1, d2, (theta) => {
        const dth = theta - th0[t];
        const ea = K[t] * dth * dth;
        e += ea;
        return { e: ea, dEdtheta: 2 * K[t] * dth };
      });
    }
    bc.acc.eangle += e;
  }
}

/**
 * Applies an angle potential: computes theta at j, asks `pot` for the energy
 * and dE/dtheta, adds forces and virial. d1/d2 are scratch arrays.
 */
export const angleTerm = (
  bc: BondedCompute, i: number, j: number, k: number, d1: number[], d2: number[],
  pot: (theta: number, c: number) => { e: number; dEdtheta?: number; dEdc?: number },
  extra?: (f1: number[], f3: number[]) => number,
): void => {
  delta(bc, j, i, d1);   // r_i - r_j
  delta(bc, j, k, d2);   // r_k - r_j
  const r1 = Math.sqrt(d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]);
  const r2 = Math.sqrt(d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]);
  let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
  if (c > 1) c = 1;
  if (c < -1) c = -1;
  const theta = libm.acos(c);
  const r = pot(theta, c);
  // dE/dc either directly or from dE/dtheta: dtheta/dc = -1/sin(theta)
  let dEdc: number;
  if (r.dEdc !== undefined) dEdc = r.dEdc;
  else {
    let s = Math.sqrt(1 - c * c);
    if (s < 0.001) s = 0.001;
    dEdc = -(r.dEdtheta ?? 0) / s;
  }
  // dc/dr_i = (d2/(r1 r2) - c d1/r1^2), dc/dr_k = (d1/(r1 r2) - c d2/r2^2)
  const f1 = [0, 0, 0], f3 = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    f1[d] = -dEdc * (d2[d] / (r1 * r2) - c * d1[d] / (r1 * r1));
    f3[d] = -dEdc * (d1[d] / (r1 * r2) - c * d2[d] / (r2 * r2));
  }
  // a term of the same angle that acts on atoms 1 and 3 (CHARMM Urey-Bradley) adds to f1 and f3 and the energy
  // before the tally, as native angle_charmm.cpp passes the combined f1, f3 to one Angle::ev_tally
  const e = r.e + (extra ? extra(f1, f3) : 0);
  tallyAngle(bc, i, j, k, e, f1, f3, d1, d2);
};

/**
 * Angle::ev_tally (angle.cpp) for the angle i-j-k with d1 = r_i - r_j, d2 = r_k - r_j and the forces f1 on i and f3
 * on k: applies the forces (f2 = -f1 - f3), adds the virial, a third of the energy and of the virial to each atom,
 * and the centroid virial (r_a - r_0) (x) f_a with r_0 = (r_i + r_j + r_k) / 3.
 */
export const tallyAngle = (
  bc: BondedCompute, i: number, j: number, k: number, e: number,
  f1: readonly number[], f3: readonly number[], d1: readonly number[], d2: readonly number[],
): void => {
  const f = bc.f;
  for (let d = 0; d < 3; d++) {
    f[3 * i + d] += f1[d];
    f[3 * j + d] -= f1[d] + f3[d];
    f[3 * k + d] += f3[d];
  }
  const w = [
    d1[0] * f1[0] + d2[0] * f3[0], d1[1] * f1[1] + d2[1] * f3[1], d1[2] * f1[2] + d2[2] * f3[2],
    d1[0] * f1[1] + d2[0] * f3[1], d1[0] * f1[2] + d2[0] * f3[2], d1[1] * f1[2] + d2[1] * f3[2],
  ];
  for (let q = 0; q < 6; q++) bc.virial[q] += w[q];
  if (bc.eatom) { const third = e / 3; bc.eatom[i] += third; bc.eatom[j] += third; bc.eatom[k] += third; }
  if (bc.vatom) for (const a of [i, j, k]) for (let q = 0; q < 6; q++) bc.vatom[6 * a + q] += w[q] / 3;
  if (bc.cvatom) {
    // a1 = (2 d1 - d2) / 3, a2 = (-d1 - d2) / 3, a3 = (-d1 + 2 d2) / 3 (angle.cpp ev_tally with cvflag_atom)
    const T = 1 / 3;
    const f2 = [-f1[0] - f3[0], -f1[1] - f3[1], -f1[2] - f3[2]];
    addCentroid(bc.cvatom, i, T * (2 * d1[0] - d2[0]), T * (2 * d1[1] - d2[1]), T * (2 * d1[2] - d2[2]), f1[0], f1[1], f1[2]);
    addCentroid(bc.cvatom, j, T * (-d1[0] - d2[0]), T * (-d1[1] - d2[1]), T * (-d1[2] - d2[2]), f2[0], f2[1], f2[2]);
    addCentroid(bc.cvatom, k, T * (-d1[0] + 2 * d2[0]), T * (-d1[1] + 2 * d2[1]), T * (-d1[2] + 2 * d2[2]), f3[0], f3[1], f3[2]);
  }
};

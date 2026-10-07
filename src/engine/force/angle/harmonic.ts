import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { parseNum } from '../util';
import { StyleError, type BondedCompute } from '../types';

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
    return [parseNum(args[0], 'K'), (parseNum(args[1], 'theta0') * Math.PI) / 180];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), th = this.params.p('theta0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${K[t]} ${(th[t] * 180) / Math.PI}`);
    return out;
  }

  equilibrium(type: number): number { return (this.params.p('theta0')[type] * 180) / Math.PI; }

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
): void => {
  delta(bc, j, i, d1);   // r_i - r_j
  delta(bc, j, k, d2);   // r_k - r_j
  const r1 = Math.sqrt(d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]);
  const r2 = Math.sqrt(d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]);
  let c = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (r1 * r2);
  if (c > 1) c = 1;
  if (c < -1) c = -1;
  const theta = Math.acos(c);
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
  if (bc.eatom) { const third = r.e / 3; bc.eatom[i] += third; bc.eatom[j] += third; bc.eatom[k] += third; }
  if (bc.vatom) for (const a of [i, j, k]) for (let q = 0; q < 6; q++) bc.vatom[6 * a + q] += w[q] / 3;
};

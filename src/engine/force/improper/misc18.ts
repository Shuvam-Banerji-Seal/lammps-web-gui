import { SimpleBonded, atomIndex, delta } from '../bonded_util';
import { parseNum } from '../util';
import { bondedVirial, StyleError, type BondedCompute } from '../types';

/*
 * improper_style distharm and sqdistharm (wave 18).
 * docs.lammps.org/improper_distharm.html: "The *distharm* improper style uses the potential", "E = K (d -
 * d_0)^2", "then the L-atom is assumed to be the central atom" and "The distance :math:`d` is oriented and
 * can take on negative values."; improper_sqdistharm.html: "E = K (d^2 - {d_0}^2)^2". That is the potential
 * E = K (d - d_0)^2
 * with K in energy/distance^2 and d_0 in distance, where d is the oriented
 * distance of the central atom from the plane of the other three atoms. The
 * page says the L atom is the central one (I,J,K,L ordering) and that d can be
 * negative.
 * docs.lammps.org/improper_sqdistharm.html gives E = K (d^2 - d_0^2)^2 with K
 * in energy/distance^4 and d_0^2 (not d_0) in distance^2.
 *
 * Measured with native LAMMPS (black box), one improper with atoms
 * I=(0,0,0), J=(1.3,0,0), K=(0.2,1.1,0), L=(0.3,0.2,0.7):
 *   - distharm with improper_coeff 1 1.0 0.5: native pe (here the only term)
 *     is 1.4761363627059, which equals (d - 0.5)^2 with d = -0.714963523. So
 *     the energy is K (d - d_0)^2 with no extra factor, and the signed d is
 *     the distance of atom I from the plane of J, K, L (the normal is
 *     b1 x b2, b1 = r_K - r_J, b2 = r_L - r_J), not the distance of L from
 *     the plane of I, J, K that the page describes.
 *   - sqdistharm with improper_coeff 1 1.0 0.25: native pe is
 *     0.0682112520957173, which equals (d^2 - 0.25)^2 for the same d.
 *   - the per-atom forces of both styles agree with the gradient of these
 *     energies, the same signed d being used (a flipped sign would change
 *     the forces), and the oracle cases w18dihimp_distharm and
 *     w18dihimp_sqdistharm agree with native over a 50-step nve run.
 * An earlier version divided both energies by 4; the forces had been correct
 * all along, so only the energy changed.
 *
 * d is differentiated analytically: with b1 = r_K - r_J, b2 = r_L - r_J,
 * w = r_I - r_J, n = b1 x b2, a = w.n, N = |n|, d = a/N,
 *   da/dr_I = n, da/dr_K = b2 x w, da/dr_L = w x b1, da/dr_J = -(sum of those)
 *   dn^2/dr_I = 0, dn^2/dr_K = 2 b2 x n, dn^2/dr_L = 2 n x b1,
 *   dn^2/dr_J = -(sum of those)
 *   dd/dr_p = [da/dr_p - (d / 2 n) dn^2/dr_p] / N   (N = |n|, so the chain
 *   rule through N = sqrt(n^2) contributes d / (2 n))
 * (a is the scalar triple product det(w, b1, b2): its gradients are
 * d/db1 = b2 x w and d/db2 = w x b1; n^2 = |b1|^2|b2|^2 - (b1.b2)^2 gives
 * dn^2/db1 = 2 b2 x n and dn^2/db2 = 2 n x b1. Checked against central finite
 * differences in tests/engineDihImp18.test.ts.)
 *
 * A degenerate plane (the three plane atoms collinear, |n| = 0) makes native
 * report nan for both the energy and the forces (measured in an earlier wave);
 * the engine returns d = 0 with a zero gradient instead, as improper_style
 * distance does for the same situation, so a run cannot be poisoned by a nan.
 */

/**
 * Oriented distance of atoms[0] (I) from the plane of atoms[1..3] (J,K,L),
 * native's convention for distharm/sqdistharm. Fills grad[3 * p + d] with
 * dd/dr of the coordinate d of atoms[p]; returns 0 with a zero gradient for a
 * degenerate (collinear) plane, as native does.
 */
export const planeDistance = (bc: BondedCompute, atoms: readonly number[], grad: number[]): number => {
  const b1 = [0, 0, 0], b2 = [0, 0, 0], w = [0, 0, 0];
  delta(bc, atoms[1], atoms[2], b1); // r_K - r_J
  delta(bc, atoms[1], atoms[3], b2); // r_L - r_J
  delta(bc, atoms[1], atoms[0], w); // r_I - r_J
  const n = [
    b1[1] * b2[2] - b1[2] * b2[1],
    b1[2] * b2[0] - b1[0] * b2[2],
    b1[0] * b2[1] - b1[1] * b2[0],
  ];
  const a = w[0] * n[0] + w[1] * n[1] + w[2] * n[2];
  const n2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  if (n2 === 0) {
    grad.fill(0);
    return 0;
  }
  const N = Math.sqrt(n2);
  const d = a / N;
  const b2xw = [b2[1] * w[2] - b2[2] * w[1], b2[2] * w[0] - b2[0] * w[2], b2[0] * w[1] - b2[1] * w[0]];
  const wxb1 = [w[1] * b1[2] - w[2] * b1[1], w[2] * b1[0] - w[0] * b1[2], w[0] * b1[1] - w[1] * b1[0]];
  const b2xn = [b2[1] * n[2] - b2[2] * n[1], b2[2] * n[0] - b2[0] * n[2], b2[0] * n[1] - b2[1] * n[0]];
  const nxb1 = [n[1] * b1[2] - n[2] * b1[1], n[2] * b1[0] - n[0] * b1[2], n[0] * b1[1] - n[1] * b1[0]];
  // da/dx and dn^2/dx for atoms I, J, K, L
  const ga = [
    [n[0], n[1], n[2]],
    [-(n[0] + b2xw[0] + wxb1[0]), -(n[1] + b2xw[1] + wxb1[1]), -(n[2] + b2xw[2] + wxb1[2])],
    [b2xw[0], b2xw[1], b2xw[2]],
    [wxb1[0], wxb1[1], wxb1[2]],
  ];
  const gn = [
    [0, 0, 0],
    [-2 * (b2xn[0] + nxb1[0]), -2 * (b2xn[1] + nxb1[1]), -2 * (b2xn[2] + nxb1[2])],
    [2 * b2xn[0], 2 * b2xn[1], 2 * b2xn[2]],
    [2 * nxb1[0], 2 * nxb1[1], 2 * nxb1[2]],
  ];
  const inv = 1 / N, k = d / (2 * N);
  for (let p = 0; p < 4; p++) {
    for (let c = 0; c < 3; c++) grad[3 * p + c] = inv * (ga[p][c] - k * gn[p][c]);
  }
  return d;
};

/** Adds f = -dEdd * dd/dr to the four atoms and tallies virial / per-atom energy. */
const applyGradD = (bc: BondedCompute, atoms: readonly number[], dEdd: number, grad: readonly number[], e: number): void => {
  const f = bc.f;
  const fk: number[] = new Array(12);
  for (let p = 0; p < 4; p++) {
    for (let c = 0; c < 3; c++) {
      const v = -dEdd * grad[3 * p + c];
      fk[3 * p + c] = v;
      f[3 * atoms[p] + c] += v;
    }
  }
  const rel: number[] = new Array(12);
  for (let p = 0; p < 4; p++) delta(bc, atoms[0], atoms[p], rel, 3 * p);
  bondedVirial(bc, atoms, rel, fk, e);
};

export class ImproperDistharm extends SimpleBonded {
  readonly name = 'distharm';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'd0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('improper_coeff distharm needs K d0');
    return [parseNum(args[0], 'K'), parseNum(args[1], 'd0')];
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), d0 = this.params.p('d0');
    const grad = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const d = planeDistance(bc, atoms, grad);
      const diff = d - d0[t];
      const ei = K[t] * diff * diff;
      e += ei;
      applyGradD(bc, atoms, 2 * K[t] * diff, grad, ei);
    }
    bc.acc.eimp += e;
  }
}

export class ImproperSqdistharm extends SimpleBonded {
  readonly name = 'sqdistharm';
  readonly kind = 'improper' as const;
  /** The second coefficient is d_0^2, not d_0 (see header). */
  readonly paramNames = ['K', 'd02'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('improper_coeff sqdistharm needs K d02');
    return [parseNum(args[0], 'K'), parseNum(args[1], 'd02')];
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), d02 = this.params.p('d02');
    const grad = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const d = planeDistance(bc, atoms, grad);
      const diff = d * d - d02[t];
      const ei = K[t] * diff * diff;
      e += ei;
      applyGradD(bc, atoms, 4 * K[t] * d * diff, grad, ei);
    }
    bc.acc.eimp += e;
  }
}
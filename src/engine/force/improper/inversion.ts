import { SimpleBonded, atomIndex, applyDihedral, termDelta } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { StyleError, type BondedCompute } from '../types';

/*
 * improper_style inversion/harmonic — docs.lammps.org/improper_inversion_harmonic.html (text in
 * plans/lammps-docs-live/improper_inversion_harmonic.rst):
 *   "The *inversion/harmonic* improper style follows the Wilson-Decius out-of-plane angle definition and uses an
 *   harmonic potential"; "E = K \left(\omega - \omega_0\right)^2"; "where K is the force constant and omega is the
 *   angle evaluated for all three axis-plane combinations centered around the atom I."
 * Coefficients: "K (energy)", "omega_0 (degrees)". "the first atom in the quadruplet is the atom of symmetry; all
 * other atoms are considered interchangeable": atom 1 of the quadruplet is the centre I.
 *
 * Energy. Measured with native LAMMPS (black box): for an improper I,J,K,L the energy is (K/3) times the sum over
 * the three axis-plane combinations (axis L with plane IJK, axis J with plane IKL, axis K with plane ILJ) of
 * (omega_c - omega_0)^2, omega_c = asin(|u_c . n_c|) in radians, u_c the unit vector I->axis and n_c the unit normal
 * of the plane. Eight random configurations with omega_0 = 0 and 20 degrees (K = 10) agree to 1e-10 relative; the
 * unsigned |omega| is required (the signed sum differs for omega_0 = 20 degrees). The factor 1/3 is the average over
 * the three combinations, measured as part of the same fit.
 *
 * Forces. The gradient is analytic: with p = I->A, q = I->B, r = I->C, D = p.(q x r), N = q x r, s = D/(|p||N|):
 * dE/ds = (2K/3)(omega - omega_0) sign(s)/sqrt(1 - s^2); ds/dp = N/(|p||N|) - D p/(|p|^3 |N|);
 * ds/dN = p/(|p||N|) - D N/(|p||N|^3); ds/dq = r x (ds/dN); ds/dr = (ds/dN) x q; ds/dI = -(ds/dp + ds/dq + ds/dr).
 * Near-degenerate geometry (|s| -> 1, axis along the plane normal) is not measured; the derivative is floored there.
 */

type V3 = [number, number, number];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3): number => Math.sqrt(dot(a, a));

/** Axis-plane combinations as slots [I, axis, planeB, planeC] into the quadruplet atoms (indices 0..3). */
const COMBOS: readonly (readonly [number, number, number, number])[] = [
  [0, 3, 1, 2], // axis L, plane IJK
  [0, 1, 2, 3], // axis J, plane IKL
  [0, 2, 3, 1], // axis K, plane ILJ
];

/*
 * Native's omega (improper_inversion_harmonic.cpp in src/MOFFF at b332f28, invang): the cosine of the angle
 * between the axis vector p and the plane spanned by q and r is taken from the projections of p on the two
 * unit bisectors (q^ + r^) and (r^ - q^) of the plane, cos = sqrt((p.u+)^2 + (p.u-)^2) / |p|, clamped at 1, and
 * omega = acos(cos). Near planar geometry omega is small and acos of this cosine loses precision (about 1e-16/omega
 * absolute), while asin(s) of the triple product does not; the energy and the 1/sin(omega) factor of the force
 * therefore use this cosine, as native does. Measured with native LAMMPS (black box): see tests/oracle/w120cf_improper_inversion_planar.
 */
const cosInversion = (q: V3, r: V3, p: V3): number => {
  const rq = 1.0 / len(q), rr = 1.0 / len(r);
  let up: V3 = [r[0] * rr + q[0] * rq, r[1] * rr + q[1] * rq, r[2] * rr + q[2] * rq];
  let um: V3 = [r[0] * rr - q[0] * rq, r[1] * rr - q[1] * rq, r[2] * rr - q[2] * rq];
  const nup = 1.0 / len(up), num = 1.0 / len(um);
  up = [up[0] * nup, up[1] * nup, up[2] * nup];
  um = [um[0] * num, um[1] * num, um[2] * num];
  const rup = dot(p, up), rum = dot(p, um);
  const c = Math.sqrt(rup * rup + rum * rum) / len(p);
  return c > 1.0 ? 1.0 : c;
};

export class ImproperInversionHarmonic extends SimpleBonded {
  readonly name = 'inversion/harmonic';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'omega0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('improper_coeff inversion/harmonic needs K omega0');
    return [parseNum(args[0], 'K'), (parseNum(args[1], 'omega0') * Math.PI) / 180];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), w0 = this.params.p('omega0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((w0[t] * 180) / Math.PI)}`);
    return out;
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), w0 = this.params.p('omega0');
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    const p: V3 = [0, 0, 0], q: V3 = [0, 0, 0], r: V3 = [0, 0, 0];
    const d = [0, 0, 0];
    let e = 0;
    for (let n = 0; n < I.n; n++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * n + w], 'improper'));
      const t = I.type[n];
      const k3 = K[t] / 3;
      let eq = 0;
      for (const [c, ax, pb, pc] of COMBOS) {
        const slots = [atoms[c], atoms[ax], atoms[pb], atoms[pc]];
        termDelta(bc, atoms[1], slots[0], slots[1], d); p[0] = d[0]; p[1] = d[1]; p[2] = d[2];
        termDelta(bc, atoms[1], slots[0], slots[2], d); q[0] = d[0]; q[1] = d[1]; q[2] = d[2];
        termDelta(bc, atoms[1], slots[0], slots[3], d); r[0] = d[0]; r[1] = d[1]; r[2] = d[2];
        const N = cross(q, r);
        const P = len(p), M = len(N);
        const D = dot(p, N);
        const pm = P * M;
        const s = pm > 0 ? D / pm : 0;
        const sa = Math.min(1, Math.abs(s));
        // energy and 1/sin(omega) factor from native's omega (cosInversion); the derivative of asin(s) is exact
        const omegaN = Math.acos(cosInversion(q, r, p));
        const dw = omegaN - w0[t];
        const ec = k3 * dw * dw;
        eq += ec;
        // native's force is 2 k3 dw / sin(omega_native) times the gradient of the cosine; the engine's exact
        // gradient of asin(s) carries sin(omega) = sa, so it is scaled by sa / sin(omega_native) (0 below 1e-12
        // rad, where native applies no angle force)
        const sinN = Math.sin(omegaN);
        const ratio = omegaN * omegaN > 1e-24 ? sa / sinN : 0;
        // ds/dp, ds/dq, ds/dr (see the header)
        const gN: V3 = pm > 0
          ? [p[0] / pm - (D * N[0]) / (P * M * M * M), p[1] / pm - (D * N[1]) / (P * M * M * M), p[2] / pm - (D * N[2]) / (P * M * M * M)]
          : [0, 0, 0];
        const gp: V3 = pm > 0
          ? [N[0] / pm - (D * p[0]) / (P * P * P * M), N[1] / pm - (D * p[1]) / (P * P * P * M), N[2] / pm - (D * p[2]) / (P * P * P * M)]
          : [0, 0, 0];
        const gq = cross(r, gN), gr = cross(gN, q);
        for (let k = 0; k < 3; k++) {
          grad[3 + k] = gp[k];
          grad[6 + k] = gq[k];
          grad[9 + k] = gr[k];
          grad[k] = -(gp[k] + gq[k] + gr[k]);
          rel[k] = 0;
          rel[3 + k] = p[k];
          rel[6 + k] = q[k];
          rel[9 + k] = r[k];
        }
        const denom = Math.sqrt(Math.max(1 - sa * sa, 1e-30));
        const dwds = (s < 0 ? -1 : 1) / denom;
        applyDihedral(bc, slots, 2 * k3 * dw * dwds * ratio, grad, rel, ec);
      }
      e += eq;
    }
    bc.acc.eimp += e;
  }
}

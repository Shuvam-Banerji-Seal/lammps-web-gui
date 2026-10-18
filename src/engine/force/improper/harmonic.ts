import { SimpleBonded, atomIndex, dihedralGeometry, applyDihedral, delta, termDelta } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { StyleError, type BondedCompute } from '../types';
import { libm } from '../../libm';

/*
 * improper_style harmonic — docs.lammps.org/improper_harmonic.html:
 *   E = K (chi - chi0)^2  "Note that the usual 1/2 factor is included in K."
 *   "If the 4 atoms in an improper quadruplet ... are ordered I,J,K,L then
 *   chi is the angle between the plane of I,J,K and the plane of J,K,L."
 *   coefficients "K (energy)", "chi0 (degrees)".
 * chi is taken as the unsigned angle between the planes, in [0, pi].
 *
 * As native LAMMPS does (src/MOLECULE/improper_harmonic.cpp at b332f28): chi is acos(c), where c is the
 * cosine of the angle between the planes built from the bond vectors (vb1 = I-J, vb2 = K-J, vb3 = L-K),
 * and the force is -2 K (chi - chi0) dc/dx / s with s = sqrt(1 - c^2) floored at 0.001. Near planar
 * geometry (chi of a few hundredths of a degree) the force is dominated by 1/s, and s computed from the
 * rounded c carries a relative error of about 1e-11. The engine therefore takes the exact gradient of
 * chi (Blondel-Karplus, bonded_util dihedralGeometry, which equals -(dc/dx)/sin(chi) with the accurate
 * sin(chi)) and multiplies it by sin(chi) / s_native, where s_native = max(sqrt(1 - c^2), 0.001) from
 * native's c. Away from planarity the factor is 1 to rounding; below sin(chi) = 0.001 it is
 * sin(chi)/0.001, as before. The energy uses chi = acos(c), as native does.
 * Pinned by tests/oracle/w120cf_improper_planar (static forces and energy of chains at chi 0.003 to 0.3 deg).
 */
const SMALL = 0.001;

const dot = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Native's cosine of the planes angle (c0, c1, c2, s1, s2, s12 and c of improper_harmonic.cpp). */
const nativeCos = (vb1: number[], vb2: number[], vb3: number[]): number => {
  const r1 = Math.sqrt(1.0 / dot(vb1, vb1));
  const r2 = Math.sqrt(1.0 / dot(vb2, vb2));
  const r3 = Math.sqrt(1.0 / dot(vb3, vb3));
  const c0 = dot(vb1, vb3) * r1 * r3;
  const c1 = dot(vb1, vb2) * r1 * r2;
  const c2 = -dot(vb3, vb2) * r3 * r2;
  let s1 = 1.0 - c1 * c1;
  if (s1 < SMALL) s1 = SMALL;
  s1 = 1.0 / s1;
  let s2 = 1.0 - c2 * c2;
  if (s2 < SMALL) s2 = SMALL;
  s2 = 1.0 / s2;
  let c = (c1 * c2 + c0) * Math.sqrt(s1 * s2);
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  return c;
};

export class ImproperHarmonic extends SimpleBonded {
  readonly name = 'harmonic';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'chi0'];

  protected parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('improper_coeff harmonic needs K chi0');
    // As native LAMMPS does (src/MOLECULE/improper_harmonic.cpp at b332f28): chi[i] = DEG2RAD * chi_one
    return [parseNum(args[0], 'K'), (Math.PI / 180) * parseNum(args[1], 'chi0')];
  }

  dataCoeffs(): string[] {
    const K = this.params.p('K'), c0 = this.params.p('chi0');
    const out: string[] = [];
    // As native LAMMPS does (src/MOLECULE/improper_harmonic.cpp at b332f28): write_data: RAD2DEG * chi[i]
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((180 / Math.PI) * c0[t])}`);
    return out;
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), c0 = this.params.p('chi0');
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    const vb1 = [0, 0, 0], vb2 = [0, 0, 0], vb3 = [0, 0, 0];
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
      // unsigned angle between the planes: chi = |phi| (exact), dchi/dphi = sign(phi)
      const sgn = phi < 0 ? -1 : 1;
      // native's chi = acos(c) and s = sqrt(1 - c^2) floored, from the vectors I-J, K-J, L-K
      delta(bc, atoms[1], atoms[0], vb1);
      delta(bc, atoms[1], atoms[2], vb2);
      termDelta(bc, atoms[1], atoms[2], atoms[3], vb3);
      const cn = nativeCos(vb1, vb2, vb3);
      const chi = libm.acos(cn);
      const sNative = Math.max(Math.sqrt(1 - cn * cn), SMALL);
      const sTrue = libm.sin(Math.abs(phi));
      const dchi = chi - c0[t];
      const ei = K[t] * dchi * dchi;
      e += ei;
      applyDihedral(bc, atoms, 2 * K[t] * dchi * sgn * (sTrue / sNative), grad, rel, ei);
    }
    bc.acc.eimp += e;
  }
}

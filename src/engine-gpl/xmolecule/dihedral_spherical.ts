/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-MOLECULE/dihedral_spherical.cpp and dihedral_spherical.h at commit b332f28 (the LAMMPS
 * developers; the spherical dihedral of Dr. Andrew Jewett, Copyright (2003) Sandia Corporation, GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * dihedral_style spherical (docs.lammps.org/dihedral_spherical.html), the sum of terms in the dihedral angle phi and
 * the two bond angles theta1 (atoms 1-2-3) and theta2 (atoms 2-3-4):
 *   E = sum_i C_i Phi_i(phi) Theta1_i(theta1) Theta2_i(theta2)
 *   Phi_i = 1 - cos(n_i (phi - phi_i))     (doc: v_i - cos(...); coefficients: C, n, phi_i, v_i)
 *   Theta1_i = v1_i - cos(m1_i (theta1 - theta1_i)),  Theta2_i = v2_i - cos(m2_i (theta2 - theta2_i))
 * dihedral_coeff N nterms, then per term 10 values: "C phi_mult phi_shift(deg) phi_offset theta1_mult
 * theta1_shift(deg) theta1_offset theta2_mult theta2_shift(deg) theta2_offset" (the doc's Coeffs). The shifts are
 * stored in radians. The phi, theta1 and theta2 expressions, the vectors and the derivatives are native's, in native's
 * order; the acos, cos and sin calls use the glibc ports of src/engine-gpl/libm.
 */
import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext, NoSettingsError } from '../../engine/force/types';
import { atomIndex } from '../../engine/force/bonded_util';
import { fmtCoeff, inumeric, parseNum } from '../../engine/force/util';
import { cos, sin } from '../libm/sincos';
import { evTallyFour } from './ev';
import { acos as libmAcos } from '../libm/acos';

// MathConst (math_const.h)
const MY_PI = 3.14159265358979323846;
const MY_2PI = 6.28318530717958647692;

/** MathExtra::dot3 */
const dot3 = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
/** MathExtra::cross3: ans = a x b */
const cross3 = (a: readonly number[], b: readonly number[], ans: number[]): void => {
  ans[0] = a[1] * b[2] - a[2] * b[1];
  ans[1] = a[2] * b[0] - a[0] * b[2];
  ans[2] = a[0] * b[1] - a[1] * b[0];
};
/** dihedral_spherical.cpp norm3safe: unit vector, unchanged when the length is 0. */
const norm3safe = (v: number[]): void => {
  const inv_scale = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  let scale = 1.0;
  if (inv_scale > 0.0) scale = 1.0 / inv_scale;
  v[0] *= scale;
  v[1] *= scale;
  v[2] *= scale;
};

/** Number of coefficient values per term (the 10 after nterms). */
const PER_TERM = 10;

export class DihedralSpherical extends Bonded {
  readonly name = 'spherical';
  readonly kind = 'dihedral' as const;
  private nterms!: Int32Array;
  /** per type: one array per coefficient, in native's order (Ccoeff, phi_mult, phi_shift, phi_offset, theta1_mult, ...). */
  private terms: (Float64Array[] | null)[] = [];
  private set!: Uint8Array;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new NoSettingsError(args[0]);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nterms = new Int32Array(ntypes + 1);
    this.terms = new Array(ntypes + 1).fill(null);
    this.set = new Uint8Array(ntypes + 1);
  }

  /**
   * dihedral_spherical.cpp coeff(): "narg < 4" and "nterms < 1" and the argument counts ("Incorrect number of arguments
   * for dihedral coefficients", 2 + 10 nterms words including the type). Native only checks for too many words; a
   * missing value is read as NULL and refused by utils::numeric.
   */
  coeff(args: string[], _ctx?: StyleContext): void {
    const narg = args.length;
    if (narg < 4) throw new StyleError('Incorrect args for dihedral coefficients');
    const [lo, hi] = typeBounds(args[0], this.ntypes);
    const nterms_one = inumeric(args[1]);
    if (nterms_one < 1) throw new StyleError('Incorrect number of terms arg for dihedral coefficients');
    if (narg > 2 + PER_TERM * nterms_one) throw new StyleError('Incorrect number of arguments for dihedral coefficients');
    for (let i = lo; i <= hi; i++) {
      this.nterms[i] = nterms_one;
      const arrs = Array.from({ length: PER_TERM }, () => new Float64Array(nterms_one));
      for (let j = 0; j < nterms_one; j++) {
        const offset = 1 + PER_TERM * j;
        const num = (k: number): number => parseNum(args[offset + k], 'dihedral coefficient');
        arrs[0][j] = num(1); // Ccoeff
        arrs[1][j] = num(2); // phi_mult
        arrs[2][j] = (num(3) * MY_PI) / 180.0; // phi_shift
        arrs[3][j] = num(4); // phi_offset
        arrs[4][j] = num(5); // theta1_mult
        arrs[5][j] = (num(6) * MY_PI) / 180.0; // theta1_shift
        arrs[6][j] = num(7); // theta1_offset
        arrs[7][j] = num(8); // theta2_mult
        arrs[8][j] = (num(9) * MY_PI) / 180.0; // theta2_shift
        arrs[9][j] = num(10); // theta2_offset
      }
      this.terms[i] = arrs;
      this.set[i] = 1;
    }
  }

  init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.set[t]) throw new StyleError(`All dihedral coeffs are not set (type ${t})`);
    }
  }

  /** write_data: "type nterms" then, per term, the 10 values as native prints them (each followed by a space). */
  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      const arrs = this.terms[t];
      if (!arrs) continue;
      let line = `${t} ${this.nterms[t]} `;
      for (let j = 0; j < this.nterms[t]; j++) {
        const vals = [
          arrs[0][j], arrs[1][j], (arrs[2][j] * 180.0) / MY_PI, arrs[3][j], arrs[4][j],
          (arrs[5][j] * 180.0) / MY_PI, arrs[6][j], arrs[7][j], (arrs[8][j] * 180.0) / MY_PI, arrs[9][j],
        ];
        for (const v of vals) line += `${fmtCoeff(v)} `;
      }
      out.push(line);
    }
    return out;
  }

  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const x = bc.s.x;
    const f = bc.f;
    const vb12 = [0, 0, 0], vb23 = [0, 0, 0], vb34 = [0, 0, 0];
    const n123 = [0, 0, 0], n234 = [0, 0, 0];
    const proj12on23 = [0, 0, 0], proj34on23 = [0, 0, 0], perp12on23 = [0, 0, 0], perp34on23 = [0, 0, 0];
    const f1 = [0, 0, 0], f2 = [0, 0, 0], f3 = [0, 0, 0], f4 = [0, 0, 0];
    for (let n = 0; n < D.n; n++) {
      const i1 = atomIndex(bc, D.atoms[4 * n], 'dihedral');
      const i2 = atomIndex(bc, D.atoms[4 * n + 1], 'dihedral');
      const i3 = atomIndex(bc, D.atoms[4 * n + 2], 'dihedral');
      const i4 = atomIndex(bc, D.atoms[4 * n + 3], 'dihedral');
      const type = D.type[n];
      const phi = this.phi(bc, x, i1, i2, i3, i4, vb12, vb23, vb34, n123, n234);
      const dphi_dx1 = [0, 0, 0], dphi_dx2 = [0, 0, 0], dphi_dx3 = [0, 0, 0], dphi_dx4 = [0, 0, 0];
      const dot123 = dot3(vb12, vb23);
      const dot234 = dot3(vb23, vb34);
      const L23sqr = dot3(vb23, vb23);
      const L23 = Math.sqrt(L23sqr);
      let inv_L23sqr = 0.0;
      let inv_L23 = 0.0;
      if (L23sqr !== 0.0) {
        inv_L23sqr = 1.0 / L23sqr;
        inv_L23 = 1.0 / L23;
      }
      const neg_inv_L23 = -inv_L23;
      const dot123_over_L23sqr = dot123 * inv_L23sqr;
      const dot234_over_L23sqr = dot234 * inv_L23sqr;
      for (let d = 0; d < 3; d++) {
        proj12on23[d] = vb23[d] * dot123_over_L23sqr;
        proj34on23[d] = vb23[d] * dot234_over_L23sqr;
        perp12on23[d] = vb12[d] - proj12on23[d];
        perp34on23[d] = vb34[d] - proj34on23[d];
      }
      const perp12on23_len = Math.sqrt(dot3(perp12on23, perp12on23));
      const perp34on23_len = Math.sqrt(dot3(perp34on23, perp34on23));
      let inv_perp12on23 = 0.0;
      if (perp12on23_len !== 0.0) inv_perp12on23 = 1.0 / perp12on23_len;
      let inv_perp34on23 = 0.0;
      if (perp34on23_len !== 0.0) inv_perp34on23 = 1.0 / perp34on23_len;
      for (let d = 0; d < 3; d++) {
        dphi_dx1[d] = n123[d] * inv_perp12on23;
        dphi_dx4[d] = n234[d] * inv_perp34on23;
      }
      const proj12on23_len = dot123 * inv_L23;
      const proj34on23_len = dot234 * inv_L23;
      const dphi123_dx2_coef = neg_inv_L23 * (L23 + proj12on23_len);
      const dphi234_dx2_coef = inv_L23 * proj34on23_len;
      const dphi234_dx3_coef = neg_inv_L23 * (L23 + proj34on23_len);
      const dphi123_dx3_coef = inv_L23 * proj12on23_len;
      for (let d = 0; d < 3; d++) {
        dphi_dx2[d] = dphi123_dx2_coef * dphi_dx1[d] + dphi234_dx2_coef * dphi_dx4[d];
        dphi_dx3[d] = dphi123_dx3_coef * dphi_dx1[d] + dphi234_dx3_coef * dphi_dx4[d];
      }
      const dth1_dx1 = [0, 0, 0], dth1_dx2 = [0, 0, 0], dth1_dx3 = [0, 0, 0];
      const dth2_dx2 = [0, 0, 0], dth2_dx3 = [0, 0, 0], dth2_dx4 = [0, 0, 0];
      const L12sqr = dot3(vb12, vb12);
      const L12 = Math.sqrt(L12sqr);
      const L34sqr = dot3(vb34, vb34);
      const L34 = Math.sqrt(L34sqr);
      let inv_L12sqr = 0.0;
      let inv_L12 = 0.0;
      let inv_L34sqr = 0.0;
      let inv_L34 = 0.0;
      if (L12sqr !== 0.0) {
        inv_L12sqr = 1.0 / L12sqr;
        inv_L12 = 1.0 / L12;
      }
      if (L34sqr !== 0.0) {
        inv_L34sqr = 1.0 / L34sqr;
        inv_L34 = 1.0 / L34;
      }
      const proj23on12 = [0, 0, 0], perp23on12 = [0, 0, 0], proj23on34 = [0, 0, 0], perp23on34 = [0, 0, 0];
      const dot123_over_L12sqr = dot123 * inv_L12sqr;
      const dot234_over_L34sqr = dot234 * inv_L34sqr;
      for (let d = 0; d < 3; d++) {
        proj23on12[d] = vb12[d] * dot123_over_L12sqr;
        proj23on34[d] = vb34[d] * dot234_over_L34sqr;
        perp23on12[d] = vb23[d] - proj23on12[d];
        perp23on34[d] = vb23[d] - proj23on34[d];
      }
      const perp23on12_len = Math.sqrt(dot3(perp23on12, perp23on12));
      const perp23on34_len = Math.sqrt(dot3(perp23on34, perp23on34));
      let inv_perp23on12 = 0.0;
      if (perp23on12_len !== 0.0) inv_perp23on12 = 1.0 / perp23on12_len;
      let inv_perp23on34 = 0.0;
      if (perp23on34_len !== 0.0) inv_perp23on34 = 1.0 / perp23on34_len;
      const coeff_dth1_dx1 = -inv_perp23on12 * inv_L12;
      const coeff_dth1_dx3 = inv_perp12on23 * inv_L23;
      const coeff_dth2_dx2 = -inv_perp34on23 * inv_L23;
      const coeff_dth2_dx4 = inv_perp23on34 * inv_L34;
      for (let d = 0; d < 3; d++) {
        dth1_dx1[d] = perp23on12[d] * coeff_dth1_dx1;
        dth1_dx3[d] = perp12on23[d] * coeff_dth1_dx3;
        dth1_dx2[d] = -(dth1_dx1[d] + dth1_dx3[d]);
        dth2_dx2[d] = perp34on23[d] * coeff_dth2_dx2;
        dth2_dx4[d] = perp23on34[d] * coeff_dth2_dx4;
        dth2_dx3[d] = -(dth2_dx2[d] + dth2_dx4[d]);
      }
      let ct1 = -dot123 * inv_L12 * inv_L23;
      if (ct1 < -1.0) ct1 = -1.0;
      else if (ct1 > 1.0) ct1 = 1.0;
      const theta1 = libmAcos(ct1);
      let ct2 = -dot234 * inv_L23 * inv_L34;
      if (ct2 < -1.0) ct2 = -1.0;
      else if (ct2 > 1.0) ct2 = 1.0;
      const theta2 = libmAcos(ct2);
      const g = this.generalized(type, phi, theta1, theta2);
      const edihedral = g.energy;
      for (let d = 0; d < 3; d++) {
        f1[d] = g.dphi * dphi_dx1[d] + g.dth1 * dth1_dx1[d];
        f2[d] = g.dphi * dphi_dx2[d] + g.dth1 * dth1_dx2[d] + g.dth2 * dth2_dx2[d];
        f3[d] = g.dphi * dphi_dx3[d] + g.dth1 * dth1_dx3[d] + g.dth2 * dth2_dx3[d];
        f4[d] = g.dphi * dphi_dx4[d] + g.dth2 * dth2_dx4[d];
      }
      for (let d = 0; d < 3; d++) {
        f[3 * i1 + d] += f1[d];
        f[3 * i2 + d] += f2[d];
        f[3 * i3 + d] += f3[d];
        f[3 * i4 + d] += f4[d];
      }
      evTallyFour(bc, [i1, i2, i3, i4], 'dihed', edihedral, f1, f3, f4, vb12, vb23, vb34);
    }
  }

  /**
   * Phi(x1, x2, x3, x4) of dihedral_spherical.cpp: the bond vectors vb12 = x2 - x1, vb23 = x3 - x2, vb34 = x4 - x3,
   * each taken with domain->minimum_image, the normals n123 = vb23 x vb12 and n234 = vb23 x vb34 (unit length), and
   * phi = acos(-n123 . n234), taken as 2 pi - phi when n123 . vb34 > 0.
   */
  private phi(
    bc: BondedCompute, x: Float64Array, i1: number, i2: number, i3: number, i4: number,
    vb12: number[], vb23: number[], vb34: number[], n123: number[], n234: number[],
  ): number {
    for (let d = 0; d < 3; d++) {
      vb12[d] = x[3 * i2 + d] - x[3 * i1 + d];
      vb23[d] = x[3 * i3 + d] - x[3 * i2 + d];
      vb34[d] = x[3 * i4 + d] - x[3 * i3 + d];
    }
    bc.geom.minimumImage(vb12);
    bc.geom.minimumImage(vb23);
    bc.geom.minimumImage(vb34);
    cross3(vb23, vb12, n123);
    cross3(vb23, vb34, n234);
    norm3safe(n123);
    norm3safe(n234);
    let cos_phi = -dot3(n123, n234);
    if (cos_phi > 1.0) cos_phi = 1.0;
    else if (cos_phi < -1.0) cos_phi = -1.0;
    let phi = libmAcos(cos_phi);
    if (dot3(n123, vb34) > 0.0) {
      phi = -phi;
      phi += MY_2PI;
    }
    return phi;
  }

  /** CalcGeneralizedForces: the energy and the three partial derivatives (dE/dphi, dE/dtheta1, dE/dtheta2). */
  private generalized(type: number, phi: number, theta1: number, theta2: number): { energy: number; dphi: number; dth1: number; dth2: number } {
    let energy = 0.0;
    let m_du_dphi = 0.0;
    let m_du_dth1 = 0.0;
    let m_du_dth2 = 0.0;
    const arrs = this.terms[type]!;
    const [C, phi_mult, phi_shift, phi_offset, theta1_mult, theta1_shift, theta1_offset, theta2_mult, theta2_shift, theta2_offset] = arrs;
    for (let j = 0; j < this.nterms[type]; j++) {
      let cp = 1.0;
      let sp = 0.0;
      if (phi_mult[j] !== 0.0) {
        const p = phi_mult[j] * (phi - phi_shift[j]);
        cp = cos(p);
        sp = sin(p);
      }
      let ct1 = 1.0;
      let st1 = 0.0;
      if (theta1_mult[j] !== 0.0) {
        const t1 = theta1_mult[j] * (theta1 - theta1_shift[j]);
        ct1 = cos(t1);
        st1 = sin(t1);
      }
      let ct2 = 1.0;
      let st2 = 0.0;
      if (theta2_mult[j] !== 0.0) {
        const t2 = theta2_mult[j] * (theta2 - theta2_shift[j]);
        ct2 = cos(t2);
        st2 = sin(t2);
      }
      energy += C[j] * (phi_offset[j] - cp) * (theta1_offset[j] - ct1) * (theta2_offset[j] - ct2);
      m_du_dphi += -C[j] * sp * phi_mult[j] * (theta1_offset[j] - ct1) * (theta2_offset[j] - ct2);
      m_du_dth1 += -C[j] * (phi_offset[j] - cp) * st1 * theta1_mult[j] * (theta2_offset[j] - ct2);
      m_du_dth2 += -C[j] * (phi_offset[j] - cp) * (theta1_offset[j] - ct1) * st2 * theta2_mult[j];
    }
    return { energy, dphi: m_du_dphi, dth1: m_du_dth1, dth2: m_du_dth2 };
  }
}

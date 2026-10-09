import { SimpleBonded, atomIndex, dihedralGeometry, delta, TypeParams } from '../bonded_util';
import { fmtCoeff, parseNum } from '../util';
import { bondedVirial, StyleError, type BondedCompute, type StyleContext } from '../types';

/*
 * dihedral_style class2 (docs.lammps.org/dihedral_class2.html). COMPASS class-2
 * torsion: an ordinary 3-term cosine plus five cross terms, all summed over the
 * quadruplet i-j-k-l.
 *
 * Docs.lammps.org/dihedral_class2.html, "The *class2* dihedral style uses the potential":
 *   E      = E_d + E_mbt + E_ebt + E_at + E_aat + E_bb13
 *   E_d    = sum_{n=1}^{3} K_n [ 1 - cos (n phi - phi_n) ]
 *   E_mbt  = (r_jk - r_2) [ A_1 cos(phi) + A_2 cos(2 phi) + A_3 cos(3 phi) ]
 *   E_ebt  = (r_ij - r_1) [ B_1 cos(phi) + B_2 cos(2 phi) + B_3 cos(3 phi) ] +
 *            (r_kl - r_3) [ C_1 cos(phi) + C_2 cos(2 phi) + C_3 cos(3 phi) ]
 *   E_at   = (theta_ijk - theta_1) [ D_1 cos(phi) + D_2 cos(2 phi) + D_3 cos(3 phi) ] +
 *            (theta_jkl - theta_2) [ E_1 cos(phi) + E_2 cos(2 phi) + E_3 cos(3 phi) ]
 *   E_aat  = M (theta_ijk - theta_1) (theta_jkl - theta_2) cos(phi)
 *   E_bb13 = N (r_ij - r_1) (r_kl - r_3)
 * with the coefficient lists of the doc: K_1 phi_1 K_2 phi_2 K_3 phi_3 (energy, degrees); mbt A_1
 * A_2 A_3 r_2; ebt B_1 B_2 B_3 C_1 C_2 C_3 r_1 r_3; at D_1 D_2 D_3 E_1 E_2 E_3 theta_1 theta_2;
 * aat M theta_1 theta_2; bb13 N r_1 r_3. theta_1 and theta_2 "are specified in degrees, but
 * LAMMPS converts them to radians internally". In a data file the keyword is left out and the
 * coefficients appear under the matching section heading (data.ts CLASS2_SECTIONS turns them
 * into dihedral_coeff ID <kw> ...).
 *
 * Measured with native LAMMPS (black box, 4-atom chains and the three-chain oracle cases,
 * scratch plans/scratch/w35c2dih/, 2026-10-09):
 *  - the six formulas above reproduce the native edihed and per-atom forces exactly (the oracle
 *    cases w35c2dih, w35c2dih_coeff and w35c2dih_phi agree at rel 1e-9). A first single-chain probe
 *    looked like an extra factor 1/4, but that is only the default `thermo_modify norm yes`: native
 *    (like this engine) reports thermo quantities per atom, so a 4-atom system prints total/4.
 *  - every one of the six coefficient sets must be set for every dihedral type; a missing line makes
 *    init abort with All dihedral coeffs are not set. (input script and data file both measured),
 *    so init rejects any type missing one of the six.
 *  - write_data writes six sections, in the order Dihedral Coeffs, AngleAngleTorsion Coeffs,
 *    EndBondTorsion Coeffs, MiddleBondTorsion Coeffs, BondBond13 Coeffs, AngleTorsion Coeffs, each
 *    line ID <coeffs> with the degrees and the equilibrium distances unchanged (dataCoeffs and
 *    dataCrossSections); read_data reads the cross terms back from those sections.
 */

const DEG2RAD = Math.PI / 180;

/** The five cross-term keywords, their coefficient names in order, and which need a degree conversion. */
const CROSS: Record<string, { names: readonly string[]; deg: readonly number[] }> = {
  mbt: { names: ['A1', 'A2', 'A3', 'r2'], deg: [] },
  ebt: { names: ['B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'r1', 'r3'], deg: [] },
  at: { names: ['D1', 'D2', 'D3', 'E1', 'E2', 'E3', 'theta1', 'theta2'], deg: [6, 7] },
  aat: { names: ['M', 'theta1', 'theta2'], deg: [1, 2] },
  bb13: { names: ['N', 'r1', 'r3'], deg: [] },
};

export class DihedralClass2 extends SimpleBonded {
  readonly name = 'class2';
  readonly kind = 'dihedral' as const;
  readonly paramNames = ['K1', 'phi1', 'K2', 'phi2', 'K3', 'phi3'];

  private main!: TypeParams;
  private mbt!: TypeParams;
  private ebt!: TypeParams;
  private at!: TypeParams;
  private aat!: TypeParams;
  private bb13!: TypeParams;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.main = new TypeParams(ntypes, ['K1', 'phi1', 'K2', 'phi2', 'K3', 'phi3']);
    this.mbt = new TypeParams(ntypes, CROSS.mbt.names);
    this.ebt = new TypeParams(ntypes, CROSS.ebt.names);
    this.at = new TypeParams(ntypes, CROSS.at.names);
    this.aat = new TypeParams(ntypes, CROSS.aat.names);
    this.bb13 = new TypeParams(ntypes, CROSS.bb13.names);
  }

  private group(kw: string): TypeParams {
    return { mbt: this.mbt, ebt: this.ebt, at: this.at, aat: this.aat, bb13: this.bb13 }[kw]!;
  }

  override coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError('usage: dihedral_coeff N coeffs');
    const word = args[0];
    const rest = args.slice(1);
    if (!rest.length) throw new StyleError('dihedral_coeff class2 needs coefficients after the type');
    const kw = rest[0];
    if (kw in CROSS) {
      const g = this.group(kw);
      const vals = rest.slice(1);
      if (vals.length !== g.names.length) {
        throw new StyleError(`dihedral_coeff class2 ${kw} needs ${g.names.length} coefficients: ${g.names.join(' ')}`);
      }
      const nums = vals.map((w, k) => parseNum(w, g.names[k]));
      for (const k of CROSS[kw].deg) nums[k] *= DEG2RAD;
      g.setRange(word, nums);
      return;
    }
    if (/[a-zA-Z]/.test(kw)) {
      throw new StyleError(`unknown dihedral_coeff class2 keyword '${kw}' (expected a numeric line or mbt, ebt, at, aat, bb13)`);
    }
    if (rest.length !== 6) throw new StyleError('dihedral_coeff class2 needs K1 phi1 K2 phi2 K3 phi3');
    const nums = rest.map((w, k) => parseNum(w, this.paramNames[k]));
    nums[1] *= DEG2RAD; nums[3] *= DEG2RAD; nums[5] *= DEG2RAD;
    this.main.setRange(word, nums);
  }

  override init(_ctx?: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.main.set[t] || !this.mbt.set[t] || !this.ebt.set[t] || !this.at.set[t] || !this.aat.set[t] || !this.bb13.set[t]) {
        throw new StyleError(`all dihedral coeffs are not set (type ${t})`);
      }
    }
  }

  // write_data: native prints the six coefficient lines as read (degrees, unscaled distances).
  override dataCoeffs(): string[] {
    const K = this.main.p('K1'), P1 = this.main.p('phi1'), K2 = this.main.p('K2'), P2 = this.main.p('phi2');
    const K3 = this.main.p('K3'), P3 = this.main.p('phi3');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff(P1[t] / DEG2RAD)} ${fmtCoeff(K2[t])} ${fmtCoeff(P2[t] / DEG2RAD)} ${fmtCoeff(K3[t])} ${fmtCoeff(P3[t] / DEG2RAD)}`);
    }
    return out;
  }

  /** write_data cross terms, in native's order (see the header and output/data.ts); angles in degrees. */
  override dataCrossSections(): { title: string; lines: string[] }[] {
    const sec = (title: string, kw: string) => {
      const g = this.group(kw);
      const lines: string[] = [];
      for (let t = 1; t <= this.ntypes; t++) {
        lines.push(`${t} ${CROSS[kw].names.map((nm, k) => fmtCoeff(CROSS[kw].deg.includes(k) ? g.p(nm)[t] / DEG2RAD : g.p(nm)[t])).join(' ')}`);
      }
      return { title, lines };
    };
    return [
      sec('AngleAngleTorsion Coeffs', 'aat'), sec('EndBondTorsion Coeffs', 'ebt'), sec('MiddleBondTorsion Coeffs', 'mbt'),
      sec('BondBond13 Coeffs', 'bb13'), sec('AngleTorsion Coeffs', 'at'),
    ];
  }

  compute(bc: BondedCompute): void {
    const D = bc.s.topo.dihedrals;
    const K = [this.main.p('K1'), this.main.p('K2'), this.main.p('K3')];
    const P = [this.main.p('phi1'), this.main.p('phi2'), this.main.p('phi3')];
    const A1 = this.mbt.p('A1'), A2 = this.mbt.p('A2'), A3 = this.mbt.p('A3'), R2 = this.mbt.p('r2');
    const B1 = this.ebt.p('B1'), B2 = this.ebt.p('B2'), B3 = this.ebt.p('B3');
    const C1 = this.ebt.p('C1'), C2 = this.ebt.p('C2'), C3 = this.ebt.p('C3');
    const ER1 = this.ebt.p('r1'), ER3 = this.ebt.p('r3');
    const D1 = this.at.p('D1'), D2 = this.at.p('D2'), D3 = this.at.p('D3');
    const E1 = this.at.p('E1'), E2 = this.at.p('E2'), E3 = this.at.p('E3');
    const AT1 = this.at.p('theta1'), AT2 = this.at.p('theta2');
    const M = this.aat.p('M'), AA1 = this.aat.p('theta1'), AA2 = this.aat.p('theta2');
    const N = this.bb13.p('N'), BR1 = this.bb13.p('r1'), BR3 = this.bb13.p('r3');

    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    const fk = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < D.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, D.atoms[4 * q + w], 'dihedral'));
      const [i, j, k, l] = atoms;
      const t = D.type[q];
      const phi = dihedralGeometry(bc, i, j, k, l, grad, rel);
      const c1 = Math.cos(phi), c2 = Math.cos(2 * phi), c3 = Math.cos(3 * phi);
      const s1 = Math.sin(phi), s2 = Math.sin(2 * phi), s3 = Math.sin(3 * phi);

      // geometry of the two end angles and the three bonds
      const b1 = [rel[3], rel[4], rel[5]], b2 = [rel[6] - rel[3], rel[7] - rel[4], rel[8] - rel[5]];
      const b3 = [rel[9] - rel[6], rel[10] - rel[7], rel[11] - rel[8]];
      const rIJ = Math.hypot(b1[0], b1[1], b1[2]);
      const rJK = Math.hypot(b2[0], b2[1], b2[2]);
      const rKL = Math.hypot(b3[0], b3[1], b3[2]);

      // E_d and dE_d/dphi
      let ed = 0, dEdphi = 0;
      for (let n = 0; n < 3; n++) {
        const arg = (n + 1) * phi - P[n][t];
        ed += K[n][t] * (1 - Math.cos(arg));
        dEdphi += K[n][t] * (n + 1) * Math.sin(arg);
      }

      // E_mbt
      const acos = A1[t] * c1 + A2[t] * c2 + A3[t] * c3;
      const embt = (rJK - R2[t]) * acos;
      dEdphi += (rJK - R2[t]) * (-A1[t] * s1 - 2 * A2[t] * s2 - 3 * A3[t] * s3);
      const dEdrJK = acos;

      // E_ebt
      const bcos = B1[t] * c1 + B2[t] * c2 + B3[t] * c3;
      const ccos = C1[t] * c1 + C2[t] * c2 + C3[t] * c3;
      const eebt = (rIJ - ER1[t]) * bcos + (rKL - ER3[t]) * ccos;
      dEdphi += (rIJ - ER1[t]) * (-B1[t] * s1 - 2 * B2[t] * s2 - 3 * B3[t] * s3);
      dEdphi += (rKL - ER3[t]) * (-C1[t] * s1 - 2 * C2[t] * s2 - 3 * C3[t] * s3);
      const dEdrIJ = bcos;
      const dEdrKL = ccos;

      // the two angle values and the angle gradients (atoms i-j-k and j-k-l)
      const d1 = [0, 0, 0], d2 = [0, 0, 0], d3 = [0, 0, 0], d4 = [0, 0, 0];
      delta(bc, j, i, d1); delta(bc, j, k, d2);   // r_i - r_j, r_k - r_j
      delta(bc, k, j, d3); delta(bc, k, l, d4);   // r_j - r_k, r_l - r_k
      const thIJK = angleOf(d1, d2), thJKL = angleOf(d3, d4);
      const g1 = angleGrad(d1, d2, thIJK);        // [g_i, g_j, g_k]
      const g2 = angleGrad(d3, d4, thJKL);        // [g_j, g_k, g_l]

      // E_at
      const dcos = D1[t] * c1 + D2[t] * c2 + D3[t] * c3;
      const ecos = E1[t] * c1 + E2[t] * c2 + E3[t] * c3;
      const eat = (thIJK - AT1[t]) * dcos + (thJKL - AT2[t]) * ecos;
      dEdphi += (thIJK - AT1[t]) * (-D1[t] * s1 - 2 * D2[t] * s2 - 3 * D3[t] * s3);
      dEdphi += (thJKL - AT2[t]) * (-E1[t] * s1 - 2 * E2[t] * s2 - 3 * E3[t] * s3);
      const dEdThIJK = dcos;
      const dEdThJKL = ecos;

      // E_aat
      const eaat = M[t] * (thIJK - AA1[t]) * (thJKL - AA2[t]) * c1;
      dEdphi += -M[t] * (thIJK - AA1[t]) * (thJKL - AA2[t]) * s1;
      const dEdThIJK2 = M[t] * (thJKL - AA2[t]) * c1;
      const dEdThJKL2 = M[t] * (thIJK - AA1[t]) * c1;

      // E_bb13
      const ebb13 = N[t] * (rIJ - BR1[t]) * (rKL - BR3[t]);
      const dEdrIJ2 = N[t] * (rKL - BR3[t]);
      const dEdrKL2 = N[t] * (rIJ - BR1[t]);

      // forces
      fk.fill(0);
      for (let a = 0; a < 4; a++) for (let d = 0; d < 3; d++) fk[3 * a + d] = -dEdphi * grad[3 * a + d];
      // bond lengths
      addBond(fk, 0, 1, b1, rIJ, dEdrIJ);
      addBond(fk, 1, 2, b2, rJK, dEdrJK);
      addBond(fk, 2, 3, b3, rKL, dEdrKL);
      addBond(fk, 0, 1, b1, rIJ, dEdrIJ2);
      addBond(fk, 2, 3, b3, rKL, dEdrKL2);
      // angles
      addAngle(fk, [0, 1, 2], g1, dEdThIJK + dEdThIJK2);
      addAngle(fk, [1, 2, 3], g2, dEdThJKL + dEdThJKL2);

      const eq = ed + embt + eebt + eat + eaat + ebb13;
      e += eq;
      for (let a = 0; a < 4; a++) for (let d = 0; d < 3; d++) bc.f[3 * atoms[a] + d] += fk[3 * a + d];
      bondedVirial(bc, atoms, rel, fk, eq);
    }
    bc.acc.edihed += e;
  }
}

/** Angle between vectors A and B (both 3-vectors). */
const angleOf = (a: number[], b: number[]): number => {
  let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(a[0], a[1], a[2]) * Math.hypot(b[0], b[1], b[2]));
  if (c > 1) c = 1;
  if (c < -1) c = -1;
  return Math.acos(c);
};

/**
 * Gradients of the angle at the vertex of A = r_i - r_j and B = r_k - r_j, dtheta/dr_i,
 * dtheta/dr_j and dtheta/dr_k (A and B are minimum-image displacements). Returns [g_i, g_j, g_k].
 */
const angleGrad = (A: number[], B: number[], theta: number): number[][] => {
  const a = Math.hypot(A[0], A[1], A[2]), b = Math.hypot(B[0], B[1], B[2]);
  let s = Math.sin(theta);
  if (s < 1e-12) s = 1e-12;
  const c = Math.cos(theta);
  const ab = a * b;
  const gi = [0, 0, 0], gk = [0, 0, 0], gj = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    gi[d] = ((c / (a * a)) * A[d] - B[d] / ab) / s;
    gk[d] = ((c / (b * b)) * B[d] - A[d] / ab) / s;
    gj[d] = -(gi[d] + gk[d]);
  }
  return [gi, gj, gk];
};

/** Adds the force from a bond length r of vector b = r_b - r_a on atoms `aIndex`/`bIndex`. */
const addBond = (fk: number[], aIndex: number, bIndex: number, b: number[], r: number, dEdr: number): void => {
  for (let d = 0; d < 3; d++) {
    const f = -dEdr * (b[d] / r);   // force on b
    fk[3 * bIndex + d] += f;
    fk[3 * aIndex + d] -= f;
  }
};

/** Adds the force from an angle with gradients g = [g_i, g_j, g_k] on the triple `idx`. */
const addAngle = (fk: number[], idx: number[], g: number[][], dEdtheta: number): void => {
  for (let m = 0; m < 3; m++) {
    for (let d = 0; d < 3; d++) fk[3 * idx[m] + d] += -dEdtheta * g[m][d];
  }
};

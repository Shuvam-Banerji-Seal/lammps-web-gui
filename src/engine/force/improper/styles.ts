import { SimpleBonded, atomIndex, delta, dihedralGeometry, applyDihedral } from '../bonded_util';
import { fmtCoeff, parseNum, parseInt_ } from '../util';
import { StyleError, bondedVirial, typeBounds, type BondedCompute, type StyleContext } from '../types';

/*
 * Improper styles (wave 1). Every class below implements exactly the formula
 * and coefficient list of its docs.lammps.org page (quoted in its header).
 *
 * Two families:
 * - cvff and cossq use the angle between the plane of I,J,K and the plane of
 *   J,K,L (an "effectively dihedral" angle) and share dihedralGeometry /
 *   applyDihedral from bonded_util.ts.
 * - umbrella, fourier and distance are functions of the out-of-plane offset
 *   of one atom w.r.t. a plane; they build their own geometry with analytic
 *   gradients (umbrellaC / distanceC below), checked against central finite
 *   differences in tests/engineImproperStyles.test.ts.
 */

const DEG2RAD = Math.PI / 180;

/**
 * Out-of-plane coordinate for the umbrella angle: with the centre atom
 * atoms[0], plane atoms atoms[1], atoms[2] and apex atoms[3],
 *   v = r_apex - r_centre,  n = (r_j - r_i) x (r_k - r_i),
 *   sin(omega) = |v.n| / (|v| |n|),  cos(omega) = sqrt(1 - c)
 * with c = (v.n)^2 / (v^2 n^2) in [0, 1] (omega is the angle between an axis
 * and a plane, so omega in [0, 90 degrees]).
 * Fills grad[3*p + d] with the analytic d(c)/d(coordinate d of atoms[p]).
 * Derivation: c = u^2 with u = a / sqrt(v^2 n^2), a = v.n; for each atom p,
 *   du/dp = w da/dp - (u/2) (dv^2/dp / v^2 + dn^2/dp / n^2),  w = 1/sqrt(v^2 n^2)
 * where da/dx are the gradients of the scalar triple product
 * a = (r_l - r_i) . [(r_j - r_i) x (r_k - r_i)]:
 *   da/dl = n;  da/dj = b2 x v;  da/dk = v x b1;  da/di = -(n + b2 x v + v x b1)
 * (sum zero: translation invariant) and dn^2/dx from n = b1 x b2:
 *   dn^2/di = 2 n x (b2 - b1);  dn^2/dj = 2 b2 x n;  dn^2/dk = 2 n x b1;  dn^2/dl = 0.
 * Verified against central finite differences in tests/engineImproperStyles.test.ts.
 */
const umbrellaC = (bc: BondedCompute, atoms: readonly number[], grad: number[]): number => {
  const b1 = [0, 0, 0], b2 = [0, 0, 0], v = [0, 0, 0];
  delta(bc, atoms[0], atoms[1], b1);
  delta(bc, atoms[0], atoms[2], b2);
  delta(bc, atoms[0], atoms[3], v);
  const n = [
    b1[1] * b2[2] - b1[2] * b2[1],
    b1[2] * b2[0] - b1[0] * b2[2],
    b1[0] * b2[1] - b1[1] * b2[0],
  ];
  const a = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
  const v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const nv2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  const den = v2 * nv2;
  if (den <= 0) {
    grad.fill(0);
    return 0;
  }
  const w = 1 / Math.sqrt(den);
  const u = a * w;
  const gjx = b2[1] * v[2] - b2[2] * v[1], gjy = b2[2] * v[0] - b2[0] * v[2], gjz = b2[0] * v[1] - b2[1] * v[0];
  const gkx = v[1] * b1[2] - v[2] * b1[1], gky = v[2] * b1[0] - v[0] * b1[2], gkz = v[0] * b1[1] - v[1] * b1[0];
  // ga: d a/d x for atoms i, j, k, l
  const ga = [
    [-(n[0] + gjx + gkx), -(n[1] + gjy + gky), -(n[2] + gjz + gkz)],
    [gjx, gjy, gjz],
    [gkx, gky, gkz],
    [n[0], n[1], n[2]],
  ];
  // gn: d n^2/d x for atoms i, j, k, l
  const db = [b2[0] - b1[0], b2[1] - b1[1], b2[2] - b1[2]];
  const gn = [
    [2 * (n[1] * db[2] - n[2] * db[1]), 2 * (n[2] * db[0] - n[0] * db[2]), 2 * (n[0] * db[1] - n[1] * db[0])],
    [2 * (b2[1] * n[2] - b2[2] * n[1]), 2 * (b2[2] * n[0] - b2[0] * n[2]), 2 * (b2[0] * n[1] - b2[1] * n[0])],
    [2 * (n[1] * b1[2] - n[2] * b1[1]), 2 * (n[2] * b1[0] - n[0] * b1[2]), 2 * (n[0] * b1[1] - n[1] * b1[0])],
    [0, 0, 0],
  ];
  // gv2: d v^2/d x for atoms i, j, k, l
  const gv2 = [[-2 * v[0], -2 * v[1], -2 * v[2]], [0, 0, 0], [0, 0, 0], [2 * v[0], 2 * v[1], 2 * v[2]]];
  for (let p = 0; p < 4; p++) {
    for (let d = 0; d < 3; d++) {
      const du = w * ga[p][d] - (u / 2) * (gv2[p][d] / v2 + gn[p][d] / nv2);
      grad[3 * p + d] = 2 * u * du;
    }
  }
  return u * u;
};

/**
 * Out-of-plane coordinate for improper_style distance: with the central atom
 * atoms[0] and plane atoms atoms[1..3], d is the distance between the central
 * atom and the plane, d^2 = a^2 / n^2 with n = (r_k - r_j) x (r_l - r_j) and
 * a = (r_i - r_j) . n. Fills grad[3*p + d] with d(c)/d(coordinate d of
 * atoms[p]) for c = d^2, from
 *   da/di = n;  da/dj = -n + (r_i - r_j) x (r_l - r_j) + (r_k - r_j) x (r_i - r_j);
 *   da/dk = (r_l - r_j) x (r_i - r_j);  da/dl = (r_i - r_j) x (r_k - r_j)
 * (sum zero) and
 *   dn^2/dj = 2 n x (p2 - p1);  dn^2/dk = 2 p2 x n;  dn^2/dl = 2 n x p1;  dn^2/di = 0.
 * Verified against central finite differences in tests/engineImproperStyles.test.ts.
 */
const distanceC = (bc: BondedCompute, atoms: readonly number[], grad: number[]): number => {
  const p1 = [0, 0, 0], p2 = [0, 0, 0], w = [0, 0, 0];
  delta(bc, atoms[1], atoms[2], p1);
  delta(bc, atoms[1], atoms[3], p2);
  delta(bc, atoms[1], atoms[0], w);
  const n = [
    p1[1] * p2[2] - p1[2] * p2[1],
    p1[2] * p2[0] - p1[0] * p2[2],
    p1[0] * p2[1] - p1[1] * p2[0],
  ];
  const a = w[0] * n[0] + w[1] * n[1] + w[2] * n[2];
  const nv2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  if (nv2 <= 0) {
    grad.fill(0);
    return 0;
  }
  const wpx2 = [
    w[1] * p2[2] - w[2] * p2[1], w[2] * p2[0] - w[0] * p2[2], w[0] * p2[1] - w[1] * p2[0],
  ];
  const p1xw = [
    p1[1] * w[2] - p1[2] * w[1], p1[2] * w[0] - p1[0] * w[2], p1[0] * w[1] - p1[1] * w[0],
  ];
  const p2xw = [
    p2[1] * w[2] - p2[2] * w[1], p2[2] * w[0] - p2[0] * w[2], p2[0] * w[1] - p2[1] * w[0],
  ];
  const wxp1 = [
    w[1] * p1[2] - w[2] * p1[1], w[2] * p1[0] - w[0] * p1[2], w[0] * p1[1] - w[1] * p1[0],
  ];
  // ga: d a/d x for atoms i, j, k, l
  const ga = [
    [n[0], n[1], n[2]],
    [-n[0] + wpx2[0] + p1xw[0], -n[1] + wpx2[1] + p1xw[1], -n[2] + wpx2[2] + p1xw[2]],
    [p2xw[0], p2xw[1], p2xw[2]],
    [wxp1[0], wxp1[1], wxp1[2]],
  ];
  const dp = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
  const gn = [
    [0, 0, 0],
    [2 * (n[1] * dp[2] - n[2] * dp[1]), 2 * (n[2] * dp[0] - n[0] * dp[2]), 2 * (n[0] * dp[1] - n[1] * dp[0])],
    [2 * (p2[1] * n[2] - p2[2] * n[1]), 2 * (p2[2] * n[0] - p2[0] * n[2]), 2 * (p2[0] * n[1] - p2[1] * n[0])],
    [2 * (n[1] * p1[2] - n[2] * p1[1]), 2 * (n[2] * p1[0] - n[0] * p1[2]), 2 * (n[0] * p1[1] - n[1] * p1[0])],
  ];
  const inv2 = 1 / (nv2 * nv2);
  for (let p = 0; p < 4; p++) {
    for (let d = 0; d < 3; d++) {
      grad[3 * p + d] = (2 * a * ga[p][d] * nv2 - a * a * gn[p][d]) * inv2;
    }
  }
  return (a * a) / nv2;
};

/** Adds f = -dE/dc * grad(c) to bc.f and tallies virial / per-atom energy. */
const applyGradC = (bc: BondedCompute, atoms: readonly number[], dEdc: number, grad: readonly number[], e: number): void => {
  const f = bc.f;
  const fk: number[] = new Array(atoms.length * 3);
  for (let p = 0; p < atoms.length; p++) {
    for (let d = 0; d < 3; d++) {
      const val = -dEdc * grad[3 * p + d];
      fk[3 * p + d] = val;
      f[3 * atoms[p] + d] += val;
    }
  }
  const rel: number[] = new Array(atoms.length * 3);
  for (let p = 0; p < atoms.length; p++) delta(bc, atoms[0], atoms[p], rel, 3 * p);
  bondedVirial(bc, atoms, rel, fk, e);
};

/*
 * improper_style cvff — docs.lammps.org/improper_cvff.html:
 *
 *    E = K [1 + d  \cos (n \phi) ]
 *
 * "where phi is the improper dihedral angle." "If the 4 atoms in an improper
 * quadruplet (listed in the data file read by the read_data command) are
 * ordered I,J,K,L then the improper dihedral angle is between the plane of
 * I,J,K and the plane of J,K,L. Note that because this is effectively a
 * dihedral angle, the formula for this improper style is the same as for
 * dihedral_style harmonic." phi is therefore the signed dihedral of I,J,K,L
 * (trans = 180 degrees); E is even in phi, so the forces are unambiguous.
 * * :math:`K` (energy)
 * * :math:`d` (+1 or -1)
 * * :math:`n` (0,1,2,3,4,6)
 */
export class ImproperCvff extends SimpleBonded {
  readonly name = 'cvff';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'd', 'n'];

  protected override parse(args: string[]): number[] {
    if (args.length !== 3) throw new StyleError('improper_coeff cvff needs K d n');
    const d = parseInt_(args[1], 'd');
    if (d !== 1 && d !== -1) throw new StyleError(`improper cvff d must be +1 or -1, got '${args[1]}'`);
    const n = parseInt_(args[2], 'n');
    if (n !== 0 && n !== 1 && n !== 2 && n !== 3 && n !== 4 && n !== 6) {
      throw new StyleError(`improper cvff n must be one of 0,1,2,3,4,6, got '${args[2]}'`);
    }
    return [parseNum(args[0], 'K'), d, n];
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), D = this.params.p('d'), N = this.params.p('n');
    const grad = new Array(12).fill(0), rel = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const phi = dihedralGeometry(bc, atoms[0], atoms[1], atoms[2], atoms[3], grad, rel);
      const arg = N[t] * phi;
      const ei = K[t] * (1 + D[t] * Math.cos(arg));
      e += ei;
      applyDihedral(bc, atoms, -K[t] * D[t] * N[t] * Math.sin(arg), grad, rel, ei);
    }
    bc.acc.eimp += e;
  }
}

/*
 * improper_style umbrella — docs.lammps.org/improper_umbrella.html:
 *
 *    E = & \frac{1}{2}K\left( \frac{1}{\sin\omega_0}\right) ^2 \left( \cos\omega - \cos\omega_0\right) ^2 \qquad \omega_0 \neq 0^o \\
 *    E = & K\left( 1-cos\omega\right)  \qquad \omega_0 = 0^o
 *
 * "where K is the force constant and omega is the angle between the IL axis
 * and the IJK plane". "If :math:`\omega_0 = 0` the potential term has a minimum for
 * the planar structure." omega in [0, 90 degrees], so cos(omega) >= 0.
 * * :math:`K` (energy)
 * * :math:`\omega_0` (degrees)
 */
export class ImproperUmbrella extends SimpleBonded {
  readonly name = 'umbrella';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'omega0'];

  protected override parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('improper_coeff umbrella needs K omega0');
    return [parseNum(args[0], 'K'), parseNum(args[1], 'omega0') * DEG2RAD];
  }

  override dataCoeffs(): string[] {
    const K = this.params.p('K'), w0 = this.params.p('omega0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${K[t]} ${(w0[t] * 180) / Math.PI}`);
    return out;
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), w0 = this.params.p('omega0');
    const grad = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const c = umbrellaC(bc, atoms, grad);
      const cw = Math.sqrt(Math.max(0, 1 - c));
      let ei: number, dEdc: number;
      if (w0[t] === 0) {
        ei = K[t] * (1 - cw);
        dEdc = K[t] / (2 * cw);
      } else {
        const s0 = Math.sin(w0[t]);
        const inv = 1 / (s0 * s0);
        const d = cw - Math.cos(w0[t]);
        ei = 0.5 * K[t] * inv * d * d;
        dEdc = -0.5 * K[t] * inv * d / cw;
      }
      e += ei;
      applyGradC(bc, atoms, dEdc, grad, ei);
    }
    bc.acc.eimp += e;
  }
}

/*
 * improper_style cossq — docs.lammps.org/improper_cossq.html:
 *
 *    E = \frac{1}{2} K \cos^2{\left(\chi - \chi_0\right)}
 *
 * "where :math:`\chi` is the improper angle, :math:`\chi_0` is its
 * equilibrium value, and :math:`K` is a prefactor."
 * * :math:`K` (energy)
 * * :math:`\chi_0` (degrees)
 *
 * The page calls chi the angle between the planes I,J,K and J,K,L. Measured
 * with native LAMMPS (black box, random four-atom geometries, chi0 = 0 and
 * 30 degrees; energies to 12 digits), chi is instead the angle between the
 * bond vectors r_J - r_I and r_L - r_K:
 *   cos(chi) = (r_J - r_I).(r_L - r_K) / (|r_J - r_I| |r_L - r_K|).
 * Forces, also measured: native applies F = -K cos(chi - chi0) grad(c) with
 * c = cos(chi), i.e. the chi0 = 0 chain rule. For chi0 != 0 that is not the
 * gradient of the energy above (one geometry at chi0 = 30 degrees: every
 * component is 0.8289 of -dE/dx, exactly cos(chi - chi0) / [d cos(chi - chi0)
 * / d cos(chi)]); the engine reproduces native's forces.
 */
const bondPairCos = (bc: BondedCompute, atoms: readonly number[], grad: number[]): number => {
  const u = [0, 0, 0], w = [0, 0, 0];
  delta(bc, atoms[0], atoms[1], u);
  delta(bc, atoms[2], atoms[3], w);
  const uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  const ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  const inv = 1 / Math.sqrt(uu * ww);
  const c = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) * inv;
  for (let d = 0; d < 3; d++) {
    const dcdu = w[d] * inv - (c * u[d]) / uu;
    const dcdw = u[d] * inv - (c * w[d]) / ww;
    grad[d] = -dcdu; grad[3 + d] = dcdu;
    grad[6 + d] = -dcdw; grad[9 + d] = dcdw;
  }
  return c;
};

export class ImproperCossq extends SimpleBonded {
  readonly name = 'cossq';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'chi0'];

  protected override parse(args: string[]): number[] {
    if (args.length !== 2) throw new StyleError('improper_coeff cossq needs K chi0');
    return [parseNum(args[0], 'K'), parseNum(args[1], 'chi0') * DEG2RAD];
  }

  override dataCoeffs(): string[] {
    const K = this.params.p('K'), c0 = this.params.p('chi0');
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${fmtCoeff(K[t])} ${fmtCoeff((c0[t] * 180) / Math.PI)}`);
    return out;
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), c0 = this.params.p('chi0');
    const grad = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const c = Math.min(1, Math.max(-1, bondPairCos(bc, atoms, grad)));
      const cd = c * Math.cos(c0[t]) + Math.sqrt(1 - c * c) * Math.sin(c0[t]); // cos(chi - chi0)
      const ei = 0.5 * K[t] * cd * cd;
      e += ei;
      applyGradC(bc, atoms, K[t] * cd, grad, ei);
    }
    bc.acc.eimp += e;
  }
}

/*
 * improper_style fourier — docs.lammps.org/improper_fourier.html:
 *
 *    E = K [C_0 + C_1 \cos ( \omega) + C_2 \cos( 2 \omega) ]
 *
 * "where K is the force constant, C0, C1, C2 are dimensionless coefficients,
 * and omega is the angle between the IL axis and the IJK plane".
 * "If all parameter (see below) is not zero, the all the three possible
 * angles will taken in account." — the three possible umbrella angles of the
 * quadruplet (axis IL in plane IJK, axis IK in plane IJL, axis IJ in plane
 * IKL), each with the full formula; with all = 0 only the IL / IJK angle.
 * cos(2 omega) = 2 cos(omega)^2 - 1 = 1 - 2 c with c from umbrellaC.
 * * :math:`K` (energy)
 * * :math:`C_0` (unitless)
 * * :math:`C_1` (unitless)
 * * :math:`C_2` (unitless)
 * * all  (0 or 1, optional)
 */
const AXES_ONE = [[0, 1, 2, 3]];
// centre first, then the two plane atoms, then the apex atom
const AXES_ALL = [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1]];

export class ImproperFourier extends SimpleBonded {
  readonly name = 'fourier';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K', 'C0', 'C1', 'C2'];
  private all!: Uint8Array;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.all = new Uint8Array(ntypes + 1);
  }

  override coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 5 && args.length !== 6) throw new StyleError('improper_coeff fourier needs type K C0 C1 C2 [all]');
    if (args.length === 6) {
      const all = parseInt_(args[5], 'all');
      if (all !== 0 && all !== 1) throw new StyleError(`improper fourier all must be 0 or 1, got '${args[5]}'`);
      const [lo, hi] = typeBounds(args[0], this.ntypes);
      for (let t = lo; t <= hi; t++) this.all[t] = all;
    }
    super.coeff([args[0], ...args.slice(1, 5)], ctx);
  }

  override dataCoeffs(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) {
      let line = `${t} ${this.paramNames.map((name) => this.params.p(name)[t]).join(' ')}`;
      if (this.all[t] === 1) line += ' 1';
      out.push(line);
    }
    return out;
  }

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K = this.params.p('K'), C0 = this.params.p('C0'), C1 = this.params.p('C1'), C2 = this.params.p('C2');
    const grad = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const axes = this.all[t] === 1 ? AXES_ALL : AXES_ONE;
      let ei = 0;
      for (const ord of axes) {
        const a4 = ord.map((w) => atoms[w]);
        const c = umbrellaC(bc, a4, grad);
        const cw = Math.sqrt(Math.max(0, 1 - c));
        const term = K[t] * (C0[t] + C1[t] * cw + C2[t] * (1 - 2 * c));
        const dEdc = K[t] * (-C1[t] / (2 * cw) - 2 * C2[t]);
        ei += term;
        applyGradC(bc, a4, dEdc, grad, term);
      }
      e += ei;
    }
    bc.acc.eimp += e;
  }
}

/*
 * improper_style distance — docs.lammps.org/improper_distance.html:
 *
 *    E = K_2 d^2 + K_4 d^4
 *
 * "where d is the distance between the central atom and the plane formed by
 * the other three atoms. If the 4 atoms in an improper quadruplet (listed in
 * the data file read by the read_data command) are ordered I,J,K,L then the
 * I-atom is assumed to be the central atom."
 * * :math:`K_2` (energy/distance\^2)
 * * :math:`K_4` (energy/distance\^4)
 */
export class ImproperDistance extends SimpleBonded {
  readonly name = 'distance';
  readonly kind = 'improper' as const;
  readonly paramNames = ['K2', 'K4'];

  compute(bc: BondedCompute): void {
    const I = bc.s.topo.impropers;
    const K2 = this.params.p('K2'), K4 = this.params.p('K4');
    const grad = new Array(12).fill(0);
    let e = 0;
    for (let q = 0; q < I.n; q++) {
      const atoms = [0, 1, 2, 3].map((w) => atomIndex(bc, I.atoms[4 * q + w], 'improper'));
      const t = I.type[q];
      const c = distanceC(bc, atoms, grad);
      const ei = K2[t] * c + K4[t] * c * c;
      e += ei;
      applyGradC(bc, atoms, K2[t] + 2 * K4[t] * c, grad, ei);
    }
    bc.acc.eimp += e;
  }
}

/*
 * improper_style zero — docs.lammps.org/improper_zero.html:
 *
 * "Using an improper style of zero means improper forces and energies are
 * not computed, but the geometry of improper quadruplets is still
 * accessible to other commands."
 *
 * "The optional nocoeff flag allows to read data files with a ImproperCoeff
 * section for any improper style. Similarly, any improper_coeff commands
 * will only be checked for the improper type number and the rest ignored."
 *
 * "Note that the improper_coeff command must be used for all improper
 * types, though no additional values are specified."
 */
export class ImproperZero extends SimpleBonded {
  readonly name = 'zero';
  readonly kind = 'improper' as const;
  readonly paramNames: readonly string[] = [];
  private nocoeff = false;

  override settings(args: string[], _ctx?: StyleContext): void {
    for (const w of args) {
      if (w !== 'nocoeff') throw new StyleError(`improper_style zero: unknown keyword '${w}' (only nocoeff)`);
      this.nocoeff = true;
    }
  }

  override coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError('usage: improper_coeff N');
    if (!this.nocoeff && args.length > 1) {
      throw new StyleError("improper_coeff zero takes no coefficient values (use 'improper_style zero nocoeff' to ignore them)");
    }
    this.params.setRange(args[0], []);
  }

  compute(_bc: BondedCompute): void {}
}

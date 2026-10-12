import { Bonded, StyleError, typeBounds, type BondedCompute, type StyleContext } from './types';
import { fmtCoeff, parseNum } from './util';

/*
 * Shared pieces for bond/angle/dihedral/improper styles.
 *
 * Coefficients — docs.lammps.org/bond_coeff.html (same for angle_coeff,
 * dihedral_coeff, improper_coeff): "For numeric values only, a wild-card
 * asterisk can be used to set the coefficients for multiple bond types. This
 * takes the form "*" or "*n" or "n*" or "m*n"." "a bond_coeff command can
 * override a previous setting for the same bond type."
 *
 * Geometry uses minimum-image displacements between the atoms of a term
 * (atoms of a bonded term are within half a box of each other).
 * Dihedral angle (dihedral_style.html: "phi is the torsional angle defined by
 * the quadruplet of atoms", trans = 180 degrees): with b1 = r_j - r_i,
 * b2 = r_k - r_j, b3 = r_l - r_k, m = b1 x b2, n = b2 x b3,
 *   phi = atan2(|b2| b1 . n, m . n)
 * and the analytic gradients of Blondel & Karplus, J Comput Chem 17, 1132
 * (1996), in these vectors (p = b1.b2/|b2|^2, q = b3.b2/|b2|^2):
 *   dphi/dr_i = -|b2|/|m|^2 m;   dphi/dr_l = |b2|/|n|^2 n;
 *   dphi/dr_j = -(1 + p) dphi/dr_i + q dphi/dr_l
 *   dphi/dr_k = -(1 + q) dphi/dr_l + p dphi/dr_i
 * (checked against finite differences in tests/engineBonded.test.ts).
 */

/** Per-type coefficient storage with set flags. */
export class TypeParams {
  readonly data: Float64Array[];
  readonly set: Uint8Array;
  constructor(readonly ntypes: number, readonly names: readonly string[]) {
    this.data = names.map(() => new Float64Array(ntypes + 1));
    this.set = new Uint8Array(ntypes + 1);
  }
  p(name: string): Float64Array {
    const k = this.names.indexOf(name);
    if (k < 0) throw new Error(`no parameter ${name}`);
    return this.data[k];
  }
  setRange(w: string, values: readonly number[]): void {
    const [lo, hi] = typeBounds(w, this.ntypes);
    for (let t = lo; t <= hi; t++) {
      for (let k = 0; k < this.names.length; k++) this.data[k][t] = values[k] ?? Number.NaN;
      this.set[t] = 1;
    }
  }
  check(kind: string): void {
    for (let t = 1; t <= this.ntypes; t++) if (!this.set[t]) throw new StyleError(`all ${kind} coeffs are not set (type ${t})`);
  }
  /** "type v1 v2 ..." lines for a data file. */
  lines(): string[] {
    const out: string[] = [];
    for (let t = 1; t <= this.ntypes; t++) out.push(`${t} ${this.data.map((d) => fmtCoeff(d[t])).join(' ')}`);
    return out;
  }
}

/** A bonded style whose coefficients are a fixed list of numbers per type. */
export abstract class SimpleBonded extends Bonded {
  params!: TypeParams;
  abstract readonly paramNames: readonly string[];
  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length) throw new StyleError(`${this.kind}_style ${this.name} takes no arguments`);
  }
  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.params = new TypeParams(ntypes, this.paramNames);
  }
  /** Parsed values (override to convert units, e.g. degrees to radians). */
  protected parse(args: string[]): number[] {
    const n = this.paramNames.length;
    if (args.length !== n) throw new StyleError(`${this.kind}_coeff ${this.name} needs ${n} coefficient(s): ${this.paramNames.join(' ')}`);
    return args.map((w, k) => parseNum(w, this.paramNames[k]));
  }
  coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError(`usage: ${this.kind}_coeff N coefficients`);
    this.params.setRange(args[0], this.parse(args.slice(1)));
  }
  init(_ctx?: StyleContext): void {
    this.params.check(this.kind);
  }
  dataCoeffs(): string[] {
    return this.params.lines();
  }
}

/** Index of an atom ID, or a StyleError naming the missing atom. */
export const atomIndex = (bc: BondedCompute, id: number, kind: string): number => {
  const i = id < bc.map.length ? bc.map[id] : -1;
  if (i < 0) throw new StyleError(`${kind} atom ${id} missing`);
  return i;
};

/** Minimum-image displacement r_b - r_a into out[o..o+2]. */
export const delta = (bc: BondedCompute, a: number, b: number, out: number[], o = 0): void => {
  const x = bc.s.x;
  out[o] = x[3 * b] - x[3 * a];
  out[o + 1] = x[3 * b + 1] - x[3 * a + 1];
  out[o + 2] = x[3 * b + 2] - x[3 * a + 2];
  const d = [out[o], out[o + 1], out[o + 2]];
  bc.geom.minimumImage(d);
  out[o] = d[0]; out[o + 1] = d[1]; out[o + 2] = d[2];
};

/*
 * Bond displacement r_i - r_j for a bond stored on atom i (its first atom)
 * with partner j, into out[o..o+2]. The partner is the periodic image of j
 * closest to i among the ghost copies, where ghost copies exist only within
 * the ghost cutoff of the box (neighbor cutghost, passed as bc.ghostCut).
 *
 * docs.lammps.org/Errors_details.html, warning "Bond/angle/dihedral extent >
 * half of periodic box length": "LAMMPS will still compute bonded interactions
 * based on the closest periodic images of the atoms and thus in most cases
 * the results will be correct." Measured with native LAMMPS (black box): two
 * atoms in a periodic x box of length 12, bond "create_bonds single/bond 1 1 2"
 * (atom 1 at x = 0.5, atom 2 at x = x2), comm cutoff at its default 2.3 (pair
 * cutoff 2.0 + skin 0.3). Bond energy K (r - r0)^2 with K = 1, r0 = 1:
 *   x2 = 11.0 (image -1.0, a ghost): 0.25;  x2 = 10.0 (image -2.0): 2.25
 *   x2 = 9.0 (image -3.0, not a ghost): 56.25 (r = 8.5, the atom as stored)
 *   x2 = 8.0: 42.25 (r = 7.5);  x2 = 7.0: 30.25 (r = 6.5)
 * and with the bond written "create_bonds single/bond 1 2 1" (owner at 9.0,
 * partner at 0.5, image 12.5 a ghost): 6.25 (r = 3.5). So the owner is the
 * first atom of the bond and the partner's images are tested against the
 * ghost region of the box: an image p = x_j + nL is a ghost when
 * lo - cut <= p <= hi + cut (periodic dimensions, orthogonal box). The atom
 * as stored is always a candidate.
 */
export const bondDelta = (bc: BondedCompute, i: number, j: number, out: number[], o = 0): void => {
  const x = bc.s.x;
  const g = bc.geom;
  const cut = bc.ghostCut ?? Number.POSITIVE_INFINITY;
  if (bc.ghostCut === undefined || g.triclinic) {
    delta(bc, j, i, out, o);
    return;
  }
  for (let k = 0; k < 3; k++) {
    const xi = x[3 * i + k];
    const xj = x[3 * j + k];
    let disp = xj - xi;
    if (g.periodic[k]) {
      const len = k === 0 ? g.lx : k === 1 ? g.ly : g.lz;
      let best = Number.POSITIVE_INFINITY;
      for (let n = -2; n <= 2; n++) {
        const p = xj + n * len;
        if (p < g.lo[k] - cut || p > g.hi[k] + cut) continue;
        const dd = p - xi;
        if (Math.abs(dd) < Math.abs(best)) best = dd;
      }
      if (Number.isFinite(best)) disp = best;
    }
    out[o + k] = -disp;
  }
};

/** Dihedral angle and its gradient for atoms i, j, k, l (all four as 3-vectors in grad). */
export const dihedralGeometry = (bc: BondedCompute, i: number, j: number, k: number, l: number, grad: number[], rel: number[]): number => {
  const b = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  delta(bc, i, j, b, 0);
  delta(bc, j, k, b, 3);
  delta(bc, k, l, b, 6);
  const [b1x, b1y, b1z, b2x, b2y, b2z, b3x, b3y, b3z] = b;
  const mx = b1y * b2z - b1z * b2y, my = b1z * b2x - b1x * b2z, mz = b1x * b2y - b1y * b2x;
  const nx = b2y * b3z - b2z * b3y, ny = b2z * b3x - b2x * b3z, nz = b2x * b3y - b2y * b3x;
  const b2len = Math.sqrt(b2x * b2x + b2y * b2y + b2z * b2z);
  const m2 = mx * mx + my * my + mz * mz, n2 = nx * nx + ny * ny + nz * nz;
  const phi = Math.atan2(b2len * (b1x * nx + b1y * ny + b1z * nz), mx * nx + my * ny + mz * nz);
  const b22 = b2len * b2len;
  const gi = m2 > 0 ? -b2len / m2 : 0, gl = n2 > 0 ? b2len / n2 : 0;
  const giv = [gi * mx, gi * my, gi * mz], glv = [gl * nx, gl * ny, gl * nz];
  const p = b22 > 0 ? (b1x * b2x + b1y * b2y + b1z * b2z) / b22 : 0;
  const q = b22 > 0 ? (b3x * b2x + b3y * b2y + b3z * b2z) / b22 : 0;
  for (let d = 0; d < 3; d++) {
    grad[d] = giv[d];
    grad[3 + d] = -(1 + p) * giv[d] + q * glv[d];
    grad[6 + d] = -(1 + q) * glv[d] + p * giv[d];
    grad[9 + d] = glv[d];
  }
  // positions relative to atom i (for the virial)
  for (let d = 0; d < 3; d++) {
    rel[d] = 0;
    rel[3 + d] = b[d];
    rel[6 + d] = b[d] + b[3 + d];
    rel[9 + d] = b[d] + b[3 + d] + b[6 + d];
  }
  return phi;
};

/** Applies forces -dE/dphi * grad to the four atoms and tallies virial/per-atom energy. */
export const applyDihedral = (bc: BondedCompute, atoms: number[], dEdphi: number, grad: number[], rel: number[], e: number, virialOnly = false): void => {
  const f = bc.f;
  const fk: number[] = new Array(12);
  for (let a = 0; a < 4; a++) {
    for (let d = 0; d < 3; d++) {
      const v = -dEdphi * grad[3 * a + d];
      fk[3 * a + d] = v;
      if (!virialOnly) f[3 * atoms[a] + d] += v;
    }
  }
  const v = bc.virial;
  const w = [0, 0, 0, 0, 0, 0];
  for (let a = 0; a < 4; a++) {
    const rx = rel[3 * a], ry = rel[3 * a + 1], rz = rel[3 * a + 2];
    const fx = fk[3 * a], fy = fk[3 * a + 1], fz = fk[3 * a + 2];
    w[0] += rx * fx; w[1] += ry * fy; w[2] += rz * fz; w[3] += rx * fy; w[4] += rx * fz; w[5] += ry * fz;
  }
  for (let c = 0; c < 6; c++) v[c] += w[c];
  if (bc.eatom) for (const i of atoms) bc.eatom[i] += e / 4;
  if (bc.vatom) for (const i of atoms) for (let c = 0; c < 6; c++) bc.vatom[6 * i + c] += w[c] / 4;
};

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

/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/AMOEBA/fix_amoeba_bitorsion.cpp (read_grid_data, create_splines, nspline, cspline, cytsy, cytsyp, cytsys,
 * chkttor, bcuint1, bcucof, read_data_header, read_data_section, pre_neighbor, post_force) at commit b332f28, adapted from
 * Tinker ktortor.f, ettors.f, bcuint1.f and bcucof.f. AMOEBA package authors: the LAMMPS developers (Sandia Corporation,
 * GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix amoeba/bitorsion: the torsion-torsion correction of the AMOEBA force field, a five-atom term (the two torsions
 * 1-2-3-4 and 2-3-4-5 around the central bond 2-3-4). The energy is a bicubic spline of a tabulated surface E(phi1, phi2)
 * per type, read from the grid file given to the fix; the forces are the derivatives of that surface. The chirality of
 * the central atom flips the signs of both angles (chkttor). Read from the data file: the "N bitorsions" header and the
 * BiTorsions section (index, type and five atom IDs per term). Each term is evaluated once, from its lowest-index atom;
 * the energy, forces and virial are the full sums over its five atoms (see fix_pitorsion.ts). The fix is off when the
 * pair style's bitorsion flag is off.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { BondedCompute } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { SimState } from '../../engine/types';
import { buildAtomMap } from '../../engine/atoms';
import { delta } from '../../engine/force/bonded_util';
import { partnerMap } from './angle_amoeba';
import { acos as libmAcos } from '../libm/acos';

const RAD2DEG = 180.0 / Math.PI;

/** Bicubic weight table of bcucof (fix_amoeba_bitorsion.cpp WT; the Numerical Recipes bcucof weights). */
const WT: number[][] = [
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  [-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
  [2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1],
  [0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1],
  [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
  [9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2],
  [-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2],
  [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
  [-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1],
  [4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1],
];

/** One bitorsion grid type: the node counts, the node coordinates (degrees) and the energies (x varies fastest). */
export interface BitorsionGrid {
  nx: number;
  ny: number;
  ttx: Float64Array;
  tty: Float64Array;
  tbf: Float64Array;
}

/** The spline data of one type: x derivatives, y derivatives and cross derivatives on the grid (Tinker tbx, tby, tbxy). */
export interface BitorsionSpline {
  tbx: Float64Array;
  tby: Float64Array;
  tbxy: Float64Array;
}

/**
 * read_grid_data: the grid file has a line to skip, the number of types, and per type a line "type nx ny" followed by
 * nx * ny lines' worth of "x y energy" triples (the x coordinate varies fastest).
 */
export const parseBitorsionGrids = (text: string, name: string): BitorsionGrid[] => {
  const lines = text.split(/\r?\n/);
  const bad = (what: string): never => {
    throw new StyleError(`fix amoeba/bitorsion file ${name}: ${what}`);
  };
  // the first line is a title (skipped); blank lines and '#' comments are skipped where a value is read (TextFileReader)
  let li = 1;
  const nextLine = (): string => {
    while (li < lines.length && (lines[li].trim() === '' || lines[li].trim().startsWith('#'))) li++;
    if (li >= lines.length) bad('unexpected end of file');
    return lines[li++].trim();
  };
  const ntypes = Number(nextLine().split(/\s+/)[0]);
  if (!Number.isInteger(ntypes) || ntypes < 1) bad('no types');
  const grids: BitorsionGrid[] = [];
  const headerLine = (): number[] => {
    const w = nextLine().split(/\s+/).map(Number);
    if (w.length < 3 || w.some((v) => !Number.isFinite(v))) bad(`invalid grid header '${w.join(' ')}'`);
    return w;
  };
  for (let t = 1; t <= ntypes; t++) {
    const [, nx, ny] = headerLine();
    if (nx < 2 || ny < 2) bad(`invalid ${nx} by ${ny} grid for type ${t}`);
    const npts = nx * ny;
    const vals: number[] = [];
    while (vals.length < 3 * npts && li < lines.length) {
      for (const tok of lines[li++].trim().split(/\s+/)) {
        if (tok === '') continue;
        const v = Number(tok);
        if (!Number.isFinite(v)) bad(`non-numeric grid value '${tok}'`);
        vals.push(v);
      }
    }
    if (vals.length < 3 * npts) bad(`type ${t} grid is incomplete`);
    const ttx = new Float64Array(nx), tty = new Float64Array(ny), tbf = new Float64Array(npts);
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        const e = 3 * (iy * nx + ix);
        if (iy === 0) ttx[ix] = vals[e];
        if (ix === 0) tty[iy] = vals[e + 1];
        tbf[iy * nx + ix] = vals[e + 2];
      }
    }
    grids.push({ nx, ny, ttx, tty, tbf });
  }
  return grids;
};

/** nspline (Tinker): natural cubic spline of the n+1 points (x0, y0); s1 gets the first derivatives at the nodes. */
const nspline = (n: number, x0: Float64Array, y0: Float64Array, s1: Float64Array): void => {
  const h = new Float64Array(n + 1), g = new Float64Array(n + 1), dy = new Float64Array(n + 1);
  const dla = new Float64Array(n + 1), dmu = new Float64Array(n + 1);
  const y21 = 0.0, y2n = 0.0;
  for (let i = 0; i <= n - 1; i++) {
    h[i] = x0[i + 1] - x0[i];
    dy[i] = (y0[i + 1] - y0[i]) / h[i];
  }
  for (let i = 1; i <= n - 1; i++) {
    dla[i] = h[i] / (h[i] + h[i - 1]);
    dmu[i] = 1.0 - dla[i];
    g[i] = 3.0 * (dla[i] * dy[i - 1] + dmu[i] * dy[i]);
  }
  dla[n] = 1.0;
  dla[0] = 0.0;
  dmu[n] = 0.0;
  dmu[0] = 1.0;
  g[0] = 3.0 * dy[0] - 0.5 * h[0] * y21;
  g[n] = 3.0 * dy[n - 1] + 0.5 * h[n - 1] * y2n;
  dmu[0] = 0.5 * dmu[0];
  g[0] = 0.5 * g[0];
  for (let i = 1; i <= n; i++) {
    const t = 2.0 - dmu[i - 1] * dla[i];
    dmu[i] = dmu[i] / t;
    g[i] = (g[i] - g[i - 1] * dla[i]) / t;
  }
  for (let i = n - 1; i >= 0; i--) g[i] = g[i] - dmu[i] * g[i + 1];
  for (let i = 0; i <= n; i++) s1[i] = g[i];
};

/** cytsyp (Tinker): factors the cyclic tridiagonal matrix in place; returns iflag (1 when it is positive definite). */
const cytsyp = (n: number, dm: Float64Array, du: Float64Array, cr: Float64Array): number => {
  const eps = 1.0e-8;
  if (n < 3) return -2;
  let row = Math.abs(dm[1]) + Math.abs(du[1]) + Math.abs(du[n]);
  if (row === 0.0) return 0;
  let d = 1.0 / row;
  if (dm[1] < 0.0) return -1;
  if (Math.abs(dm[1]) * d <= eps) return 0;
  let temp1 = du[1];
  du[1] = du[1] / dm[1];
  cr[1] = du[n] / dm[1];
  let temp2 = 0.0;
  for (let i = 2; i < n; i++) {
    row = Math.abs(dm[i]) + Math.abs(du[i]) + Math.abs(temp1);
    if (row === 0.0) return 0;
    d = 1.0 / row;
    dm[i] = dm[i] - temp1 * du[i - 1];
    if (dm[i] < 0.0) return -1;
    if (Math.abs(dm[i]) * d <= eps) return 0;
    if (i < n - 1) {
      cr[i] = -temp1 * cr[i - 1] / dm[i];
      temp1 = du[i];
      du[i] = du[i] / dm[i];
    } else {
      temp2 = du[i];
      du[i] = (du[i] - temp1 * cr[i - 1]) / dm[i];
    }
  }
  row = Math.abs(du[n]) + Math.abs(dm[n]) + Math.abs(temp2);
  if (row === 0.0) return 0;
  d = 1.0 / row;
  dm[n] = dm[n] - dm[n - 1] * du[n - 1] * du[n - 1];
  temp1 = 0.0;
  for (let i = 1; i < n - 1; i++) temp1 += dm[i] * cr[i] * cr[i];
  dm[n] = dm[n] - temp1;
  if (dm[n] < 0.0) return -1;
  if (Math.abs(dm[n]) * d <= eps) return 0;
  return 1;
};

/** cytsys (Tinker): the update and back substitution of the factored cyclic system; the solution goes to x. */
const cytsys = (n: number, dm: Float64Array, du: Float64Array, cr: Float64Array, rs: Float64Array, x: Float64Array): void => {
  let temp = rs[1];
  rs[1] = temp / dm[1];
  let sum = cr[1] * temp;
  for (let i = 2; i < n; i++) {
    temp = rs[i] - du[i - 1] * temp;
    rs[i] = temp / dm[i];
    if (i !== n - 1) sum += cr[i] * temp;
  }
  temp = rs[n] - du[n - 1] * temp;
  temp = temp - sum;
  rs[n] = temp / dm[n];
  x[n] = rs[n];
  x[n - 1] = rs[n - 1] - du[n - 1] * x[n];
  for (let i = n - 2; i > 0; i--) x[i] = rs[i] - du[i] * x[i + 1] - cr[i] * x[n];
};

/** cspline (Tinker): periodic interpolating cubic spline of the n+1 points (xn, fn); b gets the first derivatives. */
const cspline = (n: number, xn: Float64Array, fn: Float64Array, b: Float64Array): void => {
  const h = new Float64Array(n + 1), du = new Float64Array(n + 1), dm = new Float64Array(n + 1);
  const rc = new Float64Array(n + 1), rs = new Float64Array(n + 1), c = new Float64Array(n + 1);
  const average = 0.5 * (fn[0] + fn[n]);
  fn[0] = average;
  fn[n] = average;
  for (let i = 0; i < n; i++) h[i] = xn[i + 1] - xn[i];
  h[n] = h[0];
  for (let i = 1; i < n; i++) du[i] = h[i];
  du[n] = h[0];
  for (let i = 1; i <= n; i++) dm[i] = 2.0 * (h[i - 1] + h[i]);
  let temp1 = (fn[1] - fn[0]) / h[0];
  for (let i = 1; i < n; i++) {
    const temp2 = (fn[i + 1] - fn[i]) / h[i];
    rs[i] = 3.0 * (temp2 - temp1);
    temp1 = temp2;
  }
  rs[n] = 3.0 * ((fn[1] - fn[0]) / h[0] - temp1);
  const iflag = cytsyp(n, dm, du, rc);
  if (iflag !== 1) return;
  cytsys(n, dm, du, rc, rs, c);
  c[0] = c[n];
  for (let i = 0; i < n; i++) b[i] = (fn[i + 1] - fn[i]) / h[i] - h[i] / 3.0 * (c[i + 1] + 2.0 * c[i]);
  b[n] = (fn[1] - fn[n]) / h[n] - h[n] / 3.0 * (c[1] + 2.0 * c[n]);
};

/**
 * create_splines (fix_amoeba_bitorsion.cpp): the x, y and cross derivatives of each type's grid. A grid whose x and y
 * ranges are both 360 degrees is periodic (cspline), otherwise natural (nspline). As native, the y-direction passes use
 * the x node values (ttx) as their abscissae.
 */
export const createBitorsionSpline = (g: BitorsionGrid): BitorsionSpline => {
  const nx = g.nx, ny = g.ny;
  const eps = 1.0e-6;
  let cyclic = true;
  if (Math.abs(Math.abs(g.ttx[0] - g.ttx[nx - 1]) - 360.0) > eps) cyclic = false;
  if (Math.abs(Math.abs(g.tty[0] - g.tty[ny - 1]) - 360.0) > eps) cyclic = false;
  const tbx = new Float64Array(nx * ny), tby = new Float64Array(nx * ny), tbxy = new Float64Array(nx * ny);
  const solve = (n: number, xs: Float64Array, fs: Float64Array, out: Float64Array): void => {
    if (cyclic) cspline(n, xs, fs, out);
    else nspline(n, xs, fs, out);
  };
  // x derivatives along each row
  const tmp1 = Float64Array.from(g.ttx.subarray(0, nx));
  for (let j = 0; j < ny; j++) {
    const row = new Float64Array(nx);
    for (let i = 0; i < nx; i++) row[i] = g.tbf[j * nx + i];
    const bs = new Float64Array(nx);
    solve(nx - 1, tmp1, row, bs);
    for (let i = 0; i < nx; i++) tbx[j * nx + i] = bs[i];
  }
  // y derivatives along each column (abscissae ttx, as native)
  const tmpY = Float64Array.from(g.ttx.subarray(0, ny));
  for (let i = 0; i < nx; i++) {
    const col = new Float64Array(ny);
    for (let j = 0; j < ny; j++) col[j] = g.tbf[j * nx + i];
    const bs = new Float64Array(ny);
    solve(ny - 1, tmpY, col, bs);
    for (let j = 0; j < ny; j++) tby[j * nx + i] = bs[j];
  }
  // cross derivatives: the y pass of tbx
  for (let i = 0; i < nx; i++) {
    const col = new Float64Array(ny);
    for (let j = 0; j < ny; j++) col[j] = tbx[j * nx + i];
    const bs = new Float64Array(ny);
    solve(ny - 1, tmpY, col, bs);
    for (let j = 0; j < ny; j++) tbxy[j * nx + i] = bs[j];
  }
  return { tbx, tby, tbxy };
};

/** bcucof (Tinker): the 4x4 bicubic coefficients c[i][k] of one cell from its corner values and derivatives. */
const bcucof = (y: number[], y1: number[], y2: number[], y12: number[], d1: number, d2: number): number[][] => {
  const x = new Array<number>(16).fill(0);
  const d1d2 = d1 * d2;
  for (let i = 0; i < 4; i++) {
    x[i] = y[i];
    x[i + 4] = y1[i] * d1;
    x[i + 8] = y2[i] * d2;
    x[i + 12] = y12[i] * d1d2;
  }
  const cl = new Array<number>(16).fill(0);
  for (let i = 0; i < 16; i++) {
    let xx = 0.0;
    for (let k = 0; k < 16; k++) xx += WT[i][k] * x[k];
    cl[i] = xx;
  }
  const c: number[][] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  let j = 0;
  for (let i = 0; i < 4; i++) for (let k = 0; k < 4; k++) c[i][k] = cl[j++];
  return c;
};

/** bcuint1 (Tinker): the bicubic energy at (x1, x2) and its derivatives with respect to x1 and x2. */
const bcuint1 = (
  y: number[], y1: number[], y2: number[], y12: number[], x1l: number, x1u: number, x2l: number, x2u: number,
  x1: number, x2: number,
): { ansy: number; ansy1: number; ansy2: number } => {
  const c = bcucof(y, y1, y2, y12, x1u - x1l, x2u - x2l);
  const t = (x1 - x1l) / (x1u - x1l);
  const u = (x2 - x2l) / (x2u - x2l);
  let ansy = 0.0, ansy1 = 0.0, ansy2 = 0.0;
  for (let i = 3; i >= 0; i--) {
    ansy = t * ansy + ((c[i][3] * u + c[i][2]) * u + c[i][1]) * u + c[i][0];
    ansy1 = u * ansy1 + (3.0 * c[3][i] * t + 2.0 * c[2][i]) * t + c[1][i];
    ansy2 = t * ansy2 + (3.0 * c[i][3] * u + 2.0 * c[i][2]) * u + c[i][1];
  }
  ansy1 /= x1u - x1l;
  ansy2 /= x2u - x2l;
  return { ansy, ansy1, ansy2 };
};

interface BiTerm {
  type: number;
  ids: number[];
}

/** The lookups chkttor needs: a local index per atom ID, the type and atomic number, and the 1-2 partners. */
interface ChiralityContext {
  shim: BondedCompute;
  partners: Map<number, number[]>;
  idx: (id: number) => number;
  typeOf: (id: number) => number;
  atomicOf: (type: number) => number;
}

/**
 * chkttor (fix_amoeba_bitorsion.cpp): the chirality of the central atom ic (atom ids ib, ic, id of the term). Its 1-2
 * partners other than ib and id are j and k; the one with the larger atomic number (type when they tie) is atom ia. The
 * signed volume of the parallelepiped spanned by ia, ib and id at ic decides the sign: a negative volume flips both
 * torsion angles. A centre without four partners is not tested.
 */
const chirality = (
  ctx: ChiralityContext, icId: number, ibId: number, idId: number, value1: number, value2: number,
): { sign: number; value1: number; value2: number } => {
  let sign = 1.0;
  const p = ctx.partners.get(icId) ?? [];
  if (p.length !== 4) return { sign, value1, value2 };
  const others = p.filter((q) => q !== ibId && q !== idId);
  if (others.length < 2) return { sign, value1, value2 };
  const jId = others[0], kId = others[1];
  const jLocal = ctx.idx(jId), kLocal = ctx.idx(kId);
  const jtype = ctx.typeOf(jId), ktype = ctx.typeOf(kId);
  let ilocal = -1;
  if (jtype > ktype) ilocal = jLocal;
  if (ktype > jtype) ilocal = kLocal;
  if (ctx.atomicOf(jtype) > ctx.atomicOf(ktype)) ilocal = jLocal;
  if (ctx.atomicOf(ktype) > ctx.atomicOf(jtype)) ilocal = kLocal;
  if (ilocal < 0) return { sign, value1, value2 };
  // the signed volume at the centre: positions of ia, ib and id relative to ic (nearest copies)
  const icLocal = ctx.idx(icId);
  const ia = [0, 0, 0], ib = [0, 0, 0], id = [0, 0, 0];
  delta(ctx.shim, icLocal, ilocal, ia);
  delta(ctx.shim, icLocal, ctx.idx(ibId), ib);
  delta(ctx.shim, icLocal, ctx.idx(idId), id);
  const xac = ia[0], yac = ia[1], zac = ia[2];
  const xbc = ib[0], ybc = ib[1], zbc = ib[2];
  const xdc = id[0], ydc = id[1], zdc = id[2];
  const c1 = ybc * zdc - zbc * ydc;
  const c2 = ydc * zac - zdc * yac;
  const c3 = yac * zbc - zac * ybc;
  const vol = xac * c1 + xbc * c2 + xdc * c3;
  if (vol < 0.0) {
    sign = -1.0;
    value1 = -value1;
    value2 = -value2;
  }
  return { sign, value1, value2 };
};

export class FixAmoebaBiTorsion extends Fix {
  readonly style = 'amoeba/bitorsion';
  private readonly grids: BitorsionGrid[];
  private readonly splines: BitorsionSpline[];
  private nbitorsions: number | null = null;
  private readonly terms: BiTerm[] = [];
  private etotal = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError(`fix amoeba/bitorsion group ${group}: only group all is supported`);
    if (args.length !== 1) throw new StyleError('usage: fix ID group-ID amoeba/bitorsion bitorsion-file');
    this.grids = parseBitorsionGrids(sys.readFile(args[0]), args[0]);
    this.splines = this.grids.map((g) => createBitorsionSpline(g));
    this.energyGlobal = true;
    this.virialGlobal = true;
    this.thermoEnergy = true;
    this.thermoVirial = true;
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  /** read_data header line: "N bitorsions". */
  readHeader(line: string, at: number): void {
    const m = /^\s*(\d+)\s+bitorsions\s*$/.exec(line);
    if (!m) throw new StyleError(`data file line ${at}: invalid read data header line for fix amoeba/bitorsion: '${line.trim()}'`);
    this.nbitorsions = Number(m[1]);
  }

  sectionLines(_natoms: number): number {
    if (this.nbitorsions === null) throw new StyleError('read_data fix amoeba/bitorsion: the header must give the "bitorsions" count');
    return this.nbitorsions;
  }

  /** BiTorsions rows: index, type and five atom IDs. */
  readSection(rows: { w: string[]; at: number }[]): void {
    for (const { w, at } of rows) {
      if (w.length !== 7) throw new StyleError(`data file line ${at}: BiTorsions line needs 7 values (index, type, five atom IDs)`);
      const nums = w.map((t) => Number(t));
      if (nums.some((v) => !Number.isInteger(v))) throw new StyleError(`data file line ${at}: BiTorsions values must be integers`);
      this.terms.push({ type: nums[1], ids: nums.slice(2) });
    }
  }

  energy(): number { return this.etotal; }
  computeScalar(): number { return this.etotal; }
  minPostForce(): void { this.postForce(); }

  private pairFlag(): boolean {
    const pair = this.sys.ff.pair as unknown as { extract?(name: string): unknown } | null;
    if (!pair || typeof pair.extract !== 'function') throw new StyleError('Cannot use fix amoeba/bitorsion w/out pair amoeba/hippo');
    const flag = pair.extract('bitorsion_flag');
    if (flag === undefined) throw new StyleError('Cannot use fix amoeba/bitorsion w/out pair amoeba/hippo');
    return Boolean(flag);
  }

  postForce(): void {
    const s: SimState = this.sys.state;
    const f = s.f;
    const v = this.virial;
    v.fill(0);
    this.etotal = 0;
    if (!this.pairFlag()) return;
    const pair = this.sys.ff.pair as unknown as { extract(name: string): unknown };
    const atomicNum = pair.extract('atomic_num') as Map<number, number> | undefined;
    const amtype = s.custom.get('amtype');
    if (!atomicNum || !amtype) throw new StyleError('fix amoeba/bitorsion needs the pair style amoeba atom types and atomic numbers');
    const shim = { s, geom: this.sys.geom } as unknown as BondedCompute;
    const map = buildAtomMap(s);
    const idx = (id: number): number => {
      const i = id < map.length ? map[id] : -1;
      if (i < 0) throw new StyleError(`fix amoeba/bitorsion: atom ${id} missing`);
      return i;
    };
    const ctx: ChiralityContext = {
      shim, partners: partnerMap(shim), idx,
      typeOf: (id: number): number => amtype.data[idx(id)],
      atomicOf: (t: number): number => atomicNum.get(t) ?? 0,
    };
    const d = [0, 0, 0];
    for (const term of this.terms) {
      const a = term.ids.map(idx);
      let owner = a[0];
      for (const k of a) if (k < owner) owner = k;
      // positions relative to the owner (nearest copies); every atom of the term takes its share
      const rel: number[][] = [];
      for (let k = 0; k < 5; k++) {
        delta(shim, owner, a[k], d);
        rel.push([d[0], d[1], d[2]]);
      }
      const r = this.bitorsionTerm(rel, term.type, term.ids, ctx);
      if (r === null) continue;
      this.etotal += r.e;
      for (let k = 0; k < 5; k++) {
        const g = r.g[k];
        f[3 * a[k]] -= g[0];
        f[3 * a[k] + 1] -= g[1];
        f[3 * a[k] + 2] -= g[2];
      }
      for (let c = 0; c < 6; c++) v[c] += r.v[c];
    }
  }

  /**
   * The torsion-torsion energy and forces of one term (fix_amoeba_bitorsion.cpp post_force), with the five atoms' positions
   * rel relative to the owner. Returns the energy, the forces per atom in native order (ia, ib, ic, id, ie) and the virial
   * of the native tally (xx, yy, zz, xy, xz, yz); null for a degenerate geometry.
   */
  private bitorsionTerm(
    rel: number[][], btype: number, ids: number[], ctx: ChiralityContext,
  ): { e: number; g: number[][]; v: number[] } | null {
    const [ia, ib, ic, id, ie] = rel;
    const xia = ia[0], yia = ia[1], zia = ia[2];
    const xib = ib[0], yib = ib[1], zib = ib[2];
    const xic = ic[0], yic = ic[1], zic = ic[2];
    const xid = id[0], yid = id[1], zid = id[2];
    const xie = ie[0], yie = ie[1], zie = ie[2];
    const xba = xib - xia, yba = yib - yia, zba = zib - zia;
    const xcb = xic - xib, ycb = yic - yib, zcb = zic - zib;
    const xdc = xid - xic, ydc = yid - yic, zdc = zid - zic;
    const xed = xie - xid, yed = yie - yid, zed = zie - zid;
    const xt = yba * zcb - ycb * zba, yt = zba * xcb - zcb * xba, zt = xba * ycb - xcb * yba;
    const xu = ycb * zdc - ydc * zcb, yu = zcb * xdc - zdc * xcb, zu = xcb * ydc - xdc * ycb;
    const rt2 = xt * xt + yt * yt + zt * zt;
    const ru2 = xu * xu + yu * yu + zu * zu;
    const rtru = Math.sqrt(rt2 * ru2);
    const xv = ydc * zed - yed * zdc, yv = zdc * xed - zed * xdc, zv = xdc * yed - xed * ydc;
    const rv2 = xv * xv + yv * yv + zv * zv;
    const rurv = Math.sqrt(ru2 * rv2);
    if (rtru <= 0.0 || rurv <= 0.0) return null;
    const rcb = Math.sqrt(xcb * xcb + ycb * ycb + zcb * zcb);
    let cosine1 = (xt * xu + yt * yu + zt * zu) / rtru;
    cosine1 = Math.min(1.0, Math.max(-1.0, cosine1));
    let angle1 = RAD2DEG * libmAcos(cosine1);
    let dot = xba * xu + yba * yu + zba * zu;
    if (dot < 0.0) angle1 = -angle1;
    const rdc = Math.sqrt(xdc * xdc + ydc * ydc + zdc * zdc);
    let cosine2 = (xu * xv + yu * yv + zu * zv) / rurv;
    cosine2 = Math.min(1.0, Math.max(-1.0, cosine2));
    let angle2 = RAD2DEG * libmAcos(cosine2);
    dot = xcb * xv + ycb * yv + zcb * zv;
    if (dot < 0.0) angle2 = -angle2;
    const grid = this.grids[btype - 1];
    if (!grid) throw new StyleError(`fix amoeba/bitorsion: type ${btype} has no grid in the file`);
    const sp = this.splines[btype - 1];
    const ch = chirality(ctx, ids[2], ids[1], ids[3], angle1, angle2);
    const sign = ch.sign;
    const value1 = ch.value1, value2 = ch.value2;
    const nx = grid.nx, ny = grid.ny;
    // the cell containing (value1, value2): bisection on the node coordinates, as native
    let nlo = 0, nhi = nx - 1;
    while (nhi - nlo > 1) {
      const nt = Math.floor((nhi + nlo) / 2);
      if (grid.ttx[nt] > value1) nhi = nt;
      else nlo = nt;
    }
    const xlo = nlo;
    nlo = 0;
    nhi = ny - 1;
    while (nhi - nlo > 1) {
      const nt = Math.floor((nhi + nlo) / 2);
      if (grid.tty[nt] > value2) nhi = nt;
      else nlo = nt;
    }
    const ylo = nlo;
    const x1l = grid.ttx[xlo], x1u = grid.ttx[xlo + 1];
    const y1l = grid.tty[ylo], y1u = grid.tty[ylo + 1];
    const pos2 = (ylo + 1) * nx + xlo;
    const pos1 = pos2 - nx;
    const ftt = [grid.tbf[pos1], grid.tbf[pos1 + 1], grid.tbf[pos2 + 1], grid.tbf[pos2]];
    const ft1 = [sp.tbx[pos1], sp.tbx[pos1 + 1], sp.tbx[pos2 + 1], sp.tbx[pos2]];
    const ft2 = [sp.tby[pos1], sp.tby[pos1 + 1], sp.tby[pos2 + 1], sp.tby[pos2]];
    const ft12 = [sp.tbxy[pos1], sp.tbxy[pos1 + 1], sp.tbxy[pos2 + 1], sp.tbxy[pos2]];
    const bi = bcuint1(ftt, ft1, ft2, ft12, x1l, x1u, y1l, y1u, value1, value2);
    const e = bi.ansy;
    const dedang1 = sign * RAD2DEG * bi.ansy1;
    const dedang2 = sign * RAD2DEG * bi.ansy2;
    // derivatives of the first angle (around the torsion 1-2-3-4) on atoms ia..id
    const xca = xic - xia, yca = yic - yia, zca = zic - zia;
    const xdb = xid - xib, ydb = yid - yib, zdb = zid - zib;
    const dedxt = dedang1 * (yt * zcb - ycb * zt) / (rt2 * rcb);
    const dedyt = dedang1 * (zt * xcb - zcb * xt) / (rt2 * rcb);
    const dedzt = dedang1 * (xt * ycb - xcb * yt) / (rt2 * rcb);
    const dedxu = -dedang1 * (yu * zcb - ycb * zu) / (ru2 * rcb);
    const dedyu = -dedang1 * (zu * xcb - zcb * xu) / (ru2 * rcb);
    const dedzu = -dedang1 * (xu * ycb - xcb * yu) / (ru2 * rcb);
    const dedxia = zcb * dedyt - ycb * dedzt;
    const dedyia = xcb * dedzt - zcb * dedxt;
    const dedzia = ycb * dedxt - xcb * dedyt;
    const dedxib = yca * dedzt - zca * dedyt + zdc * dedyu - ydc * dedzu;
    const dedyib = zca * dedxt - xca * dedzt + xdc * dedzu - zdc * dedxu;
    const dedzib = xca * dedyt - yca * dedxt + ydc * dedxu - xdc * dedyu;
    const dedxic = zba * dedyt - yba * dedzt + ydb * dedzu - zdb * dedyu;
    const dedyic = xba * dedzt - zba * dedxt + zdb * dedxu - xdb * dedzu;
    const dedzic = yba * dedxt - xba * dedyt + xdb * dedyu - ydb * dedxu;
    const dedxid = zcb * dedyu - ycb * dedzu;
    const dedyid = xcb * dedzu - zcb * dedxu;
    const dedzid = ycb * dedxu - xcb * dedyu;
    // derivatives of the second angle (around 2-3-4-5) on atoms ib..ie
    const xec = xie - xic, yec = yie - yic, zec = zie - zic;
    const dedxu2 = dedang2 * (yu * zdc - ydc * zu) / (ru2 * rdc);
    const dedyu2 = dedang2 * (zu * xdc - zdc * xu) / (ru2 * rdc);
    const dedzu2 = dedang2 * (xu * ydc - xdc * yu) / (ru2 * rdc);
    const dedxv2 = -dedang2 * (yv * zdc - ydc * zv) / (rv2 * rdc);
    const dedyv2 = -dedang2 * (zv * xdc - zdc * xv) / (rv2 * rdc);
    const dedzv2 = -dedang2 * (xv * ydc - xdc * yv) / (rv2 * rdc);
    const dedxib2 = zdc * dedyu2 - ydc * dedzu2;
    const dedyib2 = xdc * dedzu2 - zdc * dedxu2;
    const dedzib2 = ydc * dedxu2 - xdc * dedyu2;
    const dedxic2 = ydb * dedzu2 - zdb * dedyu2 + zed * dedyv2 - yed * dedzv2;
    const dedyic2 = zdb * dedxu2 - xdb * dedzu2 + xed * dedzv2 - zed * dedxv2;
    const dedzic2 = xdb * dedyu2 - ydb * dedxu2 + yed * dedxv2 - xed * dedyv2;
    const dedxid2 = zcb * dedyu2 - ycb * dedzu2 + yec * dedzv2 - zec * dedyv2;
    const dedyid2 = xcb * dedzu2 - zcb * dedxu2 + zec * dedxv2 - xec * dedzv2;
    const dedzid2 = ycb * dedxu2 - xcb * dedyu2 + xec * dedyv2 - yec * dedxv2;
    const dedxie2 = zdc * dedyv2 - ydc * dedzv2;
    const dedyie2 = xdc * dedzv2 - zdc * dedxv2;
    const dedzie2 = ydc * dedxv2 - xdc * dedyv2;
    const g = [
      [dedxia, dedyia, dedzia],
      [dedxib + dedxib2, dedyib + dedyib2, dedzib + dedzib2],
      [dedxic + dedxic2, dedyic + dedyic2, dedzic + dedzic2],
      [dedxid + dedxid2, dedyid + dedyid2, dedzid + dedzid2],
      [dedxie2, dedyie2, dedzie2],
    ];
    // virial tally of native post_force (ev_tally with nctotal 5)
    const vxx = xcb * (dedxic + dedxid) - xba * dedxia + xdc * dedxid;
    const vyx = ycb * (dedxic + dedxid) - yba * dedxia + ydc * dedxid;
    const vzx = zcb * (dedxic + dedxid) - zba * dedxia + zdc * dedxid;
    const vyy = ycb * (dedyic + dedyid) - yba * dedyia + ydc * dedyid;
    const vzy = zcb * (dedyic + dedyid) - zba * dedyia + zdc * dedyid;
    const vzz = zcb * (dedzic + dedzid) - zba * dedzia + zdc * dedzid;
    const vxx2 = xdc * (dedxid2 + dedxie2) - xcb * dedxib2 + xed * dedxie2;
    const vyx2 = ydc * (dedxid2 + dedxie2) - ycb * dedxib2 + yed * dedxie2;
    const vzx2 = zdc * (dedxid2 + dedxie2) - zcb * dedxib2 + zed * dedxie2;
    const vyy2 = ydc * (dedyid2 + dedyie2) - ycb * dedyib2 + yed * dedyie2;
    const vzy2 = zdc * (dedyid2 + dedyie2) - zcb * dedyib2 + zed * dedyie2;
    const vzz2 = zdc * (dedzid2 + dedzie2) - zcb * dedzib2 + zed * dedzie2;
    const v = [
      -vxx - vxx2, -vyy - vyy2, -vzz - vzz2, -vyx - vyx2, -vzx - vzx2, -vzy - vzy2,
    ];
    return { e, g, v };
  }
}

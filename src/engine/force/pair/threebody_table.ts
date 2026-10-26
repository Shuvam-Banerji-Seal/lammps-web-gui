import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum, joinPotentialEntries } from '../util';
import { libm } from '../../libm';

/*
 * pair_style threebody/table — docs.lammps.org/pair_threebody_table.html
 * (source: plans/lammps-docs-live/pair_threebody_table.rst)
 *
 * Syntax: pair_style threebody/table, with no further arguments documented; any argument is refused here.
 *
 * Energy: the three-body sum over the neighbours of each center. "In contrast to the Stillinger-Weber potential, all
 * forces are not calculated analytically, but read in from a three-body force/energy table" (the page continues
 * with the .3b file).
 *
 * Potential file (.3b): "Lines that are not blank or comments (starting with #) define parameters for a triplet of
 * elements." An entry holds element 1 (the center), element 2, element 3, the cut, the filename, the keyword, the
 * style and N. "When set to 0, no interaction is calculated for this element triplet." "At the moment, only the
 * style linear is allowed and implemented." Any other style is refused by name.
 *
 * Table grid: "The tabulation is done on a three-dimensional grid of the two distances r_{ij} and r_{ik} as well as
 * the angle theta_{ijk}". The angle is always varied in 2N steps from (0.0 + 180.0/(4N)) to
 * (180.0 - 180.0/(4N)). The entry count is M = N * N * (N+1) for element 2 == element 3 and M = 2 * N * N * N
 * otherwise (the page's two cases).
 *
 * Table section: the N parameter "It should be the same than the parameter" (the .3b N), "otherwise its value is
 * overwritten." Measured with native LAMMPS (black box): a .3b file with N = 4 and a table section with N = 12 gives
 * the N = 12 forces, so the table's N is the one used. The lines are read by position (see the layout below); the r
 * and theta columns define the grid. Measured with native LAMMPS (black box): a symmetric entry read from a file in
 * the general 2N^3 layout uses the row of the symmetric layout at the same grid position, so the lookup uses the
 * positional layout, not a search of the r and theta columns.
 *
 * Lookup: "For each triplet configuration occurring in the simulation within the cutoff distance, the next nearest
 * tabulated triplet configuration is looked up." "No interpolation is done." Measured with native LAMMPS (black box):
 * the nearest grid point is taken for both distances and for the angle, on the grid values of the file; an exact tie
 * goes to the lower grid point (r = 2.8 with rmin 2.55 and dr 0.1 gives 2.75; theta = 90 degrees with N = 12 gives
 * 86.25; r = 2.9 gives 2.85).
 *
 * Forces: "The forces on all three particles I, J, and K of a triplet of this type of three-body interaction potential"
 * lie in the plane of the three distance vectors, and the six tabulated constants project them: f_i = f_i1 r_ij +
 * f_i2 r_ik, f_j = f_j1 r_ij + f_j2 r_jk, f_k = f_k1 r_ik + f_k2 r_jk. Measured with native LAMMPS (black box):
 * r_ij = x_j - x_i, r_ik = x_k - x_i, r_jk = x_k - x_j, and the three forces are added as given.
 *
 * Energy: "as well as the energy of a triplet e." Measured with native LAMMPS (black box): a single triplet gives
 * exactly its table energy, counted once.
 *
 * Triplet selection (measured with native LAMMPS, black box): the nearer of the two neighbours is j, so the table's
 * r_ij is the smaller distance (the symmetric layout requires it). The table is the entry of the triplet (center, j,
 * k); a cut of 0 there excludes it. Each leg is cut with the cut of the entry (center, leg, leg), the Stillinger-Weber
 * convention for the screening of a leg: a leg j is used when r_ij is below the cut of (i, j, j), and a distance equal
 * to the cut excludes it. A leg distance more than half a grid step outside [rmin, rmax] of the table drops the
 * triplet. Measured with native LAMMPS: with the (2, 2, 2) entry at cut 0, every triplet with a type 2 center and a
 * leg of an asymmetric entry was dropped, and raising that cut restored them.
 *
 * This pair style does not support the pair_modify shift, table, and tail options. It needs newton on (the engine
 * always evaluates each physical triplet once).
 */

/** One .3b entry: element triplet, cutoff, and the table reference. */
interface Entry3B {
  e1: string;
  e2: string;
  e3: string;
  cut: number;
  filename: string;
  keyword: string;
  style: string;
  n: number;
}

/** A built table: layout parameters and the seven values of every grid point (fi1 fi2 fj1 fj2 fk1 fk2 e). */
interface TriTable {
  cut: number;
  n: number;
  rmin: number;
  rmax: number;
  /** r grid (n values, the r_ij column of the file at b = a) and angle grid (2n values, degrees) */
  gr: Float64Array;
  gt: Float64Array;
  /** r grid step (nominal) and angle grid step (degrees) */
  dr: number;
  dth: number;
  /** symmetric (element 2 == element 3) layout: rows of (a, b >= a, m) */
  sym: boolean;
  /** symmetric layout: first row of each r_ij index a */
  off: Int32Array;
  rows: Float64Array;
}

/** Parses the .3b file: entries of 8 words (3 element names, cut, filename, keyword, style, N). */
const parse3B = (text: string, fileName: string): Entry3B[] => {
  const lines = joinPotentialEntries(text.split(/\r?\n/), 8);
  const out: Entry3B[] = [];
  for (const raw of lines) {
    const t = raw.trim();
    if (t === '' || t.startsWith('#')) continue;
    const w = t.split(/\s+/);
    if (w.length !== 8) {
      throw new StyleError(`threebody/table potential file ${fileName}: entry '${t}' needs 8 values: element1 element2 element3 cut filename keyword style N`);
    }
    out.push({
      e1: w[0], e2: w[1], e3: w[2],
      cut: parseNum(w[3], 'threebody/table cut'),
      filename: w[4], keyword: w[5], style: w[6],
      n: parseNum(w[7], 'threebody/table N'),
    });
  }
  if (out.length === 0) throw new StyleError(`threebody/table potential file ${fileName} contains no entries`);
  return out;
};

/** Reads the table section keyword of a threebody/table file and builds its lookup layout. */
const buildTable = (text: string, entry: Entry3B, sym: boolean): TriTable => {
  const fileName = entry.filename;
  const lines = text.split(/\r?\n/);
  let k = 0;
  while (k < lines.length) {
    const header = lines[k].trim();
    k++;
    if (header === '' || header.startsWith('#')) continue;
    const kw = header.split(/\s+/)[0];
    let params = '';
    while (k < lines.length) {
      const l = lines[k].trim();
      k++;
      if (l !== '' && !l.startsWith('#')) {
        params = l;
        break;
      }
    }
    if (params === '') throw new StyleError(`threebody/table table ${fileName}: section ${kw} has no parameter line`);
    const pw = params.split(/\s+/);
    let n = 0, rmin = NaN, rmax = NaN;
    for (let p = 0; p < pw.length; ) {
      const key = pw[p++];
      if (key === 'N') n = parseNum(pw[p++], 'table N');
      else if (key === 'rmin') rmin = parseNum(pw[p++], 'table rmin');
      else if (key === 'rmax') rmax = parseNum(pw[p++], 'table rmax');
      else throw new StyleError(`threebody/table table ${fileName}: section ${kw} has unknown parameter '${key}' (N, rmin and rmax)`);
    }
    if (!Number.isInteger(n) || n < 2) throw new StyleError(`threebody/table table ${fileName}: section ${kw} needs an integer N >= 2, got ${n}`);
    if (!(rmax > rmin)) throw new StyleError(`threebody/table table ${fileName}: section ${kw} needs rmin < rmax`);
    const M = sym ? n * n * (n + 1) : 2 * n * n * n;
    const rows = new Float64Array(M * 7);
    let got = 0;
    const body: number[][] = [];
    while (got < M && k < lines.length) {
      const l = lines[k].trim();
      k++;
      if (l === '' || l.startsWith('#')) continue;
      const t = l.split(/\s+/);
      if (t.length !== 11) throw new StyleError(`threebody/table table ${fileName}: data line '${l}' needs 11 values (index r_ij r_ik theta f_i1 f_i2 f_j1 f_j2 f_k1 f_k2 e)`);
      const vals = t.map((v, c) => parseNum(v, `table value ${c + 1}`));
      body.push(vals);
      got++;
    }
    if (got < M) throw new StyleError(`threebody/table table ${fileName}: section ${kw} has ${got} of ${M} data lines (N = ${n}, ${sym ? 'symmetric' : 'general'} layout)`);
    if (kw === entry.keyword) {
      for (let m = 0; m < M; m++) {
        const v = body[m];
        for (let c = 0; c < 6; c++) rows[m * 7 + c] = v[4 + c];
        rows[m * 7 + 6] = v[10];
      }
      const off = new Int32Array(n);
      for (let a = 1; a < n; a++) off[a] = off[a - 1] + (n - (a - 1)) * 2 * n;
      // the grid values as the file gives them: r_ij and theta of the grid points (decimal values, see gridIndexIn)
      const gr = new Float64Array(n), gt = new Float64Array(2 * n);
      for (let a = 0; a < n; a++) gr[a] = body[(sym ? off[a] : a * n * 2 * n)][1];
      for (let m = 0; m < 2 * n; m++) gt[m] = body[m][3];
      return { cut: entry.cut, n, rmin, rmax, gr, gt, dr: (rmax - rmin) / (n - 1), dth: 180 / (2 * n), sym, off, rows };
    }
  }
  throw new StyleError(`threebody/table table ${fileName} has no section with keyword '${entry.keyword}'`);
};

/**
 * Nearest point of the grid g (ascending) to the value r: the point with the smaller |r - g_i|, an exact tie going to
 * the lower point. Returns -1 when r lies more than half a grid step outside the grid: a distance up to half a step
 * beyond rmin or rmax still maps to the end point (measured with native LAMMPS, black box: 1.85 accepted and 1.8167
 * dropped for rmin = 2 and a step of 1/3). The grid values are the decimal values of the file: a distance on a
 * half-point (r = 2.9 between 2.85 and 2.95) is decided by |r - g| with those values, and native picks 2.85
 * (measured), so the file values are compared, not values computed as rmin + i dr.
 */
const gridIndexIn = (r: number, g: Float64Array, step: number): number => {
  const n = g.length;
  const first = g[0];
  const x = (r - first) / step;
  if (x <= -0.5 || x > n - 0.5) return -1;
  const lo = Math.min(Math.max(Math.floor(x), 0), n - 1);
  const hi = Math.min(lo + 1, n - 1);
  if (lo === hi) return lo;
  return Math.abs(r - g[lo]) <= Math.abs(r - g[hi]) ? lo : hi;
};

/**
 * pair_style threebody/table. Full neighbor list; per owned center i, each pair of neighbours (j, k) within the
 * cutoff is one triplet, evaluated once with the nearer neighbour as j (see the header).
 */
export class PairThreebodyTable extends Pair {
  readonly name = 'threebody/table';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private fileRead = false;
  private elemOf = new Int32Array(0);
  private elemNames: string[] = [];
  private fileElems = new Set<string>();
  /** tables per (ti, tj, tk) type triple, null when the entry has cut 0 or no triplet */
  private tabOf: (TriTable | null)[] = [];
  /** leg cut per (center, leg type): the cut of the entry (center, leg, leg) */
  private legCut = new Float64Array(0);
  private maxCut = 0;
  private tableCache = new Map<string, TriTable>();
  private cap = 0;
  private nIdx = new Int32Array(0);
  private nType = new Int32Array(0);
  private nDx = new Float64Array(0);
  private nDy = new Float64Array(0);
  private nDz = new Float64Array(0);
  private nR = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError(`pair_style threebody/table takes no arguments (got '${args[0]}')`);
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.elemOf = new Int32Array(nt).fill(-1);
    this.elemNames = [];
    this.fileRead = false;
    this.fileElems.clear();
    this.tabOf = new Array<TriTable | null>(nt * nt * nt).fill(null);
    this.legCut = new Float64Array(nt * nt);
    this.maxCut = 0;
    this.tableCache.clear();
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    // Pair::map_element2type(narg - 3, ...) runs first: fewer than three words is a wrong element count
    if (args.length < 3) throw new StyleError('Number of element to type mappings does not match number of atom types');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style threebody/table must be * *');
    const filename = args[2];
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    if (this.fileRead) throw new StyleError('pair_coeff for style threebody/table can only be given once (one .3b file)');
    const entries = parse3B(ctx.readFile(filename, potentialFile('threebody')), filename);
    this.fileRead = true;
    for (const e of entries) {
      this.fileElems.add(e.e1);
      this.fileElems.add(e.e2);
      this.fileElems.add(e.e3);
    }
    const nt = this.ntypes + 1;
    this.elemNames = [];
    for (let t = 1; t <= this.ntypes; t++) {
      const name = elems[t - 1];
      if (name === 'NULL') {
        this.elemOf[t] = -1;
        continue;
      }
      if (!this.fileElems.has(name)) {
        throw new StyleError(`element '${name}' is not in threebody/table potential file ${filename} (elements: ${[...this.fileElems].sort().join(' ')})`);
      }
      let idx = this.elemNames.indexOf(name);
      if (idx < 0) {
        idx = this.elemNames.length;
        this.elemNames.push(name);
      }
      this.elemOf[t] = idx;
    }
    // every triplet of mapped elements needs its entry; a cut of 0 means no interaction
    const byKey = new Map<string, Entry3B>();
    for (const e of entries) {
      const key = `${e.e1} ${e.e2} ${e.e3}`;
      if (byKey.has(key)) throw new StyleError(`threebody/table potential file ${filename}: duplicate entry for elements ${key}`);
      byKey.set(key, e);
    }
    this.maxCut = 0;
    this.legCut = new Float64Array(nt * nt);
    for (let ti = 1; ti <= this.ntypes; ti++) {
      if (this.elemOf[ti] < 0) continue;
      for (let tj = 1; tj <= this.ntypes; tj++) {
        if (this.elemOf[tj] < 0) continue;
        for (let tk = 1; tk <= this.ntypes; tk++) {
          if (this.elemOf[tk] < 0) continue;
          const n1 = this.elemNames[this.elemOf[ti]], n2 = this.elemNames[this.elemOf[tj]], n3 = this.elemNames[this.elemOf[tk]];
          const e = byKey.get(`${n1} ${n2} ${n3}`);
          if (!e) throw new StyleError(`threebody/table potential file ${filename} has no entry for elements ${n1} ${n2} ${n3} (needed by types ${ti} ${tj} ${tk})`);
          if (e.cut < 0) throw new StyleError(`threebody/table potential file ${filename} entry ${n1} ${n2} ${n3}: cut must be >= 0`);
          if (e.style !== 'linear') {
            throw new StyleError(`threebody/table potential file ${filename} entry ${n1} ${n2} ${n3}: table style '${e.style}' is not supported (only linear)`);
          }
          // the leg cut of (center ti, leg tj) is the cut of the entry (ti, tj, tj)
          if (tj === tk) {
            this.legCut[ti * nt + tj] = e.cut;
          }
          const t3 = (ti * nt + tj) * nt + tk;
          if (e.cut === 0) continue;
          const tkey = `${e.filename}\u0000${e.keyword}\u0000${n2 === n3 ? 's' : 'g'}`;
          let tab = this.tableCache.get(tkey);
          if (!tab) {
            tab = buildTable(ctx.readFile(e.filename, potentialFile('threebodytable table')), e, n2 === n3);
            this.tableCache.set(tkey, tab);
          }
          this.tabOf[t3] = tab;
          if (e.cut > this.maxCut) this.maxCut = e.cut;
        }
      }
    }
  }

  override initOne(i: number, j: number): number {
    if (this.elemOf[i] < 0 || this.elemOf[j] < 0) return 0;
    return this.maxCut;
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const elemOf = this.elemOf;
    const tabOf = this.tabOf;
    const legCut = this.legCut;
    const eatom = pc.eatom;
    const vatom = pc.vatom;
    let evdwl = 0;

    let maxn = 0;
    for (let i = 0; i < list.inum; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) {
      this.cap = maxn;
      this.nIdx = new Int32Array(maxn);
      this.nType = new Int32Array(maxn);
      this.nDx = new Float64Array(maxn);
      this.nDy = new Float64Array(maxn);
      this.nDz = new Float64Array(maxn);
      this.nR = new Float64Array(maxn);
    }
    const nIdx = this.nIdx, nType = this.nType, nDx = this.nDx, nDy = this.nDy, nDz = this.nDz, nR = this.nR;

    for (let i = 0; i < nlocal; i++) {
      const tiv = type[i];
      if (elemOf[tiv] < 0) continue;
      const ti = tiv * nt;
      const k0 = list.firstneigh[i];
      const k1 = k0 + list.numneigh[i];
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        if (elemOf[tj] < 0) continue;
        const dxx = x[3 * j] - x[3 * i];
        const dyy = x[3 * j + 1] - x[3 * i + 1];
        const dzz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dxx * dxx + dyy * dyy + dzz * dzz;
        if (rsq >= cutsq[ti + tj]) continue;
        nIdx[m] = j;
        nType[m] = tj;
        nDx[m] = dxx; nDy[m] = dyy; nDz[m] = dzz;
        nR[m] = Math.sqrt(rsq);
        m++;
      }
      for (let p = 0; p < m; p++) {
        for (let q = p + 1; q < m; q++) {
          // the nearer neighbour is j (the table's r_ij is the smaller distance)
          const J = nR[p] <= nR[q] ? p : q;
          const K = J === p ? q : p;
          const tab = tabOf[(tiv * nt + nType[J]) * nt + nType[K]];
          if (!tab) continue;
          const rJ = nR[J], rK = nR[K];
          if (!(rJ < legCut[ti + nType[J]]) || !(rK < legCut[ti + nType[K]])) continue;
          const n = tab.n;
          const a = gridIndexIn(rJ, tab.gr, tab.dr);
          const b = gridIndexIn(rK, tab.gr, tab.dr);
          if (a < 0 || b < 0) continue;
          const dJx = nDx[J], dJy = nDy[J], dJz = nDz[J];
          const dKx = nDx[K], dKy = nDy[K], dKz = nDz[K];
          const cosT = (dJx * dKx + dJy * dKy + dJz * dKz) / (rJ * rK);
          const theta = (libm.acos(cosT > 1 ? 1 : cosT < -1 ? -1 : cosT) * 180) / Math.PI;
          const mm = gridIndexIn(theta, tab.gt, tab.dth);
          if (mm < 0) continue;
          const row = tab.sym ? tab.off[a] + (b - a) * 2 * n + mm : (a * n + b) * 2 * n + mm;
          const R = tab.rows;
          const o = 7 * row;
          const fi1 = R[o], fi2 = R[o + 1], fj1 = R[o + 2], fj2 = R[o + 3], fk1 = R[o + 4], fk2 = R[o + 5], e = R[o + 6];
          // r_jk = x_k - x_j
          const rjkx = dKx - dJx, rjky = dKy - dJy, rjkz = dKz - dJz;
          const fix = fi1 * dJx + fi2 * dKx, fiy = fi1 * dJy + fi2 * dKy, fiz = fi1 * dJz + fi2 * dKz;
          const fjx = fj1 * dJx + fj2 * rjkx, fjy = fj1 * dJy + fj2 * rjky, fjz = fj1 * dJz + fj2 * rjkz;
          const fkx = fk1 * dKx + fk2 * rjkx, fky = fk1 * dKy + fk2 * rjky, fkz = fk1 * dKz + fk2 * rjkz;
          const jj = nIdx[J], kk = nIdx[K];
          f[3 * i] += fix;
          f[3 * i + 1] += fiy;
          f[3 * i + 2] += fiz;
          f[3 * jj] += fjx;
          f[3 * jj + 1] += fjy;
          f[3 * jj + 2] += fjz;
          f[3 * kk] += fkx;
          f[3 * kk + 1] += fky;
          f[3 * kk + 2] += fkz;
          evdwl += e;
          if (eatom) {
            const e3 = e / 3;
            eatom[i] += e3;
            eatom[jj] += e3;
            eatom[kk] += e3;
          }
          if (vatom) {
            // virial of the triplet relative to the center: r_j (x) f_j + r_k (x) f_k, one third per atom
            const w0 = (dJx * fjx + dKx * fkx) / 3, w1 = (dJy * fjy + dKy * fky) / 3, w2 = (dJz * fjz + dKz * fkz) / 3;
            const w3 = (dJx * fjy + dKx * fky) / 3, w4 = (dJx * fjz + dKx * fkz) / 3, w5 = (dJy * fjz + dKy * fkz) / 3;
            const va = vatom;
            for (const at of [i, jj, kk]) {
              va[6 * at] += w0; va[6 * at + 1] += w1; va[6 * at + 2] += w2;
              va[6 * at + 3] += w3; va[6 * at + 4] += w4; va[6 * at + 5] += w5;
            }
          }
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}

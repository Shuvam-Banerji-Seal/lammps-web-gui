/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair_style polymorphic
 * (src/MANYBODY/pair_polymorphic.cpp, pair_polymorphic.h) and its tabular functions (src/tabular_function.cpp,
 * tabular_function.h). Contributing authors: Reese Jones, Xiaowang Zhou (SNL); the form modifies pair_tersoff.cpp by
 * Aidan Thompson (SNL). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_polymorphic.html. "pair_coeff * * file elem1 ... elemN [epsilon]". The file holds
 * tabulated U, V, W, P, G and F functions (cubic-spline interpolated by TabularFunction) of the element pairs and triplets.
 * The compute() follows native order: the two-body U term on half the pairs (by atom IDs), the bond-order zeta from the
 * W-neighbors (eta = 1 uses the self-energy form, otherwise the V-neighbors with the triplet P and G tables).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import {
  checkOneCoeffArgs, mapElementsToTypes, nextDouble, PotentialEntryReader, potentialUnitFactor, tagOf, tallyPair,
  tallyTriplet,
} from './common';

/**
 * TabularFunction (src/tabular_function.cpp at b332f28): a cubic-spline table on [xmin, xmax] with `size` points.
 * value() picks the nearest grid cell (std::lround, clamped) and evaluates the cell's cubic polynomial.
 */
export class TabularFunction {
  size = 0;
  xmin = 0;
  xmax = 0;
  xmaxsq = 0;
  rdx = 0;
  vmax = 0;
  private xs = new Float64Array(0);
  private ys = new Float64Array(0);
  private ys1 = new Float64Array(0);
  private ys2 = new Float64Array(0);
  private ys3 = new Float64Array(0);
  private ys4 = new Float64Array(0);
  private ys5 = new Float64Array(0);
  private ys6 = new Float64Array(0);

  /** TabularFunction::set_values(n, x1, x2, values). */
  setValues(n: number, x1: number, x2: number, values: Float64Array): void {
    if (n !== this.size) {
      this.size = n;
      this.xs = new Float64Array(n);
      this.ys = new Float64Array(n);
      this.ys1 = new Float64Array(n);
      this.ys2 = new Float64Array(n);
      this.ys3 = new Float64Array(n);
      this.ys4 = new Float64Array(n);
      this.ys5 = new Float64Array(n);
      this.ys6 = new Float64Array(n);
    }
    this.xmin = x1;
    this.xmax = x2;
    this.xmaxsq = x2 * x2;
    for (let i = 0; i < n; i++) this.ys[i] = values[i];
    this.initialize();
  }

  private initialize(): void {
    const n = this.size, ys = this.ys, ys1 = this.ys1, ys2 = this.ys2, ys3 = this.ys3;
    this.rdx = (this.xmax - this.xmin) / (n - 1.0);
    this.vmax = 0.0;
    for (let i = 0; i < n; i++) if (Math.abs(ys[i]) > this.vmax) this.vmax = Math.abs(ys[i]);
    for (let i = 0; i < n; i++) this.xs[i] = this.xmin + i * this.rdx;
    this.rdx = 1.0 / this.rdx;
    ys1[0] = ys[1] - ys[0];
    ys1[1] = 0.5 * (ys[2] - ys[0]);
    ys1[n - 2] = 0.5 * (ys[n - 1] - ys[n - 3]);
    ys1[n - 1] = ys[n - 1] - ys[n - 2];
    for (let i = 2; i < n - 2; i++) ys1[i] = ((ys[i - 2] - ys[i + 2]) + 8.0 * (ys[i + 1] - ys[i - 1])) / 12.0;
    for (let i = 0; i < n - 1; i++) {
      ys2[i] = 3.0 * (ys[i + 1] - ys[i]) - 2.0 * ys1[i] - ys1[i + 1];
      ys3[i] = ys1[i] + ys1[i + 1] - 2.0 * (ys[i + 1] - ys[i]);
    }
    ys2[n - 1] = 0.0;
    ys3[n - 1] = 0.0;
    for (let i = 0; i < n; i++) {
      this.ys4[i] = ys1[i] * this.rdx;
      this.ys5[i] = 2.0 * ys2[i] * this.rdx;
      this.ys6[i] = 3.0 * ys3[i] * this.rdx;
    }
  }

  /** The cell index of x: std::lround of the scaled position, clamped to the table (TabularFunction::value). */
  private cell(x: number): { ks: number; ps: number } {
    const ps0 = (x - this.xmin) * this.rdx;
    let ks = lround(ps0);
    if (ks > this.size - 1) ks = this.size - 1;
    if (ks < 0) ks = 0;
    return { ks, ps: ps0 - ks };
  }

  /** The function value at x (value()'s y output). */
  valY(x: number): number {
    const { ks, ps } = this.cell(x);
    return ((this.ys3[ks] * ps + this.ys2[ks]) * ps + this.ys1[ks]) * ps + this.ys[ks];
  }

  /** The derivative at x (value()'s y1 output). */
  valD(x: number): number {
    const { ks, ps } = this.cell(x);
    return (this.ys6[ks] * ps + this.ys5[ks]) * ps + this.ys4[ks];
  }
}

/** std::lround: the nearest integer, halves away from zero. */
const lround = (v: number): number => {
  const a = Math.abs(v);
  let r = Math.floor(a);
  if (a - r >= 0.5) r += 1;
  return v < 0 ? -r : r;
};

/** PairPolymorphic::PairParameters: one element pair (U, V, W, F tables and the exponent xi). */
interface PairP {
  cut: number;
  cutsq: number;
  xi: number;
  U: TabularFunction;
  V: TabularFunction;
  W: TabularFunction;
  F: TabularFunction;
}

/** PairPolymorphic::TripletParameters: the P and G tables of an element triplet. */
interface TripP {
  P: TabularFunction;
  G: TabularFunction;
}

/** utils::numeric: the optional epsilon argument of pair_coeff (default 0). */
const numericArg = (w: string): number => {
  if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(w)) {
    throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  }
  return Number(w);
};

/**
 * PotentialFileReader::next_values / TextFileReader::next_dvector of the polymorphic file: a table of n values is read
 * line by line, the words of a line beyond the count are not used (native next_dvector).
 */
class PolyReader {
  readonly entries: PotentialEntryReader;
  constructor(lines: readonly string[]) { this.entries = new PotentialEntryReader(lines, 'polymorphic'); }

  values(nparams: number): string[] {
    const w = this.entries.next(nparams);
    if (!w) throw new StyleError('Missing line in polymorphic file!');
    return w;
  }

  dvector(n: number): Float64Array {
    const list = new Float64Array(n);
    let i = 0;
    while (i < n) {
      const w = this.entries.next(1);
      if (!w) {
        if (i === 0) throw new StyleError('EOF reached');
        throw new StyleError(`Incorrect format in polymorphic file! ${i}/${n} values`);
      }
      for (let k = 0; k < w.length && i < n; k++) list[i++] = nextDouble(w[k]);
    }
    return list;
  }
}

const nextInt = (w: string | undefined): number => {
  if (w === undefined) throw new StyleError('No more tokens');
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Not a valid integer number: '${w}'`);
  return Number(w);
};

export class PairPolymorphic extends Pair {
  override readonly name: string = 'polymorphic';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected elements: string[] = [];
  protected map: Int32Array = new Int32Array(0);
  /** pair_coeff optional epsilon (default 0). */
  protected epsilon = 0;
  protected eta = 0;
  protected maxX = 0;
  protected nr = 0;
  protected ng = 0;
  protected nx = 0;
  protected pairs: PairP[] = [];
  protected triples: TripP[] = [];
  /** file element index -> element index of the pair_coeff list (match in the native code). */
  protected match: number[] = [];
  /** elem2param[a*N + b] and elem3param[(a*N + b)*N + c]: parameter indices (setup_params). */
  protected elem2param = new Int32Array(0);
  protected elem3param = new Int32Array(0);
  protected cutmax = 0;
  protected cutmaxsq = 0;
  protected fileRead = false;

  override settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    let a = args;
    if (a.length === 4 + this.ntypes) {
      this.epsilon = numericArg(a[a.length - 1]);
      a = a.slice(0, -1);
    }
    checkOneCoeffArgs(a, this.ntypes, this.name);
    const { elements, map } = mapElementsToTypes(a.slice(3), this.ntypes);
    this.elements = elements;
    this.map = map;
    this.readFile(a[2], ctx);
    this.setupParams();
    this.fileRead = true;
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style polymorphic requires newton pair on');
  }

  override initOne(i: number, j: number): number {
    if (!this.fileRead || this.map[i] < 0 || this.map[j] < 0) throw new StyleError('All pair coeffs are not set');
    return this.cutmax;
  }

  /** PairPolymorphic::read_file: header, element match, pair parameters, then the tables in native order. */
  protected readFile(fileName: string, ctx: StyleContext): void {
    const nel = this.elements.length;
    const text = ctx.readFile(fileName);
    // PotentialFileReader without unit_convert: a units tag other than the unit style is an error (open_potential)
    potentialUnitFactor(text, 'polymorphic', fileName, ctx, false);
    const reader = new PolyReader(text.split(/\r?\n/));
    const hdr = reader.values(2);
    const ntypes = nextInt(hdr[0]);
    if (ntypes !== nel) {
      throw new StyleError(`Incorrect number of elements (expected: ${nel} found: ${ntypes}) in potential file`);
    }
    this.eta = nextInt(hdr[1]);
    this.match = new Array<number>(nel).fill(0);
    for (let i = 0; i < nel; i++) {
      const w = reader.values(3);
      nextDouble(w[0]); // atomic number
      nextDouble(w[1]); // atomic mass
      const name = w[2];
      const j = this.elements.indexOf(name);
      if (j < 0) throw new StyleError(`Element ${name} in potential file not used`);
      this.match[i] = j;
    }
    const h4 = reader.values(4);
    this.nr = nextInt(h4[0]);
    this.ng = nextInt(h4[1]);
    this.nx = nextInt(h4[2]);
    this.maxX = nextDouble(h4[3]);
    if (this.ng === 0 || this.nr === 0 || this.nx === 0) throw new StyleError('Error reading potential file header');

    const npair = nel * (nel + 1) / 2;
    const ntriple = nel * nel * nel;
    const pairs: PairP[] = [];
    for (let i = 0; i < npair; i++) {
      const w = reader.values(2);
      const cut = nextDouble(w[0]);
      const xi = nextDouble(w[1]);
      pairs.push({
        cut, cutsq: cut * cut, xi,
        U: new TabularFunction(), V: new TabularFunction(), W: new TabularFunction(), F: new TabularFunction(),
      });
    }
    this.pairs = pairs;
    this.triples = [];
    for (let i = 0; i < ntriple; i++) this.triples.push({ P: new TabularFunction(), G: new TabularFunction() });

    const nr = this.nr;
    for (let i = 0; i < npair; i++) pairs[i].U.setValues(nr, 0.0, pairs[i].cut, reader.dvector(nr)); // U
    for (let i = 0; i < npair; i++) pairs[i].V.setValues(nr, 0.0, pairs[i].cut, reader.dvector(nr)); // V
    for (let i = 0; i < npair; i++) pairs[i].W.setValues(nr, 0.0, pairs[i].cut, reader.dvector(nr)); // W
    this.cutmax = 0.0;
    for (let i = 0; i < npair; i++) if (pairs[i].cut > this.cutmax) this.cutmax = pairs[i].cut;
    this.cutmaxsq = this.cutmax * this.cutmax;

    if (this.eta !== 3) {
      for (let j = 0; j < nel; j++) { // P of (i, j, j)
        const table = reader.dvector(nr);
        for (let i = 0; i < nel; i++) {
          this.triples[i * nel * nel + j * nel + j].P.setValues(nr, -this.cutmax, this.cutmax, table);
        }
      }
      for (let j = 0; j < nel - 1; j++) { // P of (i, j, k) and (i, k, j), j < k
        for (let k = j + 1; k < nel; k++) {
          const table = reader.dvector(nr);
          for (let i = 0; i < nel; i++) {
            this.triples[i * nel * nel + j * nel + k].P.setValues(nr, -this.cutmax, this.cutmax, table);
            this.triples[i * nel * nel + k * nel + j].P.setValues(nr, -this.cutmax, this.cutmax, table);
          }
        }
      }
    }
    if (this.eta === 3) {
      for (let i = 0; i < ntriple; i++) this.triples[i].P.setValues(nr, -this.cutmax, this.cutmax, reader.dvector(nr));
    }
    for (let i = 0; i < ntriple; i++) this.triples[i].G.setValues(this.ng, -1.0, 1.0, reader.dvector(this.ng));
    for (let i = 0; i < npair; i++) pairs[i].F.setValues(this.nx, 0.0, this.maxX, reader.dvector(this.nx));

    for (const p of pairs) {
      p.cut = p.U.xmax;
      if (p.cut < p.V.xmax) p.cut = p.V.xmax;
      if (p.cut < p.W.xmax) p.cut = p.W.xmax;
      p.cutsq = p.cut * p.cut;
    }
    this.cutmax = 0.0;
    for (const p of pairs) if (this.cutmax < p.cut) this.cutmax = p.cut;
    this.cutmaxsq = this.cutmax * this.cutmax;
  }

  /** PairPolymorphic::setup_params: the element pair and triplet indices of the file's order. */
  protected setupParams(): void {
    const nel = this.elements.length;
    const match = this.match;
    this.elem2param = new Int32Array(nel * nel);
    this.elem3param = new Int32Array(nel * nel * nel);
    let n = 0;
    for (let i = 0; i < nel; i++) {
      this.elem2param[match[i] * nel + match[i]] = n;
      n++;
    }
    for (let i = 0; i < nel - 1; i++) {
      for (let j = i + 1; j < nel; j++) {
        this.elem2param[match[i] * nel + match[j]] = n;
        this.elem2param[match[j] * nel + match[i]] = n;
        n++;
      }
    }
    n = 0;
    for (let i = 0; i < nel; i++) {
      for (let j = 0; j < nel; j++) {
        for (let k = 0; k < nel; k++) {
          this.elem3param[(match[i] * nel + match[j]) * nel + match[k]] = n;
          n++;
        }
      }
    }
  }

  /** PairPolymorphic::compute. */
  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    const x = pc.x, f = pc.f, type = pc.type;
    const nel = this.elements.length;
    const map = this.map, e2 = this.elem2param, e3 = this.elem3param;
    const P = this.pairs, T = this.triples;
    const eps = this.epsilon;
    const eta = this.eta;
    const cutmaxsq = this.cutmaxsq;

    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const itag = pc.s.id[i];
      const itype = map[type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const k0 = list.firstneigh[i];
      const jnum = list.numneigh[i];

      let emb = 0;
      if (eta === 1) emb = P[e2[itype * nel + itype]].F.vmax;

      // neighbor lists of this atom: V (bond-order neighbors), W (zeta neighbors), with the separations
      const vJ: number[] = [], vDx: number[] = [], vDy: number[] = [], vDz: number[] = [], vR: number[] = [];
      const wJ: number[] = [], wDx: number[] = [], wDy: number[] = [], wDz: number[] = [], wR: number[] = [];

      let delx = 0, dely = 0, delz = 0;
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & NEIGHMASK;
        const jtype = map[type[j]];
        delx = xtmp - x[3 * j];
        dely = ytmp - x[3 * j + 1];
        delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq >= cutmaxsq) continue;
        const r0 = Math.sqrt(rsq);
        const p = P[e2[itype * nel + jtype]];
        if (eta === 1) {
          if (emb > eps) {
            const q = P[e2[jtype * nel + jtype]];
            if (rsq < q.W.xmaxsq && q.W.vmax > eps) {
              wJ.push(j); wDx.push(delx); wDy.push(dely); wDz.push(delz); wR.push(r0);
            }
          }
        } else if (p.F.vmax > eps) {
          if (rsq < p.V.xmaxsq && p.V.vmax > eps) {
            vJ.push(j); vDx.push(delx); vDy.push(dely); vDz.push(delz); vR.push(r0);
          }
          if (rsq < p.W.xmaxsq && p.W.vmax > eps) {
            wJ.push(j); wDx.push(delx); wDy.push(dely); wDz.push(delz); wR.push(r0);
          }
        }

        // two-body term on half of the pairs (by atom IDs)
        const jtag = tagOf(pc, j);
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < x[3 * i + 2]) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        if (rsq >= p.U.xmaxsq || p.U.vmax <= eps) continue;
        const evdwl = p.U.valY(r0);
        let fpair = p.U.valD(r0);
        fpair = -fpair / r0;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        tallyPair(pc, i, j, evdwl, fpair, delx, dely, delz);
      }

      if (eta === 1) {
        if (emb > eps) {
          const p = P[e2[itype * nel + itype]];
          let zetaIJ = 0.0;
          for (let kk = 0; kk < wJ.length; kk++) {
            const k = wJ[kk];
            const ktype = map[type[k]];
            const q = P[e2[ktype * nel + ktype]];
            zetaIJ += q.W.valY(wR[kk]);
          }
          const bij = p.F.valY(zetaIJ);
          const bijD = p.F.valD(zetaIJ);
          const prefactor = 0.5 * bijD;
          const evdwl = -0.5 * bij;
          tallyPair(pc, i, i, evdwl, 0.0, delx, dely, delz);
          for (let kk = 0; kk < wJ.length; kk++) {
            const k = wJ[kk];
            const ktype = map[type[k]];
            const d2x = -wDx[kk], d2y = -wDy[kk], d2z = -wDz[kk];
            const q = P[e2[ktype * nel + ktype]];
            let fpair = q.W.valD(wR[kk]);
            fpair = -prefactor * fpair / wR[kk];
            f[3 * i] += d2x * fpair;
            f[3 * i + 1] += d2y * fpair;
            f[3 * i + 2] += d2z * fpair;
            f[3 * k] -= d2x * fpair;
            f[3 * k + 1] -= d2y * fpair;
            f[3 * k + 2] -= d2z * fpair;
            if (pc.vatom) {
              // Pair::v_tally2 (per-atom part): half of the pair virial to i and k
              const va = pc.vatom, fp = -fpair;
              const v = [d2x * d2x * fp, d2y * d2y * fp, d2z * d2z * fp, d2x * d2y * fp, d2x * d2z * fp, d2y * d2z * fp];
              for (const q2 of [i, k]) for (let c = 0; c < 6; c++) va[6 * q2 + c] += 0.5 * v[c];
            }
          }
        }
      } else {
        const delr1 = new Float64Array(3), delr2 = new Float64Array(3);
        const fi = new Float64Array(3), fj = new Float64Array(3), fk = new Float64Array(3);
        const d1 = new Float64Array(3), d2 = new Float64Array(3);
        for (let jj = 0; jj < vJ.length; jj++) {
          const j = vJ[jj];
          const jtype = map[type[j]];
          const p = P[e2[itype * nel + jtype]];
          delr1[0] = -vDx[jj];
          delr1[1] = -vDy[jj];
          delr1[2] = -vDz[jj];
          const r1 = vR[jj];
          let zetaIJ = 0.0;
          const w1: number[] = []; // indices into the W list of the triplets that pass
          for (let kk = 0; kk < wJ.length; kk++) {
            const k = wJ[kk];
            if (j === k) continue;
            const ktype = map[type[k]];
            const trip = T[e3[(jtype * nel + itype) * nel + ktype]];
            if (trip.G.vmax <= eps) continue;
            w1.push(kk);
            delr2[0] = -wDx[kk];
            delr2[1] = -wDy[kk];
            delr2[2] = -wDz[kk];
            const r2 = wR[kk];
            const costheta = (delr1[0] * delr2[0] + delr1[1] * delr2[1] + delr1[2] * delr2[2]) / (r1 * r2);
            const q = P[e2[itype * nel + ktype]];
            const wfac = q.W.valY(r2);
            const pfac = trip.P.valY(r1 - p.xi * r2);
            const gfac = trip.G.valY(costheta);
            zetaIJ += wfac * pfac * gfac;
          }
          const fa = p.V.valY(r1);
          const faD = p.V.valD(r1);
          const bij = p.F.valY(zetaIJ);
          const bijD = p.F.valD(zetaIJ);
          let fpair = -0.5 * bij * faD / r1;
          const prefactor = 0.5 * fa * bijD;
          const evdwl = -0.5 * bij * fa;
          f[3 * i] += delr1[0] * fpair;
          f[3 * i + 1] += delr1[1] * fpair;
          f[3 * i + 2] += delr1[2] * fpair;
          f[3 * j] -= delr1[0] * fpair;
          f[3 * j + 1] -= delr1[1] * fpair;
          f[3 * j + 2] -= delr1[2] * fpair;
          tallyPair(pc, i, j, evdwl, -fpair, -delr1[0], -delr1[1], -delr1[2]);

          for (const kk of w1) {
            const k = wJ[kk];
            const ktype = map[type[k]];
            const trip = T[e3[(jtype * nel + itype) * nel + ktype]];
            delr2[0] = -wDx[kk];
            delr2[1] = -wDy[kk];
            delr2[2] = -wDz[kk];
            const r2 = wR[kk];
            const q = P[e2[itype * nel + ktype]];
            this.attractive(p, q, trip, prefactor, r1, r2, delr1, delr2, fi, fj, fk, d1, d2);
            for (let c = 0; c < 3; c++) {
              f[3 * i + c] += fi[c];
              f[3 * j + c] += fj[c];
              f[3 * k + c] += fk[c];
            }
            tallyTriplet(pc, i, j, k, fj, fk, delr1, delr2);
          }
        }
      }
    }
  }

  /** PairPolymorphic::attractive: the rij and rik unit vectors and the zeta-term derivatives. */
  protected attractive(
    p: PairP, q: PairP, trip: TripP, prefactor: number, rij: number, rik: number,
    delrij: Float64Array, delrik: Float64Array, fi: Float64Array, fj: Float64Array, fk: Float64Array,
    rijHat: Float64Array, rikHat: Float64Array,
  ): void {
    const rijinv = 1.0 / rij;
    for (let c = 0; c < 3; c++) rijHat[c] = rijinv * delrij[c];
    const rikinv = 1.0 / rik;
    for (let c = 0; c < 3; c++) rikHat[c] = rikinv * delrik[c];
    this.zetatermD(prefactor, rijHat, rij, rikHat, rik, fi, fj, fk, p, q, trip);
  }

  /** PairPolymorphic::ters_zetaterm_d. */
  protected zetatermD(
    prefactor: number, rijHat: Float64Array, rij: number, rikHat: Float64Array, rik: number,
    dri: Float64Array, drj: Float64Array, drk: Float64Array, p: PairP, q: PairP, trip: TripP,
  ): void {
    const cosTheta = rijHat[0] * rikHat[0] + rijHat[1] * rikHat[1] + rijHat[2] * rikHat[2];
    const fc = q.W.valY(rik);
    const dfc = q.W.valD(rik);
    const exDelr = trip.P.valY(rij - p.xi * rik);
    const exDelrD = trip.P.valD(rij - p.xi * rik);
    const gijk = trip.G.valY(cosTheta);
    const gijkD = trip.G.valD(cosTheta);
    const { dri: dcosdri, drj: dcosdrj, drk: dcosdrk } = costhetaD(rijHat, rij, rikHat, rik);
    for (let c = 0; c < 3; c++) {
      let v = (-dfc * gijk * exDelr) * rikHat[c];
      v = (fc * gijkD * exDelr) * dcosdri[c] + v;
      v = (fc * gijk * exDelrD * p.xi) * rikHat[c] + v;
      v = (-fc * gijk * exDelrD) * rijHat[c] + v;
      dri[c] = prefactor * v;
    }
    for (let c = 0; c < 3; c++) {
      let v = (fc * gijkD * exDelr) * dcosdrj[c];
      v = (fc * gijk * exDelrD) * rijHat[c] + v;
      drj[c] = prefactor * v;
    }
    for (let c = 0; c < 3; c++) {
      let v = (dfc * gijk * exDelr) * rikHat[c];
      v = (fc * gijkD * exDelr) * dcosdrk[c] + v;
      v = (-fc * gijk * exDelrD * p.xi) * rikHat[c] + v;
      drk[c] = prefactor * v;
    }
  }
}

/** PairPolymorphic::costheta_d: derivatives of cos(theta) with respect to Ri, Rj and Rk. */
const costhetaD = (rijHat: Float64Array, rij: number, rikHat: Float64Array, rik: number) => {
  const cosTheta = rijHat[0] * rikHat[0] + rijHat[1] * rikHat[1] + rijHat[2] * rikHat[2];
  const drj = [0, 0, 0], drk = [0, 0, 0], dri = [0, 0, 0];
  for (let c = 0; c < 3; c++) drj[c] = -cosTheta * rijHat[c] + rikHat[c];
  for (let c = 0; c < 3; c++) drj[c] = (1.0 / rij) * drj[c];
  for (let c = 0; c < 3; c++) drk[c] = -cosTheta * rikHat[c] + rijHat[c];
  for (let c = 0; c < 3; c++) drk[c] = (1.0 / rik) * drk[c];
  for (let c = 0; c < 3; c++) dri[c] = -(drj[c] + drk[c]);
  return { dri, drj, drk };
};

import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNum, parseInt_ } from '../force/util';
import { buildTriples, rawBispectrum, adjointBispectrum, WignerTables, type Triple, type Cmat, type Grad } from './sna';

/*
 * compute ID group-ID snap rcutfac rfac0 twojmax R_1 ... R_N w_1 ... w_N keyword values ...
 * — docs.lammps.org/compute_sna_atom.html. Compute snap calculates a global array with the information of
 * the three per-atom computes sna/atom, snad/atom and snav/atom. Shared parsing and bispectrum helpers are used by compute sna/grid too
 * (src/engine/compute/sna_grid.ts).
 *
 * Doc, Output info for compute snap, gives the row order: one row of sna/atom quantities summed for
 * all atoms of type I; 3N rows of snad/atom quantities (x, y and z of atom i in consecutive rows, atoms
 * sorted by atom ID); and 6 rows of snav/atom quantities summed for all atoms of type I. The last
 * column holds the energy, force component or virial stress component of the reference potential.
 * Measured with native LAMMPS (black box): row 1 holds the total potential
 * energy in that last column, the 3N snad rows hold the force component of atom id
 * (rows 2 + 3(id-1) + a), and the six snav rows hold the pressure virial (pressure NULL virial)
 * in the order xx, yy, zz, yz, xz, xy.
 *
 * bikflag (doc): with bikflag 0 a single bispectrum row is used; with bikflag 1 it is replaced by a
 * separate per-atom bispectrum row for each atom, whose final column is set to zero. Measured: the first of these rows
 * still carries the potential energy in its last column, and the rows are N per-atom rows, then
 * 3N snad rows, then 6 snav rows.
 *
 * dgradflag (doc): with dgradflag 1 the gradients are listed separately for each pair of atoms; the
 * option dgradflag=1 requires that bikflag=1; the total number of rows is N + 3N^2 + 1 and the number
 * of columns is K + 3. Measured: the first N rows hold the force
 * components and the K bispectrum components, the 3N^2 rows hold (i, j, a, dB_i,k/dr_j^a) with j the
 * outer index, then i, then a, and the final row holds the potential energy. Measured with native LAMMPS
 * (black box): the gradient rows hold the negative of dB_i,k/dr_j^a (so their sum over i equals snad/atom).
 *
 * Only the central atoms are restricted to the group for compute snap (measured with native LAMMPS:
 * neighbours outside the group still contribute); for the grid computes the neighbours are restricted
 * (doc, sna/grid paragraph, and measured). Keywords: rmin0, switchflag, bzeroflag, quadraticflag, bnormflag,
 * wselfallflag (no effect without chem, as documented), switchinnerflag with sinner and dinner, bikflag
 * and dgradflag. Not implemented here and reported as StyleError: chem (multi-element variant),
 * nnn, wmode and delta (the doc says "only implemented for compute sna/atom").
 */

export interface SnaParams {
  rcutfac: number;
  rfac0: number;
  twojmax: number;
  /** Per-type radius R_t and neighbor weight w_t, index t-1 for LAMMPS type t. */
  radius: Float64Array;
  weight: Float64Array;
  rmin0: number;
  switchflag: boolean;
  bzeroflag: boolean;
  quadraticflag: boolean;
  bnormflag: boolean;
  switchinner: boolean;
  /** Per-type S_inner and D_inner (doc: ntypes values each for sinner and dinner). */
  sinner: Float64Array;
  dinner: Float64Array;
  triples: Triple[];
  /** Number of bispectrum components K, quadratic terms Q, and columns per block K + Q. */
  K: number;
  Q: number;
  blk: number;
  /** Division per component (2j+1 with bnormflag, else 1). */
  norm: Float64Array;
  /** Bispectrum of an atom with no neighbors (identity u), zero when bzeroflag is off. */
  b0: Float64Array;
}

export interface SnaOptions {
  p: SnaParams;
  bik: boolean;
  dgrad: boolean;
}

/** Largest per-type radius (for the cutoff check). */
export const maxRadius = (p: SnaParams): number => {
  let rmax = 0;
  for (let t = 0; t < p.radius.length; t++) if (p.radius[t] > rmax) rmax = p.radius[t];
  return rmax;
};

/** Parses the arguments rcutfac rfac0 twojmax, the radii R_t and weights w_t, and keyword values, for compute snap, sna/grid and sna/grid/local. */
export const parseSnaArgs = (id: string, style: string, args: string[], ntypes: number, snapOnly: boolean): SnaOptions => {
  const need = 3 + 2 * ntypes;
  if (args.length < need) {
    throw new StyleError(`compute ${id} (${style}): expects rcutfac rfac0 twojmax, ${ntypes} radii and ${ntypes} weights (${need} values), got ${args.length}`);
  }
  const rcutfac = parseNum(args[0], `compute ${id} (${style}) rcutfac`);
  if (!(rcutfac > 0)) throw new StyleError(`compute ${id} (${style}): rcutfac must be positive (got ${args[0]})`);
  const rfac0 = parseNum(args[1], `compute ${id} (${style}) rfac0`);
  const twojmax = parseInt_(args[2], `compute ${id} (${style}) twojmax`);
  if (twojmax < 0) throw new StyleError(`compute ${id} (${style}): twojmax must be a non-negative integer (got ${args[2]})`);
  const radius = new Float64Array(ntypes);
  const weight = new Float64Array(ntypes);
  for (let t = 0; t < ntypes; t++) {
    radius[t] = parseNum(args[3 + t], `compute ${id} (${style}) R_${t + 1}`);
    weight[t] = parseNum(args[3 + ntypes + t], `compute ${id} (${style}) w_${t + 1}`);
  }
  let rmin0 = 0;
  let switchflag = true, bzeroflag = true, quadraticflag = false, bnormflag = false;
  let switchinnerflag = false, bik = false, dgrad = false;
  let sinnerList: number[] | null = null, dinnerList: number[] | null = null;
  for (let k = need; k < args.length; k++) {
    const kw = args[k];
    const value = (): string => {
      if (k + 1 >= args.length) throw new StyleError(`compute ${id} (${style}): keyword '${kw}' needs a value`);
      return args[++k];
    };
    const flag = (): boolean => {
      const w = value();
      if (w !== '0' && w !== '1') throw new StyleError(`compute ${id} (${style}): ${kw} must be 0 or 1 (got '${w}')`);
      return w === '1';
    };
    const list = (): number[] => {
      const out: number[] = [];
      for (let t = 0; t < ntypes; t++) out.push(parseNum(value(), `compute ${id} (${style}) ${kw}`));
      return out;
    };
    if (kw === 'rmin0') rmin0 = parseNum(value(), `compute ${id} (${style}) rmin0`);
    else if (kw === 'switchflag') switchflag = flag();
    else if (kw === 'bzeroflag') bzeroflag = flag();
    else if (kw === 'quadraticflag') quadraticflag = flag();
    else if (kw === 'bnormflag') bnormflag = flag();
    else if (kw === 'wselfallflag') flag(); // doc: "When the chem keyword is not used, this keyword has no effect."
    else if (kw === 'switchinnerflag') switchinnerflag = flag();
    else if (kw === 'sinner') sinnerList = list();
    else if (kw === 'dinner') dinnerList = list();
    else if (kw === 'bikflag') {
      if (!snapOnly) throw new StyleError(`compute ${id} (${style}): keyword 'bikflag' is only implemented for compute snap`);
      bik = flag();
    } else if (kw === 'dgradflag') {
      if (!snapOnly) throw new StyleError(`compute ${id} (${style}): keyword 'dgradflag' is only implemented for compute snap`);
      dgrad = flag();
    } else if (kw === 'chem') {
      throw new StyleError(`compute ${id} (${style}): keyword 'chem' (multi-element bispectrum) is not implemented in this engine`);
    } else if (kw === 'nnn' || kw === 'wmode' || kw === 'delta') {
      throw new StyleError(`compute ${id} (${style}): keyword '${kw}' is only implemented for compute sna/atom (doc), not in this engine for ${style}`);
    } else {
      throw new StyleError(`compute ${id} (${style}): unknown keyword '${kw}'`);
    }
  }
  if (switchinnerflag && (!sinnerList || !dinnerList)) {
    throw new StyleError(`compute ${id} (${style}): switchinnerflag 1 needs the keywords sinner and dinner (ntypes values each)`);
  }
  if (!switchinnerflag && (sinnerList || dinnerList)) {
    throw new StyleError(`compute ${id} (${style}): sinner and dinner are only used with switchinnerflag 1 (not supported without it in this engine)`);
  }
  if (dgrad && !bik) throw new StyleError(`compute ${id} (${style}): dgradflag 1 requires bikflag 1 (doc)`);
  if (dgrad && quadraticflag) throw new StyleError(`compute ${id} (${style}): dgradflag 1 with quadraticflag 1 is not implemented in this engine`);
  const triples = buildTriples(twojmax);
  const K = triples.length;
  const Q = quadraticflag ? (K * (K + 1)) / 2 : 0;
  const norm = new Float64Array(K);
  for (let c = 0; c < K; c++) norm[c] = bnormflag ? triples[c].J + 1 : 1;
  // B0 from the identity expansion (the self term alone)
  const b0 = new Float64Array(K);
  if (bzeroflag) {
    const id0: Cmat[] = [];
    for (let J = 0; J <= twojmax; J++) {
      const m = J + 1;
      const re = new Float64Array(m * m), im = new Float64Array(m * m);
      for (let q = 0; q < m; q++) re[q * m + q] = 1;
      id0.push({ re, im });
    }
    rawBispectrum(triples, id0, b0);
  }
  const p: SnaParams = {
    rcutfac, rfac0, twojmax, radius, weight, rmin0, switchflag, bzeroflag, quadraticflag, bnormflag,
    switchinner: switchinnerflag,
    sinner: new Float64Array(sinnerList ?? new Array(ntypes).fill(0)),
    dinner: new Float64Array(dinnerList ?? new Array(ntypes).fill(0)),
    triples, K, Q, blk: K + Q, norm, b0,
  };
  return { p, bik, dgrad };
};

/** Cutoff factor f_c(r) and its radial derivative (switchflag, rmin0 convention of compute sna/atom). */
export const cutoffFactor = (p: SnaParams, r: number, Rii: number): [number, number] => {
  const span = Rii - p.rmin0;
  const inside = p.switchflag && r >= p.rmin0;
  if (!inside) return [1, 0];
  return [0.5 * (Math.cos((Math.PI * (r - p.rmin0)) / span) + 1), -0.5 * Math.sin((Math.PI * (r - p.rmin0)) / span) * (Math.PI / span)];
};

/**
 * Inner switching function (doc, switchinnerflag): zero for r <= S_inner - D_inner, one for r above
 * S_inner + D_inner, and in between 1/2 (1 - cos(pi/2 (1 + (r - S_inner)/D_inner))). Returns
 * [f_inner, df_inner/dr].
 */
export const innerSwitch = (r: number, S: number, D: number): [number, number] => {
  if (r <= S - D) return [0, 0];
  if (r > S + D) return [1, 0];
  const th = (Math.PI / 2) * (1 + (r - S) / D);
  return [0.5 * (1 - Math.cos(th)), 0.5 * Math.sin(th) * (Math.PI / (2 * D))];
};

/** Cayley-Klein parameters (ar, ai, br, bi) and the scaled sine/cosine used by the derivative tables. */
export const cayleyKlein = (p: SnaParams, r: number, Rii: number, dx: number, dy: number, dz: number): { ar: number; ai: number; br: number; bi: number; th: number; sg: number } => {
  const th = (p.rfac0 * Math.PI * (r - p.rmin0)) / (Rii - p.rmin0);
  const sn = Math.sin(th), cs = Math.cos(th);
  const sg = sn < 0 ? -1 : 1;
  const Sx = sg * sn;
  const Cx = sg * cs;
  const g = Sx / r;
  return { ar: Cx, ai: g * dz, br: g * dy, bi: g * dx, th, sg };
};

/** Quadratic terms of a bispectrum vector, upper triangle, diagonal halved (as compute sna/atom). */
export const quadraticTerms = (b: Float64Array, K: number, out: Float64Array, off: number): void => {
  let q = off;
  for (let i = 0; i < K; i++) {
    for (let j = i; j < K; j++) out[q++] = i === j ? 0.5 * b[i] * b[i] : b[i] * b[j];
  }
};

/**
 * Cutoff of the bispectrum (rcutfac * (R_i + R_j), largest pair) against the pair cutoff and the
 * ghost cutoff. Measured with native LAMMPS (black box): compute snap refuses a cutoff longer than
 * the pair cutoff (same rule as the engine message below); the engine checks
 * it when the array is evaluated (native checks at run setup).
 */
export const checkCutoff = (id: string, style: string, p: SnaParams, nb: { cutghost: number; cutneighmax: number; skin: number }): void => {
  const cutmax = 2 * p.rcutfac * maxRadius(p);
  const paircut = nb.cutneighmax > 0 ? nb.cutneighmax - nb.skin : 0;
  if (paircut > 0 && cutmax > paircut + 1e-12) {
    throw new StyleError(`compute ${id} (${style}): cutoff ${cutmax} is longer than pairwise cutoff ${paircut}`);
  }
  if (nb.cutghost < cutmax - 1e-12) {
    throw new StyleError(`compute ${id} (${style}): cutoff rcutfac*(R_i+R_j) up to ${cutmax} exceeds the ghost cutoff ${nb.cutghost} (set a pair style with a larger cutoff)`);
  }
};

/** Voigt pairs (a, b) of the stress components xx yy zz yz xz xy, as used by the snav rows. */
const PAIRS_V: [number, number][] = [[0, 0], [1, 1], [2, 2], [2, 1], [2, 0], [1, 0]];

export class ComputeSnap extends Compute {
  readonly style = 'snap';
  private readonly p: SnaParams;
  private readonly bik: boolean;
  private readonly dgrad: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const o = parseSnaArgs(id, 'snap', args, sys.state.ntypes, true);
    this.p = o.p;
    this.bik = o.bik;
    this.dgrad = o.dgrad;
    this.arrayFlag = true;
    this.sizeArrayCols = this.dgrad ? this.p.K + 3 : sys.state.ntypes * this.p.blk + 1;
  }

  protected computeArray(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const p = this.p;
    const acc = sys.forces();
    checkCutoff(this.id, this.style, p, nb);
    const n = s.n;
    const nall = nb.nall, xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    const nt = p.radius.length, K = p.K, Q = p.Q, blk = p.blk, tj = p.twojmax, triples = p.triples;
    const gb = this.groupBit;

    // atoms in ID order: row r of the array belongs to rankOrder[r]
    const rankOrder = Array.from({ length: n }, (_, i) => i).sort((a, b) => s.id[a] - s.id[b]);
    const rank = new Int32Array(n);
    rankOrder.forEach((a, r) => { rank[a] = r; });

    const snad = new Float64Array(n * nt * 3 * blk);   // [owner][type][a][col]
    const snapV = new Float64Array(6 * nt * blk);      // [voigt][type][col]
    const sumB = new Float64Array(nt * blk);           // sna/atom summed per type
    const atomB = new Float64Array(this.bik || this.dgrad ? n * blk : 0);
    const dg = this.dgrad ? new Float64Array(n * n * 3 * K) : null; // [j rank][i rank][a][k]

    // per-J offsets into the flat Wigner tables
    const T = new WignerTables(tj);
    const u: Cmat[] = [], gr: Grad[] = [], gi: Grad[] = [];
    for (let J = 0; J <= tj; J++) {
      const nn = (J + 1) * (J + 1);
      u.push({ re: new Float64Array(nn), im: new Float64Array(nn) });
      gr.push({ r: new Float64Array(nn), i: new Float64Array(nn) });
      gi.push({ r: new Float64Array(nn), i: new Float64Array(nn) });
    }
    const offJ = new Int32Array(tj + 2);
    for (let J = 0; J <= tj; J++) offJ[J + 1] = offJ[J] + (J + 1) * (J + 1);
    const S = offJ[tj + 1];
    const cap = Math.max(nall, 1);
    const nbJ = new Int32Array(cap);
    const nDx = new Float64Array(cap), nDy = new Float64Array(cap), nDz = new Float64Array(cap);
    const nR = new Float64Array(cap), nTh = new Float64Array(cap), nRc = new Float64Array(cap);
    const nSc = new Float64Array(cap), nDsc = new Float64Array(cap);
    let WR = new Float64Array(0), WI = new Float64Array(0), G = new Float64Array(0);
    const raw = new Float64Array(K), Bf = new Float64Array(K), gbuf = new Float64Array(K);
    const vals = new Float64Array(blk);
    const dQ = new Float64Array(Q * 3);
    const pairsV = PAIRS_V;

    for (let ip = 0; ip < n; ip++) {
      if (!(s.mask[ip] & gb)) continue;
      const tI = ta[ip] - 1;
      const xi = xa[3 * ip], yi = xa[3 * ip + 1], zi = xa[3 * ip + 2];
      // neighbours inside their pair cutoff, group members only
      let m = 0;
      for (let k = 0; k < nall; k++) {
        if (k === ip) continue;
        const dx = xa[3 * k] - xi, dy = xa[3 * k + 1] - yi, dz = xa[3 * k + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Rii = p.rcutfac * (p.radius[tI] + p.radius[ta[k] - 1]);
        if (!(r < Rii) || r === 0) continue;
        nbJ[m] = k; nDx[m] = dx; nDy[m] = dy; nDz[m] = dz; nR[m] = r; nRc[m] = Rii;
        nTh[m] = (p.rfac0 * Math.PI * (r - p.rmin0)) / (Rii - p.rmin0);
        const [fc, dfc] = cutoffFactor(p, r, Rii);
        const w = p.weight[ta[k] - 1];
        let fin = 1, dfin = 0;
        if (p.switchinner) {
          const Si = 0.5 * (p.sinner[tI] + p.sinner[ta[k] - 1]);
          const Di = 0.5 * (p.dinner[tI] + p.dinner[ta[k] - 1]);
          [fin, dfin] = innerSwitch(r, Si, Di);
        }
        nSc[m] = fc * fin * w;
        nDsc[m] = (dfc * fin + fc * dfin) * w;
        m++;
      }
      if (WR.length < m * 3 * S) {
        WR = new Float64Array(m * 3 * S);
        WI = new Float64Array(m * 3 * S);
        G = new Float64Array(K * m * 3);
      }
      // pass 1: u matrices (self term = identity) and derivative tables of every neighbour
      for (let J = 0; J <= tj; J++) {
        u[J].re.fill(0); u[J].im.fill(0);
        for (let q = 0; q < J + 1; q++) u[J].re[q * (J + 1) + q] = 1;
      }
      for (let a = 0; a < m; a++) {
        const r = nR[a];
        const sn = Math.sin(nTh[a]), cs = Math.cos(nTh[a]);
        const sg = sn < 0 ? -1 : 1;
        const Sx = sg * sn, Cx = sg * cs;
        const g = Sx / r;
        const dx = nDx[a], dy = nDy[a], dz = nDz[a];
        const nm = [dx / r, dy / r, dz / r];
        const thp = (p.rfac0 * Math.PI) / (nRc[a] - p.rmin0);
        const gp = (Cx * thp * r - Sx) / (r * r);
        const dp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
        const vz = [0, 0, 1], vy = [0, 1, 0], vx = [1, 0, 0];
        for (let mm = 0; mm < 3; mm++) {
          dp[0][mm] = -Sx * thp * nm[mm];
          dp[1][mm] = gp * nm[mm] * dz + g * vz[mm];
          dp[2][mm] = gp * nm[mm] * dy + g * vy[mm];
          dp[3][mm] = gp * nm[mm] * dx + g * vx[mm];
        }
        const sc = nSc[a], dsc = nDsc[a];
        T.compute(Cx, g * dz, g * dy, g * dx, true);
        for (let J = 0; J <= tj; J++) {
          const nn = (J + 1) * (J + 1);
          const Ur = T.ur[J], Ui = T.ui[J];
          const Dr = T.dur[J], Di = T.dui[J];
          for (let q = 0; q < nn; q++) {
            u[J].re[q] += sc * Ur[q];
            u[J].im[q] += sc * Ui[q];
            const base = a * 3 * S + (offJ[J] + q) * 3;
            for (let mm = 0; mm < 3; mm++) {
              let dre = dsc * nm[mm] * Ur[q], dim = dsc * nm[mm] * Ui[q];
              for (let pp = 0; pp < 4; pp++) {
                dre += sc * Dr[pp][q] * dp[pp][mm];
                dim += sc * Di[pp][q] * dp[pp][mm];
              }
              WR[base + mm] = dre;
              WI[base + mm] = dim;
            }
          }
        }
      }
      rawBispectrum(triples, u, raw);
      for (let c = 0; c < K; c++) Bf[c] = (raw[c] - p.b0[c]) / p.norm[c];
      vals.fill(0);
      for (let c = 0; c < K; c++) vals[c] = Bf[c];
      if (Q) quadraticTerms(Bf, K, vals, K);
      for (let c = 0; c < blk; c++) sumB[tI * blk + c] += vals[c];
      if (this.bik || this.dgrad) for (let c = 0; c < blk; c++) atomB[ip * blk + c] = vals[c];

      // Jacobian of every component with respect to every neighbour displacement
      for (let k = 0; k < K; k++) {
        gbuf.fill(0);
        gbuf[k] = 1 / p.norm[k];
        for (let J = 0; J <= tj; J++) { gr[J].r.fill(0); gi[J].r.fill(0); }
        adjointBispectrum(triples, u, gbuf, gr, gi);
        for (let a = 0; a < m; a++) {
          for (let mm = 0; mm < 3; mm++) {
            let acc2 = 0;
            for (let J = 0; J <= tj; J++) {
              const nn = (J + 1) * (J + 1);
              const GR = gr[J].r, GI = gi[J].r;
              for (let q = 0; q < nn; q++) {
                const gR = GR[q], gI = GI[q];
                if (gR === 0 && gI === 0) continue;
                const idx = a * 3 * S + (offJ[J] + q) * 3 + mm;
                acc2 += gR * WR[idx] + gI * WI[idx];
              }
            }
            G[(k * m + a) * 3 + mm] = acc2;
          }
        }
      }

      // accumulate derivatives: -dB/dx_o for the owner o of every neighbour, +sum for the centre
      const xI = [xi, yi, zi];
      for (let a = 0; a < m; a++) {
        const jg = nbJ[a];
        const o = owner[jg];
        if (Q) {
          let q = 0;
          for (let kk = 0; kk < K; kk++) {
            for (let ll = kk; ll < K; ll++) {
              for (let mm = 0; mm < 3; mm++) {
                const gk = G[(kk * m + a) * 3 + mm], gl = G[(ll * m + a) * 3 + mm];
                dQ[q * 3 + mm] = kk === ll ? Bf[kk] * gk : Bf[kk] * gl + Bf[ll] * gk;
              }
              q++;
            }
          }
        }
        // snad (per owner), block of the central type
        for (let mm = 0; mm < 3; mm++) {
          const base = (o * nt + tI) * 3 * blk + mm * blk;
          const baseC = (ip * nt + tI) * 3 * blk + mm * blk;
          for (let k = 0; k < K; k++) {
            const g = G[(k * m + a) * 3 + mm];
            snad[base + k] -= g;
            snad[baseC + k] += g;
          }
          for (let q = 0; q < Q; q++) {
            const g = dQ[q * 3 + mm];
            snad[base + K + q] -= g;
            snad[baseC + K + q] += g;
          }
        }
        // snav: summed over all owners, the pair term is (x_centre - x_neighbour) (x) G
        const xj = [xa[3 * jg], xa[3 * jg + 1], xa[3 * jg + 2]];
        for (let c = 0; c < 6; c++) {
          const [pa, pb] = pairsV[c];
          const cb = (c * nt + tI) * blk;
          const dq = xI[pa] - xj[pa];
          for (let k = 0; k < K; k++) snapV[cb + k] += dq * G[(k * m + a) * 3 + pb];
          for (let q = 0; q < Q; q++) snapV[cb + K + q] += dq * dQ[q * 3 + pb];
        }
        // dgradflag: dB_i/dx_j (owner j) and dB_i/dx_i = -sum_j dB_i/dx_j
        if (dg) {
          const ri = rank[ip], rj = rank[o];
          for (let mm = 0; mm < 3; mm++) {
            for (let k = 0; k < K; k++) {
              const g = G[(k * m + a) * 3 + mm];
              dg[((rj * n + ri) * 3 + mm) * K + k] += g;
              dg[((ri * n + ri) * 3 + mm) * K + k] -= g;
            }
          }
        }
      }
    }

    // reference potential energy and virial (compute pressure NULL virial, pair ... fix terms)
    const pe = acc.evdwl + acc.ecoul + acc.ebond + acc.eangle + acc.edihed + acc.eimp + acc.elong + sys.fixEnergy();
    const vol = sys.geom.volume(sys.dimension);
    const w = new Float64Array(6);
    const addv = (v: Float64Array): void => { for (let c = 0; c < 6; c++) w[c] += v[c]; };
    addv(acc.virial); addv(acc.vbond); addv(acc.vangle); addv(acc.vdihed); addv(acc.vimp); addv(acc.vlong);
    sys.fixVirial(w);
    const tail = sys.ff.ptail(vol);
    // engine order xx yy zz xy xz yz -> Voigt order xx yy zz yz xz xy of the snav rows
    const engineOf = [0, 1, 2, 5, 4, 3];
    const press = new Float64Array(6);
    for (let c = 0; c < 6; c++) {
      let v = w[c] / vol;
      if (c < 3) v += tail;
      press[c] = v * s.units.nktv2p;
    }
    const pressV = new Float64Array(6);
    for (let c = 0; c < 6; c++) pressV[c] = press[engineOf[c]];

    const cols = this.dgrad ? K + 3 : nt * blk + 1;
    if (this.dgrad) {
      const rows = n + 3 * n * n + 1;
      const out = new Float64Array(rows * cols);
      for (let r = 0; r < n; r++) {
        const atom = rankOrder[r];
        out[r * cols] = s.f[3 * atom];
        out[r * cols + 1] = s.f[3 * atom + 1];
        out[r * cols + 2] = s.f[3 * atom + 2];
        for (let k = 0; k < K; k++) out[r * cols + 3 + k] = atomB[atom * blk + k];
      }
      for (let jr = 0; jr < n; jr++) {
        for (let ir = 0; ir < n; ir++) {
          for (let a = 0; a < 3; a++) {
            const row = n + (jr * n + ir) * 3 + a;
            out[row * cols] = ir;
            out[row * cols + 1] = jr;
            out[row * cols + 2] = a;
            // measured with native LAMMPS (black box): the rows hold -dB_i,k/dr_j^a, the sign of the snad rows
            for (let k = 0; k < K; k++) out[row * cols + 3 + k] = -dg![((jr * n + ir) * 3 + a) * K + k];
          }
        }
      }
      out[(rows - 1) * cols] = pe;
      this.array = out;
      this.sizeArrayRows = rows;
      this.sizeArrayCols = cols;
      return;
    }

    const rows = (this.bik ? n : 1) + 3 * n + 6;
    const out = new Float64Array(rows * cols);
    const base = this.bik ? n : 1;
    if (this.bik) {
      for (let r = 0; r < n; r++) {
        const atom = rankOrder[r];
        const tAtom = ta[atom] - 1;
        for (let c = 0; c < blk; c++) out[r * cols + tAtom * blk + c] = atomB[atom * blk + c];
      }
      out[0 * cols + cols - 1] = pe;
    } else {
      for (let t = 0; t < nt; t++) for (let c = 0; c < blk; c++) out[t * blk + c] = sumB[t * blk + c];
      out[cols - 1] = pe;
    }
    for (let r = 0; r < n; r++) {
      const atom = rankOrder[r];
      for (let a = 0; a < 3; a++) {
        const row = base + 3 * r + a;
        for (let t = 0; t < nt; t++) {
          const src = (atom * nt + t) * 3 * blk + a * blk;
          for (let c = 0; c < blk; c++) out[row * cols + t * blk + c] = snad[src + c];
        }
        out[row * cols + cols - 1] = s.f[3 * atom + a];
      }
    }
    for (let c = 0; c < 6; c++) {
      const row = base + 3 * n + c;
      for (let t = 0; t < nt; t++) for (let q = 0; q < blk; q++) out[row * cols + t * blk + q] = snapV[(c * nt + t) * blk + q];
      out[row * cols + cols - 1] = pressV[c];
    }
    this.array = out;
    this.sizeArrayRows = rows;
    this.sizeArrayCols = cols;
  }
}

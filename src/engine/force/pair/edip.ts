import { Pair, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { joinPotentialEntries, parseNum } from '../util';

/*
 * pair_style edip, edip/multi: docs.lammps.org/pair_edip.html (text: plans/lammps-docs-live/pair_edip.rst).
 * Three-body EDIP of Justo et al. 1998 (the paper the doc cites).
 *
 * Energy (the doc formula):
 *
 *   E = sum_{j != i} phi_2(R_ij, Z_i) + sum_{j != i} sum_{k != i, k > j} phi_3(R_ij, R_ik, Z_i)
 *   phi_2(r, Z) = A [ (B/r)^rho - exp(-beta Z^2) ] exp(sigma/(r - a))
 *   phi_3(R_ij, R_ik, Z_i) = exp(gamma/(R_ij - a)) exp(gamma/(R_ik - a)) h(cos theta_jik, Z_i)
 *   Z_i = sum_{m != i} f(R_im);  f(r) = 1 (r < c), exp(alpha/(1 - x^-3)) (c < r < a), 0 (r > a)
 *   h(l, Z) = lambda [ (1 - exp(-Q(Z) (l + tau(Z))^2)) + eta Q(Z) (l + tau(Z))^2 ]
 *   Q(Z) = Q0 exp(-mu Z),   tau(Z) = u1 + u2 (u3 exp(-u4 Z) - exp(-2 u4 Z))
 *
 * The doc leaves x in f undefined; the engine takes x = (r - c)/(a - c) (Justo et al. 1998). Here a is
 * cutoffA and c is cutoffC of the potential file.
 *
 * Potential file (the doc: "Lines that are not blank or comments (starting with #) define parameters for
 * a triplet of elements."): element 1, element 2, element 3, then A, B, cutoffA, cutoffC, alpha, beta,
 * eta, gamma, lambda, mu, rho, sigma, Q0, u1, u2, u3, u4 (the file header's order; the doc's list omits
 * rho). The doc: "For a single-element simulation, only a single entry is required (e.g. SiSiSi)."
 *
 * Parameter tables (edip/multi): the two-body term, the coordination f and the leg factor gamma of
 * center i and neighbour j use the entry (E_i, E_j, E_j); h uses the entry (E_i, E_j, E_k). For edip both
 * are the one entry. Measured with native LAMMPS (black box): Si.edip dimers (r = 1.5 to 3.0) equal
 * 2 phi_2(r, f(r)) to about 1e-10 relative; Si.edip trimers (angles 60 to 150 degrees) equal the sum
 * above to 3e-10 relative or better; SiC.edip dimers and five mixed trimers equal the multi rule to 1e-15.
 * Si.edip tetrahedral clusters agree to 1e-13 at r = 2.35. On larger cells native edip departs from the
 * analytic form, and native edip/multi does not: with the same Si.edip parameters on a displaced diamond
 * cell (tests/oracle/w48pb_edip_diamond) native edip/multi gives press -644.369072008205 at step 0, equal to
 * this engine (-644.3690720082104), while native edip gives -644.368549221789 (about 1e-6 in pressure,
 * 3e-9 in energy). The engine keeps the analytic form for both styles.
 *
 * Refused: pair_modify shift, table and tail (the doc: "This pair style does not support the pair_modify
 * shift, table, and tail options."); edip with more than one element name (the doc: "At the moment, only a
 * single element parameterization is implemented.").
 *
 * Summation: each owned atom i gives E_i over its full neighbour list, Z_i first (pass 1), then the
 * gradient of E_i with respect to every neighbour (direct phi_2 and phi_3 terms plus the chain term
 * dE_i/dZ_i f'(r_im) through Z_i; pass 2). The global virial is the x.f dot product.
 */

/** One entry of an EDIP potential file (the parameters of the doc's list, file order). */
export interface EDIPEntry {
  A: number; B: number; a: number; c: number; alpha: number; beta: number; eta: number;
  gamma: number; lambda: number; mu: number; rho: number; sigma: number; Q0: number;
  u1: number; u2: number; u3: number; u4: number;
}

const FIELDS = ['A', 'B', 'a', 'c', 'alpha', 'beta', 'eta', 'gamma', 'lambda', 'mu', 'rho', 'sigma', 'Q0', 'u1', 'u2', 'u3', 'u4'] as const;

/** Parses the EDIP potential file into entries keyed by "E1 E2 E3". */
export const parseEDIPFile = (text: string, fileName: string): Map<string, EDIPEntry> => {
  const lines = joinPotentialEntries(text.split(/\r?\n/), 20);
  const out = new Map<string, EDIPEntry>();
  for (let ln = 0; ln < lines.length; ln++) {
    const raw = lines[ln];
    const h = raw.indexOf('#');
    const t = (h >= 0 ? raw.slice(0, h) : raw).trim().split(/\s+/).filter(Boolean);
    if (t.length === 0) continue;
    if (t.length !== 20) {
      throw new StyleError(`EDIP potential file ${fileName} line ${ln + 1}: expected 3 element names and 17 numbers (20 words), got ${t.length}`);
    }
    const key = `${t[0]} ${t[1]} ${t[2]}`;
    if (out.has(key)) throw new StyleError(`EDIP potential file ${fileName} line ${ln + 1}: duplicate entry for elements ${key}`);
    const v = t.slice(3).map((w, k) => parseNum(w, `${FIELDS[k]} of the EDIP entry ${key}`));
    const e = {} as EDIPEntry;
    FIELDS.forEach((f, k) => { e[f] = v[k]; });
    if (!(e.a > e.c) || !(e.c > 0)) throw new StyleError(`EDIP entry ${key} needs 0 < cutoffC < cutoffA`);
    out.set(key, e);
  }
  if (out.size === 0) throw new StyleError(`EDIP potential file ${fileName} contains no parameter entries`);
  return out;
};

/** f(r) and f'(r) of the coordination function (doc: 1 for r < c, exp(alpha/(1-x^-3)) for c < r < a, 0 beyond). */
export const coordF = (p: EDIPEntry, r: number): { f: number; df: number } => {
  if (r <= p.c) return { f: 1, df: 0 };
  if (r >= p.a) return { f: 0, df: 0 };
  const x = (r - p.c) / (p.a - p.c);
  const x3 = x ** -3;
  const den = 1 - x3;
  const g = p.alpha / den;
  const f = Math.exp(g);
  // d g / dx = -alpha * (3 x^-4) / den^2 ; dx/dr = 1/(a - c)
  const dgdr = (-p.alpha * 3 * x ** -4) / (den * den) / (p.a - p.c);
  return { f, df: f * dgdr };
};

/** phi_2(r, Z) and its partial derivatives in r and Z. */
export const phi2Terms = (p: EDIPEntry, r: number, Z: number): { e: number; dr: number; dZ: number } => {
  const bz = Math.exp(-p.beta * Z * Z);
  const pw = (p.B / r) ** p.rho;
  const ex = Math.exp(p.sigma / (r - p.a));
  const br = pw - bz;
  const e = p.A * br * ex;
  // d/dr [(B/r)^rho] = -rho (B/r)^rho / r ; d/dr exp(sigma/(r-a)) = -sigma/(r-a)^2 exp(...)
  const dr = p.A * ((-p.rho * pw) / r * ex + br * ex * (-p.sigma / ((r - p.a) * (r - p.a))));
  const dZ = p.A * (2 * p.beta * Z * bz) * ex;
  return { e, dr, dZ };
};

/**
 * h(l, Z) and its partials in l and Z. Q(Z) = Q0 exp(-mu Z), tau(Z) = u1 + u2 (u3 exp(-u4 Z) - exp(-2 u4 Z)).
 */
export const hTerms = (p: EDIPEntry, l: number, Z: number): { h: number; dl: number; dZ: number } => {
  const Q = p.Q0 * Math.exp(-p.mu * Z);
  const dQ = -p.mu * Q;
  const e4 = Math.exp(-p.u4 * Z), e8 = Math.exp(-2 * p.u4 * Z);
  const tau = p.u1 + p.u2 * (p.u3 * e4 - e8);
  const dtau = p.u2 * (-p.u3 * p.u4 * e4 + 2 * p.u4 * e8);
  const s = l + tau;
  const t = s * s;
  const ex = Math.exp(-Q * t);
  // -expm1(-Qt) = 1 - exp(-Qt) without cancellation when Qt is small (Z = 4 with l + tau near 0)
  const h = p.lambda * (-Math.expm1(-Q * t) + p.eta * Q * t);
  // d/dl: lambda (ex + eta) * 2 Q s
  const dl = p.lambda * (ex + p.eta) * 2 * Q * s;
  // d/dZ: lambda (ex + eta) * (dQ t + Q dt/dZ), dt/dZ = 2 s dtau
  const dt = 2 * s * dtau;
  const dZ = p.lambda * (ex + p.eta) * (dQ * t + Q * dt);
  return { h, dl, dZ };
};

/**
 * pair_style edip (one element) and edip/multi (several elements). Parameter tables: the two-body term,
 * the coordination function and the screening exponent gamma of a leg from center i to neighbour j use
 * the entry (E_i, E_j, E_j); the three-body function h uses the entry (E_i, E_j, E_k). For the single
 * element style both are the one entry. Measured with native LAMMPS (black box) on SiC.edip (pair_style
 * edip/multi): Si-C and Si-Si dimers and five Si/C trimers (centre and neighbour types mixed, angles 100 to
 * 120 degrees) equal this rule to 1e-15 absolute, in particular the centre's coordination uses the
 * (E_i, E_m, E_m) entries of its neighbours.
 */
export class PairEDIP extends Pair {
  readonly name: string = 'edip';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;
  protected multi = false;
  protected fileName = '';
  protected entries = new Map<string, EDIPEntry>();
  /** Per type pair (nt*nt): entry (E_i, E_j, E_j). */
  protected pairP: (EDIPEntry | null)[] = [];
  /** Per type triple (nt*nt*nt): entry (E_i, E_j, E_k). */
  protected triP: (EDIPEntry | null)[] = [];
  protected elemOf = new Int32Array(0);
  private zArr = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError(`usage: pair_style ${this.name} (no arguments)`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.elemOf = new Int32Array(ntypes + 1).fill(-1);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError(`usage: pair_coeff * * filename elem1 ... elemN (style ${this.name})`);
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError(`the first 2 arguments of pair_coeff for style ${this.name} must be * *`);
    const elems = args.slice(3);
    if (elems.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style ${this.name} needs one element name per atom type (${this.ntypes}), got ${elems.length}`);
    }
    if (!this.multi && new Set(elems).size !== 1) {
      throw new StyleError(`pair style ${this.name} supports a single element (the doc: "At the moment, only a single element parameterization is implemented."); use edip/multi for several elements`);
    }
    this.fileName = args[2];
    this.entries = parseEDIPFile(ctx.readFile(args[2]), args[2]);
    const names = [...new Set(elems)];
    const nt = this.ntypes + 1;
    const get = (k: string): EDIPEntry => {
      const e = this.entries.get(k);
      if (!e) throw new StyleError(`EDIP potential file ${args[2]} has no entry for ${k}`);
      return e;
    };
    this.pairP = new Array(nt * nt).fill(null);
    this.triP = new Array(nt * nt * nt).fill(null);
    for (let ti = 1; ti <= this.ntypes; ti++) {
      const ei = elems[ti - 1];
      this.elemOf[ti] = names.indexOf(ei);
      for (let tj = 1; tj <= this.ntypes; tj++) {
        const ej = elems[tj - 1];
        this.pairP[ti * nt + tj] = get(`${ei} ${ej} ${ej}`);
        for (let tk = 1; tk <= this.ntypes; tk++) {
          this.triP[(ti * nt + tj) * nt + tk] = get(`${ei} ${ej} ${elems[tk - 1]}`);
        }
      }
    }
  }

  override initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError(`pair style ${this.name} does not support the pair_modify shift and tail options`);
    if (this.table !== 12) throw new StyleError(`pair style ${this.name} does not support the pair_modify table option`);
    if (this.pairP.length === 0) throw new StyleError(`pair style ${this.name} needs a pair_coeff command with an EDIP potential file`);
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const p = this.pairP[i * nt + j];
    if (!p || this.elemOf[i] < 0 || this.elemOf[j] < 0) throw new StyleError('all pair coeffs are not set');
    // the neighbour list must cover the coordination and three-body cutoffs of both legs
    const q = this.pairP[j * nt + i];
    return Math.max(p.a, q ? q.a : 0);
  }

  /** Per-atom Z_i (pass 1); owned atoms only. */
  protected coordination(pc: PairCompute, list: NonNullable<PairCompute['full']>): void {
    const { x, type } = pc;
    const cutsq = this.cutsq;
    const nt = this.ntypes + 1;
    const nb = list.neighbors;
    if (this.zArr.length < pc.nlocal) this.zArr = new Float64Array(pc.nlocal);
    for (let i = 0; i < pc.nlocal; i++) {
      const ti = type[i] * nt;
      let z = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti + type[j]]) continue;
        const r = Math.sqrt(rsq);
        if (r >= this.pairP[ti + type[j]]!.a) continue;
        z += coordF(this.pairP[ti + type[j]]!, r).f;
      }
      this.zArr[i] = z;
    }
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const nlocal = pc.nlocal;
    const nb = list.neighbors;
    this.coordination(pc, list);
    let evdwl = 0;
    let maxn = 0;
    for (let i = 0; i < nlocal; i++) if (list.numneigh[i] > maxn) maxn = list.numneigh[i];
    if (maxn > this.cap) this.grow(maxn);
    const sIdx = this.sIdx, sDx = this.sDx, sDy = this.sDy, sDz = this.sDz;
    const sR = this.sR, sDF = this.sDF, sT = this.sT;
    const gx = this.gx, gy = this.gy, gz = this.gz;
    for (let i = 0; i < nlocal; i++) {
      const ti = type[i] * nt;
      if (this.elemOf[type[i]] < 0) continue;
      const Z = this.zArr[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      // neighbours inside the cutoff of this centre (scratch arrays of the current centre)
      let m = 0;
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[ti + type[j]]) continue;
        const r = Math.sqrt(rsq);
        // the leg cutoff is that of its own (E_i, E_j, E_j) entry
        if (r >= this.pairP[ti + type[j]]!.a) continue;
        const cf = coordF(this.pairP[ti + type[j]]!, r);
        sIdx[m] = j; sDx[m] = dx; sDy[m] = dy; sDz[m] = dz; sR[m] = r; sDF[m] = cf.df; sT[m] = type[j];
        gx[m] = 0; gy[m] = 0; gz[m] = 0;
        m++;
      }
      // E_i = sum_j phi2(r_ij, Z_i) + sum_{j<k} phi3(r_ij, r_ik, Z_i); dEdZ = dE_i/dZ_i collects the chain term
      let Ei = 0, dEdZ = 0;
      for (let a = 0; a < m; a++) {
        const r = sR[a];
        const t2 = phi2Terms(this.pairP[ti + sT[a]]!, r, Z);
        Ei += t2.e;
        dEdZ += t2.dZ;
        // radial term: dE/dx_m = dr d / r with d = x_m - x_i
        const g = t2.dr / r;
        gx[a] += g * sDx[a]; gy[a] += g * sDy[a]; gz[a] += g * sDz[a];
      }
      for (let a = 0; a < m; a++) {
        const ra = sR[a];
        const pa = this.pairP[ti + sT[a]]!;
        const ux = sDx[a] / ra, uy = sDy[a] / ra, uz = sDz[a] / ra;
        const ga = Math.exp(pa.gamma / (ra - pa.a));
        for (let b = a + 1; b < m; b++) {
          const rb = sR[b];
          const pb = this.pairP[ti + sT[b]]!;
          const vx = sDx[b] / rb, vy = sDy[b] / rb, vz = sDz[b] / rb;
          const l = ux * vx + uy * vy + uz * vz;
          const gb = Math.exp(pb.gamma / (rb - pb.a));
          const P = ga * gb;
          const h = hTerms(this.triP[(ti + sT[a]) * nt + sT[b]]!, l, Z);
          Ei += P * h.h;
          dEdZ += P * h.dZ;
          // dP/dr = P (-gamma / (r - a)^2) for each leg; dl/dx_a = (v - l u)/ra, dl/dx_b = (u - l v)/rb
          const hA = P * (-pa.gamma / ((ra - pa.a) * (ra - pa.a))) * h.h;
          const hB = P * (-pb.gamma / ((rb - pb.a) * (rb - pb.a))) * h.h;
          const cA = (P * h.dl) / ra, cB = (P * h.dl) / rb;
          gx[a] += hA * ux + cA * (vx - l * ux);
          gy[a] += hA * uy + cA * (vy - l * uy);
          gz[a] += hA * uz + cA * (vz - l * uz);
          gx[b] += hB * vx + cB * (ux - l * vx);
          gy[b] += hB * vy + cB * (uy - l * vy);
          gz[b] += hB * vz + cB * (uz - l * vz);
        }
      }
      // chain term through Z_i: dE_i/dx_m += dE_i/dZ_i f'(r_im) u_im
      let cx = 0, cy = 0, cz = 0;
      for (let a = 0; a < m; a++) {
        const c = (dEdZ * sDF[a]) / sR[a];
        gx[a] += c * sDx[a]; gy[a] += c * sDy[a]; gz[a] += c * sDz[a];
        // F_m = -dE_i/dx_m ; the centre takes the opposite of the sum (translation invariance)
        f[3 * sIdx[a]] -= gx[a]; f[3 * sIdx[a] + 1] -= gy[a]; f[3 * sIdx[a] + 2] -= gz[a];
        cx += gx[a]; cy += gy[a]; cz += gz[a];
      }
      f[3 * i] += cx; f[3 * i + 1] += cy; f[3 * i + 2] += cz;
      evdwl += Ei;
      if (pc.eatom) pc.eatom[i] += Ei;
    }
    pc.acc.evdwl += evdwl;
  }

  // scratch of the current centre, grown on demand
  private cap = 0;
  private sIdx = new Int32Array(0);
  private sDx = new Float64Array(0);
  private sDy = new Float64Array(0);
  private sDz = new Float64Array(0);
  private sR = new Float64Array(0);
  private sDF = new Float64Array(0);
  private sT = new Int32Array(0);
  private gx = new Float64Array(0);
  private gy = new Float64Array(0);
  private gz = new Float64Array(0);

  private grow(n: number): void {
    this.cap = n;
    this.sIdx = new Int32Array(n); this.sT = new Int32Array(n);
    this.sDx = new Float64Array(n); this.sDy = new Float64Array(n); this.sDz = new Float64Array(n);
    this.sR = new Float64Array(n); this.sDF = new Float64Array(n);
    this.gx = new Float64Array(n); this.gy = new Float64Array(n); this.gz = new Float64Array(n);
  }
}

/** pair_style edip/multi: several elements, one entry per (E_i, E_j, E_k) as in the file (SiC.edip has all 8). */
export class PairEDIPMulti extends PairEDIP {
  override readonly name: string = 'edip/multi';
  protected override multi = true;
}

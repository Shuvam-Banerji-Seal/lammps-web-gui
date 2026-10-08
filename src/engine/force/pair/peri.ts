import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK } from '../../neighbor';
import { parseNum } from '../util';

/*
 * Peridynamic pair styles peri/pmb (bond-based) and peri/lps (state-based linear
 * peridynamic solid), in the discretisation of Howto_peri.rst (Parks et al.
 * 2008; Silling 2000, 2005, 2007). The bond family is built from the reference
 * positions x0 at the first evaluation; a bond is stored once per unordered
 * pair and carries the image offset of its partner, so forces only touch owned
 * atoms and periodic images need no ghost bookkeeping.
 *
 * docs.lammps.org/pair_peri.html: "Style *peri/pmb* implements the Peridynamic bond-based prototype"
 * and "Style *peri/lps* implements the Peridynamic state-based linear" (PMB, LPS).
 *
 * Measured with native LAMMPS (black box): a stretched bond adds no energy (two atoms pulled apart
 * give PotEng 0), while a contact below the short-range distance does: two atoms 0.0003 apart in
 * the reference lattice of 0.0005 with volume 1.25e-10 and c = 1.6863e22 give E_vdwl 237120129.49,
 * which is 0.5 (c_S/delta) V (r - d_pi)^2 with d_pi = 0.9 * 0.0005. The energy here is the contact
 * energy with V the mean volume of the pair (the two agree for equal volumes).
 */

/** Roundoff guard on the bond stretch numerator (Howto_peri.rst, the note after Algorithm 5). */
export const PERI_ROUNDOFF = 2.220446049250313e-16;

/**
 * Nodal volume scaling nu(|xi|) of Howto_peri.rst: a linear unitless scaling function, 1 inside
 * delta - r, linear down to 0 at delta, and 0 beyond.
 */
export const periNodalScale = (l0: number, delta: number, rnode: number): number => {
  if (l0 <= delta - rnode) return 1;
  if (l0 <= delta) return -l0 / (2 * rnode) + (delta / (2 * rnode) + 0.5);
  return 0;
};

/** Short-range interaction distance d_pi = min{0.9 |x_p - x_i|, 1.35 (r_p + r_i)} (Howto_peri.rst, eq. 15). */
export const periShortRangeDistance = (l0: number, rnode: number): number => Math.min(0.9 * l0, 1.35 * (2 * rnode));

interface PeriParams {
  /** PMB spring constant c (energy/distance/volume^2). */
  c: number;
  /** LPS bulk and shear moduli. */
  K: number;
  G: number;
  delta: number;
  s00: number;
  alpha: number;
  /** Short-range stiffness c_S = 15 * 18K / (pi delta^4) for LPS, 15 c for PMB. */
  cS: number;
}

export class PairPeri extends Pair {
  readonly name: string;
  needsHalf = true;
  needsFull = false;
  /** The virial is tallied bond by bond (the forces only touch owned atoms). */
  virialFdotr = false;
  /** Skips the roundoff guard on dr (the note on roundoff after Algorithm 5 in Howto_peri.rst). */
  roundoffGuard = true;

  private names: string[];
  private pp!: PairParams;
  private par: PeriParams | null = null;

  // bond family: one entry per unordered reference bond with |xi| <= horizon
  private built = false;
  private nBuilt = 0;
  private idsBuilt = new Int32Array(0);
  private nBonds = 0;
  private bA = new Int32Array(0);
  private bB = new Int32Array(0);
  private bShift = new Float64Array(0);
  private bL0 = new Float64Array(0);
  private bDp = new Float64Array(0);
  private bNu = new Float64Array(0);
  private bBroken = new Uint8Array(0);
  private rnode = 0;
  private xlattice = 1;
  /** Critical stretch per owned atom, from the previous evaluation (Infinity at the start). */
  private s0 = new Float64Array(0);
  /** Weighted volume m_i (LPS) and dilatation theta_i (LPS, from the last evaluation). */
  private wvol = new Float64Array(0);
  private theta = new Float64Array(0);

  constructor(readonly model: 'pmb' | 'lps') {
    super();
    this.name = model === 'pmb' ? 'peri/pmb' : 'peri/lps';
    this.names = model === 'pmb' ? ['c', 'horizon', 's00', 'alpha'] : ['K', 'G', 'horizon', 's00', 'alpha'];
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.pp = new PairParams(ntypes, this.names);
  }

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError(`pair_style ${this.name} takes no arguments`);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    const usage = `usage: pair_coeff I J ${this.names.join(' ')}`;
    if (args.length !== this.names.length + 2) throw new StyleError(usage);
    const vals = this.names.map((nm, k) => parseNum(args[k + 2], nm));
    const delta = vals[this.names.indexOf('horizon')];
    if (!(delta > 0)) throw new StyleError(`${this.name}: horizon must be > 0`);
    this.pp.setRange(args[0], args[1], vals);
    this.built = false;
  }

  initStyle(ctx: StyleContext): void {
    // the lattice spacing sets the node radius; LAMMPS's default lattice is none with spacing 1
    this.xlattice = ctx.xlattice ?? 1;
    // The family and its nodal volumes are shared by all particles, so one coefficient set serves
    // every type pair; a different set on another type pair is refused rather than ignored.
    let ref: number[] | null = null;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = i; j < nt; j++) {
        if (!this.pp.isSet(i, j)) throw new StyleError(`pair_coeff for ${this.name} is not set for types ${i} ${j}`);
        const v = this.names.map((nm) => this.pp.get(nm, i, j));
        if (ref && v.some((x, k) => x !== ref![k])) {
          throw new StyleError(`${this.name} needs the same coefficients for all type pairs (the bond family is shared)`);
        }
        ref = v;
      }
    }
    if (!ref) throw new StyleError(`pair_coeff for ${this.name} is not set`);
    const g = (nm: string) => (this.names.includes(nm) ? ref![this.names.indexOf(nm)] : 0);
    const delta = g('horizon');
    const K = g('K'), G = g('G');
    const c = this.model === 'pmb' ? g('c') : 18 * K / (Math.PI * delta ** 4);
    this.par = {
      c, K, G, delta, s00: g('s00'), alpha: g('alpha'),
      // Howto_peri.rst eq. 14: "c_S = 15 \frac{18K}{\pi \delta^4}"; for PMB, c is 18K/(pi delta^4)
      cS: 15 * c,
    };
  }

  initOne(i: number, j: number): number {
    return this.pp.get('horizon', i, j);
  }

  private params(): PeriParams {
    if (!this.par) throw new StyleError(`${this.name}: pair_coeff has not been set`);
    return this.par;
  }

  /** Bond family from the positions at the first force evaluation and the neighbor list (each physical pair once); contacts use the creation positions x0. */
  private buildFamily(pc: PairCompute): void {
    const s = pc.s;
    const list = pc.half;
    if (!list) throw new StyleError(`${this.name} needs a half neighbor list`);
    const P = this.params();
    const nl = pc.nlocal;
    const x = pc.x, x0 = s.x0!, owner = pc.nb.owner, nbr = list.neighbors;
    const A: number[] = [], B: number[] = [], SH: number[] = [], L0: number[] = [], LREF: number[] = [];
    for (let i = 0; i < list.inum; i++) {
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbr[k] & NEIGHMASK;
        let o = j, sx = 0, sy = 0, sz = 0;
        if (j >= nl) {
          // a periodic image: its owner and the image offset, which the reference configuration shares
          o = owner[j];
          sx = x[3 * j] - x[3 * o]; sy = x[3 * j + 1] - x[3 * o + 1]; sz = x[3 * j + 2] - x[3 * o + 2];
        }
        if (o === i && sx === 0 && sy === 0 && sz === 0) continue;
        // bond lengths are taken from the positions at the first evaluation (measured with native LAMMPS, black
        // box: a pair displaced before the run has no bond energy or force; a contact uses the reference x0)
        const dx = x[3 * o] - x[3 * i] + sx, dy = x[3 * o + 1] - x[3 * i + 1] + sy, dz = x[3 * o + 2] - x[3 * i + 2] + sz;
        const l0 = Math.sqrt(dx * dx + dy * dy + dz * dz);
        // Howto_peri.rst eq. 11: "\mathcal{F}_i = \{ p ~ | ~ \left\Vert {\textbf{x}_p - \textbf{x}_i} \right\Vert \leq \delta \}"
        if (!(l0 > 0 && l0 <= P.delta)) continue;
        const rx = x0[3 * o] - x0[3 * i] + sx, ry = x0[3 * o + 1] - x0[3 * i + 1] + sy, rz = x0[3 * o + 2] - x0[3 * i + 2] + sz;
        A.push(i); B.push(o); SH.push(sx, sy, sz); L0.push(l0); LREF.push(Math.sqrt(rx * rx + ry * ry + rz * rz));
      }
    }
    const nb = A.length;
    this.nBonds = nb;
    this.bA = Int32Array.from(A);
    this.bB = Int32Array.from(B);
    this.bShift = Float64Array.from(SH);
    this.bL0 = Float64Array.from(L0);
    // node radius r_i: half the lattice constant, as Howto_peri.rst chooses it; measured with native
    // LAMMPS (black box): without a lattice command the spacing is 1
    this.rnode = 0.5 * this.xlattice;
    this.bDp = new Float64Array(nb);
    this.bNu = new Float64Array(nb);
    for (let e = 0; e < nb; e++) {
      // Howto_peri.rst eq. 15 uses the reference distance |x_p - x_i| of the initial configuration
      this.bDp[e] = periShortRangeDistance(LREF[e], this.rnode);
      this.bNu[e] = periNodalScale(L0[e], P.delta, this.rnode);
    }
    this.bBroken = new Uint8Array(nb);
    const V = s.vfrac!;
    // Howto_peri.rst Algorithm 3: m(i) = sum over the family of omega |xi|^2 nu V_j, with omega = 1/|xi|
    this.wvol = new Float64Array(nl);
    for (let e = 0; e < nb; e++) {
      const a = A[e], b = B[e];
      this.wvol[a] += L0[e] * this.bNu[e] * V[b];
      this.wvol[b] += L0[e] * this.bNu[e] * V[a];
    }
    this.theta = new Float64Array(nl);
    this.s0 = new Float64Array(nl).fill(Infinity);
    this.idsBuilt = s.id.slice(0, nl);
    this.nBuilt = nl;
    this.built = true;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.vfrac || !s.x0) throw new StyleError(`pair_style ${this.name} needs atom_style peri`);
    const nl = pc.nlocal;
    if (!this.built) this.buildFamily(pc);
    else {
      let same = nl === this.nBuilt;
      for (let i = 0; same && i < nl; i++) same = s.id[i] === this.idsBuilt[i];
      if (!same) throw new StyleError(`pair_style ${this.name}: the peridynamic bonds do not match the atoms (atoms were added or deleted after the first run)`);
    }
    const P = this.params();
    const x = pc.x, f = pc.f, V = s.vfrac, vir = pc.acc.virial;
    const nb = this.nBonds, A = this.bA, B = this.bB, SH = this.bShift, L0 = this.bL0, DP = this.bDp, NU = this.bNu, BR = this.bBroken;
    const isLPS = this.model === 'lps';
    const wvol = this.wvol, theta = this.theta;
    const delta = P.delta;

    // separation of bond e in the current configuration: partner position minus owner position
    const dxOf = (e: number, out: Float64Array) => {
      const a = A[e], b = B[e];
      out[0] = x[3 * b] + SH[3 * e] - x[3 * a];
      out[1] = x[3 * b + 1] + SH[3 * e + 1] - x[3 * a + 1];
      out[2] = x[3 * b + 2] + SH[3 * e + 2] - x[3 * a + 2];
    };
    const d = new Float64Array(3);

    // LPS dilatation (Howto_peri.rst Algorithm 4) over the unbroken bonds
    if (isLPS) {
      theta.fill(0);
      for (let e = 0; e < nb; e++) {
        if (BR[e]) continue;
        dxOf(e, d);
        const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const ext = r - L0[e];
        theta[A[e]] += ext * NU[e] * V[B[e]];
        theta[B[e]] += ext * NU[e] * V[A[e]];
      }
      for (let i = 0; i < nl; i++) theta[i] = wvol[i] > 0 ? (3 / wvol[i]) * theta[i] : 0;
      // LPS strain energy over the unbroken bonds: sum_i (K/2) theta_i^2 plus the deviatoric part,
      // (15G/4) sum_j nu V_j (e_ij - theta_i |xi|/3)^2 / (m_i |xi|) over ordered pairs. Measured with native
      // LAMMPS (black box, random 5-atom configurations, uniform volumes): the fit is exact to 1e-15.
      let ek = 0;
      for (let i = 0; i < nl; i++) ek += 0.5 * P.K * theta[i] * theta[i];
      for (let e = 0; e < nb; e++) {
        if (BR[e]) continue;
        dxOf(e, d);
        const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        const ext = r - L0[e];
        const a = A[e], b = B[e];
        const eda = ext - theta[a] * L0[e] / 3, edb = ext - theta[b] * L0[e] / 3;
        ek += (15 / 4) * P.G * (NU[e] * V[b] * eda * eda / (wvol[a] * L0[e]) + NU[e] * V[a] * edb * edb / (wvol[b] * L0[e]));
      }
      pc.acc.evdwl += ek;
    }

    // short-range contact forces (Howto_peri.rst eq. 13), over every reference bond
    const cSd = P.cS / delta;
    let evdw = 0;
    for (let e = 0; e < nb; e++) {
      dxOf(e, d);
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (!(r > 0)) continue;
      const gap = r - DP[e];
      if (gap >= 0) continue;
      const k = cSd * gap;
      const a = A[e], b = B[e];
      evdw += 0.5 * cSd * gap * gap * (0.5 * (V[a] + V[b]));
      const ca = (k * V[b]) / r, cb = (k * V[a]) / r;
      for (let q = 0; q < 3; q++) {
        const fa = ca * d[q];
        f[3 * a + q] += fa;
        f[3 * b + q] -= cb * d[q];
        vir[q] += -d[q] * fa;
      }
      vir[3] += -d[0] * ca * d[1];
      vir[4] += -d[0] * ca * d[2];
      vir[5] += -d[1] * ca * d[2];
    }

    pc.acc.evdwl += evdw;

    // bond forces, bond breaking, and the new critical stretches (Algorithm 2 / Algorithm 5)
    const s0 = this.s0;
    let evb = 0;
    const smin = new Float64Array(nl).fill(Infinity);
    for (let e = 0; e < nb; e++) {
      if (BR[e]) continue;
      const a = A[e], b = B[e];
      dxOf(e, d);
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (!(r > 0)) continue;
      let dr = r - L0[e];
      if (this.roundoffGuard && Math.abs(dr) < PERI_ROUNDOFF) dr = 0;
      const stretch = dr / L0[e];
      let ga: number, gb: number;
      if (!isLPS) {
        // PMB: force on a is c s nu V_b along the bond
        ga = P.c * stretch * NU[e] * V[b];
        gb = P.c * stretch * NU[e] * V[a];
      } else {
        // LPS: (3K-5G)(theta_i/m_i + theta_j/m_j) |xi| omega + 15G e omega (1/m_i + 1/m_j), with omega = 1/|xi|
        const fh = (3 * P.K - 5 * P.G) * (theta[a] / wvol[a] + theta[b] / wvol[b]) + 15 * P.G * (dr / L0[e]) * (1 / wvol[a] + 1 / wvol[b]);
        ga = fh * NU[e] * V[b];
        gb = fh * NU[e] * V[a];
      }
      const ca = ga / r, cb = gb / r;
      for (let q = 0; q < 3; q++) {
        const fa = ca * d[q];
        f[3 * a + q] += fa;
        f[3 * b + q] -= cb * d[q];
        vir[q] += -d[q] * fa;
      }
      vir[3] += -d[0] * ca * d[1];
      vir[4] += -d[0] * ca * d[2];
      vir[5] += -d[1] * ca * d[2];
      // Howto_peri.rst: "Further, new bonds are never created during the course" and a bond
      // breaks when its stretch exceeds the smaller critical stretch of the previous step
      if (stretch > Math.min(s0[a], s0[b])) BR[e] = 1;
      if (stretch < smin[a]) smin[a] = stretch;
      if (stretch < smin[b]) smin[b] = stretch;
      if (!isLPS) evb += 0.5 * (P.c / L0[e]) * dr * dr * NU[e] * (0.5 * (V[a] + V[b]));
    }
    // no critical stretch is recorded at the run setup (historyUpdate false): bonds break from the second timestep
    // Howto_peri.rst eq. 9: s0 = s00 - alpha s_min, the minimum bond stretch of the particle this step
    const s0n = new Float64Array(nl);
    for (let i = 0; i < nl; i++) s0n[i] = smin[i] === Infinity ? Infinity : P.s00 - P.alpha * smin[i];
    if (pc.historyUpdate) this.s0 = s0n;
    pc.acc.evdwl += evb;
  }

  /** Per owned atom: the fraction of its family volume whose bonds are broken (compute damage/atom). */
  damage(vfrac: Float64Array, nlocal: number): Float64Array {
    const out = new Float64Array(nlocal);
    if (!this.built) return out;
    const tot = new Float64Array(nlocal), intact = new Float64Array(nlocal);
    for (let e = 0; e < this.nBonds; e++) {
      const a = this.bA[e], b = this.bB[e];
      tot[a] += vfrac[b]; tot[b] += vfrac[a];
      if (!this.bBroken[e]) { intact[a] += vfrac[b]; intact[b] += vfrac[a]; }
    }
    for (let i = 0; i < nlocal; i++) out[i] = tot[i] > 0 ? 1 - intact[i] / tot[i] : 0;
    return out;
  }

  /** LPS dilatation of the last evaluation (compute dilatation/atom), per owned atom. */
  dilatation(nlocal: number): Float64Array {
    if (this.model !== 'lps') throw new StyleError(`compute dilatation/atom needs pair_style peri/lps (this is ${this.name})`);
    const out = new Float64Array(nlocal);
    if (this.built) out.set(this.theta.subarray(0, Math.min(nlocal, this.theta.length)));
    return out;
  }

  /** Whether the family has been built (the bonds exist after the first evaluation). */
  isBuilt(): boolean { return this.built; }
}

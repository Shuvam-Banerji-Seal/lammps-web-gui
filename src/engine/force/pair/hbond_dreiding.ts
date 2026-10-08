import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { buildAtomMap } from '../../atoms';
import { parseNum } from '../util';

/*
 * pair_style hbond/dreiding/lj, hbond/dreiding/morse and their /angleoffset
 * variants — docs.lammps.org/pair_hbond_dreiding.html.
 *
 * Energy (the doc's equations, with theta the Acceptor-Hydrogen-Donor angle):
 *   "E  = & \left[LJ(r) | Morse(r) \right] \qquad \qquad \qquad r < r_\mathrm{in}"
 *   "= & S(r) * \left[LJ(r) | Morse(r) \right]" for r_in < r < r_out,
 *   "LJ(r)  = & AR^{-12}-BR^{-10}cos^n\theta" with A = 5 eps sigma^12 and
 *   B = 6 eps sigma^10 (the doc's expansion), and
 *   "Morse(r)  = & D_0\left\lbrace \chi^2 - 2\chi\right\rbrace cos^n\theta" with
 *   chi = exp(-alpha (r - r0)).
 *   S(r) = (r_out^2 - r^2)^2 (r_out^2 + 2 r^2 - 3 r_in^2) / (r_out^2 - r_in^2)^3
 *   (the doc's switching function), zero for r >= r_out.
 * The doc's text defines the symbols: r_in "is the inner spline distance cutoff",
 * r_out "is the outer distance cutoff", theta_c "is the angle cutoff" and n "is the
 * power of the cosine of the angle". The angle factor is measured, see below.
 *
 * Measured with native LAMMPS (black box, scratch inputs):
 *  - the angle factor is (cos theta)^n with theta the A-H-D angle (linear
 *    A-H-D = 180 degrees gives cos = -1, so odd n changes sign);
 *  - the angle cutoff: the energy is zero for theta below theta_c and nonzero
 *    at theta = theta_c (a >= comparison);
 *  - the inner/outer switch S(r) and the LJ and Morse forms match the doc
 *    equations to 1e-12 at r < r_in, inside the switching band and at r > r_out;
 *  - with hydrogens on a donor (type H in the pair_coeff), each bonded hydrogen
 *    of that type adds its own term (two hydrogens: the energy is the sum and
 *    the hydrogen-bond count is 2);
 *  - a second pair_coeff for the same donor/acceptor/hydrogen types replaces the
 *    earlier one (the last epsilon was used alone, not added), while different
 *    hydrogen types add (the doc says "the settings are cumulative" for
 *    different hydrogen types);
 *  - the energy of a donor-acceptor pair is multiplied by the special_bonds
 *    weight of that pair (0.5 gave half the energy; 0.0 excluded the pair from
 *    the list and gave count 0).
 *  - compute pair: the count is the number of donor-hydrogen-acceptor terms
 *    with r < r_out and theta at or above the angle cutoff (a term at theta = 90
 *    with cutoff 0 counts 1 although its energy is 0); the second value is the
 *    summed energy.
 *
 * Angle offset variants: the doc says "the referenced angle offset is the
 * supplementary angle of the equilibrium angle parameter". Measured with native
 * LAMMPS (black box): the factor is (cos phi)^n with
 *   phi = theta - theta_eq + 180 degrees
 * (equal to (-cos(theta - theta_eq))^n; theta_eq = 180 gives cos^n theta), and
 * the angle cutoff applies to phi (zero when phi < theta_c). The measured
 * boundary: for theta_eq = 166.6 and theta_c = 120 the energy is zero at theta
 * 106.5 and nonzero at 106.7.
 *
 * Coefficients (doc): "K = hydrogen atom type = 1 to Ntypes, or type label",
 * donor flag i or j, then epsilon and sigma (lj) or D0, alpha and r0 (morse),
 * then the optional n, r_in, r_out, angle cutoff and (angleoffset only)
 * equilibrium angle. "The last 3 coefficients for both styles are optional."
 * Measured: the equilibrium angle is accepted only after the other optional
 * values and only by the angleoffset styles.
 *
 * Restrictions: no mixing; pair_modify shift and tail are not supported (the doc
 * says "These styles do not support the" before its pair_modify shift and tail
 * options); no per-atom energy or virial; no labelmap type names. The hydrogen
 * partner of a donor is found from the bond topology (both bond directions).
 */

const DEG = Math.PI / 180;

interface HEntry {
  dtype: number;
  atype: number;
  htype: number;
  /** lj: epsilon, sigma. morse: D0, alpha, r0. */
  p1: number;
  p2: number;
  p3: number;
  n: number;
  rin: number;
  rout: number;
  /** Angle cutoff in degrees. */
  tc: number;
  /** Equilibrium angle in degrees (180 for the non-offset styles). */
  teq: number;
}

export class PairHbondDreiding extends Pair {
  readonly name: string;
  needsHalf = true;
  needsFull = false;
  keepExcluded = false;
  /** The engine adds sum x.f for the global virial; the hydrogen image correction is tallied below (the hybrid wrapper requires this flag). */
  virialFdotr = true;
  /** Settings from pair_style. */
  private nGlobal = 0;
  private rinGlobal = 0;
  private routGlobal = 0;
  private tcGlobal = 0;
  private teqGlobal = 180;
  /** Coefficient table: entries per (donor type, acceptor type), one per hydrogen type. */
  private hentries: HEntry[][] = [];
  /** Per-step tallies (compute pair). */
  hbCount = 0;
  hbEnergy = 0;

  constructor(readonly variant: 'lj' | 'morse', readonly offset: boolean) {
    super();
    this.name = `hbond/dreiding/${variant}${offset ? '/angleoffset' : ''}`;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.hentries = Array.from({ length: (ntypes + 1) * (ntypes + 1) }, () => []);
  }

  settings(args: string[]): void {
    const want = this.offset ? 5 : 4;
    if (args.length !== want) {
      throw new StyleError(`pair_style ${this.name} takes N inner outer angle_cutoff${this.offset ? ' equilibrium_angle' : ''}`);
    }
    if (!/^\d+$/.test(args[0])) throw new StyleError(`pair_style ${this.name}: N must be a non-negative integer, got '${args[0]}'`);
    this.nGlobal = Number(args[0]);
    this.rinGlobal = parseNum(args[1], 'inner distance cutoff');
    this.routGlobal = parseNum(args[2], 'outer distance cutoff');
    this.tcGlobal = parseNum(args[3], 'angle cutoff');
    this.teqGlobal = this.offset ? parseNum(args[4], 'equilibrium angle') : 180;
    this.checkDistances(this.rinGlobal, this.routGlobal);
    this.checkAngle(this.tcGlobal, 'angle cutoff');
    this.checkAngle(this.teqGlobal, 'equilibrium angle');
  }

  private checkDistances(rin: number, rout: number): void {
    if (!(rin >= 0)) throw new StyleError(`pair ${this.name}: inner distance cutoff must be >= 0`);
    if (!(rin < rout)) throw new StyleError(`pair ${this.name}: inner distance cutoff must be less than the outer distance cutoff`);
  }

  private checkAngle(deg: number, what: string): void {
    if (!(deg >= 0 && deg <= 180)) throw new StyleError(`pair ${this.name}: ${what} must be between 0 and 180 degrees`);
  }

  /**
   * pair_coeff I J hbond/dreiding/* K i|j params... ; the arguments reach this
   * method as [I, J, K, flag, params...] (hybrid removes the style name).
   */
  coeff(args: string[]): void {
    const nparam = this.variant === 'lj' ? [2, 6] : [3, 7];
    if (this.offset) nparam[1] += 1;
    if (args.length < 4 + nparam[0] || args.length > 4 + nparam[1]) {
      throw new StyleError(`pair_coeff for ${this.name}: expected I J K i|j with ${nparam[0]} to ${nparam[1]} parameters`);
    }
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const [hlo, hhi] = typeBounds(args[2], this.ntypes);
    const flag = args[3];
    if (flag !== 'i' && flag !== 'j') throw new StyleError(`pair_coeff for ${this.name}: donor flag must be i or j, got '${flag}'`);
    const num = (k: number, what: string) => parseNum(args[4 + k], what);
    const p1 = num(0, this.variant === 'lj' ? 'epsilon' : 'D0');
    const p2 = num(1, this.variant === 'lj' ? 'sigma' : 'alpha');
    let p3 = 0;
    let k = 2;
    if (this.variant === 'morse') { p3 = num(2, 'r0'); k = 3; }
    const nv = args.length - 4 - k;
    const n = nv > 0 ? Number(args[4 + k]) : this.nGlobal;
    if (nv > 0 && (!/^\d+$/.test(args[4 + k]))) throw new StyleError(`pair_coeff for ${this.name}: n must be a non-negative integer`);
    const rin = nv > 1 ? parseNum(args[4 + k + 1], 'inner distance cutoff') : this.rinGlobal;
    const rout = nv > 2 ? parseNum(args[4 + k + 2], 'outer distance cutoff') : this.routGlobal;
    const tc = nv > 3 ? parseNum(args[4 + k + 3], 'angle cutoff') : this.tcGlobal;
    const teq = nv > 4 ? parseNum(args[4 + k + 4], 'equilibrium angle') : this.teqGlobal;
    this.checkDistances(rin, rout);
    this.checkAngle(tc, 'angle cutoff');
    this.checkAngle(teq, 'equilibrium angle');
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const dtype = flag === 'i' ? i : j;
        const atype = flag === 'i' ? j : i;
        for (let h = hlo; h <= hhi; h++) {
          const e: HEntry = { dtype, atype, htype: h, p1, p2, p3, n, rin, rout, tc, teq };
          const list = this.hentries[dtype * (this.ntypes + 1) + atype];
          const at = list.findIndex((x) => x.htype === h);
          // a repeated pair_coeff for the same donor, acceptor and hydrogen type replaces the earlier one
          if (at >= 0) list[at] = e; else list.push(e);
        }
      }
    }
  }

  initStyle(_ctx: StyleContext): void {
    // pair_modify shift and tail are not supported by these styles (pair_hbond_dreiding.html)
    if (this.shift || this.tail) throw new StyleError(`pair_modify shift and tail are not supported by pair style ${this.name}`);
  }

  /** Cutoff of a type pair: the largest outer cutoff of its donor/acceptor entries. */
  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    let c = 0;
    for (const list of [this.hentries[i * nt + j], this.hentries[j * nt + i]]) {
      for (const e of list) if (e.rout > c) c = e.rout;
    }
    return c;
  }

  extract(name: string): unknown {
    if (name === 'hbond_count') return this.hbCount;
    if (name === 'hbond_energy') return this.hbEnergy;
    return undefined;
  }

  compute(pc: PairCompute): void {
    if (pc.eatom || pc.vatom) throw new StyleError(`per-atom energy and virial are not supported by pair style ${this.name}`);
    this.hbCount = 0;
    this.hbEnergy = 0;
    const list = pc.half!;
    const partners = this.hydrogenPartners(pc);
    const nt = this.ntypes + 1;
    const ctx = { pc, partners, nt };
    const sLJ = pc.specialLJ;
    const nb = list.neighbors;
    const { x, type } = pc;
    for (let i = 0; i < list.inum; i++) {
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        // special_bonds weight of the pair: the measured energy scales with it (0 excludes the pair)
        const factor = sLJ[jj >>> SBBITS];
        if (factor === 0) continue;
        // each half-list pair is tried as donor i / acceptor j and as donor j / acceptor i
        const ti = type[i], tj = type[j];
        if (this.hentries[ti * nt + tj].length) this.orient(ctx, i, j, factor, rsq);
        if (this.hentries[tj * nt + ti].length) this.orient(ctx, j, i, factor, rsq);
      }
    }
    pc.acc.evdwl += this.hbEnergy;
  }

  /** Owned indices of the bonded partners of each owned atom (both bond directions). */
  private hydrogenPartners(pc: PairCompute): Int32Array[] {
    const n = pc.nlocal;
    const bonds = pc.s.topo.bonds;
    const map = buildAtomMap(pc.s);
    const deg = new Int32Array(n);
    const pairs: number[] = [];
    for (let b = 0; b < bonds.n; b++) {
      const a = map[bonds.atoms[2 * b]], c = map[bonds.atoms[2 * b + 1]];
      if (a < 0 || c < 0) throw new StyleError(`pair style ${this.name}: a bond refers to an atom that is not in the system`);
      pairs.push(a, c);
      deg[a]++; deg[c]++;
    }
    const out: Int32Array[] = [];
    for (let k = 0; k < n; k++) out.push(new Int32Array(deg[k]));
    const fill = new Int32Array(n);
    for (let p = 0; p < pairs.length; p += 2) {
      const a = pairs[p], c = pairs[p + 1];
      out[a][fill[a]++] = c;
      out[c][fill[c]++] = a;
    }
    return out;
  }

  /** Donor d, acceptor a (indices into the owned+ghost arrays). */
  private orient(ctx: { pc: PairCompute; partners: Int32Array[]; nt: number }, d: number, a: number, factor: number, rsq: number): void {
    const { pc, partners, nt } = ctx;
    const dt = pc.type[d], at = pc.type[a];
    const entries = this.hentries[dt * nt + at];
    if (!entries.length) return;
    const owner = pc.nb.owner;
    const od = d < pc.nlocal ? d : owner[d];
    const x = pc.x, f = pc.f, type = pc.type, geom = pc.geom;
    const hs = partners[od];
    // pair vector A - D (images consistent with the neighbour list)
    const rv = [x[3 * a] - x[3 * d], x[3 * a + 1] - x[3 * d + 1], x[3 * a + 2] - x[3 * d + 2]];
    const r = Math.sqrt(rsq);
    for (const e of entries) {
      if (!(r < e.rout)) continue;
      for (let q = 0; q < hs.length; q++) {
        const h = hs[q];
        if (type[h] !== e.htype) continue;
        const oa = a < pc.nlocal ? a : owner[a];
        if (oa === h) throw new StyleError(`pair style ${this.name}: the acceptor is the hydrogen bonded to its donor`);
        // H - D, minimum image of the owned bond vector
        const hv = [x[3 * h] - x[3 * od], x[3 * h + 1] - x[3 * od + 1], x[3 * h + 2] - x[3 * od + 2]];
        geom.minimumImage(hv);
        this.term(pc, e, d, a, h, factor, rv, r, hv);
      }
    }
  }

  private term(pc: PairCompute, e: HEntry, d: number, a: number, h: number, factor: number, rv: number[], r: number, hv: number[]): void {
    // u = D - H, w = A - H
    const u = [-hv[0], -hv[1], -hv[2]];
    const w = [rv[0] - hv[0], rv[1] - hv[1], rv[2] - hv[2]];
    const uL = Math.hypot(u[0], u[1], u[2]), wL = Math.hypot(w[0], w[1], w[2]);
    if (!(uL > 0) || !(wL > 0)) throw new StyleError(`pair style ${this.name}: degenerate donor-hydrogen-acceptor geometry`);
    let c = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (uL * wL);
    c = Math.max(-1, Math.min(1, c));
    const theta = Math.acos(c);
    // phi = theta - theta_eq + 180 degrees; the factor is cos(phi)^n and the cutoff applies to phi
    const phi = theta - e.teq * DEG + Math.PI;
    const phiDeg = phi / DEG;
    if (phiDeg < e.tc - 1e-9) return;
    this.hbCount += 1;
    const cphi = Math.cos(phi), sphi = Math.sin(phi);
    const n = e.n;
    const G = Math.pow(cphi, n);
    const dG = n === 0 ? 0 : -n * Math.pow(cphi, n - 1) * sphi;
    // radial part R(r) and switching S(r)
    const ro2 = e.rout * e.rout, ri2 = e.rin * e.rin, r2 = r * r;
    let S = 1, dS = 0;
    if (r > e.rin) {
      const den = (ro2 - ri2) ** 3;
      S = (ro2 - r2) ** 2 * (ro2 + 2 * r2 - 3 * ri2) / den;
      dS = 12 * r * (ro2 - r2) * (ri2 - r2) / den;
    }
    let R: number, dR: number;
    if (this.variant === 'lj') {
      const eps = e.p1, sig = e.p2;
      const s12 = sig ** 12, s10 = sig ** 10;
      const ir = 1 / r;
      const ir10 = ir ** 10;
      R = eps * (5 * s12 * ir10 * ir * ir - 6 * s10 * ir10);
      dR = eps * (-60 * s12 * ir10 * ir ** 3 + 60 * s10 * ir10 * ir);
    } else {
      const D0 = e.p1, alpha = e.p2, r0 = e.p3;
      const ex = Math.exp(-alpha * (r - r0));
      const e2 = ex * ex;
      R = D0 * (e2 - 2 * ex);
      dR = D0 * (-2 * alpha * e2 + 2 * alpha * ex);
    }
    this.hbEnergy += factor * S * R * G;
    const dEdr = factor * G * (dS * R + S * dR);
    const gth = factor * S * R * dG;
    // angular derivatives dtheta/du, dtheta/dw (zero when the angle is 0 or 180 degrees, where the gradient is undefined)
    const s = Math.sqrt(Math.max(0, 1 - c * c));
    const dthu = [0, 0, 0], dthw = [0, 0, 0];
    if (s > 1e-10) {
      const uh = [u[0] / uL, u[1] / uL, u[2] / uL], wh = [w[0] / wL, w[1] / wL, w[2] / wL];
      for (let k = 0; k < 3; k++) {
        dthu[k] = -(wh[k] - c * uh[k]) / (uL * s);
        dthw[k] = -(uh[k] - c * wh[k]) / (wL * s);
      }
    }
    const f = pc.f;
    const rinv = 1 / r;
    const fh = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const rhat = rv[k] * rinv;
      const gA = dEdr * rhat + gth * dthw[k];
      const gD = -dEdr * rhat + gth * dthu[k];
      const gH = -gth * (dthu[k] + dthw[k]);
      f[3 * a + k] -= gA;
      f[3 * d + k] -= gD;
      f[3 * h + k] -= gH;
      fh[k] = -gH;
    }
    // The engine's virial is sum x_k . f_k over owned and ghost atoms, with the owned hydrogen position.
    // The hydrogen term must use the image bonded to the donor (x_D + hv), so add the difference.
    const x = pc.x, v = pc.acc.virial;
    const d0 = x[3 * d] + hv[0] - x[3 * h], d1 = x[3 * d + 1] + hv[1] - x[3 * h + 1], d2 = x[3 * d + 2] + hv[2] - x[3 * h + 2];
    v[0] += d0 * fh[0];
    v[1] += d1 * fh[1];
    v[2] += d2 * fh[2];
    v[3] += d0 * fh[1];
    v[4] += d0 * fh[2];
    v[5] += d1 * fh[2];
  }
}

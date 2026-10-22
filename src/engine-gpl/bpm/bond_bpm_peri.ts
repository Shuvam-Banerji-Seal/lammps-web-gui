/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm_peri.cpp and src/BPM/bond_bpm_peri.h at commit b332f28 (the four bond models PMB, LPS,
 * VES and EPS on the BPM framework, with the per-type-pair critical-stretch break test). Authors credited in the
 * BPM sources: Joel Clemmer (SNL), with the PERI models of Mike Parks (SNL) and Rezwanur Rahman and J. T. Foster
 * (UTSA). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * bond_style bpm/peri <model> ...: docs.lammps.org/bond_bpm_peri.html. The doc says the bond force uses "the
 * peridynamic micromodulus c" for pmb, the bulk and shear moduli K and G for lps, ves and eps, and the critical
 * stretch s00 - alpha * smin, where smin is the minimum stretch of the atom's bonds in the previous step (a bond
 * breaks when its stretch exceeds the more tensile of its two atoms' critical stretches).
 *
 * Browser-engine differences: the per-atom arrays of the C++ (smin, vinter, wvolume, theta, tdnorm and the EPS
 * plastic multiplier) are maps keyed by atom ID; every atom is owned, so the C++ ghost communication is not needed
 * (periodic images come from the bonded copy table). The per-bond history of the C++ (bondstore: r0, and the VES
 * deviator extensions or the EPS plastic extension) is keyed by the bond's two atom IDs. The nodal volume vfrac and
 * the damage are the custom per-atom properties of fix property/atom (d_vfrac, d_damage); the diagnostic per-atom
 * properties d_theta, d_s0 and d_lambda are written when the fix declares them. The lattice spacing comes from the
 * style context (lattice spacing in x). Refused by name: store/local (no break rows are written).
 */
import { atomIndex, delta } from '../../engine/force/bonded_util';
import { StyleError, type BondedCompute, type StyleContext, bondedCoeffArgs } from '../../engine/force/types';
import { numeric } from '../../engine/force/util';
import { BondBPMBase, bondKey } from './bond_bpm';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

// LAMMPS: NEAR_ZERO of bond_bpm_peri.cpp (roundoff noise in the stretch)
const NEAR_ZERO = 2.2204e-16;
const DBL_MAX = Number.MAX_VALUE;
const THIRD = 1 / 3;

// LAMMPS: the model enum of bond_bpm_peri.h (PMB, LPS, VES, EPS)
const PMB = 0, LPS = 1, VES = 2, EPS = 3;
const MODEL_OF: Record<string, number> = { pmb: PMB, lps: LPS, ves: VES, eps: EPS };

/** Linear partial-volume taper of the reference length r0 near the horizon (vfrac_taper in the C++ file). */
const vfracTaper = (r0: number, delta0: number, halfLc: number): number => {
  if (Math.abs(r0 - delta0) <= halfLc) return (-1 / (2 * halfLc)) * r0 + (1 + (delta0 - halfLc) / (2 * halfLc));
  return 1;
};

/** Per-bond geometry of one compute call (indexed by the bond index k; entries of removed bonds are unused). */
interface Geo {
  t: Int32Array; i1: Int32Array; i2: Int32Array; lo: Int32Array; hi: Int32Array;
  dx: Float64Array; dy: Float64Array; dz: Float64Array; r: Float64Array; r0: Float64Array; dr: Float64Array; scale: Float64Array;
}

/** Per-atom state of the state-based models for one compute call (indexed by the local atom index). */
interface StateArrays {
  theta: Float64Array;
  winv: Float64Array;
  tdnorm: Float64Array;
}
const NO_STATE: StateArrays = { theta: new Float64Array(0), winv: new Float64Array(0), tdnorm: new Float64Array(0) };

export class BondBpmPeri extends BondBPMBase {
  readonly name = 'bpm/peri';
  readonly kind = 'bond' as const;
  readonly paramNames = ['c', 'cut', 's00', 'alpha'];
  /** Per bond type: the model and its coefficients (bond_bpm_peri.cpp coeff; unused ones stay 0). */
  private model = new Int8Array(0);
  private c = new Float64Array(0);
  private kbulk = new Float64Array(0);
  private gshear = new Float64Array(0);
  private lamb = new Float64Array(0);
  private tau = new Float64Array(0);
  private yieldS = new Float64Array(0);
  private cut = new Float64Array(0);
  private s00 = new Float64Array(0);
  private alpha = new Float64Array(0);
  private xlattice = 1;
  /** Per bond (key): the reference length r0 (bondstore[.][0]). */
  private readonly ref = new Map<number, number>();
  /** Per bond (key): the two evolving history values (VES: dev, devback; EPS: edp). */
  private readonly hist = new Map<number, Float64Array>();
  /** Per atom ID: the minimum stretch of its bonds at the previous compute (LAMMPS smin; -DBL_MAX = none). */
  private smin = new Map<number, number>();
  /** Per atom ID: the reference interaction volume (LAMMPS vinter), set at the first compute. */
  private vinter = new Map<number, number>();
  /** Per atom ID: the weighted volume (LAMMPS wvolume), built once per run from the reference bonds. */
  private wvolume = new Map<number, number>();
  private wvolumeSetup = false;
  /** Per atom ID: the accumulated EPS plastic multiplier (LAMMPS lambdaValue, a diagnostic). */
  private lambdaAcc = new Map<number, number>();
  private stored = false;
  /** Set in init from the active models (bond_bpm_peri.cpp init_style). */
  private stateBased = false;
  private plastic = false;
  private kbulkRep = 0;
  private gshearRep = 0;
  private pointwiseYield = 0;

  settings(args: string[], _ctx?: StyleContext): void {
    const rest = this.takeCommon(args);
    if (rest.length) throw new StyleError(`Illegal bond_style bpm/peri argument: ${rest[0]}`);
  }

  /** The model coefficients are not written to a data file (LAMMPS writedata = 0). */
  dataCoeffs(): string[] { return []; }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const n = ntypes + 1;
    this.model = new Int8Array(n);
    this.c = new Float64Array(n); this.kbulk = new Float64Array(n); this.gshear = new Float64Array(n);
    this.lamb = new Float64Array(n); this.tau = new Float64Array(n); this.yieldS = new Float64Array(n);
    this.cut = new Float64Array(n); this.s00 = new Float64Array(n); this.alpha = new Float64Array(n);
  }

  /**
   * bond_coeff <type> <model> <coeffs>: pmb c horizon s00 alpha; lps K G horizon s00 alpha; ves K G lambda tau horizon
   * s00 alpha; eps K G yieldstress horizon s00 alpha (bond_bpm_peri.cpp coeff).
   */
  coeff(args: string[], _ctx?: StyleContext): void {
    // As native LAMMPS does (src/BPM/bond_bpm_peri.cpp BondBPMPeri::coeff at b332f28): the type range is read first,
    // then each model checks its own argument count before utils::numeric reads its values.
    if (args.length < 2) throw bondedCoeffArgs('bond');
    const [lo, hi] = this.bounds(args[0]);
    const model = MODEL_OF[args[1]];
    const num = (k: number): number => numeric(args[k]);
    let c = 0, kbulk = 0, gshear = 0, lamb = 0, tau = 0, yieldS = 0, cut = 0, s00 = 0, alpha = 0;
    if (model === PMB) {
      if (args.length !== 6) throw new StyleError('Incorrect args for bond_style bpm/peri pmb model (expected: <type> pmb c horizon s00 alpha)');
      c = num(2); cut = num(3); s00 = num(4); alpha = num(5);
    } else if (model === LPS) {
      if (args.length !== 7) throw new StyleError('Incorrect args for bond_style bpm/peri lps model (expected: <type> lps K G horizon s00 alpha)');
      kbulk = num(2); gshear = num(3); cut = num(4); s00 = num(5); alpha = num(6);
    } else if (model === VES) {
      if (args.length !== 9) throw new StyleError('Incorrect args for bond_style bpm/peri ves model (expected: <type> ves K G lambda tau horizon s00 alpha)');
      kbulk = num(2); gshear = num(3); lamb = num(4); tau = num(5); cut = num(6); s00 = num(7); alpha = num(8);
      if (!(tau > 0)) throw new StyleError(`Invalid relaxation time ${tau} for bond_style bpm/peri ves`);
    } else if (model === EPS) {
      if (args.length !== 8) throw new StyleError('Incorrect args for bond_style bpm/peri eps model (expected: <type> eps K G yieldstress horizon s00 alpha)');
      kbulk = num(2); gshear = num(3); yieldS = num(4); cut = num(5); s00 = num(6); alpha = num(7);
      if (!(yieldS > 0)) throw new StyleError(`Invalid yield stress ${yieldS} for bond_style bpm/peri eps`);
    } else {
      throw new StyleError(`Unknown bond_style bpm/peri model: ${args[1]}`);
    }
    if (!(cut > 0)) throw new StyleError(`Invalid horizon value ${cut} for bond_style bpm/peri`);
    for (let t = lo; t <= hi; t++) {
      this.model[t] = model;
      this.c[t] = c; this.kbulk[t] = kbulk; this.gshear[t] = gshear; this.lamb[t] = lamb;
      this.tau[t] = tau; this.yieldS[t] = yieldS;
      this.cut[t] = cut; this.s00[t] = s00; this.alpha[t] = alpha;
      this.params.set[t] = 1;
    }
  }

  private bounds(w: string): [number, number] {
    const n = Number(w);
    if (Number.isInteger(n) && n >= 1 && n <= this.ntypes) return [n, n];
    throw new StyleError(`bond_style bpm/peri: bond type ${w} out of range`);
  }

  init(ctx: StyleContext): void {
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.params.set[t]) throw new StyleError(`all bond coeffs are not set (type ${t})`);
    }
    this.xlattice = ctx.xlattice ?? 1;
    const s = ctx.s;
    const vf = s?.custom.get('vfrac');
    if (!vf || vf.cols !== 0 || vf.int) {
      throw new StyleError("Bond style bpm/peri requires a per-atom vfrac property; add 'fix <ID> all property/atom d_vfrac ghost yes' before bond_style");
    }
    // representative moduli and the EPS yield value (bond_bpm_peri.cpp init_style: the last set type of a model wins)
    this.stateBased = false; this.plastic = false;
    this.kbulkRep = 0; this.gshearRep = 0; this.pointwiseYield = 0;
    for (let t = 1; t <= this.ntypes; t++) {
      if (!this.params.set[t]) continue;
      const m = this.model[t];
      if (m === LPS || m === VES || m === EPS) {
        this.stateBased = true;
        this.kbulkRep = this.kbulk[t];
      }
      if (m === EPS) {
        this.plastic = true;
        this.gshearRep = this.gshear[t];
        this.pointwiseYield = (25 / (8 * Math.PI * libmPow(this.cut[t], 5))) * this.yieldS[t] * this.yieldS[t];
      }
    }
    // the weighted volume is rebuilt in each run (bond_bpm_peri.cpp init_style sets wvolume_setup = 0)
    this.wvolumeSetup = false;
  }

  /** The reference length of bond (key), captured from r at its first visit (bond_bpm_peri.cpp compute). */
  private refOf(key: number, r: number): number {
    let r0 = this.ref.get(key);
    if (r0 === undefined) {
      r0 = r;
      this.ref.set(key, r0);
    }
    return r0;
  }

  /** The two history values of bond (key), created zero on first use (bond_bpm_peri.cpp store_data). */
  private histOf(key: number): Float64Array {
    let h = this.hist.get(key);
    if (!h) {
      h = new Float64Array(2);
      this.hist.set(key, h);
    }
    return h;
  }

  /** Stores r0 and the reference interaction volume of every bond at the first compute (store_data). */
  private storeReference(bc: BondedCompute): void {
    const b = bc.s.topo.bonds;
    const vfrac = bc.s.custom.get('vfrac')!.data;
    const d = [0, 0, 0];
    for (let k = 0; k < b.n; k++) {
      if (b.type[k] <= 0) continue;
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const i1 = atomIndex(bc, ia, 'bond'), i2 = atomIndex(bc, ib, 'bond');
      delta(bc, i1, i2, d);
      this.ref.set(bondKey(ia, ib), Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
      this.vinter.set(ia, (this.vinter.get(ia) ?? 0) + vfrac[i2]);
      this.vinter.set(ib, (this.vinter.get(ib) ?? 0) + vfrac[i1]);
    }
    this.stored = true;
  }

  /** The weighted volume of every atom from the reference bonds (compute_wvolume): sum r0 * vfrac[j] * taper. */
  private buildWvolume(bc: BondedCompute, vfrac: Float64Array, g: Geo): void {
    const s = bc.s, b = s.topo.bonds;
    this.wvolume.clear();
    for (let i = 0; i < s.n; i++) this.wvolume.set(s.id[i], 0);
    for (let k = 0; k < b.n; k++) {
      if (g.t[k] <= 0) continue;
      const r0 = g.r0[k], scale = g.scale[k];
      const ia = g.lo[k], ib = g.hi[k];
      this.wvolume.set(ia, (this.wvolume.get(ia) ?? 0) + r0 * vfrac[g.i2[k]] * scale);
      this.wvolume.set(ib, (this.wvolume.get(ib) ?? 0) + r0 * vfrac[g.i1[k]] * scale);
    }
    this.wvolumeSetup = true;
  }

  /**
   * The geometry of every bond present at the start of the compute (type > 0): the endpoint indices, the vector
   * del = x_i1 - x_i2 with i1 the lower ID, the length r, the reference length r0, the stretch increment dr (roundoff
   * zeroed) and the partial-volume taper. All passes of the compute (dilatation, EPS norm, bond force) read it, as the
   * C++ passes read the same positions.
   */
  private buildGeo(bc: BondedCompute, halfLc: number): Geo {
    const b = bc.s.topo.bonds;
    const g: Geo = {
      t: new Int32Array(b.n), i1: new Int32Array(b.n), i2: new Int32Array(b.n),
      lo: new Int32Array(b.n), hi: new Int32Array(b.n),
      dx: new Float64Array(b.n), dy: new Float64Array(b.n), dz: new Float64Array(b.n),
      r: new Float64Array(b.n), r0: new Float64Array(b.n), dr: new Float64Array(b.n), scale: new Float64Array(b.n),
    };
    const d = [0, 0, 0];
    for (let k = 0; k < b.n; k++) {
      const t = b.type[k];
      if (t <= 0) continue;
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const lo = Math.min(ia, ib), hi = Math.max(ia, ib);
      const i1 = atomIndex(bc, lo, 'bond'), i2 = atomIndex(bc, hi, 'bond');
      delta(bc, i1, i2, d); // x_i2 - x_i1
      const dx = -d[0], dy = -d[1], dz = -d[2]; // del = x_i1 - x_i2 as the C++ bond
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      const r0 = this.refOf(bondKey(lo, hi), r);
      let dr = r - r0;
      if (Math.abs(dr) < NEAR_ZERO) dr = 0;
      g.t[k] = t; g.i1[k] = i1; g.i2[k] = i2; g.lo[k] = lo; g.hi[k] = hi;
      g.dx[k] = dx; g.dy[k] = dy; g.dz[k] = dz; g.r[k] = r; g.r0[k] = r0; g.dr[k] = dr;
      g.scale[k] = vfracTaper(r0, this.cut[t], halfLc);
    }
    return g;
  }

  /**
   * The per-step state of the state-based models, from the bonds present at the start of the compute (before this
   * step's breaks, as in the C++ passes): the inverse weighted volume, the dilatation theta (compute_dilatation,
   * published to the d_theta property when the fix declares it), the volumetric energy 0.5 K theta^2 (added by the
   * caller) and, for EPS, the deviatoric force-state norm with the yield check (compute_plastic_state).
   */
  private prepState(bc: BondedCompute, g: Geo, vfrac: Float64Array): { st: StateArrays; evol: number } {
    const s = bc.s, b = s.topo.bonds, n = s.n;
    if (!this.wvolumeSetup) this.buildWvolume(bc, vfrac, g);
    const wv = new Float64Array(n), winv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const w = this.wvolume.get(s.id[i]) ?? 0;
      wv[i] = w;
      winv[i] = w > 0 ? 1 / w : 0;
    }
    const theta = new Float64Array(n);
    for (let k = 0; k < b.n; k++) {
      const t = g.t[k];
      if (t <= 0) continue;
      const m = this.model[t];
      if (m !== LPS && m !== VES && m !== EPS) continue; // the PMB bonds carry no dilatation
      const i1 = g.i1[k], i2 = g.i2[k];
      theta[i1] += g.dr[k] * vfrac[i2] * g.scale[k];
      theta[i2] += g.dr[k] * vfrac[i1] * g.scale[k];
    }
    for (let i = 0; i < n; i++) theta[i] = wv[i] > 0 ? (3 / wv[i]) * theta[i] : 0;
    let evol = 0;
    const th = s.custom.get('theta');
    for (let i = 0; i < n; i++) {
      evol += 0.5 * this.kbulkRep * theta[i] * theta[i];
      if (th && th.cols === 0 && !th.int) th.data[i] = theta[i];
    }
    const st: StateArrays = { theta, winv, tdnorm: new Float64Array(0) };
    if (this.plastic) st.tdnorm = this.plasticState(bc, g, vfrac, st, wv);
    return { st, evol };
  }

  /** EPS: the deviatoric force-state norm per atom and the plastic multiplier (compute_plastic_state). */
  private plasticState(bc: BondedCompute, g: Geo, vfrac: Float64Array, st: StateArrays, wv: Float64Array): Float64Array {
    const s = bc.s, b = s.topo.bonds, n = s.n;
    const tdn = new Float64Array(n);
    const { theta, winv } = st;
    for (let k = 0; k < b.n; k++) {
      const t = g.t[k];
      if (t <= 0 || this.model[t] !== EPS) continue;
      const i1 = g.i1[k], i2 = g.i2[k];
      const winv1 = winv[i1], winv2 = winv[i2];
      if (winv1 <= 0 || winv2 <= 0) continue;
      const r0 = g.r0[k];
      const dev = g.dr[k] - 0.5 * (theta[i1] + theta[i2]) * r0 * THIRD;
      const edp = this.histOf(bondKey(g.lo[k], g.hi[k]))[0];
      const tdtrial = 15 * this.gshear[t] * (1 / r0) * (winv1 + winv2) * (dev - edp);
      tdn[i1] += tdtrial * tdtrial * vfrac[i2] * g.scale[k];
      tdn[i2] += tdtrial * tdtrial * vfrac[i1] * g.scale[k];
    }
    for (let i = 0; i < n; i++) {
      const norm = Math.sqrt(tdn[i]);
      tdn[i] = norm;
      if (0.5 * norm * norm - this.pointwiseYield > 0) {
        const id = s.id[i];
        this.lambdaAcc.set(id, (this.lambdaAcc.get(id) ?? 0) +
          (norm / Math.sqrt(2 * this.pointwiseYield) - 1) * wv[i] / (15 * this.gshearRep));
      }
    }
    return tdn;
  }

  compute(bc: BondedCompute): void {
    if (!this.stored) this.storeReference(bc);
    const b = bc.s.topo.bonds;
    const s = bc.s;
    const vfrac = s.custom.get('vfrac')!.data;
    const halfLc = 0.5 * this.xlattice;
    const allow = bc.historyUpdate === true && this.breakOn;
    // the previous committed minimum stretch by local index, and this step's accumulators (DBL_MAX = none)
    const sminPrev = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) sminPrev[i] = this.smin.get(s.id[i]) ?? -DBL_MAX;
    const sminNew = new Float64Array(s.n).fill(DBL_MAX);
    const s0Prop = s.custom.get('s0');
    const s0New = s0Prop && s0Prop.cols === 0 && !s0Prop.int ? new Float64Array(s.n).fill(-DBL_MAX) : null;
    const f = bc.f, v = bc.virial;
    const g = this.buildGeo(bc, halfLc);
    let e = 0;
    let st = NO_STATE;
    if (this.stateBased) {
      const p = this.prepState(bc, g, vfrac);
      st = p.st;
      e += p.evol;
      if (bc.eatom) for (let i = 0; i < s.n; i++) bc.eatom[i] += 0.5 * this.kbulkRep * st.theta[i] * st.theta[i];
    }
    for (let k = 0; k < b.n; k++) {
      const t = g.t[k];
      if (t <= 0) continue;
      const lo = g.lo[k], hi = g.hi[k];
      const i1 = g.i1[k], i2 = g.i2[k];
      const dx = g.dx[k], dy = g.dy[k], dz = g.dz[k];
      const r = g.r[k], r0 = g.r0[k], dr = g.dr[k];
      const stretch = dr / r0;
      const smin1 = sminPrev[i1], smin2 = sminPrev[i2];
      if (allow && stretch > this.s00[t] - this.alpha[t] * Math.max(smin1, smin2)) {
        this.breakBond(b.type, k);
        continue;
      }
      const scale = g.scale[k];
      const vEff = 0.5 * (vfrac[i1] + vfrac[i2]);
      const m = this.model[t];
      let fbond = 0, eb = 0;
      if (m === LPS) {
        const { theta, winv } = st;
        const w1 = winv[i1], w2 = winv[i2];
        const rinv0 = 1 / r0;
        let rk = 0;
        if (w1 > 0 && w2 > 0) {
          rk = (3 * this.kbulk[t] - 5 * this.gshear[t]) * vEff * scale * (theta[i1] * w1 + theta[i2] * w2);
          rk += 15 * this.gshear[t] * vEff * scale * rinv0 * (w1 + w2) * dr;
          // deviatoric energy per bond (vanishes for a pure dilatation)
          const dev1 = dr - theta[i1] * r0 * THIRD;
          const dev2 = dr - theta[i2] * r0 * THIRD;
          eb = 0.25 * 15 * this.gshear[t] * vEff * scale * rinv0 * (dev1 * dev1 * w1 + dev2 * dev2 * w2);
        }
        fbond = r > 0 ? -(rk / r) : 0;
      } else if (m === VES) {
        // elastic dilatation plus a relaxing deviatoric term with the back-extension recurrence (pair_peri_ves form)
        const { theta, winv } = st;
        const w1 = winv[i1], w2 = winv[i2];
        const rinv0 = 1 / r0;
        let rk = 0;
        if (w1 > 0 && w2 > 0) {
          rk = 3 * this.kbulk[t] * vEff * scale * (theta[i1] * w1 + theta[i2] * w2);
          const dev = dr - 0.5 * (theta[i1] + theta[i2]) * r0 * THIRD;
          const c1 = this.tau[t] / bc.s.dt;
          const decay = libmExp(-1 / c1);
          const beta = 1 - c1 * (1 - decay);
          const h = this.histOf(bondKey(lo, hi));
          const devPrior = h[0], devbackPrior = h[1];
          const edb = devPrior * (1 - decay) + devbackPrior * decay + beta * (dev - devPrior);
          const lam = this.lamb[t];
          const gfac = 15 * this.gshear[t] * vEff * scale * rinv0 * (w1 + w2);
          rk += gfac * ((1 - lam) * dev + lam * (dev - edb));
          eb = 0.25 * gfac * ((1 - lam) * dev * dev + lam * (dev - edb) * (dev - edb));
          // write back the evolving deviator history for the next step
          h[0] = dev;
          h[1] = edb;
        }
        fbond = r > 0 ? -(rk / r) : 0;
      } else if (m === EPS) {
        // elastic dilatation plus a deviatoric force state returned radially onto the yield surface (the norms
        // of both atoms come from prepState); the plastic extension edp is the bond's evolving history
        const { theta, winv, tdnorm } = st;
        const w1 = winv[i1], w2 = winv[i2];
        const rinv0 = 1 / r0;
        let rk = 0;
        if (w1 > 0 && w2 > 0) {
          const dev = dr - 0.5 * (theta[i1] + theta[i2]) * r0 * THIRD;
          const h = this.histOf(bondKey(lo, hi));
          const edp = h[0];
          rk = 3 * this.kbulk[t] * (theta[i1] * w1 + theta[i2] * w2);
          const tdtrial = 15 * this.gshear[t] * rinv0 * (w1 + w2) * (dev - edp);
          const tdnormB = 0.5 * (tdnorm[i1] + tdnorm[i2]);
          const yieldnorm = Math.sqrt(2 * this.pointwiseYield);
          let rkdev = tdtrial;
          let edpNew = edp;
          if (tdnormB > yieldnorm) {
            const radial = yieldnorm / tdnormB; // < 1 in the plastic regime
            rkdev = radial * tdtrial;
            edpNew = edp + (dev - edp) * (1 - radial);
          }
          rk += rkdev;
          // recoverable deviatoric elastic energy uses the returned elastic extension
          const edev = dev - edpNew;
          eb = 0.25 * 15 * this.gshear[t] * rinv0 * (w1 + w2) * edev * edev * vEff * scale;
          h[0] = edpNew;
        }
        fbond = r > 0 ? -((rk / r) * vEff * scale) : 0;
      } else {
        // PMB bond force and energy
        fbond = r > 0 ? -(this.c[t] * vEff * scale * stretch) / r : 0;
        eb = 0.5 * this.c[t] * vEff * scale * stretch * dr;
      }
      f[3 * i1] += dx * fbond; f[3 * i1 + 1] += dy * fbond; f[3 * i1 + 2] += dz * fbond;
      f[3 * i2] -= dx * fbond; f[3 * i2 + 1] -= dy * fbond; f[3 * i2 + 2] -= dz * fbond;
      e += eb;
      // ev_tally(i1, i2, ..., fbond * vfrac_eff, del): v = del (x) del * (fbond * vfrac_eff)
      const fv = fbond * vEff;
      v[0] += dx * dx * fv; v[1] += dy * dy * fv; v[2] += dz * dz * fv;
      v[3] += dx * dy * fv; v[4] += dx * dz * fv; v[5] += dy * dz * fv;
      if (bc.eatom) { bc.eatom[i1] += 0.5 * eb; bc.eatom[i2] += 0.5 * eb; }
      // per-atom: half of v to each atom (Bond::ev_tally, newton_bond on)
      if (bc.vatom) { const va = bc.vatom; const w = [dx * dx * fv, dy * dy * fv, dz * dz * fv, dx * dy * fv, dx * dz * fv, dy * dz * fv]; for (let c = 0; c < 6; c++) { va[6 * i1 + c] += 0.5 * w[c]; va[6 * i2 + c] += 0.5 * w[c]; } }
      const crit = this.s00[t] - this.alpha[t] * stretch;
      if (stretch < sminNew[i1]) sminNew[i1] = stretch;
      if (stretch < sminNew[i2]) sminNew[i2] = stretch;
      if (s0New) {
        if (crit > s0New[i1]) s0New[i1] = crit;
        if (crit > s0New[i2]) s0New[i2] = crit;
      }
    }
    bc.acc.ebond += e;
    // commit the new minimum stretch of every atom (-DBL_MAX when it has no surviving bond)
    for (let i = 0; i < s.n; i++) this.smin.set(s.id[i], sminNew[i] === DBL_MAX ? -DBL_MAX : sminNew[i]);
    this.publish(s, s0New);
    this.updateDamage(bc, vfrac);
  }

  /** Writes the diagnostic per-atom properties the fix declares: d_s0 (critical stretch) and d_lambda (EPS plastic multiplier). */
  private publish(s: BondedCompute['s'], s0New: Float64Array | null): void {
    const s0 = s.custom.get('s0');
    const lam = s.custom.get('lambda');
    for (let i = 0; i < s.n; i++) {
      if (s0 && s0New) s0.data[i] = s0New[i] === -DBL_MAX ? DBL_MAX : s0New[i];
      if (lam && lam.cols === 0 && !lam.int) lam.data[i] = this.lambdaAcc.get(s.id[i]) ?? 0;
    }
  }

  /** Damage of each atom: 1 - (surviving partner volume) / (reference interaction volume), if d_damage exists. */
  private updateDamage(bc: BondedCompute, vfrac: Float64Array): void {
    const damage = bc.s.custom.get('damage');
    if (!damage || damage.cols !== 0 || damage.int) return;
    const s = bc.s;
    const b = s.topo.bonds;
    const surviving = new Float64Array(s.n);
    for (let k = 0; k < b.n; k++) {
      if (b.type[k] <= 0) continue;
      const i1 = atomIndex(bc, b.atoms[2 * k], 'bond'), i2 = atomIndex(bc, b.atoms[2 * k + 1], 'bond');
      surviving[i1] += vfrac[i2];
      surviving[i2] += vfrac[i1];
    }
    for (let i = 0; i < s.n; i++) {
      const vi = this.vinter.get(s.id[i]) ?? 0;
      damage.data[i] = vi > 0 ? 1 - surviving[i] / vi : 0;
    }
  }
}

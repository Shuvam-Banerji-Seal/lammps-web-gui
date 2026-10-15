/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/fix_shardlow.cpp and
 * fix_shardlow.h (FixShardlow: fix shardlow, the velocity-Verlet SSA splitting of the DPD thermostat and its
 * isoenergetic variant), src/DPD-REACT/nbin_ssa.cpp (NBinSSA: the active interaction region of the ghosts),
 * src/DPD-REACT/npair_half_bin_newton_ssa.cpp (NPairHalfBinNewtonSSA::build: the work phases and work items of the
 * local atoms, and the ghost groups), src/DPD-REACT/nstencil_half_bin_3d_ssa.cpp (the subphase stencils) and
 * src/DPD-REACT/random_external_state.h (the generator, see random_external_state.ts). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE). Contributing authors (from fix_shardlow.cpp): James Larentzos (U.S. Army Research
 * Laboratory), Timothy I. Mattox (Engility Corporation), Martin Lisal (Institute of Chemical Process Fundamentals of
 * the Czech Academy of Sciences and J. E. Purkinje University), John Brennan, Joshua Moore and William Mattson (Army
 * Research Lab).
 *
 * The fix applies the dissipative and random parts of the pair dpd/fdt or dpd/fdt/energy forces (the pair keeps only
 * the conservative force while the fix is defined: pair_dpd_fdt.ts, splitFdt). Input rules (fix_shardlow.cpp setup()):
 * the deterministic integrator (fix nve or fix nph) must follow the fix, and constant temperature fixes are refused.
 * docs.lammps.org/fix_shardlow.html describes the command.
 *
 * Order of work: the atoms are processed in the native work order, so the random draws come out in the same order:
 * first the local work phases (each phase holds work items; item w uses random state w, the states persist across
 * phases and steps), then the seven ghost groups of the active interaction region (state 0). Ghost velocities are
 * copied from their owners before each ghost group and their changes are added back to the owners afterwards, as the
 * native forward and reverse communication does. Positions and the neighbor enumeration come from the native order
 * helpers (src/engine-gpl/neighbor/nativeOrder.ts) with the ghost ordering of one processor.
 *
 * Scope: orthogonal, three-dimensional boxes (native errors for triclinic boxes; 2d is refused here). Special bonds
 * are not checked against the native list (no special-bond case is tested).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { DPD_THETA, DPD_UCOND, DPD_UMECH, massOf, nativeOrder } from '../../engine/atoms';
import { nativeGhosts } from '../neighbor/nativeOrder';
import { PairDPDfdt, PairDPDfdtEnergy } from './pair_dpd_fdt';
import { ExtState, esGenNextParallelState, esInit, esNormal } from './random_external_state';
import { pairMatch } from '../dpdrx/pair_match';

/** npair_half_bin_newton_ssa / fix_shardlow.cpp: "static constexpr double EPSILON_SQUARED = 1.0e-20;". */
const EPSILON_SQUARED = 1.0e-20;
const OFF = 4096, BW = 8192;
const binKey = (ix: number, iy: number, iz: number): number => ((iz + OFF) * BW + (iy + OFF)) * BW + (ix + OFF);

/** One item of work: the atoms of one work item, as entries (atom, neighbor range in the flat list). */
interface Entries {
  atom: number[];
  nbStart: number[];
  nbEnd: number[];
  nb: number[];
}

/** Local work phases (item ranges of entries) and the ghost groups (one item each, AIR 1..7). */
interface Plan {
  local: Entries;
  phases: [number, number][][];
  ghosts: Entries[];
}

/** nstencil_half_bin_3d_ssa.cpp: stencil offsets (3 per bin) and the subphase starts nst[0..4]; full = all entries. */
interface Stencil {
  off: number[];
  nst: number[];
  full: number;
}

/** coord2ssaAIR (nbin_ssa.cpp): the active interaction region of a ghost atom (0 = owned, -1 = not in AIR). */
const coord2ssaAIR = (x: number, y: number, z: number, lo: number[], hi: number[]): number => {
  let ix = 0, iy = 0, iz = 0;
  if (z < lo[2]) iz = -1;
  if (z >= hi[2]) iz = 1;
  if (y < lo[1]) iy = -1;
  if (y >= hi[1]) iy = 1;
  if (x < lo[0]) ix = -1;
  if (x >= hi[0]) ix = 1;
  if (iz < 0) return -1;
  if (iz === 0) {
    if (iy < 0) return -1;
    if (iy === 0 && ix < 0) return -1;
    if (iy === 0 && ix === 0) return 0;
    if (iy === 0 && ix > 0) return 2;
    if (iy > 0 && ix === 0) return 1;
    if (iy > 0 && ix !== 0) return 3;
  } else {
    if (ix === 0 && iy === 0) return 4;
    if (ix === 0 && iy !== 0) return 5;
    if (ix !== 0 && iy === 0) return 6;
    return 7;
  }
  return -2;
};

/** Array of the per-atom data in native order: owners 0..n-1, then the ghosts of nativeGhosts. */
interface Work {
  n: number;
  nall: number;
  root: Int32Array;
  img: Int32Array;
  x: Float64Array;
  v: Float64Array;
  v0: Float64Array;
  uCond: Float64Array;
  uMech: Float64Array;
  theta: Float64Array;
  mass: Float64Array;
  type: Int32Array;
}

/** Tables of the pair in the form the update kernel reads (index i * nt + j). */
interface Coeffs {
  nt: number;
  cutsq: Float64Array;
  cut: Float64Array;
  sigma: Float64Array;
  kappa: Float64Array | null;
  alpha: Float64Array | null;
  cutneighsq: Float64Array;
}

export class FixShardlow extends Fix {
  readonly style = 'shardlow';
  /** ssa_maxPhaseLen: the largest work-item estimate so far (grows only, as the native neighbor list does). */
  private maxPhaseLen = 0;
  /** The random states (rand_state) and the seed they were drawn from. */
  private rng: ExtState[] = [];
  private rngFor = -1;
  /** Internal temperatures at the last communication (ghost copies of dpdTheta), indexed by engine atom. */
  private thetaSnap = new Float64Array(0);
  private readonly pair: PairDPDfdt | PairDPDfdtEnergy;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal fix shardlow command');
    // fix_shardlow.cpp constructor: force->pair_match("dpd/fdt", 1), then "dpd/fdt/energy" (a hybrid sub-style counts)
    const p = pairMatch(sys.ff.pair, 'dpd/fdt') ?? pairMatch(sys.ff.pair, 'dpd/fdt/energy');
    // fix_shardlow.cpp constructor: "Must use pair_style dpd/fdt or dpd/fdt/energy with fix shardlow"
    if (!(p instanceof PairDPDfdt) && !(p instanceof PairDPDfdtEnergy)) {
      throw new StyleError('Must use pair_style dpd/fdt or dpd/fdt/energy with fix shardlow');
    }
    this.pair = p;
    // the pair keeps only the conservative force from now on (pair_dpd_fdt.cpp init_style: splitFDT_flag)
    p.splitFdt = true;
  }

  /** fix_shardlow.cpp setup(): constant temperature fixes are refused, and the order of the integrators is checked. */
  setup(): void {
    const fixes = this.sys.fixes;
    if (fixes.some((f) => /^(nvt|npt|gle|gld)/.test(f.style))) {
      throw new StyleError('Cannot use constant temperature integration routines with DPD-REACT.');
    }
    let seenShardlow = false;
    for (let i = 0; i < fixes.length; i++) {
      const st = fixes[i].style;
      if (/^shardlow/.test(st)) seenShardlow = true;
      if (/^(nve|nph)/.test(st)) {
        if (seenShardlow) break;
        throw new StyleError('The deterministic integrator must follow fix shardlow in the input file.');
      }
      if (i === fixes.length - 1) {
        throw new StyleError('A deterministic integrator (e.g. fix nve or fix nph) is required when using fix shardlow.');
      }
    }
    this.snapshotTheta();
  }

  /** Ghost copies of dpdTheta are taken at each communication, after the post-integrate hooks (preForce). */
  preForce(): void { this.snapshotTheta(); }

  private snapshotTheta(): void {
    const th = this.sys.state.custom.get(DPD_THETA);
    if (!th) return;
    this.thetaSnap = Float64Array.from(th.data.subarray(0, this.sys.state.n));
  }

  /** Coefficients of the pair, read once per step from the pair object. */
  private coeffs(): Coeffs {
    const v = this.pair.shardlowView();
    const nt = v.nt;
    const cutsq = new Float64Array(nt * nt);
    const cutneighsq = new Float64Array(nt * nt);
    const skin = this.sys.nb.skin;
    for (let t = 0; t < nt * nt; t++) {
      const c = v.cut[t];
      cutsq[t] = c * c;
      const cn = c + skin;
      cutneighsq[t] = cn * cn;
    }
    return { nt, cutsq, cut: v.cut, sigma: v.sigma, kappa: v.kappa, alpha: v.alpha, cutneighsq };
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const nb = this.sys.nb;
    const g = this.sys.geom;
    const view = this.pair.shardlowView();
    if (g.triclinic) throw new StyleError('Fix shardlow does not yet support triclinic geometries');
    if (s.dimension === 2) throw new StyleError('fix shardlow in two dimensions is not supported by the browser engine');
    const rcut = 2.0 * nb.cutneighmax;
    const bbx = g.hi[0] - g.lo[0], bby = g.hi[1] - g.lo[1], bbz = g.hi[2] - g.lo[2];
    if (rcut >= bbx || rcut >= bby || rcut >= bbz) {
      throw new StyleError(`Shardlow algorithm requires sub-domain length > 2*(rcut+skin). rcut= ${rcut} bbx= ${bbx} bby= ${bby} bbz= ${bbz}`);
    }
    const coeffs = this.coeffs();
    const work = this.gather();
    const plan = this.buildPlan(work, coeffs);
    this.drawStates(view.seed);
    this.runPlan(plan, work, coeffs, view.energy, view.boltz, view.temperature);
    this.scatter(work, view.energy);
  }

  /** Native work order: owner data (x, v, type, mass, theta) and ghosts of one processor (nativeGhosts). */
  private gather(): Work {
    const s = this.sys.state;
    const g = this.sys.geom;
    const nb = this.sys.nb;
    const n = s.n;
    const order = nativeOrder(s);
    const xn = new Float64Array(3 * n);
    for (let k = 0; k < n; k++) {
      const e = order[k];
      xn[3 * k] = s.x[3 * e]; xn[3 * k + 1] = s.x[3 * e + 1]; xn[3 * k + 2] = s.x[3 * e + 2];
    }
    const gh = nativeGhosts(xn, n, g, nb.cutghost, false);
    const nall = gh.nall;
    const v = new Float64Array(3 * nall);
    const type = new Int32Array(nall);
    const mass = new Float64Array(nall);
    const theta = new Float64Array(nall);
    const uCond = new Float64Array(nall);
    const uMech = new Float64Array(nall);
    const energy = this.pair.shardlowView().energy;
    const th = energy ? s.custom.get(DPD_THETA)!.data : null;
    const uc = energy ? s.custom.get(DPD_UCOND)!.data : null;
    const um = energy ? s.custom.get(DPD_UMECH)!.data : null;
    const snap = this.thetaSnap.length >= n ? this.thetaSnap : null;
    for (let k = 0; k < nall; k++) {
      const e = order[gh.root[k]];
      v[3 * k] = s.v[3 * e]; v[3 * k + 1] = s.v[3 * e + 1]; v[3 * k + 2] = s.v[3 * e + 2];
      type[k] = s.type[e];
      mass[k] = massOf(s, e);
      if (th && uc && um) {
        // owners: current values; ghosts: the copies taken at the last communication (dpdTheta, uCond, uMech)
        theta[k] = k < n || !snap ? th[e] : snap[e];
        uCond[k] = uc[e];
        uMech[k] = um[e];
      }
    }
    return {
      n, nall, root: gh.root, img: gh.img, x: gh.x, v, v0: new Float64Array(3 * nall), uCond, uMech, theta, mass, type,
    };
  }

  /** The random states of the work items: es_init(me + seed), then one parallel state per item. */
  private drawStates(seed: number): void {
    if (this.maxPhaseLen > this.rngFor) {
      const serial = esInit(seed);
      this.rng = [];
      for (let i = 0; i < this.maxPhaseLen; i++) this.rng.push(esGenNextParallelState(serial));
      this.rngFor = this.maxPhaseLen;
    }
  }

  /** NBinSSA bins and NPairHalfBinNewtonSSA::build: the work phases of the local atoms and the ghost groups. */
  private buildPlan(w: Work, c: Coeffs): Plan {
    const g = this.sys.geom;
    const nb = this.sys.nb;
    const n = w.n;
    const X = w.x;
    const types = w.type;
    const nall = w.nall;

    // bin geometry (nbin.cpp coord2bin, nbin_standard.cpp setup_bins, nstencil.cpp sx, sy, sz)
    const lo = g.lo, hi = g.hi;
    const bbox = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let binsizeOpt = nb.binsize > 0 ? nb.binsize : 0.5 * nb.cutneighmax;
    if (binsizeOpt === 0) binsizeOpt = bbox[0];
    const binsizeinv = 1.0 / binsizeOpt;
    const nbin = [0, 0, 0], binsize = [0, 0, 0], bininv = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      nbin[d] = Math.trunc(bbox[d] * binsizeinv);
      if (nbin[d] === 0) nbin[d] = 1;
      binsize[d] = bbox[d] / nbin[d];
      bininv[d] = 1.0 / binsize[d];
    }
    const coord = (x: number, d: number): number => {
      if (x >= hi[d]) return Math.trunc((x - hi[d]) * bininv[d]) + nbin[d];
      if (x >= lo[d]) return Math.min(Math.trunc((x - lo[d]) * bininv[d]), nbin[d] - 1);
      return Math.trunc((x - lo[d]) * bininv[d]) - 1;
    };
    const cutmax = nb.cutneighmax;
    const reach = (d: number): number => {
      const r = Math.trunc(cutmax * bininv[d]);
      return r * binsize[d] < cutmax ? r + 1 : r;
    };
    const sx = reach(0), sy = reach(1), sz = reach(2);
    const cutmaxsq = cutmax * cutmax;
    const binDist = (i: number, j: number, k: number): number => {
      const dx = i > 0 ? (i - 1) * binsize[0] : i === 0 ? 0 : (i + 1) * binsize[0];
      const dy = j > 0 ? (j - 1) * binsize[1] : j === 0 ? 0 : (j + 1) * binsize[1];
      const dz = k > 0 ? (k - 1) * binsize[2] : k === 0 ? 0 : (k + 1) * binsize[2];
      return dx * dx + dy * dy + dz * dz;
    };
    const st = this.stencil(sx, sy, sz, (i, j, k) => binDist(i, j, k) < cutmaxsq);

    // local bins (NBinSSA::bin_atoms): owned atoms in ascending chains; extent of the local bins
    const chains = new Map<number, number[]>();
    const posIn = new Int32Array(n);
    let lbx0 = Infinity, lby0 = Infinity, lbz0 = Infinity, lbx1 = -Infinity, lby1 = -Infinity, lbz1 = -Infinity;
    const bxOf = new Int32Array(nall), byOf = new Int32Array(nall), bzOf = new Int32Array(nall);
    for (let k = 0; k < nall; k++) {
      bxOf[k] = coord(X[3 * k], 0); byOf[k] = coord(X[3 * k + 1], 1); bzOf[k] = coord(X[3 * k + 2], 2);
    }
    for (let k = 0; k < n; k++) {
      const key = binKey(bxOf[k], byOf[k], bzOf[k]);
      let ch = chains.get(key);
      if (!ch) { ch = []; chains.set(key, ch); }
      posIn[k] = ch.length;
      ch.push(k);
      lbx0 = Math.min(lbx0, bxOf[k]); lbx1 = Math.max(lbx1, bxOf[k] + 1);
      lby0 = Math.min(lby0, byOf[k]); lby1 = Math.max(lby1, byOf[k] + 1);
      lbz0 = Math.min(lbz0, bzOf[k]); lbz1 = Math.max(lbz1, bzOf[k] + 1);
    }
    const chainAt = (ix: number, iy: number, iz: number): number[] | undefined => chains.get(binKey(ix, iy, iz));

    // phase count and the work-item estimate (npair_half_bin_newton_ssa.cpp: phaseLenEstimate, ssa_maxPhaseLen)
    const sx1 = sx + 1, sy1 = sy + 1, sz1 = sz + 1;
    const xE = Math.floor((lbx1 - lbx0 + sx1 - 1) / sx1) + 1;
    const yE = Math.floor((lby1 - lby0 + sy1 - 1) / sy1) + 1;
    const zE = Math.floor((lbz1 - lbz0 + sz1 - 1) / sz1) + 1;
    this.maxPhaseLen = Math.max(this.maxPhaseLen, xE * yE * zE);

    const pairsOk = (i: number, j: number): boolean => {
      const t = types[i] * c.nt + types[j];
      const dx = X[3 * i] - X[3 * j], dy = X[3 * i + 1] - X[3 * j + 1], dz = X[3 * i + 2] - X[3 * j + 2];
      return dx * dx + dy * dy + dz * dz <= c.cutneighsq[t];
    };

    const phases: [number, number][][] = [];
    const local: Entries = { atom: [], nbStart: [], nbEnd: [], nb: [] };
    const pushEntry = (E: Entries, i: number, start: number): void => {
      if (E.nb.length > start) {
        E.atom.push(i);
        E.nbStart.push(start);
        E.nbEnd.push(E.nb.length);
      }
    };
    for (let zoff = sz; zoff >= 0; --zoff) {
      for (let yoff = sy; yoff >= 0; --yoff) {
        for (let xoff = sx; xoff >= 0; --xoff) {
          const items: [number, number][] = [];
          for (let zb = lbz0 + zoff; zb < lbz1; zb += sz1) {
            for (let yb = lby0 + yoff - sy; yb < lby1; yb += sy1) {
              for (let xb = lbx0 + xoff - sx; xb < lbx1; xb += sx1) {
                const itemStart = local.atom.length;
                for (let sub = 0; sub < 4; sub++) {
                  const sYb = yb + ((sub & 2) ? sy : 0);
                  const sXb = xb + ((sub & 1) ? sx : 0);
                  if (sYb < lby0 || sYb >= lby1) continue;
                  if (sXb < lbx0 || sXb >= lbx1) continue;
                  const ch = chainAt(sXb, sYb, zb);
                  if (!ch) continue;
                  for (const i of ch) {
                    const start = local.nb.length;
                    for (let q = st.nst[sub]; q < st.nst[sub + 1]; q++) {
                      const dx = st.off[3 * q], dy = st.off[3 * q + 1], dz = st.off[3 * q + 2];
                      const jch = chainAt(sXb + dx, sYb + dy, zb + dz);
                      if (!jch) continue;
                      const from = dx === 0 && dy === 0 && dz === 0 ? posIn[i] + 1 : 0;
                      for (let p = from; p < jch.length; p++) {
                        const j = jch[p];
                        if (pairsOk(i, j)) local.nb.push(j);
                      }
                    }
                    pushEntry(local, i, start);
                  }
                }
                if (local.atom.length > itemStart) items.push([itemStart, local.atom.length]);
              }
            }
          }
          phases.push(items);
        }
      }
    }

    // ghost groups (NPairHalfBinNewtonSSA::build, AIR ghosts only): ascending ghost order inside a group
    const ghosts: Entries[] = [];
    for (let grp = 0; grp < 7; grp++) ghosts.push({ atom: [], nbStart: [], nbEnd: [], nb: [] });
    for (let k = n; k < nall; k++) {
      const code = coord2ssaAIR(X[3 * k], X[3 * k + 1], X[3 * k + 2], lo, hi);
      if (code < 1) continue;
      const E = ghosts[code - 1];
      const start = E.nb.length;
      const gx = bxOf[k], gy = byOf[k], gz = bzOf[k];
      for (let q = 0; q < st.full; q++) {
        const dx = st.off[3 * q], dy = st.off[3 * q + 1], dz = st.off[3 * q + 2];
        const bx = gx + dx, by = gy + dy, bz = gz + dz;
        if (bx < lbx0 || bx >= lbx1 || by < lby0 || by >= lby1 || bz < lbz0 || bz >= lbz1) continue;
        const ch = chainAt(bx, by, bz);
        if (!ch) continue;
        for (const j of ch) {
          if (pairsOk(k, j)) E.nb.push(j);
        }
      }
      pushEntry(E, k, start);
    }
    return { local, phases, ghosts };
  }

  /** Stencil of the half bins plus the back bins (NStencilHalfBin3dSSA::create), in the native order. */
  private stencil(sx: number, sy: number, sz: number, within: (i: number, j: number, k: number) => boolean): Stencil {
    const off: number[] = [];
    const nst: number[] = [0];
    const add = (i: number, j: number, k: number): void => { if (within(i, j, k)) off.push(i, j, k); };
    off.push(0, 0, 0);
    for (let k = 0; k <= sz; k++) for (let j = 0; j <= sy; j++) for (let i = 0; i <= sx; i++) if (k > 0 || j > 0 || i > 0) add(i, j, k);
    nst.push(off.length / 3);
    for (let k = 0; k <= sz; k++) for (let j = 1; j <= sy; j++) for (let i = -sx; i < 0; i++) add(i, j, k);
    nst.push(off.length / 3);
    for (let k = 1; k <= sz; k++) for (let j = -sy; j < 0; j++) for (let i = 0; i <= sx; i++) add(i, j, k);
    nst.push(off.length / 3);
    for (let k = 1; k <= sz; k++) for (let j = -sy; j <= 0; j++) for (let i = -sx; i < 0; i++) add(i, j, k);
    nst.push(off.length / 3);
    for (let k = -sz; k < 0; k++) for (let j = 0; j <= sy; j++) for (let i = 0; i <= sx; i++) add(i, j, k);
    for (let k = -sz; k < 0; k++) for (let j = 1; j <= sy; j++) for (let i = -sx; i < 0; i++) add(i, j, k);
    for (let k = -sz; k <= 0; k++) for (let j = -sy; j < 0; j++) for (let i = 0; i <= sx; i++) add(i, j, k);
    for (let k = -sz; k <= 0; k++) for (let j = -sy; j <= 0; j++) for (let i = -sx; i < 0; i++) add(i, j, k);
    return { off, nst, full: off.length / 3 };
  }

  /** The SSA sweep: the local phases, then the ghost groups with their forward and reverse communication. */
  private runPlan(plan: Plan, w: Work, c: Coeffs, energy: boolean, boltz: number, temperature: number): void {
    const s = this.sys.state;
    const dt = s.dt;
    const dtsqrt = Math.sqrt(dt);
    const ftm2v = s.units.ftm2v;
    for (const items of plan.phases) {
      for (let wi = 0; wi < items.length; wi++) {
        const [a, b] = items[wi];
        this.sweep(plan.local, a, b, w, c, energy, dt, dtsqrt, ftm2v, boltz, temperature, wi);
      }
    }
    // ghost groups: forward copy of the owners' velocities, one item (state 0), reverse add-back of the deltas
    for (let grp = 0; grp < 7; grp++) {
      const E = plan.ghosts[grp];
      for (let k = w.n; k < w.nall; k++) {
        const r = w.root[k];
        for (let q = 0; q < 3; q++) { w.v[3 * k + q] = w.v[3 * r + q]; w.v0[3 * k + q] = w.v[3 * r + q]; }
        if (energy) { w.uCond[k] = 0; w.uMech[k] = 0; }
      }
      this.sweep(E, 0, E.atom.length, w, c, energy, dt, dtsqrt, ftm2v, boltz, temperature, 0);
      for (let k = w.n; k < w.nall; k++) {
        const r = w.root[k];
        for (let q = 0; q < 3; q++) w.v[3 * r + q] += w.v[3 * k + q] - w.v0[3 * k + q];
        if (energy) { w.uCond[r] += w.uCond[k]; w.uMech[r] += w.uMech[k]; }
      }
    }
  }

  /** Entries [a, b) of E, with random state index wi (fix_shardlow.cpp ssa_update_dpd / ssa_update_dpde). */
  private sweep(E: Entries, a: number, b: number, w: Work, c: Coeffs, energy: boolean,
    dt: number, dtsqrt: number, ftm2v: number, boltz: number, temperature: number, wi: number): void {
    const rs = this.rng[wi];
    if (!rs) throw new StyleError('fix shardlow ran out of random states');
    const boltzInv = 1.0 / boltz;
    const { x, v, uCond, uMech, theta, mass, type } = w;
    const nt = c.nt;
    for (let e = a; e < b; e++) {
      const i = E.atom[e];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      let vxi = v[3 * i], vyi = v[3 * i + 1], vzi = v[3 * i + 2];
      let uMechI = uMech[i], uCondI = uCond[i];
      const itype = type[i];
      const thetaIinv = energy ? 1.0 / theta[i] : 0;
      const massI = mass[i];
      const massinvI = 1.0 / massI;
      const massINeg4 = massI * (-0.25) / ftm2v;
      for (let q = E.nbStart[e]; q < E.nbEnd[e]; q++) {
        const j = E.nb[q];
        const jtype = type[j];
        const t = itype * nt + jtype;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (!((rsq < c.cutsq[t]) && (rsq >= EPSILON_SQUARED))) continue;
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        const dxr = delx * rinv, dyr = dely * rinv, dzr = delz * rinv;
        const wr = 1.0 - r / c.cut[t];
        const wdt = wr * wr * dt;
        const halfsigma = 0.5 * c.sigma[t];
        let thetaIJinv: number, thetaJinv = 0;
        if (energy) {
          thetaJinv = 1.0 / theta[j];
          thetaIJinv = 0.5 * (thetaIinv + thetaJinv);
        } else {
          thetaIJinv = 1.0 / temperature;
        }
        const halfgamma = halfsigma * halfsigma * boltzInv * thetaIJinv;
        const sigmaRand = halfsigma * wr * dtsqrt * ftm2v * esNormal(rs);
        const massJ = mass[j];
        const massinvJ = 1.0 / massJ;
        let delUCond = 0;
        if (energy) {
          const alphaIJ = c.alpha![t];
          const kappaIJ = c.kappa![t];
          delUCond = alphaIJ * wr * dtsqrt * esNormal(rs);
          delUCond += kappaIJ * (thetaIinv - thetaJinv) * wdt;
          uCond[j] -= delUCond;
          uCondI += delUCond;
        }
        const gammaFactor = halfgamma * wdt * ftm2v;
        const inv1pMuGamma = 1.0 / (1.0 + (massinvI + massinvJ) * gammaFactor);
        let vxj = v[3 * j], vyj = v[3 * j + 1], vzj = v[3 * j + 2];
        const dot4 = vxj * vxj + vyj * vyj + vzj * vzj;
        const dot3 = vxi * vxi + vyi * vyi + vzi * vzi;
        let delvx = vxi - vxj, delvy = vyi - vyj, delvz = vzi - vzj;
        let dotRinv = dxr * delvx + dyr * delvy + dzr * delvz;
        const factorA = sigmaRand - gammaFactor * dotRinv;
        vxi += dxr * factorA * massinvI; vyi += dyr * factorA * massinvI; vzi += dzr * factorA * massinvI;
        vxj -= dxr * factorA * massinvJ; vyj -= dyr * factorA * massinvJ; vzj -= dzr * factorA * massinvJ;
        delvx = vxi - vxj; delvy = vyi - vyj; delvz = vzi - vzj;
        dotRinv = dxr * delvx + dyr * delvy + dzr * delvz;
        const factorB = (sigmaRand - gammaFactor * dotRinv) * inv1pMuGamma;
        vxi += dxr * factorB * massinvI; vyi += dyr * factorB * massinvI; vzi += dzr * factorB * massinvI;
        let partialUMech = 0;
        if (energy) partialUMech = ((vxi * vxi + vyi * vyi + vzi * vzi) - dot3) * massinvJ;
        vxj -= dxr * factorB * massinvJ; vyj -= dyr * factorB * massinvJ; vzj -= dzr * factorB * massinvJ;
        if (energy) {
          partialUMech += ((vxj * vxj + vyj * vyj + vzj * vzj) - dot4) * massinvI;
          const delUMech = partialUMech * (massJ * massINeg4);
          uMechI += delUMech;
          uMech[j] += delUMech;
        }
        v[3 * j] = vxj; v[3 * j + 1] = vyj; v[3 * j + 2] = vzj;
      }
      v[3 * i] = vxi; v[3 * i + 1] = vyi; v[3 * i + 2] = vzi;
      if (energy) { uMech[i] = uMechI; uCond[i] = uCondI; }
    }
  }

  /** Writes the owners' velocities (and internal energies) back to the engine's atom order. */
  private scatter(w: Work, energy: boolean): void {
    const s = this.sys.state;
    const order = nativeOrder(s);
    const uc = energy ? s.custom.get(DPD_UCOND)!.data : null;
    const um = energy ? s.custom.get(DPD_UMECH)!.data : null;
    for (let k = 0; k < w.n; k++) {
      const e = order[k];
      s.v[3 * e] = w.v[3 * k]; s.v[3 * e + 1] = w.v[3 * k + 1]; s.v[3 * e + 2] = w.v[3 * k + 2];
      if (uc && um) { uc[e] = w.uCond[k]; um[e] = w.uMech[k]; }
    }
  }
}

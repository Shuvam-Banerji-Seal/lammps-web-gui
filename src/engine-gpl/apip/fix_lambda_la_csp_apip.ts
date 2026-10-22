/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/fix_lambda_la_csp_apip.cpp
 * and fix_lambda_la_csp_apip.h (fix lambda/la/csp/apip: the arguments, init, setup_pre_force and pre_force with the
 * dynamic and constant CSP neighbour pairs, the reverse and forward communication of the local averages and
 * prefactors, pre_reverse with calculate_forces, ev_tally2 and ev_tally3, store_la and the f_lambda statistics,
 * select2, the switching and weighting polynomials and compute_scalar), at commit b332f28. Author named in the
 * source: David Immel (FZ Juelich), with the LAMMPS Sandia copyright notice (GNU General Public License). This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/fix_lambda_la_csp_apip.html. The switching parameter of each atom is a polynomial of the local
 * average of the centro-symmetry parameter (weighted by a radial polynomial between cut_lo and cut_hi), which makes
 * the adaptive-precision potential conservative; the fix adds the forces of the derivative of lambda in pre_reverse
 * (with its own virial). With csp_mode static (the default) the CSP neighbour pairs found at the first setup are kept
 * (by atom ID, the image within half a box length); csp_mode dynamic selects them every step and counts the atoms
 * whose pairs changed (the scalar). Engine specifics: ghost atoms read their owners' values (native forward
 * communication), ghost rows of the averages are added to their owners (native reverse communication), the forces on
 * ghost images are added to their owners, and, as native compute stress/atom reads only the owned rows of a fix's
 * per-atom virial, the per-atom virial of ghost images is dropped.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import { APIP_E_FAST, APIP_E_PRECISE, APIP_LAMBDA, buildAtomMap, nativeOrder } from '../../engine/atoms';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { pow as libmPow } from '../libm/pow';

const THIRD = 1.0 / 3.0;

/** select2 (Numerical Recipes, as in the fix): the k smallest values of a[0..n) first, ia permuted alongside. */
const select2 = (k: number, n: number, a: Float64Array, ia: Int32Array): void => {
  const at = (q: number): number => a[q - 1];
  const swap = (p: number, q: number): void => {
    const t = a[p - 1]; a[p - 1] = a[q - 1]; a[q - 1] = t;
    const u = ia[p - 1]; ia[p - 1] = ia[q - 1]; ia[q - 1] = u;
  };
  let l = 1, ir = n;
  for (;;) {
    if (ir <= l + 1) {
      if (ir === l + 1 && at(ir) < at(l)) swap(l, ir);
      return;
    }
    const mid = (l + ir) >> 1;
    swap(mid, l + 1);
    if (at(l) > at(ir)) swap(l, ir);
    if (at(l + 1) > at(ir)) swap(l + 1, ir);
    if (at(l) > at(l + 1)) swap(l, l + 1);
    let i = l + 1, j = ir;
    const v = at(l + 1), iv = ia[l];
    for (;;) {
      do i++; while (at(i) < v);
      do j--; while (at(j) > v);
      if (j < i) break;
      swap(i, j);
    }
    a[l] = at(j); ia[l] = ia[j - 1];
    a[j - 1] = v; ia[j - 1] = iv;
    if (j >= k) ir = j - 1;
    if (j <= k) l = i;
  }
};

export class FixLambdaLACSPAPIP extends Fix {
  readonly style = 'lambda/la/csp/apip';
  private readonly thresholdLo: number;
  private readonly thresholdHi: number;
  private readonly thresholdWidth: number;
  private readonly cutLo: number;
  private readonly cutHi: number;
  private readonly cutWidth: number;
  private readonly cutHiSq: number;
  private readonly nnn: number;
  private readonly constNgh: boolean = true;
  private readonly cspCutsq: number = 25;
  private readonly calculateForcesFlag: boolean = true;
  private readonly lambdaNonGroup: number = 1;
  peratomFreq = 0;
  private readonly storeStats: boolean = false;
  private readonly cutsqCombined: number;
  private tagsStored = false;
  private counterChanged = 0;
  /** Per-atom prop key of the stored CSP neighbour IDs (native FixStoreAtom with nnn columns). */
  private readonly keyTags: string;
  private readonly natOrder = new NativeFullOrder();
  /** la_inp of every atom (ghosts: their owners'), la_avg and la_norm of the owned atoms, per step. */
  private csp = new Float64Array(0);
  private cspAvg = new Float64Array(0);
  private cspNorm = new Float64Array(0);
  /** ngh_pairs: the nnn neighbour indices (owned or ghost) of each owned atom. */
  private nghPairs = new Int32Array(0);
  private vatomOwned = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const num = (w: string | undefined): number => {
      if (w === undefined || w === '') throw new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file');
      const v = Number(w);
      if (!Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
      return v;
    };
    const int = (w: string): number => {
      if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
      return parseInt(w, 10);
    };
    if (args.length < 5) throw new StyleError('fix lambda/la/csp/apip requires five arguments');
    this.thresholdLo = num(args[0]);
    this.thresholdHi = num(args[1]);
    this.thresholdWidth = this.thresholdHi - this.thresholdLo;
    this.cutLo = num(args[2]);
    this.cutHi = num(args[3]);
    this.cutWidth = this.cutHi - this.cutLo;
    this.cutHiSq = this.cutHi * this.cutHi;
    this.nnn = args[4] === 'fcc' ? 12 : args[4] === 'bcc' ? 14 : int(args[4]);
    let constNgh = true, cspCutsq = 25, forces = true, nonGroup = 1, freq = 0;
    for (let k = 5; k < args.length; k++) {
      const w = args[k];
      if (w === 'csp_mode') {
        if (k + 1 === args.length) throw new StyleError('the csp_mode option of fix lambda/la/csp/apip requires an additional argument');
        if (args[k + 1] === 'dynamic') constNgh = false;
        else if (args[k + 1] === 'static') constNgh = true;
        else throw new StyleError(`expected dynamic or static instead of ${args[k + 1]}`);
        k++;
      } else if (w === 'csp_cut') {
        if (k + 1 === args.length) throw new StyleError('the csp_cut option of fix lambda/la/csp/apip requires an additional argument');
        // pow(value, 2), which gcc compiles to value * value
        const c = num(args[k + 1]);
        cspCutsq = c * c;
        k++;
      } else if (w === 'forces') {
        if (k + 1 === args.length) throw new StyleError('the forces option of fix lambda/la/csp/apip requires an additional argument');
        const v = args[k + 1];
        if (v === 'yes' || v === 'on' || v === 'true' || v === '1') forces = true;
        else if (v === 'no' || v === 'off' || v === 'false' || v === '0') forces = false;
        else throw new StyleError(`Expected boolean parameter instead of '${v}' in input script or data file`);
        k++;
      } else if (w === 'lambda_non_group') {
        if (k + 1 === args.length) throw new StyleError('the lambda_non_group option of fix lambda/la/csp/apip requires an additional argument');
        const v = args[k + 1];
        nonGroup = v === 'fast' ? 1 : v === 'precise' ? 0 : num(v);
        k++;
      } else if (w === 'store_peratom') {
        if (k + 1 === args.length) throw new StyleError('the store_stats option of fix lambda/la/csp/apip requires an additional argument');
        freq = int(args[k + 1]);
        if (freq <= 0) throw new StyleError('store_peratom frequency needs to be positive.');
        k++;
      } else {
        throw new StyleError(`unknown argument ${w}`);
      }
    }
    this.constNgh = constNgh;
    this.cspCutsq = cspCutsq;
    this.calculateForcesFlag = forces;
    this.lambdaNonGroup = nonGroup;
    if (this.cutLo > this.cutHi || this.cutLo < 0) throw new StyleError('0 <= cut_lo <= cut_hi required');
    if (this.thresholdLo > this.thresholdHi || this.thresholdLo < 0) throw new StyleError('0 <= threshold_lo <= threshold_hi required');
    if (nonGroup < 0 || nonGroup > 1) throw new StyleError('0 <= lambda_non_group <= 1 required');
    if (!constNgh) {
      this.scalarFlag = true;
      this.extscalar = 1;
    }
    this.cutsqCombined = cspCutsq > this.cutHiSq ? cspCutsq : this.cutHiSq;
    if (forces) {
      this.virialGlobal = true;
      this.thermoVirial = true;
    }
    if (!sys.state.custom.has(APIP_LAMBDA)) throw new StyleError('fix lambda/la/csp/apip requires atomic style with lambda.');
    if (freq > 0) {
      this.peratomFreq = freq;
      this.storeStats = true;
      this.peratomFlag = true;
      this.sizePeratomCols = 5;
      this.arrayAtom = new Float64Array(5 * sys.state.n);
    }
    this.keyTags = `#apip_la_csp_pairs:${id}`;
    sys.state.custom.set(this.keyTags, { int: true, cols: this.nnn, data: new Float64Array(this.nnn * sys.state.n) });
  }

  destroy(): void {
    this.sys.state.custom.delete(this.keyTags);
  }

  init(): void {
    if (this.sys.fixes.filter((f) => f.style === 'lambda/la/csp/apip').length > 1) throw new StyleError('More than one fix lambda/la/csp/apip.');
    if (this.sys.atomStyle !== 'apip' && this.sys.atomStyle !== 'apip conservative') throw new StyleError('fix lambda/la/csp/apip requires atom style apip');
  }

  /** setup(): native has none (the engine default would call postForce). */
  setup(): void {}

  /** The native order of the full list is that of the build step (native bins at the reneighboring positions). */
  private refreshOrder(): void {
    const sys = this.sys;
    if (sys.nb.full) this.natOrder.neighbors(sys.state, sys.nb, sys.geom, sys.nb.full);
  }

  postNeighbor(): void {
    this.refreshOrder();
  }

  setupPreForce(): void {
    const nb = this.sys.nb;
    // init: the cutoffs against the pair cutoff (force->pair->cutforce, set once the pair style is initialised)
    const cutforce = nb.cutneighmax > 0 ? nb.cutneighmax - nb.skin : 0;
    if (cutforce < this.cutHi) throw new StyleError(`cutoff of potential (${cutforce}) smaller than cutoff of weighting function (${this.cutHi})`);
    if (cutforce * cutforce < this.cspCutsq) throw new StyleError(`cutoff of potential (${cutforce}) smaller than cutoff of the CSP (${Math.sqrt(this.cspCutsq)})`);
    this.refreshOrder();
    if (!this.constNgh || !this.tagsStored) this.preForceDynPairs();
    else this.preForceConstPairs();
  }

  preForce(): void {
    if (this.constNgh) this.preForceConstPairs();
    else this.preForceDynPairs();
  }

  setupPreReverse(): void {
    this.preReverse();
  }

  preReverse(): void {
    this.storeLa();
    this.calculateForces();
  }

  /** The arrays of one step (nall rows for the averages, which collect the ghost rows before they go to the owners). */
  private alloc(): void {
    const nb = this.sys.nb;
    const nall = nb.nall, n = this.sys.state.n;
    this.csp = new Float64Array(nall);
    this.cspAvg = new Float64Array(nall);
    this.cspNorm = new Float64Array(nall);
    if (this.nghPairs.length < n * this.nnn) this.nghPairs = new Int32Array(n * this.nnn);
  }

  private switching(input: number): number {
    if (input <= this.thresholdLo) return 1;
    if (input >= this.thresholdHi) return 0;
    const t = 1 - 2 * (1 + (input - this.thresholdHi) / this.thresholdWidth);
    return 0.5 + 7.5 / 2. * (t / 4. - libmPow(t, 3) / 6. + libmPow(t, 5) / 20.);
  }

  private weighting(r: number): number {
    if (r <= this.cutLo) return 1;
    if (r >= this.cutHi) return 0;
    const t = 1 - 2 * (1 + (r - this.cutHi) / this.cutWidth);
    return 0.5 + 7.5 / 2. * (t / 4. - libmPow(t, 3) / 6. + libmPow(t, 5) / 20.);
  }

  private derWeighting(r: number): number {
    if (r <= this.cutLo || r >= this.cutHi) return 0;
    const t = 1 - 2 * (1 + (r - this.cutHi) / this.cutWidth);
    return -1.875 / this.cutWidth * (1 - 2 * t * t + libmPow(t, 4));
  }

  /** The reverse communication of the averages, the normalisation, lambda, and la_inp of the ghosts. */
  private finishAverages(): void {
    const s = this.sys.state, nb = this.sys.nb;
    const owner = nb.owner, n = s.n;
    for (let k = nb.nlocal; k < nb.nall; k++) {
      const o = owner[k];
      this.cspAvg[o] += this.cspAvg[k];
      this.cspNorm[o] += this.cspNorm[k];
    }
    const lambda = s.custom.get(APIP_LAMBDA)!.data;
    for (let i = 0; i < n; i++) {
      this.cspAvg[i] /= this.cspNorm[i];
      lambda[i] = (s.mask[i] & this.groupBit) ? this.switching(this.cspAvg[i]) : this.lambdaNonGroup;
    }
    // forward communication of la_inp (lambda of the ghosts is read through their owners)
    for (let k = nb.nlocal; k < nb.nall; k++) this.csp[k] = this.csp[owner[k]];
  }

  /** pre_force_dyn_pairs: select the CSP neighbour pairs of every atom. */
  private preForceDynPairs(): void {
    const sys = this.sys, s = sys.state, nb = sys.nb;
    const list = nb.full;
    if (!list) throw new StyleError('fix lambda/la/csp/apip needs a full neighbor list');
    this.alloc();
    const nbrs = this.natOrder.neighbors(s, nb, sys.geom, list);
    const x = nb.xall, nnn = this.nnn, nhalf = nnn / 2;
    const npairs = (nnn * (nnn - 1)) / 2;
    const pv = new Float64Array(npairs), pj = new Int32Array(npairs), pk = new Int32Array(npairs), pidx = new Int32Array(npairs);
    const tags = s.custom.get(this.keyTags)!.data;
    const id = s.id, owner = nb.owner;
    const tagOf = (k: number): number => id[owner[k]];
    let distsq = new Float64Array(64), nearest = new Int32Array(64);
    const csp = this.csp, avg = this.cspAvg, norm = this.cspNorm, ngh = this.nghPairs;
    for (const i of nativeOrder(s)) {
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      const k0 = list.firstneigh[i], jnum = list.numneigh[i];
      if (jnum > distsq.length) {
        distsq = new Float64Array(jnum);
        nearest = new Int32Array(jnum);
      }
      let nNear = 0;
      for (let jj = 0; jj < jnum; jj++) {
        const j = nbrs[k0 + jj] & NEIGHMASK;
        const dx = xt - x[3 * j], dy = yt - x[3 * j + 1], dz = zt - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < this.cutsqCombined) {
          distsq[nNear] = rsq;
          nearest[nNear++] = j;
        }
      }
      if (nNear < nnn) throw new StyleError(`Atom has only ${nNear} neighbours, but ${nnn} are required to calculate the CSP.`);
      select2(nnn, nNear, distsq, nearest);
      let np = 0;
      for (let a = 0; a < nnn; a++) {
        const ja = nearest[a];
        for (let b = a + 1; b < nnn; b++) {
          const kb = nearest[b];
          const dx = x[3 * ja] + x[3 * kb] - 2.0 * xt;
          const dy = x[3 * ja + 1] + x[3 * kb + 1] - 2.0 * yt;
          const dz = x[3 * ja + 2] + x[3 * kb + 2] - 2.0 * zt;
          pv[np] = dx * dx + dy * dy + dz * dz;
          pj[np] = ja;
          pk[np] = kb;
          pidx[np] = np;
          np++;
        }
      }
      select2(nhalf, npairs, pv, pidx);
      let value = 0.0;
      for (let a = 0; a < nhalf; a++) {
        value += pv[a];
        ngh[i * nnn + a] = pj[pidx[a]];
        ngh[i * nnn + a + nhalf] = pk[pidx[a]];
      }
      csp[i] = value;
      if (this.tagsStored) {
        // the pairs of atom IDs now and at the last selection, compared as sets
        const key = (t1: number, t2: number): string => (t1 < t2 ? `${t1},${t2}` : `${t2},${t1}`);
        const now: string[] = [], prev: string[] = [];
        for (let a = 0; a < nhalf; a++) {
          now.push(key(tagOf(ngh[i * nnn + a]), tagOf(ngh[i * nnn + a + nhalf])));
          prev.push(key(tags[i * nnn + a], tags[i * nnn + a + nhalf]));
        }
        now.sort();
        prev.sort();
        for (let a = 0; a < nhalf; a++) {
          if (now[a] !== prev[a]) {
            this.counterChanged++;
            break;
          }
        }
      }
      for (let a = 0; a < nnn; a++) tags[i * nnn + a] = tagOf(ngh[i * nnn + a]);
      let weight = this.weighting(0);
      avg[i] += value * weight;
      norm[i] += weight;
      for (let a = 0; a < nNear; a++) {
        if (distsq[a] <= this.cutHiSq) {
          weight = this.weighting(Math.sqrt(distsq[a]));
          const jj = nearest[a];
          avg[jj] += value * weight;
          norm[jj] += weight;
        }
      }
    }
    this.tagsStored = true;
    this.finishAverages();
  }

  /** pre_force_const_pairs: the CSP of the stored neighbour pairs, each at its image within half a box length. */
  private preForceConstPairs(): void {
    const sys = this.sys, s = sys.state, nb = sys.nb;
    const list = nb.full;
    if (!list) throw new StyleError('fix lambda/la/csp/apip needs a full neighbor list');
    this.alloc();
    const nbrs = this.natOrder.neighbors(s, nb, sys.geom, list);
    const x = nb.xall, nnn = this.nnn, nhalf = nnn / 2;
    const tags = s.custom.get(this.keyTags)!.data;
    const map = buildAtomMap(s);
    const owner = nb.owner;
    // atom->map and the sametag chain: the owned atom first, then its ghost images
    const images = new Map<number, number[]>();
    for (let k = nb.nlocal; k < nb.nall; k++) {
      const o = owner[k];
      const l = images.get(o);
      if (l) l.push(k); else images.set(o, [k]);
    }
    const b = s.box;
    const prdHalf = [0.5 * (b.hi[0] - b.lo[0]), 0.5 * (b.hi[1] - b.lo[1]), 0.5 * (b.hi[2] - b.lo[2])];
    const csp = this.csp, avg = this.cspAvg, norm = this.cspNorm, ngh = this.nghPairs;
    const order = nativeOrder(s);
    const findImage = (i: number, tag: number): number => {
      const o = tag >= 0 && tag < map.length ? map[tag] : -1;
      if (o < 0) throw new StyleError(`atom ID ${s.id[i]} csp neighbour with ID ${tag} not present`);
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      for (const k of [o, ...(images.get(o) ?? [])]) {
        const dx = x[3 * k] - xt, dy = x[3 * k + 1] - yt, dz = x[3 * k + 2] - zt;
        if (Math.abs(dx) < prdHalf[0] && Math.abs(dy) < prdHalf[1] && Math.abs(dz) < prdHalf[2]) return k;
      }
      throw new StyleError(`atom ID ${s.id[i]} no correct image of csp neighbour with ID ${tag} found`);
    };
    for (const i of order) {
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      let value = 0;
      for (let a = 0; a < nhalf; a++) {
        const jj = findImage(i, tags[i * nnn + a]);
        ngh[i * nnn + a] = jj;
        const kk = findImage(i, tags[i * nnn + a + nhalf]);
        ngh[i * nnn + a + nhalf] = kk;
        const dx = (x[3 * jj] - xt) + (x[3 * kk] - xt);
        const dy = (x[3 * jj + 1] - yt) + (x[3 * kk + 1] - yt);
        const dz = (x[3 * jj + 2] - zt) + (x[3 * kk + 2] - zt);
        value += dx * dx + dy * dy + dz * dz;
      }
      csp[i] = value;
    }
    // native local index of each owned atom (the i > j test of the half pass); ghosts come after every owned atom
    const rank = new Int32Array(s.n);
    for (let r = 0; r < order.length; r++) rank[order[r]] = r;
    for (const i of order) {
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      let weight = this.weighting(0);
      avg[i] += csp[i] * weight;
      norm[i] += weight;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        const ghost = j >= nb.nlocal;
        if (!ghost && rank[i] > rank[j]) continue;
        const dx = xt - x[3 * j], dy = yt - x[3 * j + 1], dz = zt - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq <= this.cutHiSq) {
          weight = this.weighting(Math.sqrt(rsq));
          avg[j] += csp[i] * weight;
          norm[j] += weight;
          if (ghost) continue;
          norm[i] += weight;
          avg[i] += csp[j] * weight;
        }
      }
    }
    this.finishAverages();
  }

  /** store_la: la_inp and la_avg in columns 3 and 4 of the per-atom array. */
  private storeLa(): void {
    const s = this.sys.state;
    if (!this.storeStats || s.step % this.peratomFreq !== 0) return;
    const n = s.n;
    if (this.arrayAtom.length !== 5 * n) {
      this.arrayAtom = new Float64Array(5 * n);
    }
    for (let i = 0; i < n; i++) {
      this.arrayAtom[5 * i + 3] = this.csp[i];
      this.arrayAtom[5 * i + 4] = this.cspAvg[i];
    }
  }

  /** calculate_forces: the forces of the derivative of lambda, with ev_tally2 / ev_tally3 into the fix virial. */
  private calculateForces(): void {
    if (!this.calculateForcesFlag) return;
    const sys = this.sys, s = sys.state, nb = sys.nb;
    const list = nb.full!;
    const nbrs = this.natOrder.neighbors(s, nb, sys.geom, list);
    const x = nb.xall, f = s.f, owner = nb.owner, nlocal = nb.nlocal, n = s.n;
    const nnn = this.nnn, nhalf = nnn / 2;
    const lambda = s.custom.get(APIP_LAMBDA)!.data;
    const eFast = s.custom.get(APIP_E_FAST)!.data, ePrecise = s.custom.get(APIP_E_PRECISE)!.data;
    const csp = this.csp, cspAvg = this.cspAvg, cspNorm = this.cspNorm, ngh = this.nghPairs;
    this.virial.fill(0);
    if (this.vatomOwned.length !== 6 * n) this.vatomOwned = new Float64Array(6 * n);
    else this.vatomOwned.fill(0);
    const va = this.vatomOwned;
    const prefactor1 = new Float64Array(nb.nall);
    const prefactor2 = new Float64Array(n);
    const order = nativeOrder(s);
    for (const i of order) {
      if (lambda[i] !== 0 && lambda[i] !== 1 && (s.mask[i] & this.groupBit)) {
        const t = 1 - 2 * (1 + (cspAvg[i] - this.thresholdHi) / this.thresholdWidth);
        prefactor1[i] = -1.875 / this.thresholdWidth * (1 - 2 * t * t + libmPow(t, 4));
      } else {
        prefactor1[i] = 0;
      }
      if (prefactor1[i] !== 0) prefactor1[i] *= (eFast[i] - ePrecise[i]) / cspNorm[i];
      prefactor2[i] = prefactor1[i] * this.weighting(0);
    }
    // forward communication of prefactor1
    for (let k = nlocal; k < nb.nall; k++) prefactor1[k] = prefactor1[owner[k]];
    // store_f_lambda_before / after: the forces this fix puts on the owned rows (the ghost rows reach their owners in the
    // native reverse communication, after pre_reverse)
    const stats = this.storeStats && s.step % this.peratomFreq === 0 ? this.arrayAtom : null;
    if (stats) for (let i = 0; i < n; i++) stats[5 * i] = stats[5 * i + 1] = stats[5 * i + 2] = 0;
    const addF = (k: number, fx: number, fy: number, fz: number): void => {
      const o = owner[k];
      f[3 * o] += fx; f[3 * o + 1] += fy; f[3 * o + 2] += fz;
      if (stats && k < nlocal) { stats[5 * k] += fx; stats[5 * k + 1] += fy; stats[5 * k + 2] += fz; }
    };
    const tallyAtom = (k: number, w: number, v: number[]): void => {
      if (k >= nlocal) return;
      for (let c = 0; c < 6; c++) va[6 * k + c] += w * v[c];
    };
    for (const i of order) {
      const pt = prefactor1[i];
      const xt = x[3 * i], yt = x[3 * i + 1], zt = x[3 * i + 2];
      const cspAvgI = cspAvg[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        const dx = xt - x[3 * j], dy = yt - x[3 * j + 1], dz = zt - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= this.cutHiSq) continue;
        const r = Math.sqrt(rsq);
        prefactor2[i] += prefactor1[j] * this.weighting(r);
        const fpair = pt * (csp[j] - cspAvgI) * this.derWeighting(r) / r;
        if (fpair === 0) continue;
        addF(i, -dx * fpair, -dy * fpair, -dz * fpair);
        addF(j, dx * fpair, dy * fpair, dz * fpair);
        // ev_tally2
        const v = [dx * dx * fpair, dy * dy * fpair, dz * dz * fpair, dx * dy * fpair, dx * dz * fpair, dy * dz * fpair];
        for (let c = 0; c < 6; c++) this.virial[c] += v[c];
        tallyAtom(i, 0.5, v);
        tallyAtom(j, 0.5, v);
      }
    }
    for (const i2 of order) {
      const fpair = -2 * prefactor2[i2];
      if (fpair === 0) continue;
      const xt = x[3 * i2], yt = x[3 * i2 + 1], zt = x[3 * i2 + 2];
      for (let a = 0; a < nhalf; a++) {
        const i1 = ngh[i2 * nnn + a], i3 = ngh[i2 * nnn + a + nhalf];
        const dx1 = x[3 * i1] - xt, dy1 = x[3 * i1 + 1] - yt, dz1 = x[3 * i1 + 2] - zt;
        const dx2 = x[3 * i3] - xt, dy2 = x[3 * i3 + 1] - yt, dz2 = x[3 * i3 + 2] - zt;
        const f0 = (dx1 + dx2) * fpair, f1 = (dy1 + dy2) * fpair, f2 = (dz1 + dz2) * fpair;
        addF(i1, f0, f1, f2);
        addF(i2, -2 * f0, -2 * f1, -2 * f2);
        addF(i3, f0, f1, f2);
        // ev_tally3
        const v = [(dx1 + dx2) * f0, (dy1 + dy2) * f1, (dz1 + dz2) * f2, (dx1 + dx2) * f1, (dx1 + dx2) * f2, (dy1 + dy2) * f2];
        for (let c = 0; c < 6; c++) this.virial[c] += v[c];
        tallyAtom(i1, THIRD, v);
        tallyAtom(i2, THIRD, v);
        tallyAtom(i3, THIRD, v);
      }
    }
  }

  virialAtom(out: Float64Array): void {
    const n = Math.min(this.sys.state.n * 6, this.vatomOwned.length);
    for (let k = 0; k < n; k++) out[k] += this.vatomOwned[k];
  }

  computeScalar(): number {
    return this.counterChanged;
  }
}

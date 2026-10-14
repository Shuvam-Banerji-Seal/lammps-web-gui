/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/REPLICA/fix_hyper_local.cpp and
 * fix_hyper_local.h (FixHyperLocal: fix hyper/local, the local bond-boost hyperdynamics bias), and the FixHyper
 * base of src/REPLICA/fix_hyper.cpp (hyperflag 2 = local). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). The C++ names no contributing author; the original authors are the LAMMPS development
 * team (Sandia National Laboratories), as named in those files.
 *
 * Differences from native, all for the single-processor browser engine:
 *  - Bonds, bias coefficients, maxstrain and the neighbor lists are kept per atom ID, not per local index, so
 *    native's pre_neighbor remapping (old2now, closest_image, ghost atoms) is not needed: every bond atom is
 *    found from its ID at each force evaluation, and the periodic image is the minimum image.
 *  - The full neighbor list (cutoff Dcut) is a brute-force search at each bond build (one image per atom: the
 *    periodic box must be longer than 2 Dcut in every periodic dimension).
 *  - The half list that gives the bonds and their owner atoms is the engine's half list (cutoff: pair cutoff plus
 *    skin) with the native pair order of src/engine-gpl/neighbor/nativeOrder.ts (npair_bin.cpp half/bin/newton).
 *  - keywords check/ghost and check/bias are refused (EngineError): they only add diagnostics.
 *  - The MPI reductions are plain sums, minima and maxima over the one processor.
 *
 * Timing quantities follow native: the bond build, the setup force evaluation (setup_pre_reverse, which
 * applies the bias but adds no statistics) and the dynamics evaluation (pre_reverse, which does). The bias is not
 * applied in minimization (the quench of the hyper command: fix hyper/local has no MIN_PRE_REVERSE mask).
 *
 * Output (f_ID): the scalar is the bias potential sum of the last evaluation; the vector has 26 entries
 * (compute_vector 0..25 = f_ID[1..26]) as in fix_hyper_local.cpp compute_vector().
 * The per-run entries that native reads at the current timestep divide by the timesteps since the hyper command
 * started (update->firststep); the entries that native keeps while update->whichflag is not 1 (the quench of
 * the hyper command) return their last dynamics value, as native does.
 *
 * Refused: fix_modify energy and virial (the bias energy is not added to pe, as in fix hyper/global).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import { NEIGHMASK } from '../../engine/neighbor';
import type { System } from '../../engine/system';
import { nativePairs } from '../neighbor/nativeOrder';

/** fix_hyper_local.cpp: BIG (minimum of the bias coefficients before any update) and COEFFINIT (a new bond). */
const BIG = 1.0e20;
const COEFFINIT = 1.0;
/** fix_hyper_local.cpp: FCCBONDS, the initial maximum of bonds per atom. */
const FCCBONDS = 12;

/** LAMMPS MAX / MIN macros (ternaries, not Math.max: the same results as native for every finite input). */
const MAXP = (a: number, b: number): number => (a > b ? a : b);
const MINP = (a: number, b: number): number => (a < b ? a : b);

export class FixHyperLocal extends Fix {
  readonly style = 'hyper/local';
  readonly cutbond: number;
  readonly qfactor: number;
  /** the maximum bias; reset (keyword reset) rescales it, as native does */
  vmax: number;
  readonly tequil: number;
  readonly dcut: number;
  readonly alphaUser: number;
  readonly boostTarget: number;
  private readonly boundflag: boolean;
  private readonly boundLower: number;
  private readonly boundUpper: number;
  private readonly resetfreq: number;
  private readonly cutbondsq: number;
  private readonly dcutsq: number;
  private readonly invvmax: number;
  private readonly invqfactorsq: number;
  private readonly beta: number;
  /** update->ntimestep when the fix was defined (starttime) and at the last reset (lastreset) */
  private readonly starttime: number;
  private lastreset: number;
  /** timestep at which the current hyper command started (update->firststep) */
  private runStart = 0;
  /** update->whichflag == 2: the hyper command's quench (set by the hyper command) */
  private inQuench = false;
  private firstflag = true;
  private nostrainyet = true;

  // cumulative counters (never reset by init_hyper)
  private nevent = 0;
  private neventAtom = 0;
  private nnewbond = 0;
  private aveboostRunning = 0;
  private aveboostOut = 0;
  private avebiasOut = 0;
  private maxbondperatom = FCCBONDS;

  // per hyper command (init_hyper)
  private nbiasRunning = 0;
  private nobiasRunning = 0;
  private negstrainRunning = 0;
  private maxdriftsq = 0;
  private maxbondlen = 0;
  private avebiasRunning = 0;
  private minbiasRunning = BIG;
  private maxbiasRunning = 0;

  // values of the last bias evaluation
  private mybias = 0;
  private nbias = 0;
  private sumboost = 0;
  private sumbiascoeff = 0;
  private minbiascoeff = 0;
  private maxbiascoeff = 0;
  private lastMaxstrain = new Float64Array(0);

  // the bond list: owner atom ID, partner atom ID, quenched length R0, bias coefficient C
  private bTagI = new Float64Array(0);
  private bTagJ = new Float64Array(0);
  private bOwner = new Int32Array(0);
  private bR0 = new Float64Array(0);
  private bC = new Float64Array(0);
  private nbonds = 0;

  // per atom of the last bond build (index = local index at the build; the IDs map it to the current index)
  private buildTag = new Float64Array(0);
  /** full neighbor list of the build (cutoff Dcut): CSR over the build atoms, partner IDs */
  private nbrStart = new Int32Array(1);
  private nbrTag = new Float64Array(0);
  /** positions and IDs at the last build (xold, tagold) */
  private oldX = new Float64Array(0);
  private oldIndex = new Map<number, number>();

  private readonly d = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 7) {
      throw new StyleError(`fix ${id} hyper/local: expected cutbond qfactor Vmax Tequil Dcut alpha boost [keywords], got ${args.length} arguments`);
    }
    this.cutbond = parseNum(args[0], `fix ${id} hyper/local cutbond`);
    this.qfactor = parseNum(args[1], `fix ${id} hyper/local qfactor`);
    this.vmax = parseNum(args[2], `fix ${id} hyper/local Vmax`);
    this.tequil = parseNum(args[3], `fix ${id} hyper/local Tequil`);
    this.dcut = parseNum(args[4], `fix ${id} hyper/local Dcut`);
    this.alphaUser = parseNum(args[5], `fix ${id} hyper/local alpha`);
    this.boostTarget = parseNum(args[6], `fix ${id} hyper/local boost`);
    // fix_hyper_local.cpp: "if (cutbond < 0.0 || qfactor < 0.0 || vmax < 0.0 || tequil <= 0.0 || dcut <= 0.0 ||
    // alpha_user <= 0.0 || boost_target < 1.0) error->all(FLERR,"Illegal fix hyper/local command");"
    if (this.cutbond < 0 || this.qfactor < 0 || this.vmax < 0 || this.tequil <= 0 || this.dcut <= 0 ||
        this.alphaUser <= 0 || this.boostTarget < 1) {
      throw new StyleError(`fix ${id} hyper/local: illegal arguments (cutbond, qfactor and Vmax >= 0; Tequil, Dcut, alpha > 0; boost >= 1)`);
    }
    let boundfrac = 0;
    let boundflag = false;
    let resetfreq = -1;
    for (let k = 7; k < args.length;) {
      const kw = args[k];
      if (kw === 'bound') {
        if (k + 1 >= args.length) throw new StyleError(`fix ${id} hyper/local: keyword bound needs a value`);
        boundfrac = parseNum(args[k + 1], `fix ${id} hyper/local bound`);
        boundflag = boundfrac >= 0;
        k += 2;
      } else if (kw === 'reset') {
        if (k + 1 >= args.length) throw new StyleError(`fix ${id} hyper/local: keyword reset needs a value`);
        resetfreq = parseNum(args[k + 1], `fix ${id} hyper/local reset`);
        if (!Number.isInteger(resetfreq) || resetfreq < -1) throw new StyleError(`fix ${id} hyper/local: reset must be an integer >= -1`);
        k += 2;
      } else if (kw === 'check/ghost' || kw === 'check/bias') {
        throw new StyleError(`fix ${id} hyper/local keyword ${kw} is not supported by the browser engine (it only adds diagnostics of ghost atoms and biased bonds)`);
      } else {
        throw new StyleError(`fix ${id} hyper/local: unknown keyword '${kw}' (bound, reset)`);
      }
    }
    this.invvmax = 1.0 / this.vmax;
    this.invqfactorsq = 1.0 / (this.qfactor * this.qfactor);
    this.cutbondsq = this.cutbond * this.cutbond;
    this.dcutsq = this.dcut * this.dcut;
    this.beta = 1.0 / (sys.units.boltz * this.tequil);
    this.boundflag = boundflag;
    this.boundLower = 1.0 - boundfrac;
    this.boundUpper = 1.0 + boundfrac;
    this.resetfreq = resetfreq;
    this.starttime = sys.state.step;
    this.lastreset = sys.state.step;
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 26;
    this.extscalar = 0;
    this.extvector = 0;
    this.timeIntegrate = false;
  }

  /** FixHyperLocal::init_hyper(): the per-command counters of the hyper command start from zero. */
  private initHyper(): void {
    this.maxdriftsq = 0;
    this.maxbondlen = 0;
    this.avebiasRunning = 0;
    this.minbiasRunning = BIG;
    this.maxbiasRunning = 0;
    this.nbiasRunning = 0;
    this.nobiasRunning = 0;
    this.negstrainRunning = 0;
  }

  /** The hyper command starts at this timestep: the counters reset (hyper.cpp calls init_hyper, then sets firststep). */
  beginRun(step: number): void {
    this.runStart = step;
    this.initHyper();
  }

  /** The event count and atoms are counted by buildBonds(natom) (native build_bond_list), so nothing is left to do. */
  recordEvent(): void {}

  /** Native keeps no drift of a rebond without event for this fix (its drift is measured by the bond build). */
  recordDrift(): void {}

  /** The hyper command's quench is running (update->whichflag == 2 in native). */
  setQuench(on: boolean): void {
    this.inQuench = on;
  }

  /** Cumulative values for the hyper command's final stats (the entries that query() reads in native). */
  summary(): { hyperTime: number; events: number; eventAtoms: number } {
    return { hyperTime: this.computeVector(22), events: this.nevent, eventAtoms: this.neventAtom };
  }

  /** Squared minimum-image separation of atoms i and j; the separation vector (x_i - x_j) is left in this.d. */
  private sep(i: number, j: number): number {
    const x = this.sys.state.x;
    const d = this.d;
    for (let c = 0; c < 3; c++) d[c] = x[3 * i + c] - x[3 * j + c];
    this.sys.geom.minimumImage(d);
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  }

  private groupCount(): number {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.inGroup(i)) n++;
    return n;
  }

  /** ID -> current local index. */
  private tagIndex(): Map<number, number> {
    const s = this.sys.state;
    const m = new Map<number, number>();
    for (let i = 0; i < s.n; i++) m.set(s.id[i], i);
    return m;
  }

  private indexOf(map: Map<number, number>, tag: number): number {
    const k = map.get(tag);
    if (k === undefined) throw new StyleError(`fix ${this.id} hyper/local: bond atom ${tag} not found`);
    return k;
  }

  /** fix_hyper_local.cpp: resetfreq block of build_bond_list(): the bias coefficients are rescaled to their mean. */
  private resetBias(): void {
    const step = this.sys.state.step;
    const elapsed = step - this.lastreset;
    const flag = this.resetfreq === 0 ? elapsed !== 0 : elapsed >= this.resetfreq;
    if (!flag) return;
    this.lastreset = step;
    const vmaxold = this.vmax;
    const ave = this.sumbiascoeff / this.nbonds;
    this.vmax *= ave;
    for (let m = 0; m < this.nbonds; m++) this.bC[m] *= vmaxold / this.vmax;
    if (this.boundflag) {
      for (let m = 0; m < this.nbonds; m++) {
        this.bC[m] = MAXP(this.bC[m], this.boundLower);
        this.bC[m] = MINP(this.bC[m], this.boundUpper);
      }
    }
  }

  /**
   * FixHyperLocal::build_bond_list(natom): the bond list of the current (quenched) positions. natom = atoms in the
   * event that triggered the build (0 for the first build and for a rebond without event).
   */
  buildBonds(natom: number): void {
    const s = this.sys.state;
    const n = s.n;
    if (natom) {
      this.nevent++;
      this.neventAtom += natom;
    }
    if (this.resetfreq >= 0) this.resetBias();

    // drift of the bond atoms since the last build (old bonds, positions at the last build)
    if (this.nbonds > 0) {
      const cur = this.tagIndex();
      const d = this.d;
      for (let m = 0; m < this.nbonds; m++) {
        for (const tag of [this.bTagI[m], this.bTagJ[m]]) {
          const k = this.indexOf(cur, tag);
          const o = this.oldIndex.get(tag);
          if (o === undefined) throw new StyleError(`fix ${this.id} hyper/local: bond atom ${tag} not found`);
          for (let c = 0; c < 3; c++) d[c] = s.x[3 * k + c] - this.oldX[3 * o + c];
          this.sys.geom.minimumImage(d);
          const distsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
          this.maxdriftsq = MAXP(distsq, this.maxdriftsq);
        }
      }
    }

    // maxbondperatom: the largest number of old bonds any atom is part of (the clist pages of native)
    const oldC = new Map<string, number>();
    if (!this.firstflag && this.nbonds > 0) {
      const cnt = new Map<number, number>();
      let mx = 0;
      for (let m = 0; m < this.nbonds; m++) {
        for (const tag of [this.bTagI[m], this.bTagJ[m]]) {
          const c = (cnt.get(tag) ?? 0) + 1;
          cnt.set(tag, c);
          mx = MAXP(mx, c);
        }
        // the bias coefficient of a pair of atoms, found by the partner's ID (clist search in native)
        const key = pairKey(this.bTagI[m], this.bTagJ[m]);
        if (!oldC.has(key)) oldC.set(key, this.bC[m]);
      }
      this.maxbondperatom = MAXP(this.maxbondperatom, mx);
    }

    // the full neighbor list with cutoff Dcut (occasional list of native) on the current positions
    const g = this.sys.geom;
    const lens = [g.lx, g.ly, g.lz];
    // (a 2d box has no z neighbors: its z extent does not count)
    for (let c = 0; c < (s.dimension === 2 ? 2 : 3); c++) {
      if (g.periodic[c] && lens[c] < 2 * this.dcut) {
        throw new StyleError(`fix ${this.id} hyper/local: a periodic box length must exceed 2 Dcut (the browser engine lists one image per atom)`);
      }
    }
    const start = new Int32Array(n + 1);
    const tags: number[] = [];
    for (let i = 0; i < n; i++) {
      start[i] = tags.length;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        if (this.sep(i, j) < this.dcutsq) tags.push(s.id[j]);
      }
    }
    start[n] = tags.length;
    this.nbrStart = start;
    this.nbrTag = Float64Array.from(tags);

    // the new bonds: the pairs of the half list (owner first, native pair order), one of them in the group, closer
    // than cutbond; a bond keeps the bias coefficient of the same pair of atoms when it had one
    const half = this.sys.nb.half;
    if (!half) throw new StyleError(`fix ${this.id} hyper/local needs a pair style with a half neighbor list`);
    // native build_one(listhalf) at every bond build: the owners follow the bins of the current positions
    const pairs = nativePairs(s, this.sys.nb, g, half, true);
    const tI: number[] = [], tJ: number[] = [], owner: number[] = [], r0: number[] = [], cs: number[] = [];
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const j = this.sys.nb.owner[pairs[p + 1] & NEIGHMASK];
      if (!(this.inGroup(i) || this.inGroup(j))) continue;
      const rsq = this.sep(i, j);
      if (!(rsq < this.cutbondsq)) continue;
      const key = pairKey(s.id[i], s.id[j]);
      const oldcoeff = this.firstflag ? 0 : (oldC.get(key) ?? 0);
      let c: number;
      if (oldcoeff > 0) c = oldcoeff;
      else {
        c = COEFFINIT;
        this.nnewbond++;
      }
      tI.push(s.id[i]);
      tJ.push(s.id[j]);
      owner.push(i);
      r0.push(Math.sqrt(rsq));
      cs.push(c);
    }
    if (this.firstflag) this.nnewbond = 0;
    this.bTagI = Float64Array.from(tI);
    this.bTagJ = Float64Array.from(tJ);
    this.bOwner = Int32Array.from(owner);
    this.bR0 = Float64Array.from(r0);
    this.bC = Float64Array.from(cs);
    this.nbonds = tI.length;

    // the positions and IDs of this build (xold, tagold)
    this.buildTag = Float64Array.from({ length: n }, (_, i) => s.id[i]);
    this.oldX = Float64Array.from(s.x.subarray(0, 3 * n));
    this.oldIndex = new Map<number, number>();
    for (let i = 0; i < n; i++) this.oldIndex.set(s.id[i], i);
    this.firstflag = false;
  }

  /**
   * FixHyperLocal::pre_reverse(): the maxstrain of every atom, the eligible bond of each atom, the bias forces, and
   * (in dynamics only, not setup) the bias coefficient update and the statistics.
   */
  private biasForces(setup: boolean): void {
    const s = this.sys.state;
    const n = s.n;
    const f = s.f;
    const q = this.qfactor;
    const vmax = this.vmax;
    const cur = this.tagIndex();
    const nbl = this.nbonds;
    const bi = new Int32Array(nbl);
    const bj = new Int32Array(nbl);
    for (let m = 0; m < nbl; m++) {
      bi[m] = this.indexOf(cur, this.bTagI[m]);
      bj[m] = this.indexOf(cur, this.bTagJ[m]);
    }
    const nb = this.buildTag.length;
    this.nostrainyet = false;

    // strain of each bond (estrain = |r - r0| / r0); maxstrain of each atom over all its bonds; the first largest
    // strain among the bonds each build atom owns (maxhalf / maxhalfstrain)
    const maxstrain = new Float64Array(n);
    const halfstrain = new Float64Array(nb);
    const ijhalf = new Int32Array(nb).fill(-1);
    for (let m = 0; m < nbl; m++) {
      const r = Math.sqrt(this.sep(bi[m], bj[m]));
      this.maxbondlen = MAXP(r, this.maxbondlen);
      const estrain = Math.abs(r - this.bR0[m]) / this.bR0[m];
      maxstrain[bi[m]] = MAXP(maxstrain[bi[m]], estrain);
      maxstrain[bj[m]] = MAXP(maxstrain[bj[m]], estrain);
      const o = this.bOwner[m];
      if (estrain > halfstrain[o]) {
        halfstrain[o] = estrain;
        ijhalf[o] = m;
      }
    }

    // maxstrain_domain: the largest strain in the neighborhood (Dcut) of each atom; a tie of the largest strain
    // among two or more neighbors is marked by a negative value and makes the atom ineligible
    const cb = new Int32Array(nb);
    for (let k = 0; k < nb; k++) cb[k] = this.indexOf(cur, this.buildTag[k]);
    const eligible = new Uint8Array(nb).fill(1);
    const maxdom = new Float64Array(n);
    for (let k = 0; k < nb; k++) {
      const i = cb[k];
      if (maxstrain[i] === 0) {
        eligible[k] = 0;
        continue;
      }
      const selfstrain = maxstrain[i];
      let emax = selfstrain;
      let ncount = 0;
      for (let p = this.nbrStart[k]; p < this.nbrStart[k + 1]; p++) {
        const j = this.indexOf(cur, this.nbrTag[p]);
        emax = MAXP(emax, maxstrain[j]);
        if (selfstrain === maxstrain[j]) ncount++;
      }
      if (halfstrain[k] < selfstrain) eligible[k] = 0;
      if (selfstrain < emax) {
        eligible[k] = 0;
      } else if (ncount > 1) {
        eligible[k] = 0;
        emax = -emax;
      }
      maxdom[i] = emax;
    }

    // the biased bonds: the owner's largest-strain bond, when the partner's own strain is its neighborhood maximum
    const biasList: number[] = [];
    for (let k = 0; k < nb; k++) {
      if (!eligible[k]) continue;
      const m = ijhalf[k];
      if (m < 0) continue;
      if (maxstrain[bj[m]] !== maxdom[bj[m]]) continue;
      biasList.push(m);
    }

    // bias forces on the biased bonds (applied in setup too)
    let nobias = 0;
    let negstrain = 0;
    let mybias = 0;
    for (const m of biasList) {
      const i = bi[m];
      const j = bj[m];
      if (maxstrain[i] >= q) {
        nobias++;
        continue;
      }
      const r = Math.sqrt(this.sep(i, j));
      const d = this.d;
      const r0 = this.bR0[m];
      const c = this.bC[m];
      const ebias = (r - r0) / r0;
      const vbias = c * vmax * (1.0 - ebias * ebias * this.invqfactorsq);
      const fbias = 2.0 * c * vmax * ebias * this.invqfactorsq;
      const fbiasr = fbias / r0 / r;
      for (let k = 0; k < 3; k++) {
        f[3 * i + k] += d[k] * fbiasr;
        f[3 * j + k] -= d[k] * fbiasr;
      }
      if (ebias < 0) negstrain++;
      mybias += vbias;
    }
    this.nbias = biasList.length;
    this.mybias = mybias;
    this.lastMaxstrain = maxstrain;
    if (setup) return;

    // dynamics: the statistics, then the bias coefficient of every bond moves toward the boost target
    this.nbiasRunning += this.nbias;
    this.nobiasRunning += nobias;
    this.negstrainRunning += negstrain;
    const alpha = s.dt / this.alphaUser;
    const target = this.boostTarget;
    let minC = BIG;
    let maxC = 0;
    let sumC = 0;
    let sumB = 0;
    for (let m = 0; m < nbl; m++) {
      const emax = MAXP(maxdom[bi[m]], maxdom[bj[m]]);
      const vb = emax < q ? vmax * (1.0 - emax * emax * this.invqfactorsq) : 0;
      const boost = Math.exp(this.beta * this.bC[m] * vb);
      let c = this.bC[m];
      c -= alpha * this.invvmax * (boost - target) / target;
      c = MAXP(c, 0.0);
      if (this.boundflag) {
        c = MAXP(c, this.boundLower);
        c = MINP(c, this.boundUpper);
      }
      this.bC[m] = c;
      sumC += c;
      minC = MINP(minC, c);
      maxC = MAXP(maxC, c);
      sumB += boost;
    }
    this.sumbiascoeff = sumC;
    if (nbl) this.avebiasRunning += sumC / nbl;
    this.minbiasRunning = MINP(this.minbiasRunning, minC);
    this.maxbiasRunning = MAXP(this.maxbiasRunning, maxC);
    this.minbiascoeff = minC;
    this.maxbiascoeff = maxC;
    this.sumboost = sumB;
    if (nbl) this.aveboostRunning += sumB / nbl;
  }

  preReverse(): void {
    this.biasForces(false);
  }

  setupPreReverse(): void {
    this.biasForces(true);
  }

  /** The bias is added in setupPreReverse / preReverse (no post_force of its own). */
  setup(): void {}

  computeScalar(): number {
    return this.mybias;
  }

  computeVector(i: number): number {
    const s = this.sys.state;
    const step = s.step;
    const elapsed = step - this.runStart;
    const allbonds = this.nbonds;
    const groupatoms = this.groupCount();
    switch (i) {
      case 0: return allbonds ? this.sumboost / allbonds : 1.0;
      case 1: return this.nbias;
      case 2: {
        if (this.nostrainyet) return 0;
        let emax = 0;
        for (let j = 0; j < s.n; j++) emax = MAXP(emax, this.lastMaxstrain[j]);
        return emax;
      }
      case 3: return this.vmax;
      case 4: return allbonds ? this.sumbiascoeff / allbonds : 1.0;
      case 5: return this.minbiascoeff;
      case 6: return this.maxbiascoeff;
      case 7: return allbonds / groupatoms;
      case 8: {
        const neighsperatom = this.nbrTag.length / s.n;
        const bondsperatom = allbonds / groupatoms;
        return neighsperatom * bondsperatom;
      }
      case 9: {
        if (step === this.runStart) this.aveboostOut = 0;
        else if (!this.inQuench) this.aveboostOut = this.aveboostRunning / elapsed;
        return this.aveboostOut;
      }
      case 10: return step === this.runStart ? 0 : this.nbiasRunning / elapsed;
      case 11: return this.nbiasRunning ? this.nobiasRunning / this.nbiasRunning : 0;
      case 12: return this.nbiasRunning ? this.negstrainRunning / this.nbiasRunning : 0;
      case 13: return this.maxbondlen;
      case 14: {
        if (step === this.runStart) this.avebiasOut = 0;
        else if (!this.inQuench) this.avebiasOut = this.avebiasRunning / elapsed;
        return this.avebiasOut;
      }
      case 15: return this.minbiasRunning;
      case 16: return this.maxbiasRunning;
      case 17: return Math.sqrt(this.maxdriftsq);
      case 18: return 0;
      case 19: return 0;
      case 20: return 0;
      case 21: return 0;
      case 22: return this.boostTarget * s.dt * (step - this.starttime);
      case 23: return this.nevent;
      case 24: return this.neventAtom;
      case 25: return this.nnewbond;
      default: throw new StyleError(`fix ${this.id} hyper/local vector index ${i + 1} is out of range`);
    }
  }

  modify(key: string, values: string[]): number {
    if (key === 'energy' || key === 'virial') throw new StyleError(`fix ${this.id} (hyper/local) does not support fix_modify ${key} in the browser engine`);
    return super.modify(key, values);
  }
}

/** Unordered key of a pair of atom IDs. */
const pairKey = (a: number, b: number): string => (a < b ? `${a},${b}` : `${b},${a}`);

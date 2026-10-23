/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_nonaffine_displacement.cpp and src/EXTRA-FIX/fix_nonaffine_displacement.h at commit b332f28
 * (class FixNonaffineDisplacement: the constructor, post_constructor, init, setup, post_force, integrate_velocity,
 * save_reference_state, calculate_D2Min, minimum_image0; contributing authors Joel Clemmer (SNL) and Ishan Srivastava
 * (LBNL)). Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Non-affine displacement of the atoms relative to a reference state (docs.lammps.org/fix_nonaffine_displacement.html),
 * the "integrated" style: the displacement accumulates dt * v of the group atoms from the reference step on, and the
 * per-atom output (3 columns) is that displacement, refreshed every nevery steps. The reference state is "fixed" at a
 * timestep, "update" every N steps, or "offset" (saved at the step that precedes a multiple of nevery).
 *
 * The "d2min" style (Falk and Langer): over the group neighbors within the cutoff (type: the pair cutoffs, radius:
 * the sum of the radii, custom: a distance), the best affine deformation F = X Y^-1 of each atom's neighbor vectors
 * from the reference state, and the per-atom output (9 columns) sqrt(D2min), the volumetric strain, the deviatoric
 * strain and the six components of the Green strain. The neighbor pairs come from an occasional half list over the
 * owned and ghost atoms, each periodic pair once (native newton on: both ends take the pair, then the ghost sums go to
 * their owners); the reference vectors use the owners' stored positions with native's minimum_image0 in the
 * reference box.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { missingArgs } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';
import type { CustomProp } from '../../engine/types';
import { NEIGHMASK } from '../../engine/neighbor';

/** fix_nonaffine_displacement.cpp EPSILON: the smallest determinant of Y taken as invertible. */
const EPSILON = 1.0e-15;

const logical = (w: string | undefined): boolean => {
  if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
  if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script`);
};

function inumeric(w: string | undefined): number {
  if (w === undefined || !/^[+-]?\d+$/.test(w.trim())) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return Number(w);
}

type RefStyle = 'fixed' | 'update' | 'offset';

export class FixNonaffineDisplacement extends Fix {
  readonly style = 'nonaffine/displacement';
  peratomFlag = true;
  sizePeratomCols = 3;
  private readonly refStyle: RefStyle;
  private readonly referenceTimestep: number;
  private readonly updateTimestep: number;
  private readonly offsetTimestep: number;
  private referenceSaved = false;
  private readonly key: string;
  private dtv = 0;
  private readonly d2min: boolean = false;
  private cutStyle: 'type' | 'radius' | 'custom' = 'type';
  private cutoffCustom = 0;
  private cutsqCustom = 0;
  private zMin = 0;
  private intensive = true;
  /** the reference box (xprd0 ... yz0 of save_reference_state) */
  private box0 = { xprd: 0, yprd: 0, zprd: 0, xh: 0, yh: 0, zh: 0, xy: 0, xz: 0, yz: 0 };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // native narg counts "fix ID group nonaffine/displacement": narg < 4 is fewer than one argument
    if (args.length < 1) throw missingArgs('fix nonaffine/displacement');
    this.nevery = inumeric(args[0]);
    // peratom_freq = nevery (the source's constructor)
    this.peratomFreq = this.nevery;
    if (this.nevery <= 0) throw new StyleError(`Illegal nevery value ${this.nevery} in fix nonaffine/displacement`);
    let iarg = 1;
    if (args[iarg] === 'integrated') {
      iarg += 1;
    } else if (args[iarg] === 'd2min') {
      if (iarg + 1 >= args.length) throw missingArgs('fix nonaffine/displacement d2min');
      this.d2min = true;
      const cs = args[iarg + 1];
      if (cs === 'type' || cs === 'radius') this.cutStyle = cs;
      else if (cs === 'custom') {
        if (iarg + 2 >= args.length) throw missingArgs('fix nonaffine/displacement custom');
        if (sys.nb.multiStyle) throw new StyleError("Fix nonaffine/displacement with custom cutoff requires neighbor style 'bin' or 'nsq'");
        this.cutStyle = 'custom';
        const c = Number(args[iarg + 2]);
        if (!Number.isFinite(c)) throw new StyleError(`Expected floating point parameter instead of '${args[iarg + 2]}' in input script or data file`);
        this.cutoffCustom = c;
        this.cutsqCustom = c * c;
        if (c <= 0) throw new StyleError(`Illegal custom cutoff length ${args[iarg + 2]}`);
        iarg += 1;
      } else throw new StyleError(`Illegal cutoff style ${cs} in fix nonaffine/displacement`);
      iarg += 2;
    } else {
      throw new StyleError(`Illegal nonaffine displacement style ${args[iarg] ?? ''} in fix nonaffine/displacement`);
    }
    if (iarg + 2 > args.length) throw missingArgs('fix nonaffine/displacement');
    let refStyle: RefStyle;
    let refTs = -1, updTs = -1, offTs = -1;
    if (args[iarg] === 'fixed') {
      refStyle = 'fixed';
      refTs = inumeric(args[iarg + 1]);
      if (refTs < 0) throw new StyleError(`Illegal reference timestep ${args[iarg + 1]} in fix nonaffine/displacement`);
    } else if (args[iarg] === 'update') {
      refStyle = 'update';
      updTs = inumeric(args[iarg + 1]);
      if (updTs <= 0) throw new StyleError(`Illegal update timestep ${args[iarg + 1]} in fix nonaffine/displacement`);
    } else if (args[iarg] === 'offset') {
      refStyle = 'offset';
      offTs = inumeric(args[iarg + 1]);
      if (offTs <= 0 || offTs > this.nevery) throw new StyleError(`Illegal offset timestep ${args[iarg + 1]} in fix nonaffine/displacement`);
    } else {
      throw new StyleError(`Illegal reference style ${args[iarg]} in fix nonaffine/displacement`);
    }
    this.refStyle = refStyle;
    this.referenceTimestep = refTs;
    this.updateTimestep = updTs;
    this.offsetTimestep = offTs;
    iarg += 2;
    while (iarg < args.length) {
      if (args[iarg] === 'z/min') {
        if (iarg + 2 > args.length) throw missingArgs('fix nonaffine/displacement');
        this.zMin = inumeric(args[iarg + 1]);
        if (this.zMin < 0) throw new StyleError('Minimum coordination must be positive');
        iarg += 2;
      } else if (args[iarg] === 'intensive/d2min') {
        if (iarg + 2 > args.length) throw missingArgs('fix nonaffine/displacement');
        this.intensive = logical(args[iarg + 1]);
        if (!this.d2min) throw new StyleError('Can only specify extensive/intensive d2min with style d2min');
        // native advances by one here, so its value is read next as a keyword (and stops the command)
        iarg += 1;
      } else {
        throw new StyleError(`Illegal keyword ${args[iarg]} in fix nonaffine/displacement`);
      }
    }
    if (this.d2min && this.cutStyle === 'radius' && !sys.state.radius) {
      throw new StyleError('Fix nonaffine/displacement radius style requires atom attribute radius');
    }
    if (this.d2min) this.sizePeratomCols = 9;
    this.key = `#nad:${id}`;
    this.ensureStore();
  }

  /** The stored reference-relative displacement, 3 values per atom (the native STORE/ATOM fix id_fix). */
  private ensureStore(): Float64Array {
    const s = this.sys.state;
    let c: CustomProp | undefined = s.custom.get(this.key);
    if (!c) {
      c = { int: false, cols: 3, data: new Float64Array(3 * s.n) };
      s.custom.set(this.key, c);
    }
    return c.data;
  }

  init(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    if (!this.referenceSaved && this.refStyle === 'fixed' && s.step > this.referenceTimestep) {
      throw new StyleError('Initial timestep exceeds that of the reference state in fix nonaffine/displacement');
    }
    if (this.d2min) {
      const pair = this.sys.ff.pair;
      if (!pair && this.cutStyle === 'type') {
        throw new StyleError('Fix nonaffine/displacement D2Min option requires a pair style be defined or cutoff specified');
      }
    }
  }

  /**
   * init's ghost range check of the custom cutoff (fix_nonaffine_displacement.cpp: "cutghost = MAX(force->pair->cutforce
   * + skin, comm->cutghostuser)"), made at setup, once the pair's cutoffs are set.
   */
  private checkCustomGhost(): void {
    if (!this.d2min || this.cutStyle !== 'custom') return;
    const nb = this.sys.nb;
    const mycutneigh = this.cutoffCustom + nb.skin;
    const cutghost = this.sys.ff.pair ? Math.max(this.pairMaxCut() + nb.skin, nb.commCutoff) : nb.commCutoff;
    if (mycutneigh > cutghost) {
      throw new StyleError('Fix nonaffine/displacement D2Min option cutoff exceeds ghost atom range - use comm_modify cutoff command');
    }
  }

  /** The largest pair cutoff (the pair's cutforce). */
  private pairMaxCut(): number {
    const pair = this.sys.ff.pair;
    let m = 0;
    if (pair) for (let k = 0; k < pair.cut.length; k++) if (pair.cut[k] > m) m = pair.cut[k];
    return m;
  }

  setup(): void {
    this.checkCustomGhost();
    this.postForceStep(true);
  }

  postForce(): void { this.postForceStep(false); }

  /** post_force (setupflag: the setup call of the run, where the displacement is zero). */
  private postForceStep(setupFlag: boolean): void {
    const s = this.sys.state;
    const n = s.n;
    const store = this.ensureStore();
    const ncol = this.sizePeratomCols;
    if (this.arrayAtom.length !== ncol * n) this.arrayAtom = new Float64Array(ncol * n);
    if (this.referenceSaved && !setupFlag) {
      if (this.d2min) {
        if (s.step % this.nevery === 0) this.calculateD2Min();
      } else {
        this.integrateVelocity();
        if (s.step % this.nevery === 0) {
          for (let i = 0; i < n; i++) for (let m = 0; m < 3; m++) this.arrayAtom[3 * i + m] = store[3 * i + m];
        }
      }
    } else {
      this.arrayAtom.fill(0);
    }
    if (this.refStyle === 'fixed' && s.step === this.referenceTimestep) this.saveReferenceState();
    if (this.refStyle === 'update' && s.step % this.updateTimestep === 0) this.saveReferenceState();
    if (this.refStyle === 'offset' && (s.step + this.offsetTimestep) % this.nevery === 0) this.saveReferenceState();
  }

  /** integrate_velocity: the displacement of each group atom grows by dt v. */
  private integrateVelocity(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    const store = this.ensureStore();
    const { v, mask } = s;
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      for (let m = 0; m < 3; m++) store[3 * i + m] += this.dtv * v[3 * i + m];
    }
  }

  /**
   * save_reference_state: the integrated displacement restarts at zero, or (d2min) the reference positions of the group
   * atoms and the reference box are stored.
   */
  private saveReferenceState(): void {
    const s = this.sys.state;
    const store = this.ensureStore();
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) for (let m = 0; m < 3; m++) store[3 * i + m] = this.d2min ? s.x[3 * i + m] : 0.0;
    }
    if (this.d2min) {
      const g = this.sys.geom;
      this.box0 = { xprd: g.lx, yprd: g.ly, zprd: g.lz, xh: 0.5 * g.lx, yh: 0.5 * g.ly, zh: 0.5 * g.lz, xy: g.xy, xz: g.xz, yz: g.yz };
    }
    this.referenceSaved = true;
  }

  /** minimum_image0: a reference separation folded into the reference box (z, then y, then x for a triclinic one). */
  private minimumImage0(d: number[]): void {
    const b = this.box0;
    const per = this.sys.state.box.periodic;
    if (!this.sys.state.box.triclinic) {
      if (per[0]) while (Math.abs(d[0]) > b.xh) { if (d[0] < 0.0) d[0] += b.xprd; else d[0] -= b.xprd; }
      if (per[1]) while (Math.abs(d[1]) > b.yh) { if (d[1] < 0.0) d[1] += b.yprd; else d[1] -= b.yprd; }
      if (per[2]) while (Math.abs(d[2]) > b.zh) { if (d[2] < 0.0) d[2] += b.zprd; else d[2] -= b.zprd; }
    } else {
      if (per[2]) {
        while (Math.abs(d[2]) > b.zh) {
          if (d[2] < 0.0) { d[2] += b.zprd; d[1] += b.yz; d[0] += b.xz; }
          else { d[2] -= b.zprd; d[1] -= b.yz; d[0] -= b.xz; }
        }
      }
      if (per[1]) {
        while (Math.abs(d[1]) > b.yh) {
          if (d[1] < 0.0) { d[1] += b.yprd; d[0] += b.xy; }
          else { d[1] -= b.yprd; d[0] -= b.xy; }
        }
      }
      if (per[0]) while (Math.abs(d[0]) > b.xh) { if (d[0] < 0.0) d[0] += b.xprd; else d[0] -= b.xprd; }
    }
  }

  /** calculate_D2Min: the two passes over the group's neighbor pairs and the strains of each atom. */
  private calculateD2Min(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const n = s.n;
    const dim = s.dimension;
    const pair = sys.ff.pair;
    const radius = s.radius;
    const nt = s.ntypes + 1;
    // the occasional list: the pair cutoffs (type), the radii sums (radius) or the custom distance, plus the skin
    let cut: number;
    if (this.cutStyle === 'custom') cut = this.cutoffCustom;
    else if (this.cutStyle === 'radius') {
      let rmax = 0;
      for (let i = 0; i < n; i++) if (radius![i] > rmax) rmax = radius![i];
      cut = 2 * rmax;
    } else cut = this.pairMaxCut();
    const list = nb.occasionalList(s, cut, false, s.ntypes);
    const nall = nb.nall, owner = nb.owner, xa = nb.xall, ta = nb.typeall;
    const x0 = this.ensureStore();
    const X = new Float64Array(9 * nall), Y = new Float64Array(9 * nall);
    const inGroup = (k: number): boolean => (s.mask[owner[k]] & this.groupBit) !== 0;
    const outside = (i: number, j: number, rsq: number, strict: boolean): boolean => {
      let c2: number;
      if (this.cutStyle === 'type') c2 = pair!.cutsq[ta[i] * nt + ta[j]];
      else if (this.cutStyle === 'custom') c2 = this.cutsqCustom;
      else { const rs = radius![owner[i]] + radius![owner[j]]; c2 = rs * rs; }
      return strict ? rsq > c2 : rsq >= c2;
    };
    const r = [0, 0, 0], r0 = [0, 0, 0];
    const sep = (i: number, j: number): number => {
      r[0] = xa[3 * i] - xa[3 * j]; r[1] = xa[3 * i + 1] - xa[3 * j + 1]; r[2] = xa[3 * i + 2] - xa[3 * j + 2];
      const oi = owner[i], oj = owner[j];
      r0[0] = x0[3 * oi] - x0[3 * oj]; r0[1] = x0[3 * oi + 1] - x0[3 * oj + 1]; r0[2] = x0[3 * oi + 2] - x0[3 * oj + 2];
      return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    };
    // first pass: X = sum r (x) r0, Y = sum r0 (x) r0 ("Only include contributions from atoms that are CURRENTLY neighbors")
    for (let i = 0; i < list.inum; i++) {
      if (!inGroup(i)) continue;
      for (let p = list.firstneigh[i]; p < list.firstneigh[i + 1]; p++) {
        const j = list.neighbors[p] & NEIGHMASK;
        if (!inGroup(j)) continue;
        const rsq = sep(i, j);
        if (outside(i, j, rsq, true)) continue;
        this.minimumImage0(r0);
        for (let k = 0; k < 3; k++) {
          for (let l = 0; l < 3; l++) {
            const xk = r[k] * r0[l], yk = r0[k] * r0[l];
            X[9 * i + 3 * k + l] += xk; Y[9 * i + 3 * k + l] += yk;
            X[9 * j + 3 * k + l] += xk; Y[9 * j + 3 * k + l] += yk;
          }
        }
      }
    }
    const Xo = new Float64Array(9 * n), Yo = new Float64Array(9 * n);
    nb.reverseSum(X, 9, Xo);
    nb.reverseSum(Y, 9, Yo);
    // F = X Y^-1 per owned atom (zero inverse when Y is singular), then copied to the ghosts
    const F = new Float64Array(9 * nall);
    const singular = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const y = Yo.subarray(9 * i, 9 * i + 9), xm = Xo.subarray(9 * i, 9 * i + 9);
      const yi = new Float64Array(9);
      if (dim === 3) {
        const det = y[0] * y[4] * y[8] - y[0] * y[5] * y[7] - y[3] * y[1] * y[8] + y[3] * y[2] * y[7] + y[6] * y[1] * y[5] - y[6] * y[2] * y[4];
        if (Math.abs(det) < EPSILON) singular[i] = 1;
        else {
          // MathExtra::invert3
          let den = y[0] * y[4] * y[8] - y[0] * y[5] * y[7];
          den += -y[3] * y[1] * y[8] + y[3] * y[2] * y[7];
          den += y[6] * y[1] * y[5] - y[6] * y[2] * y[4];
          yi[0] = (y[4] * y[8] - y[5] * y[7]) / den;
          yi[1] = -(y[1] * y[8] - y[2] * y[7]) / den;
          yi[2] = (y[1] * y[5] - y[2] * y[4]) / den;
          yi[3] = -(y[3] * y[8] - y[5] * y[6]) / den;
          yi[4] = (y[0] * y[8] - y[2] * y[6]) / den;
          yi[5] = -(y[0] * y[5] - y[2] * y[3]) / den;
          yi[6] = (y[3] * y[7] - y[4] * y[6]) / den;
          yi[7] = -(y[0] * y[7] - y[1] * y[6]) / den;
          yi[8] = (y[0] * y[4] - y[1] * y[3]) / den;
        }
      } else {
        let den = y[0] * y[4] - y[1] * y[3];
        if (Math.abs(den) < EPSILON) singular[i] = 1;
        else {
          den = 1.0 / den;
          yi[0] = y[4] * den; yi[1] = -y[1] * den; yi[3] = -y[3] * den; yi[4] = y[0] * den;
        }
      }
      // MathExtra::times3
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) F[9 * i + 3 * a + b] = xm[3 * a] * yi[b] + xm[3 * a + 1] * yi[3 + b] + xm[3 * a + 2] * yi[6 + b];
      }
    }
    nb.forwardCopy(F, 9);
    // second pass: D2min = sum |r - F r0|^2 over the current neighbors, and their count
    const D2 = new Float64Array(nall), norm = new Float64Array(nall);
    const tmp = [0, 0, 0];
    const residual = (f: number): number => {
      for (let k = 0; k < 3; k++) tmp[k] = F[f + 3 * k] * r0[0] + F[f + 3 * k + 1] * r0[1] + F[f + 3 * k + 2] * r0[2];
      const a = r[0] - tmp[0], b = r[1] - tmp[1], c = r[2] - tmp[2];
      return a * a + b * b + c * c;
    };
    for (let i = 0; i < list.inum; i++) {
      if (!inGroup(i)) continue;
      for (let p = list.firstneigh[i]; p < list.firstneigh[i + 1]; p++) {
        const j = list.neighbors[p] & NEIGHMASK;
        if (!inGroup(j)) continue;
        const rsq = sep(i, j);
        if (outside(i, j, rsq, false)) continue;
        this.minimumImage0(r0);
        D2[i] += residual(9 * i);
        norm[i] += 1;
        D2[j] += residual(9 * j);
        norm[j] += 1;
      }
    }
    const D2o = new Float64Array(n), normo = new Float64Array(n);
    nb.reverseSum(D2, 1, D2o);
    nb.reverseSum(norm, 1, normo);
    const out = this.arrayAtom;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (normo[i] < this.zMin || singular[i] === 1) {
        if (normo[i] >= this.zMin) sys.warn(`Singular matrix detected for atom ${s.id[i]}, defaulting output to zero`);
        out[9 * i] = 0.0; out[9 * i + 1] = 0.0; out[9 * i + 2] = 0.0;
        continue;
      }
      let d2 = D2o[i];
      if (this.intensive) d2 /= normo[i];
      // E = F^T F - 1 (MathExtra::transpose_times3), then its trace part and the deviatoric part
      const f = F.subarray(9 * i, 9 * i + 9);
      const E = new Float64Array(9);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) E[3 * a + b] = f[a] * f[b] + f[3 + a] * f[3 + b] + f[6 + a] * f[6 + b];
      for (let a = 0; a < dim; a++) E[4 * a] -= 1.0;
      const evol = (E[0] + E[4] + E[8]) / dim;
      for (let a = 0; a < dim; a++) E[4 * a] -= evol;
      let j2 = 0.0;
      for (let a = 0; a < 9; a++) j2 += E[a] * E[a];
      out[9 * i] = Math.sqrt(d2);
      out[9 * i + 1] = evol;
      out[9 * i + 2] = Math.sqrt(0.5 * j2);
      out[9 * i + 3] = E[0]; out[9 * i + 4] = E[4]; out[9 * i + 5] = E[8];
      out[9 * i + 6] = E[1]; out[9 * i + 7] = E[2]; out[9 * i + 8] = E[5];
    }
  }
}

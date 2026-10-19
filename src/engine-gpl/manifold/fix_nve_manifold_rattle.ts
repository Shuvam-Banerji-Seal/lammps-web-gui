/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANIFOLD/fix_nve_manifold_rattle.cpp
 * and fix_nve_manifold_rattle.h at commit b332f28 (fix nve/manifold/rattle: velocity Verlet with the RATTLE
 * constraint of each particle to a manifold). MANIFOLD package by Stefan Paquay (Eindhoven University of
 * Technology), GNU General Public License; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/fix_nve_manifold_rattle.html. Syntax: fix ID group nve/manifold/rattle tol maxit manifold
 * manifold-args keyword value; a manifold argument may be v_name of an equal-style variable, read at init and
 * before every step. Each atom of the group removes one degree of freedom (dof()), counted for every temperature
 * compute whatever its group, as native.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { massOf } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { createManifold, infnorm, type Manifold } from './manifold';

/** utils::numeric. */
export const numeric = (w: string | undefined): number => {
  const t = (w ?? '').trim();
  if (!(/^[+-]?\d+\.?\d*$/.test(t) || /^[+-]?\d+\.?\d*[eE][+-]?\d+$/.test(t) || /^[+-]?\d*\.?\d+$/.test(t) || /^[+-]?\d*\.?\d+[eE][+-]?\d+$/.test(t))) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return Number(t);
};

/** utils::inumeric. */
export const inumeric = (w: string | undefined): number => {
  const t = (w ?? '').trim();
  if (!/^[+-]?\d+$/.test(t)) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  return Number(t);
};

/** was_var: a manifold argument v_name (longer than two characters). */
const wasVar = (s: string): boolean => s.length > 2 && s[0] === 'v' && s[1] === '_';

export class FixNVEManifoldRattle extends Fix {
  readonly style: string = 'nve/manifold/rattle';
  protected dtv = 0;
  protected dtf = 0;
  protected readonly tolerance: number;
  protected readonly maxIter: number;
  protected readonly man: Manifold;
  protected readonly nvars: number;
  private readonly tstrs: string[] = [];
  private readonly isVar: boolean[] = [];
  /** The stats output interval (native nevery; 0 = none). */
  protected statsEvery = 0;
  private nextOutput = 0;
  private readonly stats = { xIters: 0, vIters: 0, xItersPerAtom: 0, vItersPerAtom: 0, natoms: 0, dofsRemoved: 0, lastOut: 0 };

  constructor(sys: System, id: string, group: string, args: string[], errorOnUnknownKeyword = true) {
    super(sys, id, group, args);
    // native narg counts "ID group style": narg < 6 is fewer than 3 arguments here
    if (args.length < 3) throw new StyleError('Illegal fix nve/manifold/rattle command');
    this.timeIntegrate = true;
    this.nevery = 1;
    this.tolerance = numeric(args[0]);
    this.maxIter = inumeric(args[1]);
    const s = sys.state;
    const m = createManifold(args[2], { box: { lo: s.box.lo, hi: s.box.hi }, warn: (t) => sys.warn(t) });
    if (!m) throw new StyleError('Error creating manifold pointer');
    this.man = m;
    this.nvars = m.nparams;
    if (3 + this.nvars > args.length) {
      throw new StyleError(`Not enough args for manifold ${m.id}, ${this.nvars} expected but got ${args.length - 3}`);
    }
    for (let i = 0; i < this.nvars; ++i) {
      const a = args[i + 3];
      if (wasVar(a)) {
        this.isVar[i] = true;
        this.tstrs[i] = a.slice(2);
      } else {
        numeric(a);
        this.isVar[i] = false;
        this.tstrs[i] = a;
      }
    }
    m.params = new Float64Array(this.nvars);
    for (let i = 0; i < this.nvars; ++i) m.params[i] = this.isVar[i] ? 0.0 : numeric(args[i + 3]);
    m.postParamInit();
    let argi = 3 + this.nvars;
    while (argi < args.length) {
      if (args[argi] === 'every') {
        this.statsEvery = inumeric(args[argi + 1]);
        this.nextOutput = s.step + this.statsEvery;
        this.sys.log(`Outputting every ${this.statsEvery} steps, next is ${this.nextOutput}`);
        argi += 2;
      } else if (errorOnUnknownKeyword) {
        throw new StyleError(`Error parsing arg "${args[argi]}".`);
      } else {
        argi += 1;
      }
    }
    if (this.statsEvery > 0) this.nevery = this.statsEvery;
  }

  /** init: the manifold parameters from their variables, and the timestep. */
  init(): void {
    this.updateVarParams();
    this.resetDt();
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  /** update_var_params: each v_ parameter is the value of its equal-style variable now. */
  protected updateVarParams(): void {
    const p = this.man.params;
    for (let i = 0; i < this.nvars; ++i) {
      if (!this.isVar[i]) continue;
      const v = this.sys.vars.get(this.tstrs[i]);
      if (!v) throw new StyleError('Variable name for fix nve/manifold/rattle does not exist');
      if (v.style !== 'equal' && v.style !== 'timer' && v.style !== 'internal') {
        throw new StyleError('Variable for fix nve/manifold/rattle is invalid style');
      }
      p[i] = this.sys.equalVariable(this.tstrs[i]);
    }
  }

  /** dof: one per atom of the fix group (none for one atom or fewer), for any compute's group. */
  override dofRemoved(_groupBit: number): number {
    const s = this.sys.state;
    let natoms = 0;
    for (let i = 0; i < s.n; ++i) if (s.mask[i] & this.groupBit) ++natoms;
    const dofs = natoms <= 1 ? 0 : natoms;
    this.stats.dofsRemoved = dofs;
    return dofs;
  }

  initialIntegrate(): void {
    this.updateVarParams();
    this.nveXRattle();
  }

  finalIntegrate(): void {
    this.nveVRattle();
  }

  endOfStep(): void {
    if (this.statsEvery && this.sys.state.step === this.nextOutput) {
      this.printStats('nve/manifold/rattle');
      this.nextOutput += this.statsEvery;
    }
  }

  protected printStats(header: string): void {
    const st = this.stats;
    const n = st.natoms;
    if (n > 0) {
      st.xItersPerAtom += st.xIters / n;
      st.vItersPerAtom += st.vIters / n;
    }
    const xIters = st.xItersPerAtom, vIters = st.vItersPerAtom;
    st.xItersPerAtom = st.xIters = 0;
    st.vItersPerAtom = st.vIters = 0;
    const step = this.sys.state.step;
    const invTdiff = 1.0 / (step - st.lastOut);
    st.lastOut = step;
    this.sys.log(`${header} stats for time step ${step} on ${st.natoms} atoms:`);
    this.sys.log(`  iters/atom: x = ${xIters * invTdiff}, v = ${vIters * invTdiff}, dofs removed = ${st.dofsRemoved}`);
  }

  protected nveXRattle(): void {
    const s = this.sys.state;
    const { x, v, f, mask, id } = s;
    const bit = this.groupBit;
    let natoms = 0;
    const xi = new Float64Array(3), vi = new Float64Array(3), fi = new Float64Array(3);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      natoms++;
      const dtfm = this.dtf / massOf(s, i);
      for (let d = 0; d < 3; d++) { xi[d] = x[3 * i + d]; vi[d] = v[3 * i + d]; fi[d] = f[3 * i + d]; }
      this.rattleManifoldX(xi, vi, fi, this.dtv, dtfm, id[i]);
      for (let d = 0; d < 3; d++) { x[3 * i + d] = xi[d]; v[3 * i + d] = vi[d]; }
    }
    if (this.statsEvery > 0) this.stats.natoms = natoms;
  }

  protected nveVRattle(): void {
    const s = this.sys.state;
    const { x, v, f, mask, id } = s;
    const bit = this.groupBit;
    const xi = new Float64Array(3), vi = new Float64Array(3), fi = new Float64Array(3);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dtfm = this.dtf / massOf(s, i);
      for (let d = 0; d < 3; d++) { xi[d] = x[3 * i + d]; vi[d] = v[3 * i + d]; fi[d] = f[3 * i + d]; }
      this.rattleManifoldV(vi, fi, xi, dtfm, id[i]);
      for (let d = 0; d < 3; d++) v[3 * i + d] = vi[d];
    }
  }

  /**
   * rattle_manifold_x: solve x_old - x_new + dtv v_1/2(l) = 0 and g(x_new) = 0 for x_new and l by Newton iteration,
   * v_1/2(l) = v_0 + dtfm (f - l n_old).
   */
  private rattleManifoldX(x: Float64Array, v: Float64Array, f: Float64Array, dtv: number, dtfm: number, tagi: number): void {
    const xo = [x[0], x[1], x[2]];
    const vo = [v[0], v[1], v[2]];
    let l = 0;
    const R = [0, 0, 0, 0];
    const dx = [0, 0, 0, 0];
    const no = [0, 0, 0];
    const nn = [0, 0, 0];
    let res: number;
    let iters = 0;
    const c = dtfm * dtv;
    let gg = this.man.gAndN(x, no);
    nn[0] = no[0]; nn[1] = no[1]; nn[2] = no[2];
    const vt = [vo[0] + dtfm * f[0], vo[1] + dtfm * f[1], vo[2] + dtfm * f[2]];
    const noDt = [dtfm * no[0], dtfm * no[1], dtfm * no[2]];
    const cInv = 1.0 / c;
    for (;;) {
      v[0] = vt[0] - l * noDt[0];
      v[1] = vt[1] - l * noDt[1];
      v[2] = vt[2] - l * noDt[2];
      R[0] = xo[0] - x[0] + dtv * v[0];
      R[1] = xo[1] - x[1] + dtv * v[1];
      R[2] = xo[2] - x[2] + dtv * v[2];
      R[3] = gg;
      const nnR = nn[0] * R[0] + nn[1] * R[1] + nn[2] * R[2];
      const noNn = no[0] * nn[0] + no[1] * nn[1] + no[2] * nn[2];
      const nInv = 1.0 / noNn;
      dx[3] = -nnR - R[3];
      dx[3] *= nInv;
      dx[0] = -R[0] - no[0] * dx[3];
      dx[1] = -R[1] - no[1] * dx[3];
      dx[2] = -R[2] - no[2] * dx[3];
      dx[3] *= cInv;
      x[0] -= dx[0];
      x[1] -= dx[1];
      x[2] -= dx[2];
      l -= dx[3];
      res = infnorm(R, 4);
      ++iters;
      if (res < this.tolerance || iters >= this.maxIter) break;
      gg = this.man.g(x);
      this.man.n(x, nn);
    }
    if (iters >= this.maxIter && res > this.tolerance) {
      throw new StyleError(`Failed to constrain atom ${tagi} (x = (${x[0]}, ${x[1]}, ${x[2]})! res = ${res}, iters = ${iters}`);
    }
    v[0] = vt[0] - l * noDt[0];
    v[1] = vt[1] - l * noDt[1];
    v[2] = vt[2] - l * noDt[2];
    this.stats.xIters += iters;
  }

  /** rattle_manifold_v: v_old - v_new + dtfm (f - mu n_new) = 0 and v_new . n_new = 0, by Newton iteration. */
  private rattleManifoldV(v: Float64Array, f: Float64Array, x: Float64Array, dtfm: number, tagi: number): void {
    const vo = [v[0], v[1], v[2]];
    let l = 0;
    const R = [0, 0, 0, 0];
    const dv = [0, 0, 0, 0];
    const n = [0, 0, 0];
    let res: number;
    let iters = 0;
    const c = dtfm;
    v[0] += dtfm * f[0];
    v[1] += dtfm * f[1];
    v[2] += dtfm * f[2];
    this.man.n(x, n);
    const vt = [vo[0] + dtfm * f[0], vo[1] + dtfm * f[1], vo[2] + dtfm * f[2]];
    const noDt = [dtfm * n[0], dtfm * n[1], dtfm * n[2]];
    const nn2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    const nInv = 1.0 / nn2;
    const cInv = 1.0 / c;
    do {
      R[0] = vt[0] - v[0] - l * noDt[0];
      R[1] = vt[1] - v[1] - l * noDt[1];
      R[2] = vt[2] - v[2] - l * noDt[2];
      R[3] = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
      const nnR = n[0] * R[0] + n[1] * R[1] + n[2] * R[2];
      dv[3] = -nnR - R[3];
      dv[3] *= nInv;
      dv[0] = -n[0] * dv[3] - R[0];
      dv[1] = -n[1] * dv[3] - R[1];
      dv[2] = -n[2] * dv[3] - R[2];
      dv[3] *= cInv;
      v[0] -= dv[0];
      v[1] -= dv[1];
      v[2] -= dv[2];
      l -= dv[3];
      res = infnorm(R, 4);
      ++iters;
    } while (res > this.tolerance && iters < this.maxIter);
    if (iters >= this.maxIter && res >= this.tolerance) {
      throw new StyleError(`Failed to constrain atom ${tagi} (x = (${x[0]}, ${x[1]}, ${x[2]})! res = ${res}, iters = ${iters}`);
    }
    this.stats.vIters += iters;
  }
}

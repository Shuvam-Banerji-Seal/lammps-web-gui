import type { System } from '../system';
import { StyleError } from '../force/types';
import { massOf } from '../atoms';
import type { BoxDof } from '../fix/box_relax';
import { checkMinHalt } from './halt';
import { computeSetup } from './verlet';
import { MINIMIZERS } from './minimizers';
import { gplHook } from '../gplHooks';

/*
 * Energy minimization — docs.lammps.org/minimize.html, min_style.html,
 * min_modify.html. Written from the cited algorithms, not from LAMMPS code.
 *
 * Objective: "the total potential energy of the system as a function of the
 * N atom coordinates" including fix energies (fix_modify energy yes).
 * Stopping: "the change in energy between outer iterations is less than etol"
 * ("met when the energy change between successive iterations divided by the
 * energy magnitude is less than or equal to the tolerance"); "the 2-norm
 * (length) of the global force vector is less than the ftol"; "the line
 * search fails because the step distance backtracks to 0.0"; "the number of
 * outer iterations or timesteps exceeds maxiter"; "the number of total force
 * evaluations exceeds maxeval". "During a minimization, the outer iteration
 * count is treated as a timestep."
 * cg: "the Polak-Ribiere version of the conjugate gradient (CG) algorithm";
 * sd: "the search direction is set to the downhill direction corresponding
 * to the force vector". Line search: "a backtracking algorithm ... described
 * in Nocedal and Wright's Numerical Optimization (Procedure 3.1 on p 41)";
 * quadratic: "once the system gets close to a local minimum and the
 * linesearch steps get small, so that the energy is approximately quadratic
 * in the step length, it uses the estimated location of zero gradient as the
 * linesearch step, provided the energy change is downhill". "The dmax
 * parameter is how far any atom can move in a single line search in any
 * dimension (x, y, or z)." Defaults: "dmax = 0.1, line = quadratic and norm =
 * two". fire (Bitzek 2006; Guenole 2020): defaults "integrator =
 * eulerimplicit, tmax = 10.0, tmin = 0.02, delaystep = 20, dtgrow = 1.1,
 * dtshrink = 0.5, alpha0 = 0.25, alphashrink = 0.99, vdfmax = 2000,
 * halfstepback = yes and initialdelay = yes". quickmin (Sheppard 2008): "the
 * damping parameter is related to the projection of the velocity vector along
 * the current force vector". "The velocity of each atom is initialized to 0.0
 * by this style, at the beginning of a minimization." "LAMMPS will
 * temporarily apply a neigh_modify every 1 delay 0 check yes setting during
 * the minimization".
 */

export const MIN_STYLES = ['cg', 'sd', 'fire', 'quickmin'] as const;

export interface MinSettings {
  dmax: number;
  line: 'backtrack' | 'quadratic' | 'forcezero';
  norm: 'two' | 'max' | 'inf';
  fire: Record<string, string>;
}

export const defaultMinSettings = (): MinSettings => ({ dmax: 0.1, line: 'quadratic', norm: 'two', fire: {} });

export interface MinResult {
  reason: string;
  e0: number; ePrev: number; e1: number;
  fnorm0: number; fnorm1: number;
  fmax0: number; fmax1: number;
  iterations: number;
  evaluations: number;
}

export interface MinHooks {
  cancelled(): boolean;
  /** Output for the current iteration (thermo/dumps); iterDone = final call. */
  thermo(iterDone: boolean): void;
}

/** native min_modify line backtrack: Armijo slope 0.4 and halving (measured, see the backtrack branch) */
const BACKTRACK_SLOPE = 0.4;
const BACKTRACK = 0.5;
/** native ALPHA_MAX (src/min_linesearch.cpp at b332f28): largest alpha of a line search trial */
const LS_ALPHA_MAX = 1.0;
const EPS_ENERGY = 1e-8;
const EMACH = 1e-8;

/** The trial state of the quadratic line search (native MinLineSearch, src/engine-gpl/min/linemin.ts). */
export interface QuadraticLinesearch {
  /** search direction */
  readonly h: Float64Array;
  /** force at the start point */
  readonly f0: Float64Array;
  /** force at the point of the last evaluation (step fills it) */
  readonly force: Float64Array;
  /** largest alpha the dof bounds allow (dmax, box, per-atom); the search caps it at 1 */
  readonly alphaMax: number;
  /** resets to the start point and steps by alpha along h; returns the energy and fills force */
  step(alpha: number): number;
}
export interface QuadraticResult { status: 'ok' | 'downhill' | 'zeroquad' | 'zeroalpha'; energy: number }
/** the lazily loaded helper of the 'minimize' HOOKS entry (src/engine-gpl/registry/linemin.ts) */
export interface LineminHook { quadratic(eoriginal: number, ls: QuadraticLinesearch): QuadraticResult }

export const minimize = async (
  sys: System, p: { etol: number; ftol: number; maxiter: number; maxeval: number }, hooks: MinHooks,
): Promise<MinResult> => {
  const s = sys.state;
  const set = sys.minSettings;
  const style = sys.minStyle;
  // a minimizer of a lazily loaded registry (min_style spin, ...; run/minimizers.ts) runs its own iteration
  if (Object.prototype.hasOwnProperty.call(MINIMIZERS, style)) return MINIMIZERS[style](sys, p, hooks);
  if (!(MIN_STYLES as readonly string[]).includes(style)) throw new StyleError(`min_style ${style} is not supported`);
  // min_modify line spin_cubic / spin_none belong to the spin minimizers (Measured with native LAMMPS (black box): on
  // a non-spin minimization both stop with a segmentation fault)
  if (set.fire.line === 'spin_cubic' || set.fire.line === 'spin_none') throw new StyleError(`min_modify line ${set.fire.line} is only for the spin minimizers (min_style spin/cg)`);
  // a pair style's per-atom degrees of freedom cannot be carried by the damped dynamics styles (quickmin, fire).
  // Measured with native LAMMPS (black box): pair eff/cut with min_style fire or quickmin stops with
  // Cannot use a damped dynamics min style with per-atom DOF.
  const perAtom = sys.ff.pair?.minPerAtom?.() ?? null;
  if (perAtom && style !== 'cg' && style !== 'sd') throw new StyleError('Cannot use a damped dynamics min style with per-atom DOF');
  // with per-atom DOF native's max norm groups the extra values differently from the atoms' per-atom vector norm;
  // that grouping is not reproduced, so norm max is refused rather than giving a different stopping point
  if (perAtom && set.norm === 'max') throw new StyleError('min_modify norm max with per-atom degrees of freedom (e.g. pair eff/cut) is not supported by the browser engine; use norm two or inf');
  const K = perAtom ? s.n : 0;
  const xe = new Float64Array(K);
  const eF = new Float64Array(K);
  // measured with native LAMMPS (black box): min_modify line forcezero gives different energies from quadratic
  // (chain, dmax 0.1: 13.4795499709744 vs 13.4813750335936 after one iteration); not reproduced yet
  if (set.line === 'forcezero') throw new StyleError('min_modify line forcezero is not supported by the browser engine yet; use line quadratic or backtrack');
  const nb = sys.nb;
  const saved = { every: nb.every, delay: nb.delay, check: nb.check };
  if (nb.every !== 1 || nb.delay !== 0) {
    sys.log('Switching to \'neigh_modify every 1 delay 0 check yes\' setting during minimization');
    nb.every = 1; nb.delay = 0; nb.check = true;
  }
  const n3 = 3 * s.n;
  // fix box/relax (docs fix_box_relax.html): its DOF join the vector after the 3N coordinates; the count is known after minSetup
  let box: BoxDof | null = null;
  let M = 0;
  let boxForce = new Float64Array(0);
  let evaluations = 0;
  // native neval: the evaluations of the line search (the setup evaluation is not one)
  let lsEvals = 0;
  const fixesPost = sys.fixes.filter((f) => f.minPostForce);
  const fixesPre = sys.fixes.filter((f) => f.minPreForce);
  // energy and forces at the current positions
  const evaluate = (): number => {
    evaluations++;
    sys.bump();
    if (evaluations === 1) { sys.setupNeighbors(); sys.sortAtoms(true); }
    else if (nb.decide(s.step, s, sys.geom)) {
      sys.pbc();
      // native's periodic atom sort (System.sortAtoms) only rewrites SimState.order; the engine's
      // arrays, and so the line search's per-atom snapshots, keep their indices
      sys.sortAtoms(false);
      nb.build(s, sys.geom, s.step);
    } else nb.forwardComm(s, sys.geom);
    for (const f of fixesPre) f.minPreForce!();
    sys.computeForcesInRun(sys.computeFlags());
    sys.forcesCurrent();
    // the pair's per-atom coordinates and forces follow the forces of this evaluation
    if (perAtom) perAtom.pull(s, xe, eF);
    // group.html: "For an energy minimization, via the :doc:`minimize <minimize>` command, an
    // assignment is made at the beginning of the minimization, but not during the iterations of
    // the minimizer."
    if (evaluations === 1) sys.assignDynamicGroups(true);
    for (const f of fixesPost) f.minPostForce!();
    sys.refreshComputes();
    const a = sys.ff.acc;
    const eBox = box ? box.evaluate(boxForce) : 0;
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp + sys.fixEnergy() + eBox;
  };
  const norms = () => {
    let two = 0, inf = 0, mx = 0;
    for (let i = 0; i < s.n; i++) {
      const fx = s.f[3 * i], fy = s.f[3 * i + 1], fz = s.f[3 * i + 2];
      const a2 = fx * fx + fy * fy + fz * fz;
      two += a2;
      mx = Math.max(mx, Math.sqrt(a2));
      inf = Math.max(inf, Math.abs(fx), Math.abs(fy), Math.abs(fz));
    }
    // the force norms include the per-atom forces (one value per atom)
    for (let k = 0; k < K; k++) { two += eF[k] * eF[k]; inf = Math.max(inf, Math.abs(eF[k])); mx = Math.max(mx, Math.abs(eF[k])); }
    // the box degrees of freedom are part of the global force vector (native: "Force two-norm initial" of a
    // box-only system equals the box gradient)
    for (let k = 0; k < M; k++) { two += boxForce[k] * boxForce[k]; inf = Math.max(inf, Math.abs(boxForce[k])); mx = Math.max(mx, Math.abs(boxForce[k])); }
    return { two: Math.sqrt(two), inf, max: mx };
  };
  const fnormOf = () => { const n = norms(); return set.norm === 'two' ? n.two : set.norm === 'max' ? n.max : n.inf; };

  computeSetup(sys);
  for (const f of sys.fixes) f.minSetup?.();
  const boxFixes = sys.fixes.filter((f) => (f as unknown as { boxDof?: BoxDof }).boxDof);
  if (boxFixes.length > 1) throw new StyleError('only one fix box/relax may be defined');
  if (boxFixes.length) {
    if (style !== 'cg' && style !== 'sd') throw new StyleError(`min_style ${style} with fix box/relax is not supported by the browser engine; use cg or sd`);
    box = (boxFixes[0] as unknown as { boxDof: BoxDof }).boxDof;
    M = box.dofCount;
    boxForce = new Float64Array(M);
  }
  let e = evaluate();
  const e0 = e;
  let ePrev = e;
  const n0 = norms();
  hooks.thermo(false);
  let reason = 'max iterations';
  let iter = 0;
  const yieldNow = () => new Promise<void>((r) => setTimeout(r, 0));
  let lastYield = performance.now();

  if (style === 'cg' || style === 'sd') {
    // the default line search is the GPL port (lazy 'minimize' HOOKS entry, see gplHooks.ts); backtrack is core
    const lineminQuad = set.line === 'backtrack' ? null : gplHook<LineminHook>('minimize').quadratic;
    // vector: atoms, then box DOF, then the pair's per-atom DOF; the cg restart period counts all of them
    const L = n3 + M + K;
    const eOff = n3 + M;
    const x0 = new Float64Array(n3);
    const xe0 = new Float64Array(K);
    const u0 = new Float64Array(M);
    const uTry = new Float64Array(M);
    const g = new Float64Array(L);   // force = -gradient: atoms, then box DOF, then per-atom DOF
    const h = new Float64Array(L);   // search direction
    const gNew = new Float64Array(L);
    const packForce = (out: Float64Array) => { out.set(s.f.subarray(0, n3)); if (M) out.set(boxForce, n3); if (K) out.set(eF, eOff); };
    packForce(g);
    h.set(g);
    let gg = dot(g, g);
    for (iter = 1; iter <= p.maxiter; iter++) {
      if (hooks.cancelled()) { reason = 'cancelled'; break; }
      // line search along h, starting with the step that moves the farthest atom by dmax
      // (and the box DOF by at most vmax per iteration, as documented for fix box/relax)
      let hmax = 0, hbox = 0, hext = 0;
      for (let k = 0; k < n3; k++) hmax = Math.max(hmax, Math.abs(h[k]));
      for (let k = n3; k < eOff; k++) hbox = Math.max(hbox, Math.abs(h[k]));
      for (let k = eOff; k < L; k++) hext = Math.max(hext, Math.abs(h[k]));
      if (hmax === 0 && hbox === 0 && hext === 0) { reason = 'forces are zero'; break; }
      const fh0 = dot(g, h);
      if (fh0 <= 0) { h.set(g); }
      const slope = dot(g, h);
      let alphaMax = hmax > 0 ? set.dmax / hmax : Infinity;
      if (box && hbox > 0) alphaMax = Math.min(alphaMax, box.vmax / hbox);
      // no per-atom DOF changes by more than the pair style's maxStep in one iteration
      if (perAtom && hext > 0) alphaMax = Math.min(alphaMax, perAtom.maxStep / hext);
      x0.set(s.x.subarray(0, n3));
      xe0.set(xe);
      const img0 = Int32Array.from(s.image.subarray(0, n3));
      if (box) { box.begin(); box.dof(u0); }
      const eStart = e;
      const moveTo = (al: number) => {
        s.image.set(img0);
        if (box) {
          for (let k = 0; k < M; k++) uTry[k] = u0[k] + al * h[n3 + k];
          box.trial(uTry, x0, s.x);
          for (let k = 0; k < n3; k++) s.x[k] += al * h[k];
        } else {
          for (let k = 0; k < n3; k++) s.x[k] = x0[k] + al * h[k];
        }
        if (s.dimension === 2) for (let k = 2; k < n3; k += 3) s.x[k] = x0[k];
        if (perAtom) {
          for (let k = 0; k < K; k++) xe[k] = xe0[k] + al * h[eOff + k];
          perAtom.push(s, xe);
        }
      };
      // one line-search evaluation at x0 + alpha h (native alpha_step): the energy, and the force in gNew
      const stepLS = (al: number): number => {
        lsEvals++;
        moveTo(al);
        const en = evaluate();
        packForce(gNew);
        return en;
      };
      let eTry: number;
      let status: 'ok' | 'downhill' | 'zeroquad' | 'zeroalpha' | 'etol';
      if (set.line === 'backtrack') {
        // Measured with native LAMMPS (black box, tests/oracle/w16min_backtrack.in): line backtrack accepts the
        // first trial alpha = min(1, dmax/max|h|), halved (alpha *= 0.5), whose energy satisfies
        // E(alpha) <= E(0) + 0.4 alpha s0, with s0 = dE/dalpha at 0 = -F.h. Acceptance does not test the slope
        // at the accepted point (native accepted a point with positive slope). Native's linemin_backtrack stops
        // halving when alpha reaches 0 or when the Armijo target is at the round-off level (de_ideal >= -EMACH):
        // "linesearch alpha is zero", or "energy tolerance" when the energy did go down (de < 0).
        let alpha = Math.min(LS_ALPHA_MAX, alphaMax);
        for (;;) {
          eTry = stepLS(alpha);
          const de = eTry - eStart;
          const deIdeal = -BACKTRACK_SLOPE * alpha * slope;
          if (de <= deIdeal) { status = 'ok'; break; }
          alpha *= BACKTRACK;
          if (alpha <= 0 || deIdeal >= -EMACH) { eTry = stepLS(0); status = de < 0 ? 'etol' : 'zeroalpha'; break; }
        }
      } else {
        // native min_modify line quadratic (linemin_quadratic, src/engine-gpl/min/linemin.ts): the hook decides the
        // trials and leaves the state at the point it returns (or at x0 after a failure)
        const r = lineminQuad!(e, { h, f0: g, force: gNew, alphaMax, step: stepLS });
        status = r.status;
        eTry = r.energy;
      }
      if (status !== 'ok') {
        // restored start point (native: ZEROALPHA, ZEROQUAD, DOWNHILL end the minimization). Native advances the
        // timestep before the line search, so the final thermo row of a failed search is at the next step.
        e = eTry;
        s.step++;
        reason = status === 'downhill' ? 'search direction is not downhill' : status === 'zeroquad' ? 'quadratic factors are zero' : status === 'etol' ? 'energy tolerance' : 'linesearch alpha is zero';
        break;
      }
      ePrev = e;
      e = eTry;
      s.step++;
      checkMinHalt(sys);
      hooks.thermo(false);
      // function evaluation criterion (native checks it right after the line search, before the tolerances)
      if (lsEvals >= p.maxeval) { reason = 'max force evaluations'; break; }
      // convergence checks (the force test uses the atoms only: a box that is still moving does not hold the minimizer)
      if (Math.abs(e - ePrev) < p.etol * 0.5 * (Math.abs(e) + Math.abs(ePrev) + EPS_ENERGY)) { reason = 'energy tolerance'; break; }
      const fn = fnormOf();
      if (fn < p.ftol) { reason = 'force tolerance'; break; }
      if (box && box.nreset > 0 && iter % box.nreset === 0) {
        // nreset: the current box becomes the reference; the objective changes, so restart the search
        box.resetReference();
        e = evaluate();
        packForce(gNew);
        g.set(gNew); h.set(gNew); gg = dot(g, g);
      } else if (style === 'sd') {
        // steepest descent: the search direction is the new force
        g.set(gNew); h.set(gNew); gg = dot(g, g);
      } else {
        // Polak-Ribiere: beta = F1.(F1 - F0) / F0.F0, restarted when negative
        let num = 0;
        for (let k = 0; k < L; k++) num += gNew[k] * (gNew[k] - g[k]);
        // Measured with native LAMMPS (black box, tests/oracle/w16min_backtrack.in): cg restarted the direction at
        // iteration 24 of the 8-atom chain (24 degrees of freedom) where beta was 1.853, so the direction restarts
        // every 3N iterations; seen once, with line backtrack
        const restartNow = iter % L === L - 1;
        const beta = gg > 0 && !restartNow ? Math.max(0, num / gg) : 0;
        g.set(gNew);
        gg = dot(g, g);
        for (let k = 0; k < L; k++) h[k] = g[k] + beta * h[k];
        if (dot(g, h) <= 0) h.set(g);
      }
      if (performance.now() - lastYield > 30) { await yieldNow(); lastYield = performance.now(); }
      if (iter === p.maxiter) reason = 'max iterations';
    }
    if (iter > p.maxiter) iter = p.maxiter;
    // Measured with native LAMMPS (black box, tests/engineLineSearch.test.ts): the cg/sd summary counts only the
    // line-search trials, not the setup evaluation (minimize 0 0 1 with line quadratic: 1 force evaluation)
    evaluations--;
  } else {
    // damped dynamics: quickmin or fire
    const fp = set.fire;
    const getNum = (k: string, d: number) => (fp[k] !== undefined ? Number(fp[k]) : d);
    const tmax = getNum('tmax', 10), tmin = getNum('tmin', 0.02), delaystep = getNum('delaystep', 20);
    const dtgrow = getNum('dtgrow', 1.1), dtshrink = getNum('dtshrink', 0.5), alpha0 = getNum('alpha0', 0.25);
    const alphashrink = getNum('alphashrink', 0.99);
    const halfstepback = (fp.halfstepback ?? 'yes') === 'yes';
    const initialdelay = (fp.initialdelay ?? 'yes') === 'yes';
    const dtBase = s.dt;
    let dt = dtBase;
    const dtMax = tmax * dtBase, dtMin = tmin * dtBase;
    let alpha = alpha0;
    let npos = 0;
    let lastNeg = 0;
    const v = s.v;
    v.fill(0, 0, n3);
    const ftm2v = s.units.ftm2v;
    for (iter = 1; iter <= p.maxiter; iter++) {
      if (hooks.cancelled()) { reason = 'cancelled'; break; }
      const f = s.f;
      let P = 0;
      for (let k = 0; k < n3; k++) P += f[k] * v[k];
      // fire mixes with the length of v from before this iteration's kick and with alpha from before this
      // iteration's alphashrink (below)
      let vNorm = 0, fNorm = 0;
      const mixAlpha = alpha;
      if (style === 'quickmin') {
        // "the damping parameter is related to the projection of the velocity vector along the current force
        // vector" (min_style.html). Measured with native LAMMPS (black box): one projection of the whole 3N vector,
        // v = (v.F / F.F) F, or v = 0 when v.F <= 0; the atoms then move with that v and only afterwards
        // v += dt F/m, so the first iteration does not move any atom (pe at step 1 equals step 0). When dmax limits
        // the step (largest velocity component times dt above dmax), the limited dt is used for the move AND the kick.
        let ff = 0;
        for (let k = 0; k < n3; k++) ff += f[k] * f[k];
        const sc = P > 0 && ff > 0 ? P / ff : 0;
        for (let k = 0; k < n3; k++) v[k] = sc * f[k];
        let vmax = 0;
        for (let k = 0; k < n3; k++) vmax = Math.max(vmax, Math.abs(v[k]));
        const dtx = vmax * dt > set.dmax ? set.dmax / vmax : dt;
        for (let k = 0; k < n3; k++) s.x[k] += dtx * v[k];
        for (let i = 0; i < s.n; i++) {
          const c = dtx * ftm2v / massOf(s, i);
          v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
        }
        if (s.dimension === 2) for (let i = 0; i < s.n; i++) v[3 * i + 2] = 0;
      } else if (P > 0) {
        for (let k = 0; k < n3; k++) { vNorm += v[k] * v[k]; fNorm += f[k] * f[k]; }
        npos++;
        if (npos > delaystep) { dt = Math.min(dt * dtgrow, dtMax); alpha *= alphashrink; }
      } else {
        lastNeg = iter;
        npos = 0;
        if (!(initialdelay && iter < delaystep)) { dt = Math.max(dt * dtshrink, dtMin); alpha = alpha0; }
        // Measured with native LAMMPS (black box): after the half step back the forces are evaluated again at the
        // new positions (one more force evaluation per reset, also on iteration 1 where v = 0: 150 iterations with 7
        // resets report 157 force evaluations), and this iteration's kick uses those forces.
        if (halfstepback) {
          for (let k = 0; k < n3; k++) s.x[k] -= 0.5 * dt * v[k];
          v.fill(0, 0, n3);
          e = evaluate();
        } else v.fill(0, 0, n3);
      }
      if (style === 'fire') {
        // semi-implicit Euler: v += dt F/m, mix, x += dt v (limited by dmax). Measured with native LAMMPS (black
        // box): the mixing v = (1 - alpha) v + alpha (|v|/|F|) F happens only on iterations with P > 0, and |v| is
        // the length before this iteration's kick (iteration 2 gives v = 0.75 (v1 + dt F1) + 0.25 |v1|/|F1| F1).
        // On the first iteration with npos > delaystep the grown dt is already used but alpha is still alpha0
        // (dt 0.0055 with 1 - alpha = 0.75 at iteration 22 of a timestep-0.005 run; 1 - alpha = 0.7525 at 23).
        // The dmax limit: on an iteration with P > 0 the largest component of the velocity BEFORE the kick sets it,
        // dtx = dmax / max|v| when max|v| dt > dmax; after a reset (v = 0) the largest component of the kick dt F/m
        // sets it. dtx is then used for both the kick and the move (measured: dt 0.03, dmax 0.02 matches native to
        // 1e-16 in x and v over 7 iterations, and iteration 1 kicks with dmax / (dt max|F|)).
        const ff = s.f;
        let vmax = 0;
        if (P > 0) for (let k = 0; k < n3; k++) vmax = Math.max(vmax, Math.abs(v[k]));
        else for (let i = 0; i < s.n; i++) {
          const c = dt * ftm2v / massOf(s, i);
          vmax = Math.max(vmax, Math.abs(c * ff[3 * i]), Math.abs(c * ff[3 * i + 1]), Math.abs(c * ff[3 * i + 2]));
        }
        const dtx = vmax * dt > set.dmax ? set.dmax / vmax : dt;
        for (let i = 0; i < s.n; i++) {
          const c = dtx * ftm2v / massOf(s, i);
          v[3 * i] += c * ff[3 * i]; v[3 * i + 1] += c * ff[3 * i + 1]; v[3 * i + 2] += c * ff[3 * i + 2];
        }
        if (fNorm > 0) {
          const sc = mixAlpha * Math.sqrt(vNorm / fNorm);
          for (let k = 0; k < n3; k++) v[k] = (1 - mixAlpha) * v[k] + sc * ff[k];
        }
        for (let k = 0; k < n3; k++) s.x[k] += dtx * v[k];
        if (s.dimension === 2) for (let i = 0; i < s.n; i++) v[3 * i + 2] = 0;
      }
      ePrev = e;
      e = evaluate();
      s.step++;
      checkMinHalt(sys);
      hooks.thermo(false);
      // "For the damped dynamics minimizers this check is not performed for a few steps after velocities are reset to 0"
      if (iter - lastNeg > delaystep && Math.abs(e - ePrev) < p.etol * 0.5 * (Math.abs(e) + Math.abs(ePrev) + EPS_ENERGY)) { reason = 'energy tolerance'; break; }
      if (fnormOf() < p.ftol) { reason = 'force tolerance'; break; }
      // Measured with native LAMMPS (black box): quickmin and fire never stop on maxeval (minimize 0 0 1000 12 runs
      // 1000 iterations with 1000 / 1037 force evaluations), and the setup evaluation is not in their count.
      if (performance.now() - lastYield > 30) { await yieldNow(); lastYield = performance.now(); }
    }
    if (iter > p.maxiter) iter = p.maxiter;
    evaluations--;
    // Measured with native LAMMPS (black box): quickmin and fire leave the damped-dynamics velocities in place when
    // the minimization ends (they are only zeroed at its start).
  }
  nb.every = saved.every; nb.delay = saved.delay; nb.check = saved.check;
  hooks.thermo(true);
  const n1 = norms();
  return {
    reason, e0, ePrev, e1: e, fnorm0: n0.two, fnorm1: n1.two, fmax0: n0.inf, fmax1: n1.inf,
    iterations: Math.max(0, iter), evaluations,
  };
};

const dot = (a: Float64Array, b: Float64Array): number => {
  let t = 0;
  for (let k = 0; k < a.length; k++) t += a[k] * b[k];
  return t;
};

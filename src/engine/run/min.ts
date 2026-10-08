import type { System } from '../system';
import { StyleError } from '../force/types';
import { massOf } from '../atoms';
import type { BoxDof } from '../fix/box_relax';

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

const ALPHA_ARMIJO = 1e-4;
const BACKTRACK = 0.5;
const EPS_ENERGY = 1e-8;
const EMACH = 1e-8;
/** Energy resolution in units of the machine epsilon (sums of ~N terms). */
const ROUNDOFF = 64 * Number.EPSILON;

export const minimize = async (
  sys: System, p: { etol: number; ftol: number; maxiter: number; maxeval: number }, hooks: MinHooks,
): Promise<MinResult> => {
  const s = sys.state;
  const set = sys.minSettings;
  const style = sys.minStyle;
  if (!(MIN_STYLES as readonly string[]).includes(style)) throw new StyleError(`min_style ${style} is not supported`);
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
    // the box degrees of freedom are part of the global force vector (native: "Force two-norm initial" of a
    // box-only system equals the box gradient)
    for (let k = 0; k < M; k++) { two += boxForce[k] * boxForce[k]; inf = Math.max(inf, Math.abs(boxForce[k])); mx = Math.max(mx, Math.abs(boxForce[k])); }
    return { two: Math.sqrt(two), inf, max: mx };
  };
  const fnormOf = () => { const n = norms(); return set.norm === 'two' ? n.two : set.norm === 'max' ? n.max : n.inf; };

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
    const L = n3 + M;
    const x0 = new Float64Array(n3);
    const u0 = new Float64Array(M);
    const uTry = new Float64Array(M);
    const g = new Float64Array(L);   // force = -gradient: atoms, then box DOF
    const h = new Float64Array(L);   // search direction
    const gNew = new Float64Array(L);
    const packForce = (out: Float64Array) => { out.set(s.f.subarray(0, n3)); if (M) out.set(boxForce, n3); };
    packForce(g);
    h.set(g);
    let gg = dot(g, g);
    for (iter = 1; iter <= p.maxiter; iter++) {
      if (hooks.cancelled()) { reason = 'cancelled'; break; }
      // line search along h, starting with the step that moves the farthest atom by dmax
      // (and the box DOF by at most vmax per iteration, as documented for fix box/relax)
      let hmax = 0, hbox = 0;
      for (let k = 0; k < n3; k++) hmax = Math.max(hmax, Math.abs(h[k]));
      for (let k = n3; k < L; k++) hbox = Math.max(hbox, Math.abs(h[k]));
      if (hmax === 0 && hbox === 0) { reason = 'forces are zero'; break; }
      const fh0 = dot(g, h);
      if (fh0 <= 0) { h.set(g); }
      const slope = dot(g, h);
      let alphaMax = hmax > 0 ? set.dmax / hmax : Infinity;
      if (box && hbox > 0) alphaMax = Math.min(alphaMax, box.vmax / hbox);
      x0.set(s.x.subarray(0, n3));
      const img0 = Int32Array.from(s.image.subarray(0, n3));
      if (box) { box.begin(); box.dof(u0); }
      const eStart = e;
      let alpha = alphaMax;
      let accepted = false;
      let eTry = e;
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
      };
      for (;;) {
        moveTo(alpha);
        eTry = evaluate();
        if (evaluations >= p.maxeval) break;
        packForce(gNew);
        const slopeNew = dot(gNew, h);
        // quadratic refinement near the minimum: zero of the interpolated directional force
        // E(a) ~ E0 - slope a + c a^2 through the trial: its minimum a0 = slope / (2 c) is the estimated zero of the gradient
        if ((set.line === 'quadratic' || set.line === 'forcezero') && Math.abs(eTry - eStart) < 1e-6 * Math.max(1, Math.abs(eStart))) {
          const c2 = (eTry - eStart + slope * alpha) / (alpha * alpha);
          const a0 = c2 > 0 ? slope / (2 * c2) : -1;
          if (a0 > 0 && a0 <= alphaMax) {
            moveTo(a0);
            const eq = evaluate();
            if (eq <= eStart + ROUNDOFF * (Math.abs(eStart) + 1)) { eTry = eq; alpha = a0; accepted = true; packForce(gNew); break; }
            moveTo(alpha);
            eTry = evaluate();
            packForce(gNew);
          }
        }
        // Armijo sufficient decrease: E(alpha) <= E(0) - c alpha (F . h)
        if (eTry <= eStart - ALPHA_ARMIJO * alpha * slope) { accepted = true; break; }
        // round-off floor: when the predicted decrease is below the energy's resolution, the energy cannot
        // decide the step. Then the forces decide it: a step that does not raise the energy beyond that
        // resolution and that flattens the directional force (strong Wolfe curvature, |F1.h| <= 0.9 |F0.h|)
        // is accepted, so the search does not stall at a tiny force
        if (slope > 0 && eTry <= eStart + ROUNDOFF * (Math.abs(eStart) + 1) && Math.abs(slopeNew) <= 0.9 * slope) { accepted = true; break; }
        alpha *= BACKTRACK;
        if (alpha * Math.max(hmax, hbox) < EMACH) break;
      }
      if (!accepted) {
        // restore the starting point
        moveTo(0);
        e = evaluate();
        reason = evaluations >= p.maxeval ? 'max force evaluations' : 'linesearch alpha is zero';
        break;
      }
      ePrev = e;
      e = eTry;
      s.step++;
      hooks.thermo(false);
      // convergence checks (the force test uses the atoms only: a box that is still moving does not hold the minimizer)
      if (Math.abs(e - ePrev) < p.etol * 0.5 * (Math.abs(e) + Math.abs(ePrev) + EPS_ENERGY)) { reason = 'energy tolerance'; break; }
      const fn = fnormOf();
      if (fn < p.ftol) { reason = 'force tolerance'; break; }
      if (evaluations >= p.maxeval) { reason = 'max force evaluations'; break; }
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
        const beta = gg > 0 ? Math.max(0, num / gg) : 0;
        g.set(gNew);
        gg = dot(g, g);
        for (let k = 0; k < L; k++) h[k] = g[k] + beta * h[k];
        if (dot(g, h) <= 0) h.set(g);
      }
      if (performance.now() - lastYield > 30) { await yieldNow(); lastYield = performance.now(); }
      if (iter === p.maxiter) reason = 'max iterations';
    }
    if (iter > p.maxiter) iter = p.maxiter;
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
      if (style === 'quickmin') {
        // project v onto F per atom ("related to the projection of the velocity vector along the current force")
        for (let i = 0; i < s.n; i++) {
          const fx = f[3 * i], fy = f[3 * i + 1], fz = f[3 * i + 2];
          const vdotf = v[3 * i] * fx + v[3 * i + 1] * fy + v[3 * i + 2] * fz;
          const ff = fx * fx + fy * fy + fz * fz;
          const sc = vdotf > 0 && ff > 0 ? vdotf / ff : 0;
          v[3 * i] = sc * fx; v[3 * i + 1] = sc * fy; v[3 * i + 2] = sc * fz;
        }
      } else if (P > 0) {
        npos++;
        if (npos > delaystep) { dt = Math.min(dt * dtgrow, dtMax); alpha *= alphashrink; }
      } else {
        lastNeg = iter;
        npos = 0;
        if (!(initialdelay && iter < delaystep)) { dt = Math.max(dt * dtshrink, dtMin); alpha = alpha0; }
        if (halfstepback) for (let k = 0; k < n3; k++) s.x[k] -= 0.5 * dt * v[k];
        v.fill(0, 0, n3);
      }
      // semi-implicit Euler: v += dt F/m, mix, x += dt v (limited by dmax)
      for (let i = 0; i < s.n; i++) {
        const c = dt * ftm2v / massOf(s, i);
        v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
      }
      if (style === 'fire') {
        let vv = 0, ff = 0;
        for (let k = 0; k < n3; k++) { vv += v[k] * v[k]; ff += f[k] * f[k]; }
        const sc = ff > 0 ? alpha * Math.sqrt(vv / ff) : 0;
        for (let k = 0; k < n3; k++) v[k] = (1 - alpha) * v[k] + sc * f[k];
      }
      let vmax = 0;
      for (let k = 0; k < n3; k++) vmax = Math.max(vmax, Math.abs(v[k]));
      let dtx = dt;
      if (vmax * dtx > set.dmax) dtx = set.dmax / vmax;
      for (let k = 0; k < n3; k++) s.x[k] += dtx * v[k];
      if (s.dimension === 2) for (let i = 0; i < s.n; i++) v[3 * i + 2] = 0;
      ePrev = e;
      e = evaluate();
      s.step++;
      hooks.thermo(false);
      // "For the damped dynamics minimizers this check is not performed for a few steps after velocities are reset to 0"
      if (iter - lastNeg > delaystep && Math.abs(e - ePrev) < p.etol * 0.5 * (Math.abs(e) + Math.abs(ePrev) + EPS_ENERGY)) { reason = 'energy tolerance'; break; }
      if (fnormOf() < p.ftol) { reason = 'force tolerance'; break; }
      if (evaluations >= p.maxeval) { reason = 'max force evaluations'; break; }
      if (performance.now() - lastYield > 30) { await yieldNow(); lastYield = performance.now(); }
    }
    if (iter > p.maxiter) iter = p.maxiter;
    v.fill(0, 0, n3);
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

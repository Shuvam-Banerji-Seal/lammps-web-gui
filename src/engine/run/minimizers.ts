import type { System } from '../system';
import type { MinHooks, MinResult } from './min';

/*
 * Minimizers that a lazily loaded registry provides (min_style names of src/engine-gpl/spin/ such as spin, filled
 * by src/engine/lazyStyles.ts group gpl_spin). min_style accepts a name registered here, and minimize() hands the
 * run to it. The core minimizers (cg, sd, fire, quickmin) stay in run/min.ts.
 */
export interface MinimizeParams { etol: number; ftol: number; maxiter: number; maxeval: number }
export type MinimizerFn = (sys: System, p: MinimizeParams, hooks: MinHooks) => Promise<MinResult>;
export const MINIMIZERS: Record<string, MinimizerFn> = {};

/**
 * Energy and forces (and the precession vectors of the spin fixes) at the current positions and spins, as one
 * evaluation of a minimizer (native Min::energy_force): neighbor list when it is due, forces, fixes that act after the
 * force evaluation, and the total energy with the fix energies. `first` is the first evaluation of a minimization.
 */
export const minEnergyForce = (sys: System, first: boolean): number => {
  const s = sys.state;
  const nb = sys.nb;
  sys.bump();
  if (first) { sys.setupNeighbors(); sys.sortAtoms(true); }
  else if (nb.decide(s.step, s, sys.geom)) {
    sys.pbc();
    sys.sortAtoms(false);
    nb.build(s, sys.geom, s.step);
  } else nb.forwardComm(s, sys.geom);
  for (const f of sys.fixes) f.minPreForce?.();
  sys.computeForcesInRun(sys.computeFlags());
  sys.forcesCurrent();
  for (const f of sys.fixes) f.minPostForce?.();
  if (first) sys.assignDynamicGroups(true);
  sys.refreshComputes();
  const a = sys.ff.acc;
  return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp + sys.fixEnergy();
};

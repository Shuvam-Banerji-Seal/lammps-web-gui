/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/min_linesearch.cpp, src/min_fire.cpp and src/REPLICA/fix_neb.cpp at commit b332f28 (see
 * src/engine-gpl/min/linemin.ts, src/engine-gpl/min/min_fire.ts and src/engine-gpl/replica/fix_neb.ts).
 *
 * The lazy HOOKS entries 'minimize', 'hyper', 'prd' and 'tad' (lazyStyles.ts group gpl_linemin): the core minimizer
 * (run/min.ts) reads the default line search and the fire minimizer through the slot of gplHook('minimize'). The
 * commands hyper, prd, tad and neb run the same minimizers (their quench, neb's damped dynamics over the replicas), so
 * their words load the group too; neb also reads the nudged forces of fix neb (nebForces) from it.
 */
import { lineminQuadratic } from '../min/linemin';
import { MinFire, type FireParams, type FireReplica } from '../min/min_fire';
import { nebForces } from '../replica/fix_neb';

const helper = {
  quadratic: lineminQuadratic,
  fire: (p: FireParams, reps: FireReplica[], evaluateAll: () => number[], multireplica: boolean) => new MinFire(p, reps, evaluateAll, multireplica),
  nebForces,
};
export const HOOKS: Record<string, typeof helper> = { minimize: helper, hyper: helper, prd: helper, tad: helper, neb: helper };

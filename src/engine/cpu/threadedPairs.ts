import type { Pair, PairCompute } from '../force/types';
/*
 * Pair styles the shared-memory threads (pairThreads.ts) may run. Each is a plain half-list pair
 * loop: every energy and force term belongs to a neighbor-list entry (no per-atom self terms such
 * as the dsf/wolf Coulomb self energy, which every thread would add again), and compute() reads
 * only the coordinate, type, charge and list arrays of PairCompute, never s, nb or geom. A thread
 * runs the style on its own range of owned atoms by seeing a list whose other atoms have no
 * neighbors. The EAM styles are two-pass styles (TwoPassPair below) whose two loops are such
 * loops. The three-body styles read the full list instead: each row adds its two- and three-body
 * terms to the atoms of that row (i, j, k) and nothing else, so the rows run in any order on any
 * thread, each thread into its own force array. The worker builds them from cpu/threadedPairFactories.ts (only the worker
 * imports the style modules, so the engine's first chunk does not hold them).
 */
export const THREADED_NAMES: ReadonlySet<string> = new Set([
  'lj/cut',
  'lj/cut/coul/cut',
  'lj/cut/coul/debye',
  'lj/cut/coul/long',
  'coul/cut',
  'coul/debye',
  'coul/long',
  'lj/charmm/coul/charmm',
  'lj/charmm/coul/charmm/implicit',
  'lj/charmm/coul/long',
  'lj/charmmfsw/coul/charmmfsh',
  'lj/charmmfsw/coul/long',
  'buck',
  'buck/coul/cut',
  'buck/coul/long',
  'born',
  'born/coul/long',
  'morse',
  'lj/cut/coul/dsf',
  'lj/cut/coul/wolf',
  'coul/dsf',
  'coul/wolf',
  'lj/class2',
  'lj/class2/coul/cut',
  'lj/class2/coul/long',
  'lj96/cut',
  'lj/expand',
  'lj/smooth',
  'lj/smooth/linear',
  'lj/gromacs',
  'lj/cubic',
  'mie/cut',
  'soft',
  'yukawa',
  'gauss',
  'table',
  'zbl',
  'eam',
  'eam/alloy',
  'eam/fs',
  'tersoff',
  'tersoff/mod',
  'tersoff/mod/c',
  'tersoff/zbl',
  'sw/mod',
 'snap',
 'meam/spline',
 'meam/sw/spline',
 'vashishta',
  // M sites, hydrogens and owners as shared inputs (Pair.threadExtras)
  'lj/cut/tip4p/cut',
  'lj/cut/tip4p/long',
  'tip4p/cut',
  'tip4p/long',
  // GPL-ported (lazyStyles.ts GPL_THREADED_PAIRS); sw gets the neighbors in native order and the atom IDs
  'sw',
  'eff/cut',
  'born/coul/long/cs',
  'buck/coul/long/cs',
  'born/coul/dsf/cs',
  'born/coul/wolf/cs',
  'coul/wolf/cs',
  'coul/long/cs',
  'lj/cut/coul/long/cs',
  'lj/class2/coul/long/cs',
  'buck/long/coul/long',
]);

/**
 * A many-body style whose energy needs a per-atom quantity summed over the neighbors first (the EAM densities). The
 * threads run its two loops over their ranges of the half list (PairCompute.half ilo .. inum): the first sums into a
 * per-atom array of all atoms, owned and ghost (densities); the engine thread adds the threads' arrays and turns the sums
 * into the per-atom values the second loop reads (embed: the ghosts summed to their owners, the embedding energy, the
 * derivative copied to the ghosts); the second loop (forces) adds the forces and pair energies as a plain pair style does.
 */
export interface TwoPassPair {
  /** The style can run in two passes now. */
  twoPassReady(): boolean;
  densities(pc: PairCompute, out: Float64Array): void;
  embed(pc: PairCompute, sums: Float64Array, values: Float64Array): void;
  forces(pc: PairCompute, values: Float64Array): void;
}

export const isTwoPass = (p: Pair): p is Pair & TwoPassPair =>
  typeof (p as Partial<TwoPassPair>).densities === 'function' && typeof (p as Partial<TwoPassPair>).twoPassReady === 'function';

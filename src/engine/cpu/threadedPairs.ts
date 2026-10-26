import type { Pair, PairCompute } from '../force/types';
import { PairEAMKernel } from '../force/pair/eamKernel';
import { PairLJCut } from '../force/pair/lj_cut';
import { PairCoulLong, PairLJCutCoulLong } from '../force/pair/coul_long';
import {
  PairLJCharmmCoulCharmm, PairLJCharmmCoulCharmmImplicit, PairLJCharmmCoulLong, PairLJCharmmfswCoulCharmmfsh, PairLJCharmmfswCoulLong,
} from '../force/pair/charmm';
import { PairLJCutCoulCut, PairLJCutCoulDebye } from '../force/pair/lj_coul';
import { PairCoulCut, PairCoulDebye } from '../force/pair/coul';
import { PairBorn, PairBuck, PairBuckCoulCut, PairMorse } from '../force/pair/simple';
import { PairBornCoulLong, PairBuckCoulLong } from '../force/pair/coul_long2';

/*
 * Pair styles the shared-memory threads (pairThreads.ts) may run. Each is a plain half-list pair
 * loop: every energy and force term belongs to a neighbor-list entry (no per-atom self terms such
 * as the dsf/wolf Coulomb self energy, which every thread would add again), and compute() reads
 * only the coordinate, type, charge and list arrays of PairCompute, never s, nb or geom. A thread
 * runs the style on its own range of owned atoms by seeing a list whose other atoms have no
 * neighbors. The EAM styles are two-pass styles (TwoPassPair below) whose two loops are such
 * loops. The worker imports only these modules.
 */
export const THREADED_PAIRS: Record<string, () => Pair> = {
  'lj/cut': () => new PairLJCut(),
  'lj/cut/coul/cut': () => new PairLJCutCoulCut(),
  'lj/cut/coul/debye': () => new PairLJCutCoulDebye(),
  'lj/cut/coul/long': () => new PairLJCutCoulLong(),
  'coul/cut': () => new PairCoulCut(),
  'coul/debye': () => new PairCoulDebye(),
  'coul/long': () => new PairCoulLong(),
  'lj/charmm/coul/charmm': () => new PairLJCharmmCoulCharmm(),
  'lj/charmm/coul/charmm/implicit': () => new PairLJCharmmCoulCharmmImplicit(),
  'lj/charmm/coul/long': () => new PairLJCharmmCoulLong(),
  'lj/charmmfsw/coul/charmmfsh': () => new PairLJCharmmfswCoulCharmmfsh(),
  'lj/charmmfsw/coul/long': () => new PairLJCharmmfswCoulLong(),
  buck: () => new PairBuck(),
  'buck/coul/cut': () => new PairBuckCoulCut(),
  'buck/coul/long': () => new PairBuckCoulLong(),
  born: () => new PairBorn(),
  'born/coul/long': () => new PairBornCoulLong(),
  morse: () => new PairMorse(),
  // two-pass styles (TwoPassPair): the threads run each loop over their ranges of the half list, with a copy that holds
  // only the loops (the engine's style reads the potential files and runs the embedding step between the loops)
  eam: () => new PairEAMKernel('eam'),
  'eam/alloy': () => new PairEAMKernel('eam/alloy'),
  'eam/fs': () => new PairEAMKernel('eam/fs'),
};

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

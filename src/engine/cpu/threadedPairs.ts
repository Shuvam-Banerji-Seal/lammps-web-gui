import type { Pair } from '../force/types';
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
 * neighbors. The worker imports only these modules.
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
};

import type { Pair } from '../force/types';
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
import { PairTersoff } from '../force/pair/tersoff';
import { PairTersoffMod, PairTersoffModC, PairTersoffZBL } from '../force/pair/tersoff_variants';
import { PairSWMod } from '../force/pair/sw';

/*
 * The factories of the styles cpu/threadedPairs.ts names (THREADED_NAMES): the pair worker (src/workers/pair.worker.ts)
 * builds its copy of the engine's style with them. Only the worker imports this module, and with it the style modules.
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
  // full-list three-body styles: each thread runs its range of rows
  tersoff: () => new PairTersoff(),
  'tersoff/mod': () => new PairTersoffMod(),
  'tersoff/mod/c': () => new PairTersoffModC(),
  'tersoff/zbl': () => new PairTersoffZBL(),
  'sw/mod': () => new PairSWMod(),
};

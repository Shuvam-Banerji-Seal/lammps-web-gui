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
import { PairSnap } from '../force/pair/snap';
import { PairMeamSpline, PairMeamSWSpline } from '../force/pair/meam_spline';
import { PairVashishta } from '../force/pair/vashishta';
import { PairCoulDsf, PairCoulWolf } from '../force/pair/coul';
import { PairLJCutCoulDsf, PairLJCutCoulWolf } from '../force/pair/lj_coul';
import { PairLJClass2, PairLJCubic, PairLJGromacs, PairMieCut } from '../force/pair/lj_variants2';
import { PairLJClass2Coul } from '../force/pair/class2_spica51';
import { PairLJ96Cut, PairLJExpand, PairLJSmooth, PairLJSmoothLinear } from '../force/pair/lj_variants';
import { PairGauss, PairSoft, PairYukawa } from '../force/pair/simple2';
import { PairTable } from '../force/pair/table';
import { PairZBL } from '../force/pair/zbl';
import { PairLJCutTIP4PCut, PairLJCutTIP4PLong, PairTIP4PCut, PairTIP4PLong } from '../force/pair/tip4p';

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
  'lj/cut/coul/dsf': () => new PairLJCutCoulDsf(),
  'lj/cut/coul/wolf': () => new PairLJCutCoulWolf(),
  'coul/dsf': () => new PairCoulDsf(),
  'coul/wolf': () => new PairCoulWolf(),
  'lj/class2': () => new PairLJClass2(),
  'lj/class2/coul/cut': () => new PairLJClass2Coul('lj/class2/coul/cut'),
  'lj/class2/coul/long': () => new PairLJClass2Coul('lj/class2/coul/long'),
  'lj96/cut': () => new PairLJ96Cut(),
  'lj/expand': () => new PairLJExpand(),
  'lj/smooth': () => new PairLJSmooth(),
  'lj/smooth/linear': () => new PairLJSmoothLinear(),
  'lj/gromacs': () => new PairLJGromacs(),
  'lj/cubic': () => new PairLJCubic(),
  'mie/cut': () => new PairMieCut(),
  soft: () => new PairSoft(),
  yukawa: () => new PairYukawa(),
  gauss: () => new PairGauss(),
  table: () => new PairTable(),
  zbl: () => new PairZBL(),
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
  // full-list many-body styles whose rows add only to their own atoms (bispectrum, spline embedding)
  snap: () => new PairSnap(),
  'meam/spline': () => new PairMeamSpline(),
  'meam/sw/spline': () => new PairMeamSWSpline(),
  vashishta: () => new PairVashishta(),
  // the TIP4P styles read the M sites, hydrogens and owners as shared inputs (Pair.threadExtras)
  'lj/cut/tip4p/cut': () => new PairLJCutTIP4PCut(),
  'lj/cut/tip4p/long': () => new PairLJCutTIP4PLong(),
  'tip4p/cut': () => new PairTIP4PCut(),
  'tip4p/long': () => new PairTIP4PLong(),
};

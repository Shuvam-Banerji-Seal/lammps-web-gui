import type { Pair } from '../force/types';
import type { PairLoader } from './pairThreadsCore';

/*
 * The pair worker's styles, loaded when the engine first sends one (a 'pair' message): the worker's first chunk holds
 * the protocol only, and each style module is its own chunk. The same styles and classes as cpu/threadedPairFactories.ts
 * (which the in-process tests use; tests/enginePairThreads.test.ts checks that the two agree).
 */
export const loadThreadedPair: PairLoader = async (name) => {
  switch (name) {
    case 'lj/cut': { const m = await import('../force/pair/lj_cut'); return () => new m.PairLJCut(); }
    case 'lj/cut/coul/cut': { const m = await import('../force/pair/lj_coul'); return () => new m.PairLJCutCoulCut(); }
    case 'lj/cut/coul/debye': { const m = await import('../force/pair/lj_coul'); return () => new m.PairLJCutCoulDebye(); }
    case 'lj/cut/coul/long': { const m = await import('../force/pair/coul_long'); return () => new m.PairLJCutCoulLong(); }
    case 'coul/cut': { const m = await import('../force/pair/coul'); return () => new m.PairCoulCut(); }
    case 'coul/debye': { const m = await import('../force/pair/coul'); return () => new m.PairCoulDebye(); }
    case 'coul/long': { const m = await import('../force/pair/coul_long'); return () => new m.PairCoulLong(); }
    case 'lj/charmm/coul/charmm': { const m = await import('../force/pair/charmm'); return () => new m.PairLJCharmmCoulCharmm(); }
    case 'lj/charmm/coul/charmm/implicit': { const m = await import('../force/pair/charmm'); return () => new m.PairLJCharmmCoulCharmmImplicit(); }
    case 'lj/charmm/coul/long': { const m = await import('../force/pair/charmm'); return () => new m.PairLJCharmmCoulLong(); }
    case 'lj/charmmfsw/coul/charmmfsh': { const m = await import('../force/pair/charmm'); return () => new m.PairLJCharmmfswCoulCharmmfsh(); }
    case 'lj/charmmfsw/coul/long': { const m = await import('../force/pair/charmm'); return () => new m.PairLJCharmmfswCoulLong(); }
    case 'buck': { const m = await import('../force/pair/simple'); return () => new m.PairBuck(); }
    case 'buck/coul/cut': { const m = await import('../force/pair/simple'); return () => new m.PairBuckCoulCut(); }
    case 'buck/coul/long': { const m = await import('../force/pair/coul_long2'); return () => new m.PairBuckCoulLong(); }
    case 'born': { const m = await import('../force/pair/simple'); return () => new m.PairBorn(); }
    case 'born/coul/long': { const m = await import('../force/pair/coul_long2'); return () => new m.PairBornCoulLong(); }
    case 'morse': { const m = await import('../force/pair/simple'); return () => new m.PairMorse(); }
    case 'eam': case 'eam/alloy': case 'eam/fs': { const m = await import('../force/pair/eamKernel'); return () => new m.PairEAMKernel(name); }
    case 'tersoff': { const m = await import('../force/pair/tersoff'); return () => new m.PairTersoff(); }
    case 'tersoff/mod': { const m = await import('../force/pair/tersoff_variants'); return () => new m.PairTersoffMod(); }
    case 'tersoff/mod/c': { const m = await import('../force/pair/tersoff_variants'); return () => new m.PairTersoffModC(); }
    case 'tersoff/zbl': { const m = await import('../force/pair/tersoff_variants'); return () => new m.PairTersoffZBL(); }
    case 'sw/mod': { const m = await import('../force/pair/sw'); return () => new m.PairSWMod(); }
    default: return null;
  }
};

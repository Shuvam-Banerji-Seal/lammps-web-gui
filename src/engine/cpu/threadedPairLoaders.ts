import type { Pair } from '../force/types';
import type { PairLoader } from './pairThreadsCore';

/*
 * The pair worker's styles, loaded when the engine first sends one (a 'pair' message): the worker's first chunk holds
 * the protocol only, and each style module is its own chunk. The same styles and classes as cpu/threadedPairFactories.ts
 * (which the in-process tests use; tests/enginePairThreads.test.ts checks that the two agree).
 */
export const loadThreadedPair: PairLoader = async (name, url) => {
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
    case 'lj/cut/coul/dsf': { const m = await import('../force/pair/lj_coul'); return () => new m.PairLJCutCoulDsf(); }
    case 'lj/cut/coul/wolf': { const m = await import('../force/pair/lj_coul'); return () => new m.PairLJCutCoulWolf(); }
    case 'coul/dsf': { const m = await import('../force/pair/coul'); return () => new m.PairCoulDsf(); }
    case 'coul/wolf': { const m = await import('../force/pair/coul'); return () => new m.PairCoulWolf(); }
    case 'lj/class2': { const m = await import('../force/pair/lj_variants2'); return () => new m.PairLJClass2(); }
    case 'lj/class2/coul/cut': case 'lj/class2/coul/long': { const m = await import('../force/pair/class2_spica51'); return () => new m.PairLJClass2Coul(name); }
    case 'lj96/cut': { const m = await import('../force/pair/lj_variants'); return () => new m.PairLJ96Cut(); }
    case 'lj/expand': { const m = await import('../force/pair/lj_variants'); return () => new m.PairLJExpand(); }
    case 'lj/smooth': { const m = await import('../force/pair/lj_variants'); return () => new m.PairLJSmooth(); }
    case 'lj/smooth/linear': { const m = await import('../force/pair/lj_variants'); return () => new m.PairLJSmoothLinear(); }
    case 'lj/gromacs': { const m = await import('../force/pair/lj_variants2'); return () => new m.PairLJGromacs(); }
    case 'lj/cubic': { const m = await import('../force/pair/lj_variants2'); return () => new m.PairLJCubic(); }
    case 'mie/cut': { const m = await import('../force/pair/lj_variants2'); return () => new m.PairMieCut(); }
    case 'soft': { const m = await import('../force/pair/simple2'); return () => new m.PairSoft(); }
    case 'yukawa': { const m = await import('../force/pair/simple2'); return () => new m.PairYukawa(); }
    case 'gauss': { const m = await import('../force/pair/simple2'); return () => new m.PairGauss(); }
    case 'table': { const m = await import('../force/pair/table'); return () => new m.PairTable(); }
    case 'zbl': { const m = await import('../force/pair/zbl'); return () => new m.PairZBL(); }
    case 'eam': case 'eam/alloy': case 'eam/fs': { const m = await import('../force/pair/eamKernel'); return () => new m.PairEAMKernel(name); }
    case 'tersoff': { const m = await import('../force/pair/tersoff'); return () => new m.PairTersoff(); }
    case 'tersoff/mod': { const m = await import('../force/pair/tersoff_variants'); return () => new m.PairTersoffMod(); }
    case 'tersoff/mod/c': { const m = await import('../force/pair/tersoff_variants'); return () => new m.PairTersoffModC(); }
    case 'tersoff/zbl': { const m = await import('../force/pair/tersoff_variants'); return () => new m.PairTersoffZBL(); }
    case 'sw/mod': { const m = await import('../force/pair/sw'); return () => new m.PairSWMod(); }
    // the GPL-ported styles (src/engine-gpl), through the lazy registries' loader (rule 7)
    // a GPL-ported style (src/engine-gpl): the engine thread loaded its registry module through the lazy registries
    // (lazyStyles.ts loadGplThreadedPair, rule 7) and sends that module's own URL, which the worker imports as it is
    default: {
      if (!url) return null;
      const m = (await import(/* @vite-ignore */ url)) as { THREADED?: Record<string, () => import('../force/types').Pair> };
      return m.THREADED?.[name] ?? null;
    }
  }
};

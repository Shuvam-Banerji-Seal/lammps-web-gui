import type { System } from './system';
import type { Bonded, KSpace, Pair } from './force/types';
import type { Fix } from './fix/fix';
import type { Compute } from './compute/compute';
import { LAZY_GROUPS, type LazyGroup, type StyleKind, type StyleModule } from './lazyStyles';
import { isRefusedStyle } from './refusedStyles';
import { PairLJCut } from './force/pair/lj_cut';
import { PairHybrid } from './force/pair/hybrid';
import { BondHarmonic } from './force/bond/harmonic';
import { BondedHybrid } from './force/bonded_hybrid';
import { AngleHarmonic } from './force/angle/harmonic';
import { DihedralHarmonic } from './force/dihedral/harmonic';
import { ImproperHarmonic } from './force/improper/harmonic';
import * as pairCoul from './registry/pair_coul';
import { PairCoulLong, PairLJCutCoulLong } from './force/pair/coul_long';
import { KSpaceEwald } from './force/kspace/ewald';
import { KSpacePPPM, KSpacePPPMCG } from './force/kspace/pppm';
import * as pairSimple from './registry/pair_simple';
import * as pairLJ from './registry/pair_lj';
import * as pairLJCoul from './registry/pair_ljcoul';
import * as pairSimple2 from './registry/pair_simple2';
import * as pairLJ2 from './registry/pair_lj2';
import * as bondedA from './registry/bonded_a';
import * as bondedB from './registry/bonded_b';
import * as bondedC from './registry/bonded_c';
import * as pairEAM from './registry/pair_eam';
import * as fixForce from './registry/fix_force';
import * as fixWall from './registry/fix_wall';
import * as fixOutput from './registry/fix_output';
import * as fixMotion from './registry/fix_motion';
import * as fixExt from './registry/fix_ext';
import * as fixMom from './registry/fix_mom';
import * as fixWref from './registry/fix_wref';
import * as fixAvg from './registry/fix_avg';
import * as computeRed from './registry/compute_red';
import * as computeTemp from './registry/compute_temp';
import * as fixDeform from './registry/fix_deform';
import * as computeDeform from './registry/compute_deform';
import * as pairZBL from './registry/pair_zbl';
import * as fixRigid2 from './registry/fix_rigid2';
import * as fixNHSphere from './registry/fix_nhsphere';
import * as computeChunk from './registry/compute_chunk';
import * as fixAveChunk from './registry/fix_avechunk';
import * as fixMisc3 from './registry/fix_misc3';
import * as fixSllod from './registry/fix_sllod';
import * as fixGjf from './registry/fix_gjf';
import * as fixAdapt from './registry/fix_adapt';
import * as computeLocal from './registry/compute_local';
import * as dihedralCharmm from './registry/dihedral_charmm';
import * as fixWall16 from './registry/fix_wall16';
import * as w17fixmove17 from './registry/fix_move17';
import * as w17pairsoft17 from './registry/pair_soft17';
import * as w17pairmisc17 from './registry/pair_misc17';
import * as w17bond17 from './registry/bond17';
import * as w17angle17 from './registry/angle17';
import * as w17dihimp17 from './registry/dihimp17';
import * as w17computechunk17 from './registry/compute_chunk17';
import * as w17computemisc17 from './registry/compute_misc17';
import * as w17fixmisc17 from './registry/fix_misc17';
import * as w18pairmisc18 from './registry/pair_misc18';
import * as w18computemisc18 from './registry/compute_misc18';
import * as w18angle18 from './registry/angle18';
import * as pairTable from './registry/pair_table';
import * as pairCharmm from './registry/pair_charmm';
import * as pairCoulLong2 from './registry/pair_coullong2';
import * as computeAtom from './registry/compute_atom';
import * as computeGlobal from './registry/compute_global';
import { FixNVE } from './fix/nve';
import { FixNVESphere } from './fix/nve_sphere';
import { FixPropertyAtom } from './fix/property_atom';
import { FixBalance } from './fix/balance';
import { ComputeERotateSphere, ComputeTempSphere } from './compute/sphere';
import { FixShake } from './fix/shake';
import { FixRigid } from './fix/rigid';
import { FixEnforce2d } from './fix/enforce2d';
import { FixNeb } from './fix/neb';
import { FixNH } from './fix/nh';
import { FixLangevin, FixTempBerendsen, FixTempCSVR, FixTempRescale } from './fix/thermostats';
import { ComputeTemp } from './compute/temp';
import { ComputeKE } from './compute/ke';
import { ComputePE } from './compute/pe';
import { ComputePressure } from './compute/pressure';

/*
 * Style registries: LAMMPS style name -> implementation. Each style lives in
 * its own file (force/pair/*, force/bond/*, ..., fix/*, compute/*) and is
 * listed here once. Anything not listed is reported as unsupported by name.
 */

export type FixFactory = (sys: System, id: string, group: string, args: string[]) => Fix;
export type ComputeFactory = (sys: System, id: string, group: string, args: string[]) => Compute;

export const PAIR_STYLES: Record<string, () => Pair> = {
  hybrid: () => new PairHybrid('hybrid', PAIR_STYLES),
  'hybrid/overlay': () => new PairHybrid('hybrid/overlay', PAIR_STYLES),
  'hybrid/scaled': () => new PairHybrid('hybrid/scaled', PAIR_STYLES),
  'hybrid/molecular': () => new PairHybrid('hybrid/molecular', PAIR_STYLES),
  'lj/cut': () => new PairLJCut(),
  'coul/long': () => new PairCoulLong(),
  'lj/cut/coul/long': () => new PairLJCutCoulLong(),
  ...pairCoul.PAIRS, ...pairSimple.PAIRS, ...pairLJ.PAIRS, ...pairLJCoul.PAIRS, ...pairSimple2.PAIRS, ...pairLJ2.PAIRS,
  ...pairEAM.PAIRS,   ...pairZBL.PAIRS, ...pairTable.PAIRS, ...pairCharmm.PAIRS, ...pairCoulLong2.PAIRS,
  ...w17pairsoft17.PAIRS, ...w17pairmisc17.PAIRS, ...w18pairmisc18.PAIRS, };

export const BOND_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new BondHarmonic(),
  hybrid: () => new BondedHybrid('bond', BOND_STYLES),
  ...bondedA.BONDS, ...w17bond17.BONDS,
};
export const ANGLE_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new AngleHarmonic(),
  hybrid: () => new BondedHybrid('angle', ANGLE_STYLES),
  ...bondedA.ANGLES, ...w17angle17.ANGLES, ...w18angle18.ANGLES,
  };
export const DIHEDRAL_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new DihedralHarmonic(),
  hybrid: () => new BondedHybrid('dihedral', DIHEDRAL_STYLES),
  ...bondedB.DIHEDRALS,
  ...dihedralCharmm.DIHEDRALS, ...w17dihimp17.DIHEDRALS,
  };
export const IMPROPER_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new ImproperHarmonic(),
  hybrid: () => new BondedHybrid('improper', IMPROPER_STYLES),
  ...bondedB.IMPROPERS, ...bondedC.IMPROPERS, ...w17dihimp17.IMPROPERS,
  };
export const KSPACE_STYLES: Record<string, () => KSpace> = {
  ewald: () => new KSpaceEwald(),
  pppm: () => new KSpacePPPM(),
  'pppm/cg': () => new KSpacePPPMCG(),
  };

export const FIX_STYLES: Record<string, FixFactory> = {
  nve: (s, i, g, a) => new FixNVE(s, i, g, a),
  'nve/sphere': (s, i, g, a) => new FixNVESphere(s, i, g, a),
  'property/atom': (s, i, g, a) => new FixPropertyAtom(s, i, g, a),
  balance: (s, i, g, a) => new FixBalance(s, i, g, a),
  enforce2d: (s, i, g, a) => new FixEnforce2d(s, i, g, a),
  neb: (s, i, g, a) => new FixNeb(s, i, g, a),
  nvt: (s, i, g, a) => new FixNH(s, i, g, a, 'nvt'),
  npt: (s, i, g, a) => new FixNH(s, i, g, a, 'npt'),
  nph: (s, i, g, a) => new FixNH(s, i, g, a, 'nph'),
  langevin: (s, i, g, a) => new FixLangevin(s, i, g, a),
  'temp/berendsen': (s, i, g, a) => new FixTempBerendsen(s, i, g, a),
  'temp/rescale': (s, i, g, a) => new FixTempRescale(s, i, g, a),
  'temp/csvr': (s, i, g, a) => new FixTempCSVR(s, i, g, a, 'temp/csvr'),
  'temp/csld': (s, i, g, a) => new FixTempCSVR(s, i, g, a, 'temp/csld'),
  shake: (s, i, g, a) => new FixShake(s, i, g, a, 'shake'),
  rattle: (s, i, g, a) => new FixShake(s, i, g, a, 'rattle'),
  ...Object.fromEntries(['rigid', 'rigid/nve', 'rigid/small', 'rigid/nve/small'].map((st) => [st, (s: System, i: string, g: string, a: string[]) => new FixRigid(s, i, g, a, st)])),
  ...fixForce.FIXES, ...fixWall.FIXES, ...fixOutput.FIXES, ...fixMotion.FIXES,
  ...fixExt.FIXES, ...fixMom.FIXES, ...fixWref.FIXES, ...fixAvg.FIXES, ...fixDeform.FIXES, ...fixRigid2.FIXES, ...fixNHSphere.FIXES, ...fixAveChunk.FIXES, ...fixMisc3.FIXES, ...fixSllod.FIXES, ...fixGjf.FIXES, ...fixAdapt.FIXES, ...fixWall16.FIXES, ...w17fixmove17.FIXES, ...w17fixmisc17.FIXES, };

export const COMPUTE_STYLES: Record<string, ComputeFactory> = {
  temp: (s, i, g, a) => new ComputeTemp(s, i, g, a),
  ke: (s, i, g, a) => new ComputeKE(s, i, g, a),
  pe: (s, i, g, a) => new ComputePE(s, i, g, a),
  pressure: (s, i, g, a) => new ComputePressure(s, i, g, a),
  'erotate/sphere': (s, i, g, a) => new ComputeERotateSphere(s, i, g, a),
  'temp/sphere': (s, i, g, a) => new ComputeTempSphere(s, i, g, a),
  ...computeAtom.COMPUTES, ...computeGlobal.COMPUTES, ...computeRed.COMPUTES, ...computeTemp.COMPUTES, ...computeDeform.COMPUTES, ...computeChunk.COMPUTES, ...computeLocal.COMPUTES, ...w17computechunk17.COMPUTES, ...w17computemisc17.COMPUTES, ...w18computemisc18.COMPUTES, };

/** The registry object of each style kind (lazy groups are merged into these when loaded). */
const REGISTRY_OF: Record<StyleKind, Record<string, unknown>> = {
  PAIRS: PAIR_STYLES, BONDS: BOND_STYLES, ANGLES: ANGLE_STYLES, DIHEDRALS: DIHEDRAL_STYLES,
  IMPROPERS: IMPROPER_STYLES, KSPACES: KSPACE_STYLES, FIXES: FIX_STYLES, COMPUTES: COMPUTE_STYLES,
};

/** Registered names of one kind, including the lazy groups that are not loaded yet. */
export const registeredNamesOf = (kind: StyleKind): string[] => {
  const set = new Set(Object.keys(REGISTRY_OF[kind]));
  for (const g of LAZY_GROUPS) for (const n of g.names[kind] ?? []) set.add(n);
  return [...set].sort();
};

/** Supported names of one kind: the registered ones without the refusal-only styles (refusedStyles.ts). */
const namesOf = (kind: StyleKind): string[] => registeredNamesOf(kind).filter((n) => !isRefusedStyle(kind, n));

/** Lists for messages and is_available(). */
export const styleNames = (): Record<string, string[]> => ({
  pair_style: namesOf('PAIRS'),
  bond_style: namesOf('BONDS'),
  angle_style: namesOf('ANGLES'),
  dihedral_style: namesOf('DIHEDRALS'),
  improper_style: namesOf('IMPROPERS'),
  kspace_style: namesOf('KSPACES'),
  fix: namesOf('FIXES'),
  compute: namesOf('COMPUTES'),
});

/** Style name -> the lazy groups that register it (a name may be in several kinds, e.g. snap). */
const LAZY_BY_NAME = new Map<string, LazyGroup[]>();
for (const g of LAZY_GROUPS) {
  for (const list of Object.values(g.names)) {
    for (const n of list ?? []) LAZY_BY_NAME.set(n, [...(LAZY_BY_NAME.get(n) ?? []), g]);
  }
}
const loaded = new Map<string, Promise<void>>();

/** Loads one lazy group and merges its factories into the registries (once). */
export const loadStyleGroup = (g: LazyGroup): Promise<void> => {
  let p = loaded.get(g.id);
  if (!p) {
    p = g.load().then((mod: StyleModule) => {
      for (const kind of Object.keys(REGISTRY_OF) as StyleKind[]) {
        const add = mod[kind];
        if (add) Object.assign(REGISTRY_OF[kind], add);
      }
    });
    loaded.set(g.id, p);
  }
  return p;
};

/**
 * Loads every lazy group that registers a style named by one of these words (the words of a command, or of a
 * restart file). Over-loading on a word that only looks like a style name is harmless.
 */
export const ensureStyles = async (words: Iterable<string>): Promise<void> => {
  const todo: Promise<void>[] = [];
  for (const w of words) {
    const groups = LAZY_BY_NAME.get(w);
    if (groups) for (const g of groups) if (!loaded.has(g.id)) todo.push(loadStyleGroup(g));
  }
  if (todo.length) await Promise.all(todo);
};

/** Loads every lazy group (tests and tools that need the full registries synchronously). */
export const loadAllStyles = (): Promise<void> => Promise.all(LAZY_GROUPS.map(loadStyleGroup)).then(() => undefined);


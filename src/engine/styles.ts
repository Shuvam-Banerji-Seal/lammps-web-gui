import type { System } from './system';
import type { Bonded, KSpace, Pair } from './force/types';
import type { Fix } from './fix/fix';
import type { Compute } from './compute/compute';
import { PairLJCut } from './force/pair/lj_cut';
import { PairHybrid } from './force/pair/hybrid';
import { BondHarmonic } from './force/bond/harmonic';
import { AngleHarmonic } from './force/angle/harmonic';
import { DihedralHarmonic } from './force/dihedral/harmonic';
import { ImproperHarmonic } from './force/improper/harmonic';
import * as pairCoul from './registry/pair_coul';
import { PairCoulLong, PairLJCutCoulLong } from './force/pair/coul_long';
import { KSpaceEwald } from './force/kspace/ewald';
import { KSpacePPPM } from './force/kspace/pppm';
import * as pairSimple from './registry/pair_simple';
import * as pairLJ from './registry/pair_lj';
import * as pairLJCoul from './registry/pair_ljcoul';
import * as pairSimple2 from './registry/pair_simple2';
import * as pairLJ2 from './registry/pair_lj2';
import * as bondedA from './registry/bonded_a';
import * as bondedB from './registry/bonded_b';
import * as bondedC from './registry/bonded_c';
import * as pairEAM from './registry/pair_eam';
import * as pair3Body from './registry/pair_3body';
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
import * as pairSW from './registry/pair_sw';
import * as fixDeform from './registry/fix_deform';
import * as computeDeform from './registry/compute_deform';
import * as pairZBL from './registry/pair_zbl';
import * as pairGran from './registry/pair_gran';
import * as pairColloid from './registry/pair_colloid';
import * as pairYColloid from './registry/pair_ycolloid';
import * as pairVashishta from './registry/pair_vashishta';
import * as computeOrient from './registry/compute_orient';
import * as fixWallGran from './registry/fix_wallgran';
import * as fixRigid2 from './registry/fix_rigid2';
import * as computeVoro from './registry/compute_voro';
import * as pairRelres from './registry/pair_relres';
import * as pairTable from './registry/pair_table';
import * as pairCharmm from './registry/pair_charmm';
import * as pairCoulLong2 from './registry/pair_coullong2';
import * as computeAtom from './registry/compute_atom';
import * as computeGlobal from './registry/compute_global';
import { FixNVE } from './fix/nve';
import { FixNVESphere } from './fix/nve_sphere';
import { ComputeERotateSphere, ComputeTempSphere } from './compute/sphere';
import { FixShake } from './fix/shake';
import { FixRigid } from './fix/rigid';
import { FixEnforce2d } from './fix/enforce2d';
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
  ...pairEAM.PAIRS, ...pair3Body.PAIRS, ...pairSW.PAIRS,
  ...pairZBL.PAIRS, ...pairTable.PAIRS, ...pairCharmm.PAIRS, ...pairCoulLong2.PAIRS,
  ...pairGran.PAIRS, ...pairColloid.PAIRS, ...pairYColloid.PAIRS, ...pairVashishta.PAIRS, ...pairRelres.PAIRS,
};

export const BOND_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new BondHarmonic(),
  ...bondedA.BONDS,
};
export const ANGLE_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new AngleHarmonic(),
  ...bondedA.ANGLES,
};
export const DIHEDRAL_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new DihedralHarmonic(),
  ...bondedB.DIHEDRALS,
};
export const IMPROPER_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new ImproperHarmonic(),
  ...bondedB.IMPROPERS, ...bondedC.IMPROPERS,
};
export const KSPACE_STYLES: Record<string, () => KSpace> = {
  ewald: () => new KSpaceEwald(),
  pppm: () => new KSpacePPPM(),
};

export const FIX_STYLES: Record<string, FixFactory> = {
  nve: (s, i, g, a) => new FixNVE(s, i, g, a),
  'nve/sphere': (s, i, g, a) => new FixNVESphere(s, i, g, a),
  enforce2d: (s, i, g, a) => new FixEnforce2d(s, i, g, a),
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
  ...fixExt.FIXES, ...fixMom.FIXES, ...fixWref.FIXES, ...fixAvg.FIXES, ...fixDeform.FIXES, ...fixWallGran.FIXES, ...fixRigid2.FIXES,
};

export const COMPUTE_STYLES: Record<string, ComputeFactory> = {
  temp: (s, i, g, a) => new ComputeTemp(s, i, g, a),
  ke: (s, i, g, a) => new ComputeKE(s, i, g, a),
  pe: (s, i, g, a) => new ComputePE(s, i, g, a),
  pressure: (s, i, g, a) => new ComputePressure(s, i, g, a),
  'erotate/sphere': (s, i, g, a) => new ComputeERotateSphere(s, i, g, a),
  'temp/sphere': (s, i, g, a) => new ComputeTempSphere(s, i, g, a),
  ...computeAtom.COMPUTES, ...computeGlobal.COMPUTES, ...computeRed.COMPUTES, ...computeTemp.COMPUTES, ...computeDeform.COMPUTES, ...computeOrient.COMPUTES, ...computeVoro.COMPUTES,
};

/** Lists for messages and is_available(). */
export const styleNames = (): Record<string, string[]> => ({
  pair_style: Object.keys(PAIR_STYLES).sort(),
  bond_style: Object.keys(BOND_STYLES).sort(),
  angle_style: Object.keys(ANGLE_STYLES).sort(),
  dihedral_style: Object.keys(DIHEDRAL_STYLES).sort(),
  improper_style: Object.keys(IMPROPER_STYLES).sort(),
  kspace_style: Object.keys(KSPACE_STYLES).sort(),
  fix: Object.keys(FIX_STYLES).sort(),
  compute: Object.keys(COMPUTE_STYLES).sort(),
});

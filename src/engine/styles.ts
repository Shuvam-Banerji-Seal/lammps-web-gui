import type { System } from './system';
import type { Bonded, KSpace, Pair } from './force/types';
import type { Fix } from './fix/fix';
import type { Compute } from './compute/compute';
import { PairLJCut } from './force/pair/lj_cut';
import { BondHarmonic } from './force/bond/harmonic';
import { AngleHarmonic } from './force/angle/harmonic';
import { DihedralHarmonic } from './force/dihedral/harmonic';
import { ImproperHarmonic } from './force/improper/harmonic';
import { FixNVE } from './fix/nve';
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
  'lj/cut': () => new PairLJCut(),
};

export const BOND_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new BondHarmonic(),
};
export const ANGLE_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new AngleHarmonic(),
};
export const DIHEDRAL_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new DihedralHarmonic(),
};
export const IMPROPER_STYLES: Record<string, () => Bonded> = {
  harmonic: () => new ImproperHarmonic(),
};
export const KSPACE_STYLES: Record<string, () => KSpace> = {};

export const FIX_STYLES: Record<string, FixFactory> = {
  nve: (s, i, g, a) => new FixNVE(s, i, g, a),
  enforce2d: (s, i, g, a) => new FixEnforce2d(s, i, g, a),
  nvt: (s, i, g, a) => new FixNH(s, i, g, a, 'nvt'),
  npt: (s, i, g, a) => new FixNH(s, i, g, a, 'npt'),
  nph: (s, i, g, a) => new FixNH(s, i, g, a, 'nph'),
  langevin: (s, i, g, a) => new FixLangevin(s, i, g, a),
  'temp/berendsen': (s, i, g, a) => new FixTempBerendsen(s, i, g, a),
  'temp/rescale': (s, i, g, a) => new FixTempRescale(s, i, g, a),
  'temp/csvr': (s, i, g, a) => new FixTempCSVR(s, i, g, a, 'temp/csvr'),
  'temp/csld': (s, i, g, a) => new FixTempCSVR(s, i, g, a, 'temp/csld'),
};

export const COMPUTE_STYLES: Record<string, ComputeFactory> = {
  temp: (s, i, g, a) => new ComputeTemp(s, i, g, a),
  ke: (s, i, g, a) => new ComputeKE(s, i, g, a),
  pe: (s, i, g, a) => new ComputePE(s, i, g, a),
  pressure: (s, i, g, a) => new ComputePressure(s, i, g, a),
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

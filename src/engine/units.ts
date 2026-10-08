import type { UnitStyle, UnitSystem } from './types';

/*
 * Unit systems — docs.lammps.org/units.html.
 *
 *   "For all units except lj, LAMMPS uses physical constants from
 *    www.physics.nist.gov. For the definition of kcal in real units, LAMMPS
 *    uses the thermochemical calorie = 4.184 J."
 *   Defaults: "For style lj these are dt = 0.005 tau and skin = 0.3 sigma.
 *    For style real these are dt = 1.0 femtoseconds and skin = 2.0 Angstroms.
 *    For style metal these are dt = 0.001 picoseconds and skin = 2.0
 *    Angstroms. For style si these are dt = 1.0e-8 seconds and skin = 0.001
 *    meters. For style cgs these are dt = 1.0e-8 seconds and skin = 0.1
 *    centimeters. For style electron these are dt = 0.001 femtoseconds and
 *    skin = 2.0 Bohr. For style micro these are dt = 2.0 microseconds and
 *    skin = 0.1 micrometers. For style nano these are dt = 0.00045
 *    nanoseconds and skin = 0.1 nanometers."
 *   docs.lammps.org/thermo_modify.html: "norm = yes for unit style of lj,
 *    norm = no for unit style of real and metal" (and no for the others:
 *    native LAMMPS prints the total ke of 2 atoms in si, cgs, electron, micro
 *    and nano units).
 *
 * The conversion factors are the NIST values (2006 CODATA era) that LAMMPS
 * output reflects. The docs do not list them, so they were MEASURED from
 * native LAMMPS (2Sep2026) output with scripts/oracle/constants.in: two
 * unit-mass, unit-charge atoms 3 apart, moving at unit speed in a 10^3 box,
 *   mvv2e  = ke                              (ke = mvv2e * m v^2 / 2, x2)
 *   boltz  = 2 ke / (dof temp)               (dof = 3N - 3 = 3)
 *   qqr2e  = 3 pe                            (coul/cut: E = qqr2e q q / r)
 *   nktv2p = 3 V press / (2 ke)              (no pair virial)
 *   mv2d   = density V / (2 m)
 *   ftm2v  = vx / dt after one fix nve step under fix addforce 1 0 0
 *   qe2f   = fx under fix efield 1 0 0
 * and rounded to the significant digits of the underlying constant (e.g.
 * real boltz 0.0019872067 kcal/mol/K, qqr2e 332.06371).
 */

// real: one (g/mol)(Å/fs)^2 in kcal/mol is 48.88821291^2.
const REAL_V = 48.88821291;
// one g/mol per Å^3 in g/cm^3: 1 / 0.602214129 (Avogadro's number / 1e24).
const GMOL_A3 = 1 / 0.602214129;

const sys = (
  style: UnitStyle, c: Omit<UnitSystem, 'style' | 'normDefault'>,
): UnitSystem => ({ style, normDefault: style === 'lj', ...c });

export const UNIT_SYSTEMS: Record<UnitStyle, UnitSystem> = {
  lj: sys('lj', { boltz: 1, mvv2e: 1, ftm2v: 1, nktv2p: 1, qqr2e: 1, qe2f: 1, mv2d: 1, dt: 0.005, skin: 0.3 }),
  real: sys('real', {
    boltz: 0.0019872067, mvv2e: REAL_V * REAL_V, ftm2v: 1 / REAL_V / REAL_V,
    nktv2p: 68568.415, qqr2e: 332.06371, qe2f: 23.060549, mv2d: GMOL_A3, dt: 1.0, skin: 2.0,
  }),
  metal: sys('metal', {
    boltz: 8.617343e-5, mvv2e: 1.0364269e-4, ftm2v: 1 / 1.0364269e-4,
    nktv2p: 1.6021765e6, qqr2e: 14.399645, qe2f: 1.0, mv2d: GMOL_A3, dt: 0.001, skin: 2.0,
  }),
  si: sys('si', { boltz: 1.3806504e-23, mvv2e: 1, ftm2v: 1, nktv2p: 1, qqr2e: 8.9876e9, qe2f: 1, mv2d: 1, dt: 1.0e-8, skin: 0.001 }),
  cgs: sys('cgs', { boltz: 1.3806504e-16, mvv2e: 1, ftm2v: 1, nktv2p: 1, qqr2e: 1, qe2f: 1, mv2d: 1, dt: 1.0e-8, skin: 0.1 }),
  electron: sys('electron', {
    boltz: 3.16681534e-6, mvv2e: 1.06657236, ftm2v: 0.937582899,
    nktv2p: 2.94210108e13, qqr2e: 1.0, qe2f: 1.94469051e-10, mv2d: 1, dt: 0.001, skin: 2.0,
  }),
  micro: sys('micro', { boltz: 1.3806504e-8, mvv2e: 1, ftm2v: 1, nktv2p: 1, qqr2e: 8.987556e6, qe2f: 1, mv2d: 1, dt: 2.0, skin: 0.1 }),
  nano: sys('nano', { boltz: 0.013806504, mvv2e: 1, ftm2v: 1, nktv2p: 1, qqr2e: 230.7078669, qe2f: 1, mv2d: 1, dt: 0.00045, skin: 0.1 }),
};

/**
 * Density conversion, "density = mass/volume" (units.html). In 2d the volume
 * is the area lx*ly and the same mv2d applies: native LAMMPS (real, 2d,
 * one atom of mass 1 in a 10x10 box) prints vol 100, density 0.016605389.
 */
export const densityFactor = (units: UnitSystem, _dimension: 2 | 3): number => units.mv2d;

export const isUnitStyle = (s: string): s is UnitStyle => s in UNIT_SYSTEMS;

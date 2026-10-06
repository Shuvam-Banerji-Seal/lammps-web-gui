import type { UnitStyle, UnitSystem } from './types';

/*
 * Unit systems, from docs.lammps.org/units.html:
 *
 *   lj    "all quantities are unitless. Without loss of generality, LAMMPS
 *          sets the fundamental quantities mass, σ, ε, and the Boltzmann
 *          constant kB = 1."  Default "dt = 0.005 τ".
 *   real  mass grams/mole, distance Angstroms, time femtoseconds, energy
 *          kcal/mol, temperature Kelvin, pressure atmospheres, density
 *          g/cm^dim.  Default "dt = 1.0 femtoseconds".
 *   metal mass grams/mole, distance Angstroms, time picoseconds, energy eV,
 *          temperature Kelvin, pressure bars, density gram/cm^dim.
 *          Default "dt = 0.001 picoseconds".
 *
 * The conversion factors are DERIVED here from the exact SI constants
 * (CODATA 2018 / SI 2019) rather than copied from anywhere, so they differ
 * from LAMMPS's own values in the 7th significant figure at most.
 */

const NA = 6.02214076e23;          // 1/mol, exact
const E_CHARGE = 1.602176634e-19;  // C, exact
const KB_SI = 1.380649e-23;        // J/K, exact
const KCAL = 4184;                 // J, thermochemical calorie
const ATM = 101325;                // Pa
const BAR = 1e5;                   // Pa

// (g/mol)/Å^3 -> g/cm^3: one g/mol is 1/NA grams, one Å^3 is 1e-24 cm^3.
const GMOL_PER_A3_TO_G_PER_CM3 = 1e24 / NA;
// (g/mol)/Å^2 -> g/cm^2 (2D "density = g/cm^dim").
const GMOL_PER_A2_TO_G_PER_CM2 = 1e16 / NA;

/** g/mol · Å^2/fs^2 = 1e-3 kg/mol · 1e10 m^2/s^2 = 1e7 J/mol. */
const REAL_MVV2E = 1e7 / KCAL;
/** g/mol · Å^2/ps^2 = 1e-3 kg/mol · 1e4 m^2/s^2 = 10 J/mol, per particle in eV. */
const METAL_MVV2E = 10 / (NA * E_CHARGE);

export const UNIT_SYSTEMS: Record<UnitStyle, UnitSystem> = {
  lj: {
    style: 'lj', boltz: 1, mvv2e: 1, ftm2v: 1, nktv2p: 1, dt: 0.005, normDefault: true,
  },
  real: {
    style: 'real',
    boltz: (KB_SI * NA) / KCAL,                 // kcal/mol/K
    mvv2e: REAL_MVV2E,
    ftm2v: 1 / REAL_MVV2E,
    nktv2p: (KCAL / NA) / 1e-30 / ATM,          // (kcal/mol)/Å^3 -> atm
    dt: 1.0, normDefault: false,
  },
  metal: {
    style: 'metal',
    boltz: KB_SI / E_CHARGE,                    // eV/K
    mvv2e: METAL_MVV2E,
    ftm2v: 1 / METAL_MVV2E,
    nktv2p: E_CHARGE / 1e-30 / BAR,             // eV/Å^3 -> bar
    dt: 0.001, normDefault: false,
  },
};

/** Density conversion for the given dimensionality ("g/cm^dim"). */
export const densityFactor = (units: UnitSystem, dimension: 2 | 3): number =>
  units.style === 'lj' ? 1 : dimension === 3 ? GMOL_PER_A3_TO_G_PER_CM3 : GMOL_PER_A2_TO_G_PER_CM2;

export const isUnitStyle = (s: string): s is UnitStyle => s === 'lj' || s === 'real' || s === 'metal';

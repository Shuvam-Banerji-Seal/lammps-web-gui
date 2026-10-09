/*
 * Units of thermo output columns for each units style — docs.lammps.org/units.html
 * (plans/lammps-docs/units.rst). The page lists them per style, e.g.
 *   "For style *real*, these are the units:" ... "energy = kcal/mol", "temperature = Kelvin",
 *   "pressure = atmospheres", "distance = Angstroms", "time = femtoseconds", "density = g/cm\^dim"
 *   "For style *metal*, these are the units:" ... "energy = eV", "pressure = bars", "time = picoseconds"
 *   "For style *lj*, all quantities are unitless."
 * The strings below are the page's, with "\^" written as "^".
 */
export type UnitDim = 'energy' | 'temperature' | 'pressure' | 'distance' | 'time' | 'density';

const UNITS: Record<string, Partial<Record<UnitDim, string>>> = {
  real: { distance: 'Angstroms', time: 'femtoseconds', energy: 'kcal/mol', temperature: 'Kelvin', pressure: 'atmospheres', density: 'g/cm^dim' },
  metal: { distance: 'Angstroms', time: 'picoseconds', energy: 'eV', temperature: 'Kelvin', pressure: 'bars', density: 'gram/cm^dim' },
  si: { distance: 'meters', time: 'seconds', energy: 'Joules', temperature: 'Kelvin', pressure: 'Pascals', density: 'kilograms/meter^dim' },
  cgs: { distance: 'centimeters', time: 'seconds', energy: 'ergs', temperature: 'Kelvin', pressure: 'dyne/cm^2 or barye = 1.0e-6 bars', density: 'grams/cm^dim' },
  electron: { distance: 'Bohr', time: 'femtoseconds', energy: 'Hartrees', temperature: 'Kelvin', pressure: 'Pascals' },
  micro: { distance: 'micrometers', time: 'microseconds', energy: 'picogram-micrometer^2/microsecond^2', temperature: 'Kelvin', pressure: 'picogram/(micrometer-microsecond^2)', density: 'picograms/micrometer^dim' },
  nano: { distance: 'nanometers', time: 'nanoseconds', energy: 'attogram-nanometer^2/nanosecond^2', temperature: 'Kelvin', pressure: 'attogram/(nanometer-nanosecond^2)', density: 'attograms/nanometer^dim' },
};

/** Thermo keywords (docs.lammps.org/thermo_style.html) whose values carry one of the units above. */
const KEYWORD_DIM: Record<string, UnitDim> = {
  temp: 'temperature',
  press: 'pressure', pxx: 'pressure', pyy: 'pressure', pzz: 'pressure', pxy: 'pressure', pxz: 'pressure', pyz: 'pressure',
  pe: 'energy', ke: 'energy', etotal: 'energy', enthalpy: 'energy', evdwl: 'energy', ecoul: 'energy', epair: 'energy',
  ebond: 'energy', eangle: 'energy', edihed: 'energy', eimp: 'energy', emol: 'energy', elong: 'energy', etail: 'energy',
  lx: 'distance', ly: 'distance', lz: 'distance', xlo: 'distance', xhi: 'distance', ylo: 'distance', yhi: 'distance',
  zlo: 'distance', zhi: 'distance', xy: 'distance', xz: 'distance', yz: 'distance', xlat: 'distance', ylat: 'distance', zlat: 'distance',
  time: 'time', dt: 'time',
  density: 'density',
};

/** The unit of a thermo keyword under a units style, or null (unitless, a count, or not listed). */
export const thermoUnit = (units: string | undefined, keyword: string): string | null => {
  if (!units || units === 'lj') return null;
  const dim = KEYWORD_DIM[keyword];
  return dim ? UNITS[units]?.[dim] ?? null : null;
};

/** One-line summary for a thermo table: which units the numbers are in. */
export const unitsCaption = (units: string | undefined): string | null => {
  if (!units) return null;
  if (units === 'lj') return 'units lj: all quantities are unitless (reduced units)';
  const u = UNITS[units];
  if (!u) return null;
  return `units ${units}: energy ${u.energy}, temperature ${u.temperature}, pressure ${u.pressure}, distance ${u.distance}, time ${u.time}`;
};

const TIME_SECONDS: Record<string, number> = {
  femtoseconds: 1e-15, picoseconds: 1e-12, nanoseconds: 1e-9, microseconds: 1e-6, seconds: 1,
};

/**
 * Simulated time per wall-clock day at `stepsPerSec` with timestep `dt` (in the units style's time unit,
 * from the table above): "12.3 ns/day", or "4.1e+05 τ/day" for lj (whose time is unitless).
 */
export const simulatedPerDay = (units: string | undefined, dt: number, stepsPerSec: number): string | null => {
  if (!units || !(dt > 0) || !(stepsPerSec > 0)) return null;
  const perDay = dt * stepsPerSec * 86400;
  if (units === 'lj') return `${perDay < 1e4 ? perDay.toFixed(0) : perDay.toExponential(1)} τ/day`;
  const unit = UNITS[units]?.time;
  const sec = unit ? TIME_SECONDS[unit] : undefined;
  if (!sec) return null;
  const ns = (perDay * sec) / 1e-9;
  return ns >= 1000 ? `${(ns / 1000).toFixed(ns >= 1e4 ? 0 : 1)} µs/day` : ns >= 1 ? `${ns.toFixed(ns >= 100 ? 0 : 1)} ns/day` : `${(ns * 1000).toFixed(1)} ps/day`;
};

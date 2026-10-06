import type { ForceResult, SimState, ThermoKeyword, ThermoRow } from './types';
import { densityFactor } from './units';

/*
 * Thermodynamic observables, following the documented definitions:
 *
 * docs.lammps.org/compute_temp.html: "T = (2 E_kin) / (N_DOF k_B) with
 *   E_kin = Σ(1/2 m_i v²_i) and N_DOF = n_dim N_atoms - n_dim - N_fix DOFs".
 * docs.lammps.org/compute_pressure.html: "P = N k_B T / V + 1/(V d) Σ r_i·f_i"
 *   where "the N in the first formula above is really degrees-of-freedom
 *   divided by d = dimensionality".
 * docs.lammps.org/thermo_modify.html: "norm = yes for unit style of lj,
 *   norm = no for unit style of real and metal"; only extensive quantities
 *   (energies) are normalised by the number of atoms.
 * docs.lammps.org/thermo_style.html: etotal = pe + ke, enthalpy = etotal +
 *   press*vol, epair = evdwl + ecoul + elong, emol = bonded energy.
 */

/** Degrees of freedom of the whole system (no constraints in v1). */
export const degreesOfFreedom = (s: SimState): number =>
  Math.max(0, s.dimension * s.n - s.dimension);

export const kineticEnergy = (s: SimState): number => {
  let sum = 0;
  const { v, type, massByType } = s;
  for (let i = 0; i < s.n; i++) {
    const m = massByType[type[i]];
    sum += m * (v[3 * i] ** 2 + v[3 * i + 1] ** 2 + v[3 * i + 2] ** 2);
  }
  return 0.5 * sum * s.units.mvv2e;
};

export const temperature = (s: SimState, ke = kineticEnergy(s)): number => {
  const dof = degreesOfFreedom(s);
  return dof > 0 ? (2 * ke) / (dof * s.units.boltz) : 0;
};

/** Volume in 3D, area in 2D. */
export const volume = (s: SimState): number => {
  const lx = s.box.hi[0] - s.box.lo[0];
  const ly = s.box.hi[1] - s.box.lo[1];
  return s.dimension === 3 ? lx * ly * (s.box.hi[2] - s.box.lo[2]) : lx * ly;
};

/** Pressure in the style's pressure units. */
export const pressure = (s: SimState, temp: number, virial: number): number => {
  const d = s.dimension;
  const dof = degreesOfFreedom(s);
  return ((dof * s.units.boltz * temp) / d + virial / d) / volume(s) * s.units.nktv2p;
};

export const totalMass = (s: SimState): number => {
  let m = 0;
  for (let i = 0; i < s.n; i++) m += s.massByType[s.type[i]];
  return m;
};

export interface ThermoOptions {
  /** Normalise extensive values by atom count (defaults to the units style's). */
  norm?: boolean;
  /** Step at which the current run started (for `elapsed`). */
  runStart?: number;
}

/** One thermo row with exactly the requested keywords. */
export const thermoRow = (
  s: SimState, keywords: readonly ThermoKeyword[], forces: ForceResult, opts: ThermoOptions = {},
): ThermoRow => {
  const norm = (opts.norm ?? s.units.normDefault) && s.n > 0 ? 1 / s.n : 1;
  const ke = kineticEnergy(s);
  const temp = temperature(s, ke);
  const press = pressure(s, temp, forces.virial);
  const vol = volume(s);
  const row: ThermoRow = {};
  for (const k of keywords) {
    switch (k) {
      case 'step': row.step = s.step; break;
      case 'elapsed': row.elapsed = s.step - (opts.runStart ?? s.step); break;
      case 'time': row.time = s.step * s.dt; break;
      case 'temp': row.temp = temp; break;
      case 'press': row.press = press; break;
      case 'pe': row.pe = forces.pe * norm; break;
      case 'ke': row.ke = ke * norm; break;
      case 'etotal': row.etotal = (forces.pe + ke) * norm; break;
      case 'enthalpy': row.enthalpy = (forces.pe + ke + (press / s.units.nktv2p) * vol) * norm; break;
      case 'evdwl': row.evdwl = forces.pe * norm; break;
      case 'epair': row.epair = forces.pe * norm; break;
      case 'ecoul': row.ecoul = 0; break;
      case 'emol': row.emol = 0; break;
      case 'vol': row.vol = vol; break;
      case 'density': row.density = (totalMass(s) / vol) * densityFactor(s.units, s.dimension); break;
      case 'lx': row.lx = s.box.hi[0] - s.box.lo[0]; break;
      case 'ly': row.ly = s.box.hi[1] - s.box.lo[1]; break;
      case 'lz': row.lz = s.box.hi[2] - s.box.lo[2]; break;
      case 'xlo': row.xlo = s.box.lo[0]; break;
      case 'xhi': row.xhi = s.box.hi[0]; break;
      case 'ylo': row.ylo = s.box.lo[1]; break;
      case 'yhi': row.yhi = s.box.hi[1]; break;
      case 'zlo': row.zlo = s.box.lo[2]; break;
      case 'zhi': row.zhi = s.box.hi[2]; break;
      case 'dt': row.dt = s.dt; break;
      case 'atoms': row.atoms = s.n; break;
    }
  }
  return row;
};

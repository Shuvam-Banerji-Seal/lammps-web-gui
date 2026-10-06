/**
 * Contracts shared by every part of the in-browser MD engine
 * (docs/design/notebook.md). Code against these; do not change them without
 * updating every implementation.
 *
 * The engine is an independent implementation written from textbook physics
 * (Allen & Tildesley; Frenkel & Smit) and from the DOCUMENTED behaviour on
 * docs.lammps.org. It contains no LAMMPS source code: LAMMPS is GPL-2.0 and
 * this project is not.
 *
 * Layout conventions
 *  - Per-atom vectors are flat Float64Array of length 3N: [x0, y0, z0, x1, …].
 *    2D runs keep z (fixed at its initial value) so every array stays 3N.
 *  - Positions are kept WRAPPED into the box; image flags in `image` make the
 *    unwrapped position exact: x + image * L.
 *  - Atom types are 1-based (LAMMPS convention); index 0 of per-type tables is
 *    unused.
 */

/** Orthogonal box. Triclinic is out of scope for v1. */
export interface SimBox {
  lo: [number, number, number];
  hi: [number, number, number];
  periodic: [boolean, boolean, boolean];
}

export type UnitStyle = 'lj' | 'real' | 'metal';

/**
 * Conversion constants for one units style (see units.ts for derivations).
 *  boltz   Boltzmann constant in energy/temperature units
 *  mvv2e   mass * velocity^2 -> energy
 *  ftm2v   force / mass * time -> velocity
 *  nktv2p  energy / volume -> pressure
 *  dt      the style's default timestep
 */
export interface UnitSystem {
  style: UnitStyle;
  boltz: number;
  mvv2e: number;
  ftm2v: number;
  nktv2p: number;
  dt: number;
  /** thermo_modify norm default: yes for lj, no otherwise. */
  normDefault: boolean;
}

export interface SimState {
  n: number;
  dimension: 2 | 3;
  box: SimBox;
  units: UnitSystem;
  /** Number of atom types (create_box N). */
  ntypes: number;
  /** 1-based type per atom, length n. */
  type: Int32Array;
  /** Per-type mass, length ntypes + 1, index 0 unused; NaN = not set. */
  massByType: Float64Array;
  x: Float64Array;
  v: Float64Array;
  f: Float64Array;
  image: Int32Array;
  /** Atom IDs, 1-based and stable for the whole session. */
  id: Int32Array;
  step: number;
  dt: number;
}

/** Lennard-Jones 12-6 parameters of one type pair. */
export interface LJPair {
  epsilon: number;
  sigma: number;
  cutoff: number;
}

export interface PairTable {
  style: 'lj/cut';
  ntypes: number;
  /** Global cutoff from `pair_style lj/cut rc`. */
  globalCutoff: number;
  /**
   * Coefficients, row-major (ntypes+1)^2, 1-based, symmetric once resolved.
   * `explicit` marks pairs set by pair_coeff; the rest come from mixing.
   */
  pairs: (LJPair | undefined)[];
  explicit: boolean[];
  mix: 'geometric' | 'arithmetic' | 'sixthpower';
  /** pair_modify shift yes: energy shifted to zero at the cutoff. */
  shift: boolean;
}

export interface ForceResult {
  /** Total pair potential energy (not per atom), energy units. */
  pe: number;
  /**
   * Scalar pair virial W = sum over interacting pairs of r_ij . F_ij, energy
   * units. Pressure (docs.lammps.org/compute_pressure.html):
   *   P = (dof * kB * T / d + W / d) / V   * nktv2p
   * — "the N in the first formula above is really degrees-of-freedom divided
   * by d = dimensionality, where the DOF value is calculated by the
   * temperature compute."
   */
  virial: number;
}

/** One way of evaluating forces: fp64 CPU (reference) or fp32 WebGPU. */
export interface ForceBackend {
  readonly kind: 'cpu' | 'webgpu';
  /** Human-readable, e.g. "CPU · fp64" or "WebGPU · <adapter>". */
  readonly label: string;
  /** Overwrites state.f; returns the pair energy and virial. */
  compute(state: SimState, pairs: PairTable): ForceResult | Promise<ForceResult>;
  dispose(): void;
}

export type FixSpec =
  | { style: 'nve'; id: string; group: 'all' }
  | { style: 'langevin'; id: string; group: 'all'; tStart: number; tStop: number; damp: number; seed: number }
  | { style: 'temp/berendsen'; id: string; group: 'all'; tStart: number; tStop: number; damp: number }
  | { style: 'temp/rescale'; id: string; group: 'all'; every: number; tStart: number; tStop: number; window: number; fraction: number }
  | { style: 'nvt'; id: string; group: 'all'; tStart: number; tStop: number; damp: number }
  | { style: 'enforce2d'; id: string; group: 'all' };

/** Supported thermo_style custom keywords (v1). */
export const THERMO_KEYWORDS = [
  'step', 'elapsed', 'time', 'temp', 'press', 'pe', 'ke', 'etotal', 'enthalpy',
  'evdwl', 'ecoul', 'epair', 'emol', 'vol', 'density', 'lx', 'ly', 'lz', 'atoms',
] as const;
export type ThermoKeyword = typeof THERMO_KEYWORDS[number];

/** One thermo output row keyed by thermo keyword. */
export type ThermoRow = Partial<Record<ThermoKeyword, number>>;

/** Interpreter errors carry the 1-based source line and the command word. */
export class EngineError extends Error {
  constructor(message: string, public readonly line: number, public readonly command: string) {
    super(line > 0 ? `line ${line}: ${message}` : message);
    this.name = 'EngineError';
  }
}

/** What the engine streams back while running (worker -> UI). */
export type EngineEvent =
  | { kind: 'log'; text: string }
  | { kind: 'thermo-header'; keywords: ThermoKeyword[] }
  | { kind: 'thermo'; row: ThermoRow }
  | { kind: 'frame'; step: number; x: Float64Array; image: Int32Array; type: Int32Array; id: Int32Array; box: SimBox }
  | { kind: 'error'; message: string; line: number; command: string }
  | { kind: 'done'; steps: number; seconds: number; backend: string };

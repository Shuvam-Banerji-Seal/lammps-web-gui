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

/**
 * Boundary style of one box face — docs.lammps.org/boundary.html:
 * "p is periodic", "f is non-periodic and fixed", "s is non-periodic and
 * shrink-wrapped", "m is non-periodic and shrink-wrapped with a minimum value".
 */
export type BoundaryStyle = 'p' | 'f' | 's' | 'm';

/**
 * Simulation box. Orthogonal, or restricted triclinic (docs.lammps.org/
 * Howto_triclinic.html): edge vectors A = (lx, 0, 0), B = (xy, ly, 0),
 * C = (xz, yz, lz), with lx = xhi - xlo etc. and the tilt factors xy, xz, yz.
 */
export interface SimBox {
  lo: [number, number, number];
  hi: [number, number, number];
  periodic: [boolean, boolean, boolean];
  /** Tilt factors [xy, xz, yz]; all 0 for an orthogonal box. */
  tilt: [number, number, number];
  triclinic: boolean;
  /** [lower, upper] face style per dimension; periodic dims are ['p', 'p']. */
  boundary: [[BoundaryStyle, BoundaryStyle], [BoundaryStyle, BoundaryStyle], [BoundaryStyle, BoundaryStyle]];
  /** For 'm' faces: the box never shrinks inside these bounds (docs: "minimum value"). */
  minLo: [number, number, number];
  minHi: [number, number, number];
  /**
   * Set when the box was created as general triclinic (create_box NULL, or a general
   * triclinic read_data header). Q is the rotation general -> restricted (rows e1, e2, e3,
   * triclinic_general.ts); the box itself stays restricted. Kept so write_data, dump and
   * thermo can rotate back to the general frame.
   */
  general?: { Q: [[number, number, number], [number, number, number], [number, number, number]] };
}

export type UnitStyle = 'lj' | 'real' | 'metal' | 'si' | 'cgs' | 'electron' | 'micro' | 'nano';

/**
 * Conversion constants for one units style (see units.ts for their source).
 *  boltz   Boltzmann constant in energy/temperature units
 *  mvv2e   mass * velocity^2 -> energy
 *  ftm2v   force / mass * time -> velocity
 *  nktv2p  energy / volume -> pressure
 *  qqr2e   q_i q_j / r -> energy (Coulomb constant)
 *  qe2f    charge * electric field -> force
 *  mv2d    mass / volume -> density
 *  dt      the style's default timestep;  skin  the default neighbor skin
 */
export interface UnitSystem {
  style: UnitStyle;
  boltz: number;
  mvv2e: number;
  ftm2v: number;
  nktv2p: number;
  qqr2e: number;
  qe2f: number;
  mv2d: number;
  dt: number;
  skin: number;
  /** thermo_modify norm default: yes for lj, no otherwise. */
  normDefault: boolean;
}

/** docs.lammps.org/atom_style.html styles the engine implements. */
/** An atom style; atom_style hybrid is stored as 'hybrid' followed by its sub-styles. */
/**
 * `template:<ID>` is atom_style template ID: the molecule template ID is part of the style string so that
 * it survives write_restart / read_restart (atoms.ts templateStyleId reads it back).
 */
export type AtomStyle = 'atomic' | 'charge' | 'bond' | 'angle' | 'molecular' | 'full' | 'sphere' | 'dipole' | 'ellipsoid' | 'peri' | `template:${string}` | `hybrid ${string}`;

/**
 * Bonded topology entries of one kind, stored by atom ID (not index) so that
 * deleting or adding atoms never invalidates them. Entry k has type type[k]
 * and atoms atoms[width*k .. width*k + width).
 */
export interface TopoList {
  n: number;
  width: 2 | 3 | 4;
  type: Int32Array;
  atoms: Int32Array;
}

export interface Topology {
  nbondtypes: number;
  nangletypes: number;
  ndihedraltypes: number;
  nimpropertypes: number;
  bonds: TopoList;
  angles: TopoList;
  dihedrals: TopoList;
  impropers: TopoList;
}

export interface SimState {
  n: number;
  dimension: 2 | 3;
  box: SimBox;
  units: UnitSystem;
  atomStyle: AtomStyle;
  /** Number of atom types (create_box N). */
  ntypes: number;
  /** 1-based type per atom, length n. */
  type: Int32Array;
  /** Per-type mass, length ntypes + 1, index 0 unused; NaN = not set. */
  massByType: Float64Array;
  /** Per-atom masses (atom styles with a per-atom mass, e.g. sphere), else null: use massOf(). */
  rmass: Float64Array | null;
  /** atom_style sphere: per-atom radius (0 = point particle), angular velocity and torque (3N); else null. */
  radius: Float64Array | null;
  omega: Float64Array | null;
  torque: Float64Array | null;
  /** atom_style dipole: point dipole per atom as mux, muy, muz and its length (4N); else null. */
  mu: Float64Array | null;
  /**
   * atom_style ellipsoid: half-axes (3N; all 0 for a point particle), orientation quaternion w i j k
   * (4N) and angular momentum (3N); else null.
   */
  shape: Float64Array | null;
  quat: Float64Array | null;
  angmom: Float64Array | null;
  /**
   * atom_style peri: per-atom volume (vfrac, 1 by default) and the reference (strain-free) position
   * x0 (3N), set when the atom is created; else null.
   */
  vfrac: Float64Array | null;
  x0: Float64Array | null;
  /**
   * fix property/atom: custom per-atom vectors and arrays by name (i_name /
   * d_name: cols 0; i2_name / d2_name: cols N), values stored as doubles
   * (integer properties are truncated when set).
   */
  custom: Map<string, CustomProp>;
  /** fix property/atom mol / q: molecule IDs or charges the atom style itself lacks. */
  propMol: boolean;
  propQ: boolean;
  x: Float64Array;
  v: Float64Array;
  f: Float64Array;
  image: Int32Array;
  /** Atom IDs, 1-based and stable for the whole session. */
  id: Int32Array;
  /**
   * Native LAMMPS storage order: order[k] is the index of the atom that native LAMMPS (one process)
   * keeps at position k of its atom list. The engine never permutes its own arrays; commands whose
   * results depend on that order (per-atom random draws, unsorted dumps, write_data, ID
   * compression) walk this list. Kept by atoms.ts (append, delete) and System.sortAtoms.
   */
  order: Int32Array;
  /** Group membership bits (bit 0 = group all). */
  mask: Int32Array;
  /** Molecule ID per atom (0 = none). */
  molecule: Int32Array;
  /**
   * atom_style template: per atom the molecule index within the template (1..Nmols, 0 = not a template
   * atom) and the atom index within that molecule (1..Natoms, 0 = not a template atom); else null.
   */
  tmplIndex: Int32Array | null;
  tmplAtom: Int32Array | null;
  /** Charge per atom. */
  q: Float64Array;
  topo: Topology;
  step: number;
  dt: number;
  /** Simulation time at step `timeStep` (thermo 'time' advances by dt from there). */
  time: number;
  timeStep: number;
}

/** One custom per-atom property of fix property/atom. */
export interface CustomProp {
  int: boolean;
  /** 0 for a vector, N for an N-column array. */
  cols: number;
  /** Length n * max(cols, 1). */
  data: Float64Array;
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
  /** Virial tensor [xx, yy, zz, xy, xz, yz] when the backend computes it. */
  virialTensor?: number[];
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

/** Built-in thermo_style custom keywords (docs.lammps.org/thermo_style.html) the engine implements. */
export const THERMO_KEYWORDS = [
  'step', 'elapsed', 'elaplong', 'dt', 'time', 'cpu', 'tpcpu', 'spcpu', 'cpuremain', 'part', 'timeremain',
  'atoms', 'temp', 'press', 'pe', 'ke', 'etotal', 'enthalpy',
  'evdwl', 'ecoul', 'epair', 'ebond', 'eangle', 'edihed', 'eimp', 'emol', 'elong', 'etail',
  'vol', 'density', 'lx', 'ly', 'lz', 'xlo', 'xhi', 'ylo', 'yhi', 'zlo', 'zhi',
  'xy', 'xz', 'yz', 'xlat', 'ylat', 'zlat', 'bonds', 'angles', 'dihedrals', 'impropers',
  'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz', 'fmax', 'fnorm', 'nbuild', 'ndanger',
  'cella', 'cellb', 'cellc', 'cellalpha', 'cellbeta', 'cellgamma',
] as const;
/** A thermo column: a built-in keyword or c_ID, c_ID[i], f_ID, f_ID[i], v_name. */
export type ThermoKeyword = string;

/** One thermo output row keyed by thermo column. */
export type ThermoRow = Record<string, number>;

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
  /** labels: header text per column where thermo_modify colname renamed it (else the keyword). */
  | { kind: 'thermo-header'; keywords: ThermoKeyword[]; labels?: string[]; units?: UnitStyle }
  | { kind: 'thermo'; row: ThermoRow }
  /** A run (or rerun/minimize-free MD run) starts: it goes from step `from` to step `to` (for progress display). */
  | { kind: 'run'; from: number; to: number }
  | { kind: 'frame'; step: number; x: Float64Array; image: Int32Array; type: Int32Array; id: Int32Array; box: SimBox }
  | { kind: 'error'; message: string; line: number; command: string }
  | { kind: 'done'; steps: number; seconds: number; backend: string };

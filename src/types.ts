export interface Atom {
  id: number;
  molId: number;
  type: number;
  charge: number;
  /** WRAPPED position, as the file gives it — this is what gets rendered. */
  x: number;
  y: number;
  z: number;
  vx?: number;
  vy?: number;
  vz?: number;
  /**
   * LAMMPS periodic image flags, when the dump carries `ix iy iz`.
   *
   * docs.lammps.org/dump.html: "For periodic dimensions, they specify which
   * image of the simulation box the atom is considered to be in. An image of
   * 0 means it is inside the box as defined. A value of 2 means add 2 box
   * lengths to get the true value."
   *
   * Kept separate from x/y/z deliberately: rendering wants the WRAPPED
   * position (otherwise a diffusing system scatters across box images and
   * looks broken), while MSD needs the UNWRAPPED one or it saturates at
   * (L/2)² and can never show linear diffusion.
   */
  ix?: number;
  iy?: number;
  iz?: number;
}

export interface Bond {
  id: number;
  type: number;
  atom1Id: number;
  atom2Id: number;
}

export interface AtomTypeInfo {
  id: number;
  mass: number;
  element: string; // e.g., "C", "H", or "X" if unknown
  label: string;   // Display name
  count: number;
}

/** Simulation box boundaries. Tilt factors present for triclinic cells. */
export interface BoxBounds {
  xlo: number; xhi: number;
  ylo: number; yhi: number;
  zlo: number; zhi: number;
  xy?: number; xz?: number; yz?: number;
}

export interface MoleculeData {
  atoms: Atom[];
  bonds: Bond[];
  atomTypes: Record<number, AtomTypeInfo>;
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
  center: { x: number; y: number; z: number };
  /** Simulation box parsed from LAMMPS box bounds / PDB CRYST1 / CIF cell. Optional for backward compat. */
  box?: BoxBounds;
  /** All frames of a trajectory (XYZ). Absent/length-1 for static structures. */
  frames?: TrajectoryFrame[];
}

/** One frame of a multi-frame (trajectory) structure. */
export interface TrajectoryFrame {
  comment?: string;
  atoms: Atom[];
  /**
   * This frame's own cell. A LAMMPS dump repeats BOX BOUNDS every frame, and
   * under NPT the cell breathes, so a single box taken from frame 0 is wrong
   * for every later frame.
   */
  box?: BoxBounds;
}

export enum ParseSection {
  NONE,
  MASSES,
  ATOMS,
  BONDS,
  ANGLES,
  DIHEDRALS,
  IMPROPERS
}

export type MaterialType = 'realistic' | 'plastic' | 'toon' | 'metallic';

export type VisualizationMode = 'ball-and-stick' | 'space-fill' | 'wireframe' | 'licorice';

export type FileFormat = 'lammps' | 'xyz' | 'pdb' | 'cif' | 'lammpsdump';

export type LightingPreset = 'studio' | 'lab' | 'outdoor' | 'space' | 'soft';

export type CameraPreset = 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom' | 'iso';

export interface VisualizationConfig {
  atomScale: number;
  bondScale: number;
  materialType: MaterialType;
  backgroundColor: string;
  showBonds: boolean;
  customColors: Record<number, string>;
  visualizationMode: VisualizationMode;
  lightingPreset: LightingPreset;
  showBox: boolean;
  showAxes: boolean;
  showLabels: boolean;
  shadowsEnabled: boolean;
  autoRotateSpeed: number;
  fov: number;
}

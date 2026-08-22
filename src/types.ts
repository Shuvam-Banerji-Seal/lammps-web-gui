export interface Atom {
  id: number;
  molId: number;
  type: number;
  charge: number;
  x: number;
  y: number;
  z: number;
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

export type FileFormat = 'lammps' | 'xyz' | 'pdb' | 'cif';

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

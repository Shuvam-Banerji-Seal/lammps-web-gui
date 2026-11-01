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
  /** True when `bonds` were guessed from distances by the parser (CIF, XYZ, dump), not read from the file. */
  bondsInferred?: boolean;
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
  /**
   * True when x/y/z are ABSOLUTE (a dump's `xu yu zu` / `xsu ysu zsu`).
   *
   * Displacement maths depends on this. Applying the minimum-image convention
   * to already-unwrapped coordinates does not just lose precision, it destroys
   * the signal: a genuine one-box drift folds to exactly zero.
   */
  coordsUnwrapped?: boolean;
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

export type VisualizationMode =
  | 'ball-and-stick' | 'space-fill' | 'wireframe' | 'licorice' | 'element-balls' | 'points'
  | 'lines' | 'cubes' | 'glass' | 'flat';

export type FileFormat = 'lammps' | 'xyz' | 'pdb' | 'cif' | 'lammpsdump';

export type LightingPreset = 'studio' | 'lab' | 'outdoor' | 'space' | 'soft';

export type CameraPreset = 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom' | 'iso';

/** What decides an atom's colour: its type (the per-type colours), or a per-atom quantity through a colour map. */
export type ColorBy = 'type' | 'charge' | 'speed' | 'id' | 'molecule' | 'x' | 'y' | 'z';

export type ColorMapName = 'viridis' | 'plasma' | 'inferno' | 'turbo' | 'coolwarm' | 'rainbow' | 'grayscale';

/** Rendering quality: sphere and cylinder tessellation, pixel ratio and shadow resolution ('auto' scales with size). */
export type RenderQuality = 'low' | 'auto' | 'high' | 'ultra';

export type ToneMappingMode = 'aces' | 'agx' | 'neutral' | 'none';

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
  /** Colour by type (default) or by a per-atom quantity through colorMap. */
  colorBy?: ColorBy;
  colorMap?: ColorMapName;
  /** Reverse the colour map. */
  colorReverse?: boolean;
  /** Map range from the data (all frames) when true or absent; colorMin..colorMax otherwise. */
  colorAuto?: boolean;
  colorMin?: number;
  colorMax?: number;
  quality?: RenderQuality;
  toneMapping?: ToneMappingMode;
  /** Tone-mapping exposure (1 = neutral). */
  exposure?: number;
  /** Image-based lighting from a procedural room environment (reflections on the PBR materials). */
  envLighting?: boolean;
  /** Depth cueing: fog toward the background colour, 0 (off) to 1 (strong). */
  depthCue?: number;
  /** One colour for every bond; empty or absent: each half takes its atom's colour. */
  bondColor?: string;
  /** Simulation box edge colour. */
  boxColor?: string;
  /** Radius multiplier per atom type (1 when absent). */
  typeRadii?: Record<number, number>;
}

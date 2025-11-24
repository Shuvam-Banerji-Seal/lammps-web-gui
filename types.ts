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
  element: string; // e.g., "C", "H", or "Type 1"
  label: string;   // Display name
  count: number;
}

export interface MoleculeData {
  atoms: Atom[];
  bonds: Bond[];
  atomTypes: Record<number, AtomTypeInfo>;
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
  center: { x: number; y: number; z: number };
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

export type MaterialType = 'realistic' | 'plastic' | 'toon';

export interface VisualizationConfig {
  atomScale: number;
  bondScale: number;
  materialType: MaterialType;
  backgroundColor: string;
  showBonds: boolean;
  customColors: Record<number, string>; // Map Atom Type ID -> Hex Color
}

// Fix for React Three Fiber intrinsic elements in TypeScript
// These elements are provided by @react-three/fiber but may not be recognized
// by TypeScript's JSX.IntrinsicElements interface in all environments.
declare global {
  namespace JSX {
    interface IntrinsicElements {
      mesh: any;
      group: any;
      ambientLight: any;
      directionalLight: any;
      spotLight: any;
      pointLight: any;
      meshPhysicalMaterial: any;
      meshStandardMaterial: any;
      meshToonMaterial: any;
      cylinderGeometry: any;
      color: any;
    }
  }
}
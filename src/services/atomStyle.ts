import { Atom, RenderQuality, VisualizationConfig, VisualizationMode } from '../types';
import { COVALENT_RADII, ELEMENT_RADII } from '../constants';

/**
 * Single source of truth for on-screen atom radius across renderers,
 * selection rings and labels — keeps overlays perfectly registered
 * with the instanced mesh.
 */
export const atomDisplayRadius = (atom: Atom, config: VisualizationConfig): number => {
  const s = config.atomScale * (config.typeRadii?.[atom.type] ?? 1);
  if (config.visualizationMode === 'space-fill' || config.visualizationMode === 'glass') {
    return (ELEMENT_RADII[atom.type] ?? 1.7) * s * 0.5;
  }
  if (config.visualizationMode === 'lines') {
    // a small dot at each atom, so atoms without bonds stay visible
    return 0.1 * s;
  }
  if (config.visualizationMode === 'flat') {
    // touching discs for unit-spaced particles (LAMMPS lj units: sigma = 1)
    return 0.5 * s;
  }
  if (config.visualizationMode === 'wireframe') {
    return 0.16 * s;
  }
  if (config.visualizationMode === 'licorice') {
    return 0.14 * s;
  }
  if (config.visualizationMode === 'element-balls') {
    return (COVALENT_RADII[atom.type] ?? 0.75) * s * 0.6;
  }
  if (config.visualizationMode === 'points') {
    return 0.4 * s;
  }
  return 0.45 * s; // ball-and-stick, and the half edge of a cube
};

/** Modes drawn with the instanced mesh of InstancedAtomMesh (the others are point sprites, PointAtoms). */
export const meshMode = (mode: VisualizationMode): boolean =>
  mode !== 'points' && mode !== 'flat' && mode !== 'lines';

/** Modes that draw bonds as cylinders (InstancedBondMesh); 'lines' draws them as lines (BondLines). */
export const cylinderBondMode = (mode: VisualizationMode): boolean =>
  mode !== 'space-fill' && mode !== 'glass' && mode !== 'points' && mode !== 'flat' && mode !== 'lines';

/** A key of everything atomDisplayRadius reads from the view (instance matrices are rewritten when it changes). */
export const radiusKeyOf = (config: VisualizationConfig): string => {
  const t = config.typeRadii;
  const types = t ? Object.keys(t).map((k) => `${k}:${t[Number(k)]}`).join(';') : '';
  return `${config.visualizationMode}|${config.atomScale}|${types}`;
};

/** Sphere segments by quality for up to 1k, 10k, 30k atoms and above ('auto' is the size-adaptive default). */
const SPHERE_SEGMENTS: Record<RenderQuality, readonly number[]> = {
  low: [16, 12, 8, 6], auto: [32, 20, 12, 8], high: [48, 32, 20, 12], ultra: [64, 48, 32, 20],
};

export const sphereSegments = (quality: RenderQuality | undefined, atoms: number): number => {
  const t = SPHERE_SEGMENTS[quality ?? 'auto'] ?? SPHERE_SEGMENTS.auto;
  return atoms > 30000 ? t[3] : atoms > 10000 ? t[2] : atoms > 1000 ? t[1] : t[0];
};

/** Radial segments of the bond cylinders. */
export const bondSegments = (quality: RenderQuality | undefined, bonds: number): number => {
  const auto = Math.min(16, Math.max(6, 18 - Math.floor(bonds / 2000)));
  if (quality === 'low') return Math.max(6, Math.round(auto / 2));
  if (quality === 'high') return auto + 8;
  if (quality === 'ultra') return 32;
  return auto;
};

/** Device pixel ratio range: 'auto' renders large systems below native resolution, 'ultra' supersamples. */
export const qualityDpr = (quality: RenderQuality | undefined, atoms: number): [number, number] => {
  if (quality === 'low') return [0.75, 1];
  if (quality === 'high') return [1, 2];
  if (quality === 'ultra') return [2, 3];
  if (atoms > 20000) return [0.75, 1.25];
  if (atoms > 5000) return [1, 1.5];
  return [1, 2];
};

/** Largest system that casts shadows, and the key light's shadow map size. */
export const shadowLimit = (quality: RenderQuality | undefined): number =>
  quality === 'low' ? 2000 : quality === 'high' ? 20000 : quality === 'ultra' ? 50000 : 8000;

export const shadowMapSize = (quality: RenderQuality | undefined): number =>
  quality === 'low' ? 1024 : quality === 'high' || quality === 'ultra' ? 4096 : 2048;

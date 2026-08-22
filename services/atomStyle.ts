import { Atom, VisualizationConfig } from '../types';
import { ELEMENT_RADII } from '../constants';

/**
 * Single source of truth for on-screen atom radius across renderers,
 * selection rings and labels — keeps overlays perfectly registered
 * with the instanced mesh.
 */
export const atomDisplayRadius = (atom: Atom, config: VisualizationConfig): number => {
  if (config.visualizationMode === 'space-fill') {
    return (ELEMENT_RADII[atom.type] ?? 1.7) * config.atomScale * 0.5;
  }
  if (config.visualizationMode === 'wireframe') {
    return 0.16 * config.atomScale;
  }
  if (config.visualizationMode === 'licorice') {
    return 0.14 * config.atomScale;
  }
  return 0.45 * config.atomScale; // ball-and-stick
};

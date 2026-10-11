/**
 * Render-only overrides for the instanced atom mesh: per-atom sRGB colours (one per instance) and a radius scale.
 * The Render panel sets them before drawing each frame of a clip and clears them after; the mesh applies them
 * synchronously to its existing instance buffers (no new meshes). The default is "no override", which leaves the
 * view exactly as it is, so the Structure Viewer and the notebook's live view are unaffected.
 */

export interface AtomRenderOverride {
  /** Per-atom sRGB colours, 3 floats per atom in instance order; null keeps the view's type colours. */
  colours: Float32Array | null;
  /** Multiplies every atom's display radius. */
  radiusScale: number;
}

export const NO_OVERRIDE: AtomRenderOverride = { colours: null, radiusScale: 1 };

let current: AtomRenderOverride = NO_OVERRIDE;
const listeners = new Set<() => void>();

export const getAtomRenderOverride = (): AtomRenderOverride => current;

/** Sets the override and tells every mounted atom mesh to apply it now. */
export const setAtomRenderOverride = (next: AtomRenderOverride): void => {
  current = next;
  listeners.forEach((fn) => fn());
};

/** Subscribes a mesh to override changes; returns the unsubscribe function. */
export const onAtomRenderOverride = (fn: () => void): (() => void) => {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
};

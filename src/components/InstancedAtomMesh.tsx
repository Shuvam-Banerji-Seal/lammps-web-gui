import React, { useRef, useMemo, useEffect, useState } from 'react';
import * as THREE from 'three';
import { ThreeEvent, useThree } from '@react-three/fiber';
import { Atom, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { atomDisplayRadius, radiusKeyOf, sphereSegments } from '../services/atomStyle';
import { colorKeyOf, makeAtomColorer } from '../services/colorMaps';
import { writeInstanceTransform } from '../services/instanceMatrix';
import { getAtomRenderOverride, onAtomRenderOverride } from './workbench/notebook/render/atomOverride';

interface InstancedAtomMeshProps {
  atoms: Atom[];
  config: VisualizationConfig;
  onHover?: (atom: Atom | null, screenX: number, screenY: number) => void;
  /** Fired on genuine clicks (not orbit drags) for the measurement tool. */
  onSelectAtom?: (id: number) => void;
}

/** Above this size, per-move raycasting is disabled to keep the UI fluid. */
export const PICKING_MAX_ATOMS = 50_000;

/**
 * Optimized instanced rendering for atoms — one draw call regardless of
 * system size. Sphere tessellation follows the quality setting and adapts to
 * system size (sphereSegments: 'auto' gives 32 segments up to 1k atoms, 20 up
 * to 10k, 12 up to 30k, 8 above).
 *
 * Performance contract:
 *  - Matrices rewrite ONLY when positions or radius-affecting fields change.
 *    Toggling labels/lighting/materials costs zero matrix work.
 *  - Colors rewrite ONLY when the color mapping changes.
 *  - Bounding sphere is recomputed after writes so frustum culling stays
 *    correct (three.js cannot infer instance extents from a unit sphere).
 */
/** Writes every instance's matrix: a translation and a uniform radius (times the render radius scale). */
const writeMatrices = (mesh: THREE.InstancedMesh, atoms: Atom[], config: VisualizationConfig, scale: number): void => {
  // An instance is only ever a translation plus a UNIFORM scale, so the
  // matrix is known in closed form. Writing the 16 floats straight into the
  // instance buffer skips Object3D.updateMatrix()'s quaternion compose and
  // the setMatrixAt copy for every atom — which matters when a trajectory
  // rewrites 60k instances on every frame.
  const m = mesh.instanceMatrix.array as Float32Array;
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i];
    writeInstanceTransform(m, i, atom.x, atom.y, atom.z, atomDisplayRadius(atom, config) * scale);
  }
};

/**
 * Writes every instance's colour: the override's sRGB colours when they cover the atoms, else the view's colouring
 * (makeAtomColorer: by type, by molecule, or a quantity through the colour map).
 */
const writeColours = (mesh: THREE.InstancedMesh, atoms: Atom[], config: VisualizationConfig, override: Float32Array | null): void => {
  const c = new THREE.Color();
  if (override && override.length >= 3 * atoms.length) {
    for (let i = 0; i < atoms.length; i++) {
      c.setRGB(override[3 * i], override[3 * i + 1], override[3 * i + 2], THREE.SRGBColorSpace);
      mesh.setColorAt(i, c);
    }
  } else {
    const colorOf = makeAtomColorer(config, DEFAULT_ATOM_COLOR);
    const rgb = new Float32Array(3);
    for (let i = 0; i < atoms.length; i++) {
      colorOf(atoms[i], rgb);
      c.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
      mesh.setColorAt(i, c);
    }
  }
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
};

const InstancedAtomMesh: React.FC<InstancedAtomMeshProps> = ({ atoms, config, onHover, onSelectAtom }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const invalidate = useThree((s) => s.invalidate);
  const [hoverId, setHoverId] = useState<number | null>(null);

  // Narrow dependency keys — object identity of `config` must NOT trigger O(n) work.
  const radiusKey = radiusKeyOf(config);
  const colorKey = colorKeyOf(config);

  const segments = sphereSegments(config.quality, atoms.length);
  // Tessellation tiers: visual quality is indistinguishable at these
  // densities, while triangle count drops ~4x per tier.
  const geometry = useMemo(
    () => new THREE.SphereGeometry(1, segments, Math.max(6, Math.round(segments / 2))),
    [segments],
  );

  // R3F disposes what it constructs from `args`, but a useMemo'd geometry that
  // is REPLACED when the tier changes is not R3F's to clean up — without this
  // the old buffers leak on the GPU every time a differently sized structure
  // is loaded.
  useEffect(() => () => geometry.dispose(), [geometry]);

  // latest props for the render-override listener (which runs outside React's render)
  const latest = useRef({ atoms, config });
  useEffect(() => { latest.current = { atoms, config }; }, [atoms, config]);
  // what the mesh currently shows, so an override is cleared (and re-applied) only when it changes
  const applied = useRef({ scale: 1, colours: false });

  // --- Matrices: positions + radius inputs only ---
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const scale = getAtomRenderOverride().radiusScale;
    writeMatrices(mesh, atoms, config, scale);
    applied.current.scale = scale;
    mesh.instanceMatrix.needsUpdate = true;
    // the canvas renders on demand: the effect runs after React's commit, so request the frame that shows these
    // matrices (without it the last trajectory/engine frame could stay unpainted until the next interaction)
    invalidate();
    // Correct frustum culling: derive bounds from actual instance placements.
    mesh.computeBoundingSphere();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, radiusKey, segments]);

  // --- Colors: only when the mapping changes ---
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const colours = getAtomRenderOverride().colours;
    writeColours(mesh, atoms, config, colours);
    applied.current.colours = colours !== null;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, colorKey, segments]);

  // --- Render overrides (the notebook's Render panel): applied at once, without a React render ---
  useEffect(() => onAtomRenderOverride(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const o = getAtomRenderOverride();
    const { atoms: a, config: c } = latest.current;
    if (o.radiusScale !== applied.current.scale) {
      writeMatrices(mesh, a, c, o.radiusScale);
      applied.current.scale = o.radiusScale;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
    }
    if (o.colours !== null || applied.current.colours) {
      writeColours(mesh, a, c, o.colours);
      applied.current.colours = o.colours !== null;
    }
    invalidate();
  }), [invalidate]);

  const pickingEnabled = atoms.length <= PICKING_MAX_ATOMS && !!(onHover || onSelectAtom);

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!onHover) return;
    e.stopPropagation();
    const idx = e.instanceId;
    if (idx !== undefined && idx !== hoverId) {
      setHoverId(idx);
    }
    onHover(idx !== undefined ? atoms[idx] : null, e.nativeEvent.clientX, e.nativeEvent.clientY);
  };

  const handlePointerOut = () => {
    setHoverId(null);
    onHover?.(null, 0, 0);
  };

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    if (!onSelectAtom) return;
    if (e.delta > 5) return; // orbit drag, not a click
    e.stopPropagation();
    const idx = e.instanceId;
    if (idx !== undefined) onSelectAtom(atoms[idx].id);
  };

  if (atoms.length === 0) return null;

  const pointerProps = pickingEnabled
    ? {
        onPointerMove: handlePointerMove,
        onPointerOut: handlePointerOut,
        onClick: handleClick,
      }
    : {};

  return (
    <instancedMesh
      // a new mesh for a new size or tessellation (the effects above write its matrices and colours)
      key={`${atoms.length}|${segments}`}
      ref={meshRef}
      args={[geometry, undefined, atoms.length]}
      castShadow={config.shadowsEnabled}
      receiveShadow={config.shadowsEnabled}
      {...pointerProps}
    >
      {config.materialType === 'realistic' && (
        <meshPhysicalMaterial
          roughness={0.15}
          metalness={0.05}
          clearcoat={1.0}
          clearcoatRoughness={0.15}
          envMapIntensity={1.1}
        />
      )}
      {config.materialType === 'plastic' && (
        <meshStandardMaterial roughness={0.4} metalness={0.0} envMapIntensity={0.7} />
      )}
      {config.materialType === 'metallic' && (
        <meshStandardMaterial roughness={0.22} metalness={0.9} envMapIntensity={1.3} />
      )}
      {config.materialType === 'toon' && <meshToonMaterial />}
    </instancedMesh>
  );
};

export default InstancedAtomMesh;

import React, { useRef, useMemo, useEffect, useState } from 'react';
import * as THREE from 'three';
import { ThreeEvent } from '@react-three/fiber';
import { Atom, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { atomDisplayRadius } from '../services/atomStyle';
import { writeInstanceTransform } from '../services/instanceMatrix';

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
 * system size. Sphere tessellation adapts to system size:
 *   <=1k atoms -> 32 segs | <=10k -> 20 | >10k -> 12
 *
 * Performance contract:
 *  - Matrices rewrite ONLY when positions or radius-affecting fields change.
 *    Toggling labels/lighting/materials costs zero matrix work.
 *  - Colors rewrite ONLY when the color mapping changes.
 *  - Bounding sphere is recomputed after writes so frustum culling stays
 *    correct (three.js cannot infer instance extents from a unit sphere).
 */
const InstancedAtomMesh: React.FC<InstancedAtomMeshProps> = ({ atoms, config, onHover, onSelectAtom }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);

  // Narrow dependency keys — object identity of `config` must NOT trigger O(n) work.
  const radiusKey = `${config.visualizationMode}|${config.atomScale}`;
  const colorKey = useMemo(
    () =>
      Object.keys(config.customColors)
        .sort((a, b) => Number(a) - Number(b))
        .map(k => `${k}:${config.customColors[Number(k)]}`)
        .join(';'),
    [config.customColors]
  );

  const geometry = useMemo(() => {
    // Tessellation tiers: visual quality is indistinguishable at these
    // densities, while triangle count drops ~4x per tier.
    const baseSegments =
      atoms.length > 30000 ? 8 : atoms.length > 10000 ? 12 : atoms.length > 1000 ? 20 : 32;
    return new THREE.SphereGeometry(1, baseSegments, Math.max(6, Math.round(baseSegments / 2)));
  }, [atoms.length]);

  // R3F disposes what it constructs from `args`, but a useMemo'd geometry that
  // is REPLACED when the tier changes is not R3F's to clean up — without this
  // the old buffers leak on the GPU every time a differently sized structure
  // is loaded.
  useEffect(() => () => geometry.dispose(), [geometry]);

  // --- Matrices: positions + radius inputs only ---
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    // An instance is only ever a translation plus a UNIFORM scale, so the
    // matrix is known in closed form. Writing the 16 floats straight into the
    // instance buffer skips Object3D.updateMatrix()'s quaternion compose and
    // the setMatrixAt copy for every atom — which matters when a trajectory
    // rewrites 60k instances on every frame.
    const m = mesh.instanceMatrix.array as Float32Array;
    for (let i = 0; i < atoms.length; i++) {
      const atom = atoms[i];
      writeInstanceTransform(
        m, i, atom.x, atom.y, atom.z, atomDisplayRadius(atom, config),
      );
    }
    mesh.instanceMatrix.needsUpdate = true;
    // Correct frustum culling: derive bounds from actual instance placements.
    mesh.computeBoundingSphere();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, radiusKey]);

  // --- Colors: only when the mapping changes ---
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    // Colour depends only on atom TYPE, of which there are a handful. The
    // previous version called color.set(hexString) once per ATOM, re-parsing
    // the same few strings tens of thousands of times per trajectory frame.
    const byType = new Map<number, THREE.Color>();
    const colorFor = (type: number): THREE.Color => {
      let c = byType.get(type);
      if (!c) {
        c = new THREE.Color(config.customColors[type] || DEFAULT_ATOM_COLOR);
        byType.set(type, c);
      }
      return c;
    };
    for (let i = 0; i < atoms.length; i++) {
      mesh.setColorAt(i, colorFor(atoms[i].type));
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atoms, colorKey]);

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
      key={atoms.length}
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

import React, { useRef, useMemo, useEffect, useState } from 'react';
import * as THREE from 'three';
import { ThreeEvent } from '@react-three/fiber';
import { Atom, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { atomDisplayRadius } from '../services/atomStyle';

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

  // --- Matrices: positions + radius inputs only ---
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const dummy = new THREE.Object3D();
    for (let i = 0; i < atoms.length; i++) {
      const atom = atoms[i];
      dummy.position.set(atom.x, atom.y, atom.z);
      dummy.scale.setScalar(atomDisplayRadius(atom, config));
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
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
    const color = new THREE.Color();
    for (let i = 0; i < atoms.length; i++) {
      color.set(config.customColors[atoms[i].type] || DEFAULT_ATOM_COLOR);
      mesh.setColorAt(i, color);
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

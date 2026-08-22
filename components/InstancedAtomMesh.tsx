import React, { useRef, useMemo, useEffect, useState } from 'react';
import * as THREE from 'three';
import { ThreeEvent } from '@react-three/fiber';
import { Atom, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR, ELEMENT_RADII, ELEMENT_DATA } from '../constants';

interface InstancedAtomMeshProps {
  atoms: Atom[];
  config: VisualizationConfig;
  onHover?: (atom: Atom | null, screenX: number, screenY: number) => void;
  /** Fired on genuine clicks (not orbit drags) for the measurement tool. */
  onSelectAtom?: (id: number) => void;
}

/**
 * Optimized instanced rendering for atoms — one draw call regardless of
 * system size. Sphere tessellation adapts to system size:
 *   <=1k atoms -> 32 segs | <=10k -> 20 | >10k -> 12
 */
const InstancedAtomMesh: React.FC<InstancedAtomMeshProps> = ({ atoms, config, onHover, onSelectAtom }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);

  const geometry = useMemo(() => {
    const baseSegments =
      atoms.length > 10000 ? 12 : atoms.length > 1000 ? 20 : 32;
    return new THREE.SphereGeometry(1, baseSegments, Math.max(8, baseSegments / 2));
  }, [atoms.length]);

  const getAtomRadius = (atom: Atom): number => {
    if (config.visualizationMode === 'space-fill') {
      // van der Waals radius scaled down slightly so molecules stay readable
      const vdwRadius = ELEMENT_RADII[atom.type] ?? 1.7;
      return vdwRadius * config.atomScale * 0.5;
    }
    if (config.visualizationMode === 'wireframe') {
      return 0.16 * config.atomScale;
    }
    if (config.visualizationMode === 'licorice') {
      return 0.14 * config.atomScale;
    }
    // ball-and-stick
    return 0.45 * config.atomScale;
  };

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < atoms.length; i++) {
      const atom = atoms[i];
      dummy.position.set(atom.x, atom.y, atom.z);
      dummy.scale.setScalar(getAtomRadius(atom));
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      color.set(config.customColors[atom.type] || DEFAULT_ATOM_COLOR);
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [atoms, config]); // eslint-disable-line react-hooks/exhaustive-deps

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

  return (
    <instancedMesh
      key={atoms.length}
      ref={meshRef}
      args={[geometry, undefined, atoms.length]}
      castShadow={config.shadowsEnabled}
      receiveShadow={config.shadowsEnabled}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
      onClick={handleClick}
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

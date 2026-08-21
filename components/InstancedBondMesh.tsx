import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { Atom, Bond, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';

interface InstancedBondMeshProps {
  bonds: Bond[];
  atomMap: Map<number, Atom>;
  config: VisualizationConfig;
  /** Bonds longer than this are skipped (periodic wrap-around guard). */
  maxBondLength?: number;
}

const UP = new THREE.Vector3(0, 1, 0);

/**
 * Renders ALL bonds in a single THREE.InstancedMesh draw call.
 * Each bond is drawn as two half-cylinders (atom1->mid colored by atom1,
 * mid->atom2 colored by atom2) — the standard Jmol/VMD look — using
 * per-instance colors. Instance count = 2 * bonds.length.
 */
const InstancedBondMesh: React.FC<InstancedBondMeshProps> = ({
  bonds,
  atomMap,
  config,
  maxBondLength = Infinity,
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const radius = Math.max(0.02, 0.12 * config.bondScale);

  // Precompute per-bond transforms; skip degenerate/oversized bonds.
  const instances = useMemo(() => {
    const list: {
      position: THREE.Vector3;
      quaternion: THREE.Quaternion;
      length: number;
      color: string;
    }[] = [];

    for (const bond of bonds) {
      const a1 = atomMap.get(bond.atom1Id);
      const a2 = atomMap.get(bond.atom2Id);
      if (!a1 || !a2) continue;

      const start = new THREE.Vector3(a1.x, a1.y, a1.z);
      const end = new THREE.Vector3(a2.x, a2.y, a2.z);
      const length = start.distanceTo(end);
      if (length < 1e-6 || length > maxBondLength) continue;

      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const dir = new THREE.Vector3().subVectors(end, start).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(UP, dir);

      const c1 = config.customColors[a1.type] ?? DEFAULT_ATOM_COLOR;
      const c2 = config.customColors[a2.type] ?? DEFAULT_ATOM_COLOR;

      list.push({ position: start.clone().lerp(mid, 0.5), quaternion: q, length: length / 2, color: c1 });
      list.push({ position: mid.clone().lerp(end, 0.5), quaternion: q, length: length / 2, color: c2 });
    }
    return list;
  }, [bonds, atomMap, config.customColors, config.bondScale, maxBondLength]);

  const geometry = useMemo(
    () => new THREE.CylinderGeometry(1, 1, 1, Math.min(16, Math.max(6, 18 - Math.floor(bonds.length / 2000)))),
    [bonds.length]
  );

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < instances.length; i++) {
      const inst = instances[i];
      dummy.position.copy(inst.position);
      dummy.quaternion.copy(inst.quaternion);
      dummy.scale.set(radius, inst.length, radius);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      color.set(inst.color);
      mesh.setColorAt(i, color);
    }
    mesh.count = instances.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [instances, radius]);

  if (instances.length === 0) return null;

  return (
    <instancedMesh
      key={instances.length}
      ref={meshRef}
      args={[geometry, undefined, instances.length]}
    >
      {config.materialType === 'realistic' && (
        <meshPhysicalMaterial roughness={0.35} metalness={0.05} envMapIntensity={0.9} />
      )}
      {config.materialType === 'plastic' && (
        <meshStandardMaterial roughness={0.55} metalness={0.0} />
      )}
      {config.materialType === 'metallic' && (
        <meshStandardMaterial roughness={0.25} metalness={0.85} envMapIntensity={1.2} />
      )}
      {config.materialType === 'toon' && <meshToonMaterial />}
    </instancedMesh>
  );
};

export default InstancedBondMesh;

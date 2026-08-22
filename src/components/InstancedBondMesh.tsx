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

interface BondSegment {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  halfLength: number;
  color: string;
}

/**
 * Renders ALL bonds in a single THREE.InstancedMesh draw call.
 * Each bond is drawn as two half-cylinders (atom1->mid colored by atom1,
 * mid->atom2 colored by atom2) using per-instance colors.
 *
 * Performance contract: expensive position/quaternion math runs only when
 * topology inputs change; moving the thickness slider only rewrites
 * matrices from cached segments (no trig), and colors only when the
 * element color mapping changes.
 */
const InstancedBondMesh: React.FC<InstancedBondMeshProps> = ({
  bonds,
  atomMap,
  config,
  maxBondLength = Infinity,
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const radius = Math.max(0.02, 0.12 * config.bondScale);

  // Expensive pass: world positions/orientations per half-bond.
  const segments = useMemo<BondSegment[]>(() => {
    const list: BondSegment[] = [];
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

      list.push({ position: start.clone().lerp(mid, 0.5), quaternion: q, halfLength: length / 2, color: c1 });
      list.push({ position: mid.clone().lerp(end, 0.5), quaternion: q.clone(), halfLength: length / 2, color: c2 });
    }
    return list;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bonds, atomMap, maxBondLength]);

  // Tessellation adapts to system size (fewer segments for huge systems).
  const geometry = useMemo(
    () =>
      new THREE.CylinderGeometry(
        1, 1, 1,
        Math.min(16, Math.max(6, 18 - Math.floor(bonds.length / 2000)))
      ),
    [bonds.length]
  );

  // Matrices: cheap rewrite from cache when only thickness changes.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = new THREE.Object3D();
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      dummy.position.copy(seg.position);
      dummy.quaternion.copy(seg.quaternion);
      dummy.scale.set(radius, seg.halfLength, radius);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.count = segments.length;
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
  }, [segments, radius]);

  // Per-instance colors.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const color = new THREE.Color();
    for (let i = 0; i < segments.length; i++) {
      color.set(segments[i].color);
      mesh.setColorAt(i, color);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [segments]);

  if (segments.length === 0) return null;

  return (
    <instancedMesh
      key={segments.length}
      ref={meshRef}
      args={[geometry, undefined, segments.length]}
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

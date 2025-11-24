
import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Atom, VisualizationConfig } from '../types';

interface BondMeshProps {
  startAtom: Atom;
  endAtom: Atom;
  config: VisualizationConfig;
}

const BondMesh: React.FC<BondMeshProps> = ({ startAtom, endAtom, config }) => {
  const { position, quaternion, length } = useMemo(() => {
    const start = new THREE.Vector3(startAtom.x, startAtom.y, startAtom.z);
    const end = new THREE.Vector3(endAtom.x, endAtom.y, endAtom.z);

    const length = start.distanceTo(end);
    
    // Midpoint for position
    const position = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

    // Orientation
    const direction = new THREE.Vector3().subVectors(end, start).normalize();
    const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

    return { position, quaternion, length };
  }, [startAtom, endAtom]);

  // Don't render extremely long bonds
  if (length > 5) return null;

  // Base bond radius is 0.12, scaled by config
  const radius = 0.12 * config.bondScale;

  return (
    <mesh position={position} quaternion={quaternion} castShadow receiveShadow>
      <cylinderGeometry args={[radius, radius, length, 16, 1]} />
      {config.materialType === 'realistic' && (
        <meshPhysicalMaterial 
          color="#F0F0F0" 
          roughness={0.2} 
          metalness={0.1} 
          clearcoat={0.8}
          envMapIntensity={1.2}
        />
      )}
      {config.materialType === 'plastic' && (
        <meshStandardMaterial 
          color="#E0E0E0" 
          roughness={0.5} 
          metalness={0.0}
        />
      )}
      {config.materialType === 'toon' && (
        <meshToonMaterial 
          color="#FFFFFF" 
        />
      )}
    </mesh>
  );
};

export default BondMesh;

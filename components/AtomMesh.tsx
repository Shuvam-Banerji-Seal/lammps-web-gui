
import React, { useMemo } from 'react';
import { Atom, VisualizationConfig } from '../types';
import { ATOM_COLORS, DEFAULT_ATOM_COLOR } from '../constants';
import * as THREE from 'three';

interface AtomMeshProps {
  atom: Atom;
  config: VisualizationConfig;
}

const AtomMesh: React.FC<AtomMeshProps> = ({ atom, config }) => {
  // Priority: Custom Config > Type-based Color > Default
  const color = config.customColors[atom.type] || ATOM_COLORS[atom.type] || DEFAULT_ATOM_COLOR;
  
  // Base radius is 0.45, scaled by config
  const radius = 0.45 * config.atomScale; 

  const geometry = useMemo(() => new THREE.SphereGeometry(radius, 32, 32), [radius]);

  return (
    <mesh position={[atom.x, atom.y, atom.z]} geometry={geometry} castShadow receiveShadow>
      {config.materialType === 'realistic' && (
        <meshPhysicalMaterial 
          color={color} 
          roughness={0.15} 
          metalness={0.2}
          clearcoat={1.0}
          clearcoatRoughness={0.1}
          reflectivity={1.0}
          envMapIntensity={1.5}
        />
      )}
      {config.materialType === 'plastic' && (
        <meshStandardMaterial 
          color={color} 
          roughness={0.4} 
          metalness={0.0}
          envMapIntensity={0.8}
        />
      )}
      {config.materialType === 'toon' && (
        <meshToonMaterial 
          color={color} 
          gradientMap={null}
        />
      )}
    </mesh>
  );
};

export default AtomMesh;
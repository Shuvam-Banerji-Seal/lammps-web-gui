import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { Atom, VisualizationConfig } from '../types';
import { ATOM_COLORS, DEFAULT_ATOM_COLOR, ELEMENT_RADII, ELEMENT_DATA } from '../constants';

interface InstancedAtomMeshProps {
  atoms: Atom[];
  config: VisualizationConfig;
}

/**
 * Optimized instanced rendering for atoms. Uses THREE.InstancedMesh
 * for dramatically better performance with large datasets.
 */
const InstancedAtomMesh: React.FC<InstancedAtomMeshProps> = ({ atoms, config }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tempObject = useMemo(() => new THREE.Object3D(), []);
  const tempColor = useMemo(() => new THREE.Color(), []);

  // Group atoms by color to batch materials
  const geometry = useMemo(() => {
    const baseSegments = atoms.length > 500 ? 16 : 32;
    return new THREE.SphereGeometry(1, baseSegments, baseSegments);
  }, [atoms.length]);

  // Compute scale for each atom based on visualization mode
  const getAtomRadius = (atom: Atom): number => {
    if (config.visualizationMode === 'space-fill') {
      // Use van der Waals radius
      const elem = ELEMENT_DATA.find(e => e.number === atom.type);
      const atomicNum = elem?.number ?? atom.type;
      const vdwRadius = ELEMENT_RADII[atomicNum] ?? 1.7;
      return vdwRadius * config.atomScale * 0.5;
    }
    if (config.visualizationMode === 'wireframe') {
      return 0.2 * config.atomScale;
    }
    // ball-and-stick
    return 0.45 * config.atomScale;
  };

  const getAtomColor = (atom: Atom): string => {
    return config.customColors[atom.type] || ATOM_COLORS[atom.type] || DEFAULT_ATOM_COLOR;
  };

  useEffect(() => {
    if (!meshRef.current) return;

    const mesh = meshRef.current;

    for (let i = 0; i < atoms.length; i++) {
      const atom = atoms[i];
      const radius = getAtomRadius(atom);

      tempObject.position.set(atom.x, atom.y, atom.z);
      tempObject.scale.setScalar(radius);
      tempObject.updateMatrix();
      mesh.setMatrixAt(i, tempObject.matrix);

      tempColor.set(getAtomColor(atom));
      mesh.setColorAt(i, tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [atoms, config]);

  if (atoms.length === 0) return null;

  return (
    <instancedMesh ref={meshRef} args={[geometry, undefined, atoms.length]} castShadow receiveShadow>
      {config.materialType === 'realistic' && (
        <meshPhysicalMaterial
          vertexColors
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
          vertexColors
          roughness={0.4}
          metalness={0.0}
          envMapIntensity={0.8}
        />
      )}
      {config.materialType === 'toon' && (
        <meshToonMaterial
          vertexColors
        />
      )}
    </instancedMesh>
  );
};

export default InstancedAtomMesh;

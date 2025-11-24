
import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, ContactShadows, Environment } from '@react-three/drei';
import { MoleculeData, VisualizationConfig } from '../types';
import AtomMesh from './AtomMesh';
import BondMesh from './BondMesh';
import * as THREE from 'three';

interface MoleculeCanvasProps {
  data: MoleculeData;
  autoRotate: boolean;
  config: VisualizationConfig;
}

const MoleculeCanvas: React.FC<MoleculeCanvasProps> = ({ data, autoRotate, config }) => {
  const { atoms, bonds, center } = data;

  // Create a map for fast atom lookup by ID when building bonds
  const atomMap = useMemo(() => {
    const map = new Map<number, typeof atoms[0]>();
    atoms.forEach(atom => map.set(atom.id, atom));
    return map;
  }, [atoms]);

  // Center the scene
  const groupPosition = useMemo(() => {
    return new THREE.Vector3(-center.x, -center.y, -center.z);
  }, [center]);

  return (
    <Canvas
      shadows
      camera={{ position: [0, 0, 20], fov: 40 }}
      dpr={[1, 2]}
      className="w-full h-full"
    >
      {/* Dynamic Background Color */}
      <color attach="background" args={[config.backgroundColor]} />
      
      {/* Environment lighting - Use studio for realistic/plastic, maybe simpler for toon but studio works well */}
      <Environment preset="studio" />

      {/* Lighting Setup */}
      <ambientLight intensity={0.6} />
      <directionalLight 
        position={[10, 10, 10]} 
        intensity={config.materialType === 'toon' ? 1.0 : 1.5} 
        castShadow 
        shadow-bias={-0.0001}
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <spotLight position={[-10, 0, -10]} intensity={1.5} angle={0.5} penumbra={1} color="#4040ff" />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#ffffff" />

      <group position={groupPosition}>
        {atoms.map((atom) => (
          <AtomMesh key={`atom-${atom.id}`} atom={atom} config={config} />
        ))}

        {config.showBonds && bonds.map((bond) => {
          const atom1 = atomMap.get(bond.atom1Id);
          const atom2 = atomMap.get(bond.atom2Id);
          
          if (atom1 && atom2) {
            return <BondMesh key={`bond-${bond.id}`} startAtom={atom1} endAtom={atom2} config={config} />;
          }
          return null;
        })}
      </group>

      <ContactShadows 
        opacity={0.4} 
        scale={50} 
        blur={2} 
        far={10} 
        resolution={512} 
        color="#000000" 
      />
      
      <OrbitControls autoRotate={autoRotate} autoRotateSpeed={0.5} makeDefault />
    </Canvas>
  );
};

export default MoleculeCanvas;

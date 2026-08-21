import React, { useMemo, useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import * as THREE from 'three';
import { MoleculeData, VisualizationConfig, Atom } from '../types';
import { ELEMENT_DATA } from '../constants';
import InstancedAtomMesh from './InstancedAtomMesh';
import InstancedBondMesh from './InstancedBondMesh';
import SimulationBox from './SimulationBox';
import LightingRig from './LightingRig';
import CameraRig from './CameraRig';
import AtomLabels from './AtomLabels';

interface MoleculeCanvasProps {
  data: MoleculeData;
  autoRotate: boolean;
  config: VisualizationConfig;
}

interface HoverInfo {
  atom: Atom;
  x: number;
  y: number;
}

/**
 * Scene sizing: prefers the simulation box extent when present (periodic
 * systems often have atoms clustered away from box walls).
 */
const sceneExtent = (data: MoleculeData): { radius: number; center: THREE.Vector3 } => {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  const consider = (x: number, y: number, z: number) => {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
  };

  if (data.box) {
    // include tilted corners
    const lx = data.box.xhi - data.box.xlo;
    const ly = data.box.yhi - data.box.ylo;
    const lz = data.box.zhi - data.box.zlo;
    const xy = data.box.xy ?? 0, xz = data.box.xz ?? 0, yz = data.box.yz ?? 0;
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++)
        for (let k = 0; k < 2; k++) {
          consider(
            data.box.xlo + i * lx + j * xy + k * xz,
            data.box.ylo + j * ly + k * yz,
            data.box.zlo + k * lz
          );
        }
  }
  for (const a of data.atoms) consider(a.x, a.y, a.z);

  if (!Number.isFinite(minX)) {
    return { radius: 10, center: new THREE.Vector3() };
  }

  const size = new THREE.Vector3(maxX - minX, maxY - minY, maxZ - minZ);
  const center = new THREE.Vector3((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
  return { radius: Math.max(size.length() / 2, 5), center };
};

const MoleculeCanvas: React.FC<MoleculeCanvasProps> = ({ data, autoRotate, config }) => {
  const [hover, setHover] = useState<HoverInfo | null>(null);

  const { atoms, bonds } = data;

  const atomMap = useMemo(() => {
    const map = new Map<number, Atom>();
    atoms.forEach(atom => map.set(atom.id, atom));
    return map;
  }, [atoms]);

  // Center the molecule group at the origin so camera math is trivial.
  const groupPosition = useMemo(
    () => new THREE.Vector3(-data.center.x, -data.center.y, -data.center.z),
    [data.center]
  );

  const { radius: boundingRadius } = useMemo(() => sceneExtent(data), [data]);

  // Periodic-image guard: drop explicit bonds that span nearly the whole cell.
  const maxBondLength = useMemo(() => {
    if (!data.box) return 8;
    const lx = data.box.xhi - data.box.xlo;
    const ly = data.box.yhi - data.box.ylo;
    const lz = data.box.zhi - data.box.zlo;
    return Math.max(lx, ly, lz) * 0.9;
  }, [data.box]);

  // Adaptive device pixel ratio — huge systems render below native res.
  const dpr: [number, number] = useMemo(() => {
    if (atoms.length > 20000) return [0.75, 1.25];
    if (atoms.length > 5000) return [1, 1.5];
    return [1, 2];
  }, [atoms.length]);

  const shadowsEnabled = config.shadowsEnabled && atoms.length <= 8000;

  const shouldShowBonds =
    config.showBonds &&
    config.visualizationMode !== 'space-fill' &&
    bonds.length > 0;

  const handleHover = useCallback(
    (atom: Atom | null, screenX: number, screenY: number) => {
      setHover(atom ? { atom, x: screenX, y: screenY } : null);
    },
    []
  );

  const hoverMeta = hover
    ? ELEMENT_DATA.find(e => e.number === hover.atom.type)
    : undefined;

  return (
    <>
      <Canvas
        shadows={shadowsEnabled}
        camera={{ position: [boundingRadius, boundingRadius * 0.8, boundingRadius], fov: config.fov }}
        dpr={dpr}
        className="w-full h-full outline-none"
        gl={{
          antialias: atoms.length <= 20000,
          alpha: false,
          powerPreference: 'high-performance',
          stencil: false,
          preserveDrawingBuffer: true, // reliable canvas.toDataURL screenshots
        }}
      >
        <color attach="background" args={[config.backgroundColor]} />

        <LightingRig preset={config.lightingPreset} shadows={shadowsEnabled} />

        <group position={groupPosition}>
          {config.showAxes && <axesHelper args={[boundingRadius * 1.2]} />}
          <InstancedAtomMesh atoms={atoms} config={config} onHover={handleHover} />
          {shouldShowBonds && (
            <InstancedBondMesh
              bonds={bonds}
              atomMap={atomMap}
              config={config}
              maxBondLength={maxBondLength}
            />
          )}
          {config.showBox && data.box && (
            <SimulationBox box={data.box} showFaces={false} />
          )}
          <AtomLabels atoms={atoms} config={config} />
        </group>

        <CameraRig
          boundingRadius={boundingRadius}
          autoRotate={autoRotate}
          autoRotateSpeed={config.autoRotateSpeed}
          fov={config.fov}
        />
      </Canvas>

      {/* Hover tooltip — DOM overlay, no in-scene text cost */}
      {hover && (
        <div
          className="pointer-events-none fixed z-50 px-3 py-2 rounded-lg border border-gray-600/60 bg-gray-900/95 text-gray-100 text-xs shadow-xl"
          style={{ left: hover.x + 14, top: hover.y + 14 }}
          role="status"
        >
          <div className="font-bold">
            {hoverMeta ? `${hoverMeta.name} (${hoverMeta.symbol})` : `Type ${hover.atom.type}`}
          </div>
          <div className="opacity-70 font-mono">
            #{hover.atom.id} · Z={hover.atom.type} · {hover.atom.charge !== 0 ? `q=${hover.atom.charge.toFixed(2)} · ` : ''}
            {`(${hover.atom.x.toFixed(2)}, ${hover.atom.y.toFixed(2)}, ${hover.atom.z.toFixed(2)})`}
          </div>
        </div>
      )}
    </>
  );
};

export default MoleculeCanvas;

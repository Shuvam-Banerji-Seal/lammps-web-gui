import React, { useMemo, useState, useCallback, useRef, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { PerformanceMonitor } from '@react-three/drei';
import * as THREE from 'three';
import { MoleculeData, VisualizationConfig, Atom, BoxBounds } from '../types';
import { ELEMENT_DATA } from '../constants';
import { measureSelection, MeasurementResult } from '../services/measure';
import { getActiveGL, registerActiveGL } from '../services/glRegistry';
import InstancedAtomMesh from './InstancedAtomMesh';
import PointAtoms from './PointAtoms';
import InstancedBondMesh from './InstancedBondMesh';
import BondLines from './BondLines';
import SimulationBox from './SimulationBox';
import LightingRig from './LightingRig';
import CameraRig from './CameraRig';
import AtomLabels from './AtomLabels';
import MeasurementOverlay from './MeasurementOverlay';
import { CanvasErrorBoundary } from './CanvasErrorBoundary';
import SceneEffects from './SceneEffects';
import { qualityDpr, shadowLimit, shadowMapSize, meshMode, cylinderBondMode } from '../services/atomStyle';
import { COLOR_BY_LABELS, colorMapGradient, isContinuous, scalarRange } from '../services/colorMaps';

interface MoleculeCanvasProps {
  data: MoleculeData;
  autoRotate: boolean;
  config: VisualizationConfig;
  /** Atom ids picked by the measurement tool. */
  selectedIds?: number[];
  onSelectAtom?: (id: number) => void;
  /** Video recording: keep frames flowing even when idle. */
  forceContinuousRender?: boolean;
  /**
   * Cell to DRAW, when the current trajectory frame has its own (NPT).
   * Camera framing deliberately stays on `data.box` — keying it to a
   * per-frame box would make the view distance pump during playback.
   */
  displayBox?: BoxBounds;
  /**
   * Periodic cell for measurements (minimum image), passed only for formats
   * whose coordinates are wrapped into a periodic cell.
   */
  measureBox?: BoxBounds;
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

const MoleculeCanvas: React.FC<MoleculeCanvasProps> = ({
  data,
  autoRotate,
  config: viewConfig,
  selectedIds = [],
  onSelectAtom,
  forceContinuousRender = false,
  displayBox,
  measureBox,
}) => {
  const [hover, setHover] = useState<HoverInfo | null>(null);
  // The GL registry must not pin this renderer (and its scene) after unmount.
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  useEffect(() => () => {
    if (glRef.current && getActiveGL()?.gl === glRef.current) registerActiveGL(null);
  }, []);
  // Adaptive quality: PerformanceMonitor lowers this when FPS dips (P6).
  const [perfFactor, setPerfFactor] = useState(1);

  const { atoms, bonds } = data;

  // a colour map needs its range: the caller's (the viewer resolves it over all frames), else this frame's
  const config = useMemo((): VisualizationConfig => {
    const by = viewConfig.colorBy;
    if (!isContinuous(by) || (viewConfig.colorMin !== undefined && viewConfig.colorMax !== undefined)) return viewConfig;
    const r = scalarRange([atoms], by!) ?? [0, 1];
    return { ...viewConfig, colorMin: r[0], colorMax: r[1] };
  }, [viewConfig, atoms]);
  const quality = config.quality;

  // Only bond rendering and the measurement tool look atoms up by id.
  // A LAMMPS dump has no bonds, so building a Map of every atom on every
  // playback frame was pure waste — 60k inserts per frame at 30 fps.
  const needsAtomMap = bonds.length > 0 || selectedIds.length > 0;
  const atomMap = useMemo(() => {
    const map = new Map<number, Atom>();
    if (!needsAtomMap) return map;
    for (let i = 0; i < atoms.length; i++) map.set(atoms[i].id, atoms[i]);
    return map;
  }, [atoms, needsAtomMap]);

  // Center the molecule group at the origin so camera math is trivial.
  const groupPosition = useMemo(
    () => new THREE.Vector3(-data.center.x, -data.center.y, -data.center.z),
    [data.center]
  );

  /**
   * Camera framing radius.
   *
   * Keyed on the BOX and the atom count rather than on `data` identity: during
   * trajectory playback `data` is a new object every frame, so this used to
   * rescan every atom per frame AND hand CameraRig a slightly different radius
   * each time, which made the camera distance visibly pump. A trajectory's
   * framing should come from its cell, which does not move.
   */
  const extentKey = data.box
    ? `box:${data.box.xlo},${data.box.xhi},${data.box.ylo},${data.box.yhi},${data.box.zlo},${data.box.zhi}`
    : `atoms:${atoms.length}`;
  const boundingRadius = useMemo(
    () => sceneExtent(data).radius,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [extentKey],
  );

  // Periodic-image guard: drop explicit bonds that span nearly the whole cell.
  const maxBondLength = useMemo(() => {
    if (!data.box) return 8;
    const lx = data.box.xhi - data.box.xlo;
    const ly = data.box.yhi - data.box.ylo;
    const lz = data.box.zhi - data.box.zlo;
    return Math.max(lx, ly, lz) * 0.9;
  }, [data.box]);

  // Device pixel ratio from the quality setting — 'auto' renders huge systems below native res, 'ultra'
  // supersamples — and PerformanceMonitor can degrade 'auto' and 'low' further on weak GPUs (P6).
  const dpr: [number, number] = useMemo(() => qualityDpr(quality, atoms.length), [quality, atoms.length]);

  const effectiveDpr: [number, number] = useMemo(
    () => [
      Math.max(0.5, dpr[0] * perfFactor),
      Math.max(0.5, dpr[1] * perfFactor),
    ],
    [dpr, perfFactor]
  );

  const shadowsEnabled = config.shadowsEnabled && atoms.length <= shadowLimit(quality);

  const shouldShowBonds = config.showBonds && cylinderBondMode(config.visualizationMode) && bonds.length > 0;
  const bondLines = config.showBonds && config.visualizationMode === 'lines' && bonds.length > 0;

  // Hover updates: immediate on atom change, position refresh throttled to
  // ~12 Hz. Returning the previous state object when unchanged lets React
  // bail out of re-renders entirely — critical for frameloop="demand".
  const lastHoverRefresh = useRef(0);
  const handleHover = useCallback(
    (atom: Atom | null, screenX: number, screenY: number) => {
      setHover(prev => {
        const changed = (prev === null) !== (atom === null) ||
          (prev !== null && atom !== null && prev.atom.id !== atom.id);
        const now = performance.now();
        if (!changed && now - lastHoverRefresh.current < 80) return prev;
        lastHoverRefresh.current = now;
        return atom ? { atom, x: screenX, y: screenY } : null;
      });
    },
    []
  );

  const hoverMeta = hover
    ? ELEMENT_DATA.find(e => e.number === hover.atom.type)
    : undefined;

  const selectedAtoms = useMemo(
    () => selectedIds.map(id => atomMap.get(id)).filter((a): a is Atom => !!a),
    [selectedIds, atomMap]
  );

  const measurement: MeasurementResult | null = useMemo(
    () => measureSelection(selectedAtoms, measureBox),
    [selectedAtoms, measureBox]
  );

  // Detect WebGL availability ONCE before mounting — prevents R3F from
  // flooding the console with retry errors in GPU-less environments.
  const webglAvailable = useMemo(() => {
    try {
      const c = document.createElement('canvas');
      return !!(c.getContext('webgl2') || c.getContext('webgl'));
    } catch {
      return false;
    }
  }, []);

  if (!webglAvailable) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-8 text-center">
        <div className="text-3xl opacity-40">🔬</div>
        <p className="text-sm font-medium text-gray-400">3D view unavailable</p>
        <p className="max-w-xs text-xs leading-relaxed text-gray-500">
          WebGL could not be initialised. The Script Builder and Compiler
          Helper remain fully functional. Try enabling hardware acceleration
          or closing other GPU-heavy tabs.
        </p>
      </div>
    );
  }

  return (
    <CanvasErrorBoundary>
      <Canvas
        shadows={shadowsEnabled ? 'percentage' : false}
        camera={{ position: [boundingRadius, boundingRadius * 0.8, boundingRadius], fov: config.fov }}
        dpr={effectiveDpr}
        className="w-full h-full outline-none"
        // Demand mode = zero idle GPU/CPU burn; OrbitControls' onChange
        // invalidates during interaction and damping settles naturally.
        // Recording forces continuous frames so captured video has motion.
        frameloop={autoRotate || forceContinuousRender ? 'always' : 'demand'}
        onCreated={({ gl, scene, camera }) => { glRef.current = gl; registerActiveGL({ gl, scene, camera }); }}
        gl={{
          antialias: atoms.length <= 20000,
          alpha: false,
          powerPreference: 'high-performance',
          stencil: false,
          // NOTE: preserveDrawingBuffer intentionally OFF — screenshots use
          // an explicit render-before-capture via the GL registry instead.
        }}
      >
        <PerformanceMonitor
          onDecline={() => setPerfFactor(quality === 'high' || quality === 'ultra' ? 1 : 0.6)}
          onIncline={() => setPerfFactor(1)}
          flipflops={3}
        />
        <color attach="background" args={[config.backgroundColor]} />
        <SceneEffects
          toneMapping={config.toneMapping ?? 'aces'}
          exposure={config.exposure ?? 1}
          envLighting={!!config.envLighting}
          depthCue={config.depthCue ?? 0}
          background={config.backgroundColor}
          radius={boundingRadius}
        />

        <LightingRig
          preset={config.lightingPreset}
          shadows={shadowsEnabled}
          radius={boundingRadius}
          shadowMapSize={shadowMapSize(quality)}
        />

        <group position={groupPosition}>
          {config.showAxes && <axesHelper args={[boundingRadius * 1.2]} />}
          {!meshMode(config.visualizationMode) ? (
            <PointAtoms atoms={atoms} config={config} flat={config.visualizationMode === 'flat'} />
          ) : (
            <InstancedAtomMesh
              atoms={atoms}
              config={config}
              onHover={handleHover}
              onSelectAtom={onSelectAtom}
            />
          )}
          {bondLines && <BondLines bonds={bonds} atomMap={atomMap} config={config} maxBondLength={maxBondLength} />}
          {shouldShowBonds && (
            <InstancedBondMesh
              bonds={bonds}
              atomMap={atomMap}
              config={config}
              maxBondLength={maxBondLength}
            />
          )}
          {config.showBox && (displayBox ?? data.box) && (
            <SimulationBox box={(displayBox ?? data.box)!} showFaces={false} color={config.boxColor} />
          )}
          <AtomLabels atoms={atoms} config={config} />
          <MeasurementOverlay selected={selectedAtoms} config={config} result={measurement} />
        </group>

        <CameraRig
          boundingRadius={boundingRadius}
          autoRotate={autoRotate}
          autoRotateSpeed={config.autoRotateSpeed}
          fov={config.fov}
        />
      </Canvas>

      {/* Colour-map legend — DOM overlay */}
      {isContinuous(config.colorBy) && (
        <div
          className="pointer-events-none absolute bottom-3 left-3 z-10 w-44 rounded-lg border border-gray-600/50 bg-gray-900/80 px-2.5 py-1.5 text-[10px] text-gray-100"
          aria-label="Colour legend"
        >
          <div className="mb-1 font-semibold">{COLOR_BY_LABELS[config.colorBy!]}</div>
          <div className="h-2.5 rounded" style={{ background: colorMapGradient(config.colorMap ?? 'viridis', !!config.colorReverse) }} />
          <div className="mt-0.5 flex justify-between font-mono">
            <span>{Number(config.colorMin).toPrecision(4)}</span>
            <span>{Number(config.colorMax).toPrecision(4)}</span>
          </div>
        </div>
      )}

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
    </CanvasErrorBoundary>
  );
};

export default MoleculeCanvas;

import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { Atom, Bond, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { writeInstanceSegment } from '../services/instanceMatrix';
import { bondSegments } from '../services/atomStyle';
import { colorKeyOf, hexToRgb, makeAtomColorer } from '../services/colorMaps';

interface InstancedBondMeshProps {
  bonds: Bond[];
  atomMap: Map<number, Atom>;
  config: VisualizationConfig;
  /** Bonds longer than this are skipped (periodic wrap-around guard). */
  maxBondLength?: number;
}

/**
 * Half-bond endpoints in flat typed arrays.
 *
 * The previous version allocated two THREE.Vector3 plus a THREE.Quaternion
 * for every half-bond — 300k short-lived objects for a 50k-bond structure,
 * rebuilt whenever `atomMap` changed identity (i.e. every frame of a
 * trajectory that carries bonds). Typed arrays make the rebuild allocation-
 * free apart from the buffers themselves.
 */
interface BondBuffers {
  count: number;
  ax: Float32Array; ay: Float32Array; az: Float32Array;
  bx: Float32Array; by: Float32Array; bz: Float32Array;
  /** The atom whose colour this half-bond takes. */
  atom: Atom[];
}

const EMPTY: BondBuffers = {
  count: 0,
  ax: new Float32Array(0), ay: new Float32Array(0), az: new Float32Array(0),
  bx: new Float32Array(0), by: new Float32Array(0), bz: new Float32Array(0),
  atom: [],
};

/**
 * Renders ALL bonds in a single THREE.InstancedMesh draw call.
 * Each bond is drawn as two half-cylinders (atom1->mid coloured by atom1,
 * mid->atom2 coloured by atom2, in the view's atom colouring) using
 * per-instance colours, or all in config.bondColor when that is set.
 *
 * Performance contract: endpoint extraction runs only when topology inputs
 * change; moving the thickness slider only rewrites matrices from the cached
 * endpoints, and colours only when the element colour mapping changes.
 */
const InstancedBondMesh: React.FC<InstancedBondMeshProps> = ({
  bonds,
  atomMap,
  config,
  maxBondLength = Infinity,
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const invalidate = useThree((s) => s.invalidate);
  const radius = Math.max(0.02, 0.12 * config.bondScale);

  // Narrow colour dependency: identity of `config` must not rebuild buffers.
  const colorKey = config.bondColor ? `bond:${config.bondColor}` : colorKeyOf(config);

  const buf = useMemo<BondBuffers>(() => {
    if (bonds.length === 0) return EMPTY;
    const cap = bonds.length * 2;
    const ax = new Float32Array(cap), ay = new Float32Array(cap), az = new Float32Array(cap);
    const bx = new Float32Array(cap), by = new Float32Array(cap), bz = new Float32Array(cap);
    const atom: Atom[] = new Array(cap);

    let n = 0;
    for (let i = 0; i < bonds.length; i++) {
      const bond = bonds[i];
      const a1 = atomMap.get(bond.atom1Id);
      const a2 = atomMap.get(bond.atom2Id);
      if (!a1 || !a2) continue;

      const dx = a2.x - a1.x, dy = a2.y - a1.y, dz = a2.z - a1.z;
      const len = Math.hypot(dx, dy, dz);
      if (len < 1e-6 || len > maxBondLength) continue;

      const mx = (a1.x + a2.x) / 2, my = (a1.y + a2.y) / 2, mz = (a1.z + a2.z) / 2;

      ax[n] = a1.x; ay[n] = a1.y; az[n] = a1.z;
      bx[n] = mx;   by[n] = my;   bz[n] = mz;
      atom[n] = a1;
      n++;

      ax[n] = mx;   ay[n] = my;   az[n] = mz;
      bx[n] = a2.x; by[n] = a2.y; bz[n] = a2.z;
      atom[n] = a2;
      n++;
    }
    return { count: n, ax, ay, az, bx, by, bz, atom };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bonds, atomMap, maxBondLength]);

  // Tessellation follows the quality setting and adapts to system size (fewer segments for huge systems).
  const segments = bondSegments(config.quality, bonds.length);
  const geometry = useMemo(() => new THREE.CylinderGeometry(1, 1, 1, segments), [segments]);

  // A useMemo'd geometry that gets REPLACED is not R3F's to dispose; without
  // this the old buffers leak on the GPU on every differently sized load.
  useEffect(() => () => geometry.dispose(), [geometry]);

  // Matrices: cheap rewrite from the cached endpoints when thickness changes.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const m = mesh.instanceMatrix.array as Float32Array;
    const basis = new Float32Array(9); // scratch, reused for every instance
    for (let i = 0; i < buf.count; i++) {
      writeInstanceSegment(
        m, i,
        buf.ax[i], buf.ay[i], buf.az[i],
        buf.bx[i], buf.by[i], buf.bz[i],
        radius, basis,
      );
    }
    mesh.count = buf.count;
    mesh.instanceMatrix.needsUpdate = true;
    // the canvas renders on demand: the effect runs after React's commit, so request the frame that shows these
    // matrices (without it the last trajectory/engine frame could stay unpainted until the next interaction)
    invalidate();
    mesh.computeBoundingSphere();
  }, [buf, radius, segments]);

  // Per-instance colours: the bond colour, or each half its atom's colour (makeAtomColorer caches type colours).
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const rgb = new Float32Array(3);
    const c = new THREE.Color();
    const colorOf = makeAtomColorer(config, DEFAULT_ATOM_COLOR);
    if (config.bondColor) hexToRgb(config.bondColor, rgb);
    for (let i = 0; i < buf.count; i++) {
      if (!config.bondColor) colorOf(buf.atom[i], rgb);
      c.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buf, colorKey, segments]);

  if (buf.count === 0) return null;

  return (
    <instancedMesh
      key={`${buf.count}|${segments}`}
      ref={meshRef}
      args={[geometry, undefined, buf.count]}
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

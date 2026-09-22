import React, { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { Atom, Bond, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { writeInstanceSegment } from '../services/instanceMatrix';

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
  /** Atom type whose colour this half-bond takes. */
  type: Int32Array;
}

const EMPTY: BondBuffers = {
  count: 0,
  ax: new Float32Array(0), ay: new Float32Array(0), az: new Float32Array(0),
  bx: new Float32Array(0), by: new Float32Array(0), bz: new Float32Array(0),
  type: new Int32Array(0),
};

/**
 * Renders ALL bonds in a single THREE.InstancedMesh draw call.
 * Each bond is drawn as two half-cylinders (atom1->mid coloured by atom1,
 * mid->atom2 coloured by atom2) using per-instance colours.
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
  const radius = Math.max(0.02, 0.12 * config.bondScale);

  // Narrow colour dependency: identity of `config` must not rebuild buffers.
  const colorKey = useMemo(
    () =>
      Object.keys(config.customColors)
        .sort((a, b) => Number(a) - Number(b))
        .map(k => `${k}:${config.customColors[Number(k)]}`)
        .join(';'),
    [config.customColors],
  );

  const buf = useMemo<BondBuffers>(() => {
    if (bonds.length === 0) return EMPTY;
    const cap = bonds.length * 2;
    const ax = new Float32Array(cap), ay = new Float32Array(cap), az = new Float32Array(cap);
    const bx = new Float32Array(cap), by = new Float32Array(cap), bz = new Float32Array(cap);
    const type = new Int32Array(cap);

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
      type[n] = a1.type;
      n++;

      ax[n] = mx;   ay[n] = my;   az[n] = mz;
      bx[n] = a2.x; by[n] = a2.y; bz[n] = a2.z;
      type[n] = a2.type;
      n++;
    }
    return { count: n, ax, ay, az, bx, by, bz, type };
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
    mesh.computeBoundingSphere();
  }, [buf, radius]);

  // Per-instance colours, cached per atom type rather than re-parsing the
  // same handful of hex strings once per half-bond.
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const byType = new Map<number, THREE.Color>();
    for (let i = 0; i < buf.count; i++) {
      const t = buf.type[i];
      let c = byType.get(t);
      if (!c) {
        c = new THREE.Color(config.customColors[t] ?? DEFAULT_ATOM_COLOR);
        byType.set(t, c);
      }
      mesh.setColorAt(i, c);
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [buf, colorKey]);

  if (buf.count === 0) return null;

  return (
    <instancedMesh
      key={buf.count}
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

import React, { useMemo } from 'react';
import * as THREE from 'three';
import { BoxBounds } from '../types';

interface SimulationBoxProps {
  box: BoxBounds;
  color?: string;
  /** Render translucent faces in addition to edges. */
  showFaces?: boolean;
}

/**
 * Reconstructs the 8 corners of a (possibly triclinic) simulation box using
 * the LAMMPS convention:
 *   corner(i,j,k) = (xlo + i*lx + j*xy + k*xz,
 *                    ylo + j*ly + k*yz,
 *                    zlo + k*lz)
 */
export const boxCorners = (box: BoxBounds): THREE.Vector3[] => {
  const lx = box.xhi - box.xlo;
  const ly = box.yhi - box.ylo;
  const lz = box.zhi - box.zlo;
  const xy = box.xy ?? 0;
  const xz = box.xz ?? 0;
  const yz = box.yz ?? 0;

  const corner = (i: number, j: number, k: number) =>
    new THREE.Vector3(
      box.xlo + i * lx + j * xy + k * xz,
      box.ylo + j * ly + k * yz,
      box.zlo + k * lz
    );

  // bit order: i*4 + j*2 + k
  const c: THREE.Vector3[] = [];
  for (let idx = 0; idx < 8; idx++) {
    c.push(corner((idx >> 2) & 1, (idx >> 1) & 1, idx & 1));
  }
  return c;
};

/** The 12 box edges as index pairs into boxCorners() output. */
const EDGE_PAIRS: [number, number][] = [
  [0, 1], [0, 2], [0, 4],          // from origin corner
  [1, 3], [1, 5],
  [2, 3], [2, 6],
  [4, 5], [4, 6],
  [7, 3], [7, 5], [7, 6],
];

/**
 * Renders the simulation cell as crisp line-segment edges (optionally with
 * translucent faces), correctly handling triclinic tilt.
 */
const SimulationBox: React.FC<SimulationBoxProps> = ({
  box,
  color = '#5b8def',
  showFaces = false,
}) => {
  const lineGeometry = useMemo(() => {
    const corners = boxCorners(box);
    const positions = new Float32Array(EDGE_PAIRS.length * 2 * 3);
    EDGE_PAIRS.forEach(([a, b], i) => {
      positions.set([corners[a].x, corners[a].y, corners[a].z], i * 6);
      positions.set([corners[b].x, corners[b].y, corners[b].z], i * 6 + 3);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [box]);

  const faceGeometry = useMemo(() => {
    if (!showFaces) return null;
    const c = boxCorners(box);
    // 6 faces of the parallelepiped, each two triangles
    const quads: [number, number, number, number][] = [
      [0, 1, 3, 2], [4, 5, 7, 6], // z- / z+
      [0, 1, 5, 4], [2, 3, 7, 6], // y- / y+
      [0, 2, 6, 4], [1, 3, 7, 5], // x- / x+
    ];
    const positions: number[] = [];
    for (const [a, b, d, e] of quads) {
      positions.push(c[a].x, c[a].y, c[a].z, c[b].x, c[b].y, c[b].z, c[d].x, c[d].y, c[d].z);
      positions.push(c[a].x, c[a].y, c[a].z, c[d].x, c[d].y, c[d].z, c[e].x, c[e].y, c[e].z);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.computeVertexNormals();
    return geo;
  }, [box, showFaces]);

  return (
    <group>
      <lineSegments geometry={lineGeometry} raycast={() => null}>
        <lineBasicMaterial color={color} transparent opacity={0.75} depthWrite={false} />
      </lineSegments>
      {faceGeometry && (
        <mesh geometry={faceGeometry} raycast={() => null}>
          <meshBasicMaterial
            color={color}
            transparent
            opacity={0.06}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
};

export default SimulationBox;

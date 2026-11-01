import React, { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { Atom, Bond, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { colorKeyOf, hexToRgb, makeAtomColorer } from '../services/colorMaps';
import { halfBonds } from './InstancedBondMesh';

interface BondLinesProps {
  bonds: Bond[];
  atomMap: Map<number, Atom>;
  config: VisualizationConfig;
  /** Bonds longer than this are skipped (periodic wrap-around guard). */
  maxBondLength?: number;
}

/**
 * The 'lines' representation's bonds: every bond two line segments of one THREE.LineSegments draw call, atom1 to the
 * midpoint in atom1's colour and the midpoint to atom2 in atom2's (or all in config.bondColor), the same half bonds as
 * InstancedBondMesh. Lines are one pixel wide whatever the zoom (WebGL ignores a line width above 1), which is what
 * keeps a large bonded system readable.
 */
const BondLines: React.FC<BondLinesProps> = ({ bonds, atomMap, config, maxBondLength = Infinity }) => {
  const invalidate = useThree((s) => s.invalidate);
  const buf = useMemo(() => halfBonds(bonds, atomMap, maxBondLength), [bonds, atomMap, maxBondLength]);
  const colorKey = config.bondColor ? `bond:${config.bondColor}` : colorKeyOf(config);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(6 * buf.count);
    for (let i = 0; i < buf.count; i++) {
      const o = 6 * i;
      pos[o] = buf.ax[i]; pos[o + 1] = buf.ay[i]; pos[o + 2] = buf.az[i];
      pos[o + 3] = buf.bx[i]; pos[o + 4] = buf.by[i]; pos[o + 5] = buf.bz[i];
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(6 * buf.count), 3));
    g.computeBoundingSphere();
    return g;
  }, [buf]);
  useEffect(() => () => geometry.dispose(), [geometry]);

  useEffect(() => {
    const col = geometry.getAttribute('color') as THREE.BufferAttribute;
    const c = col.array as Float32Array;
    const rgb = new Float32Array(3);
    const lin = new THREE.Color();
    const colorOf = makeAtomColorer(config, DEFAULT_ATOM_COLOR);
    if (config.bondColor) hexToRgb(config.bondColor, rgb);
    for (let i = 0; i < buf.count; i++) {
      if (!config.bondColor) colorOf(buf.atom[i], rgb);
      lin.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
      const o = 6 * i;
      c[o] = c[o + 3] = lin.r; c[o + 1] = c[o + 4] = lin.g; c[o + 2] = c[o + 5] = lin.b;
    }
    col.needsUpdate = true;
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, colorKey]);

  const material = useMemo(() => new THREE.LineBasicMaterial({ vertexColors: true }), []);
  useEffect(() => () => material.dispose(), [material]);

  return buf.count > 0 ? <lineSegments geometry={geometry} material={material} /> : null;
};

export default BondLines;

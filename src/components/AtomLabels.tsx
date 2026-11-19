import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Atom, AtomTypeInfo, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR } from '../constants';
import { typeDisplayName } from '../services/elementTypes';

interface AtomLabelsProps {
  atoms: Atom[];
  config: VisualizationConfig;
  /** The structure's types: a label names the element, else the file's type label, else the type number. */
  atomTypes: Readonly<Record<number, AtomTypeInfo>>;
  /** Hard cap — sprites are DOM-free but still per-object; beyond this labels auto-hide. */
  maxLabels?: number;
}

const textureCache = new Map<string, THREE.Texture>();

/**
 * Draws an element badge (symbol on colored disc) into a canvas texture.
 * Canvas-rendered text = zero runtime font/HDR fetches, fully offline.
 */
const getLabelTexture = (symbol: string, colorHex: string): THREE.Texture => {
  const key = `${symbol}|${colorHex}`;
  const cached = textureCache.get(key);
  if (cached) return cached;

  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  // disc
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 6, 0, Math.PI * 2);
  ctx.fillStyle = colorHex;
  ctx.fill();
  ctx.lineWidth = 6;
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.stroke();

  // symbol
  const fontSize = symbol.length > 2 ? 40 : symbol.length > 1 ? 52 : 64;
  ctx.font = `bold ${fontSize}px ui-sans-serif, system-ui, Arial, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  // contrast-aware text color
  const c = new THREE.Color(colorHex);
  const luminance = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  ctx.fillStyle = luminance > 0.45 ? '#111318' : '#ffffff';
  ctx.fillText(symbol, size / 2, size / 2 + 4);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  textureCache.set(key, texture);
  return texture;
};

/**
 * Billboarded element-symbol badges floating at each atom position.
 * Auto-caps: for very large systems labeling every atom is visual noise
 * and draw-call overhead — only the first maxLabels atoms are labeled and
 * the UI surfaces that limitation.
 */
const AtomLabels: React.FC<AtomLabelsProps> = ({ atoms, config, atomTypes, maxLabels = 400 }) => {
  const sprites = useMemo(() => {
    if (!config.showLabels || atoms.length > maxLabels) return null;
    const names = new Map<number, string>();
    for (const a of atoms) if (!names.has(a.type)) names.set(a.type, typeDisplayName(atomTypes, a.type));

    return atoms.map(atom => {
      // the element of the atom's type, else the file's name for the type ("OW"), else its number
      const symbol = names.get(atom.type)!;
      const color = config.customColors[atom.type] ?? DEFAULT_ATOM_COLOR;
      return {
        id: atom.id,
        position: new THREE.Vector3(atom.x, atom.y, atom.z),
        texture: getLabelTexture(symbol, color),
      };
    });
  }, [atoms, atomTypes, config.showLabels, config.customColors, maxLabels]);

  if (!sprites) return null;

  const scale = 0.9 * Math.max(0.5, config.atomScale);

  return (
    <group raycast={() => null}>
      {sprites.map(s => (
        <sprite key={s.id} position={s.position} scale={[scale, scale, scale]}>
          <spriteMaterial map={s.texture} transparent depthTest={false} depthWrite={false} />
        </sprite>
      ))}
    </group>
  );
};

export default AtomLabels;

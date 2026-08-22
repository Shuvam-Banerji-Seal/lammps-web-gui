import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Atom, VisualizationConfig } from '../types';
import { DEFAULT_ATOM_COLOR, ELEMENT_DATA } from '../constants';

interface AtomLabelsProps {
  atoms: Atom[];
  config: VisualizationConfig;
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
const AtomLabels: React.FC<AtomLabelsProps> = ({ atoms, config, maxLabels = 400 }) => {
  const sprites = useMemo(() => {
    if (!config.showLabels || atoms.length > maxLabels) return null;

    return atoms.map(atom => {
      const elemMeta = ELEMENT_DATA.find(e => e.number === atom.type);
      const symbol = elemMeta?.symbol ?? '?';
      const color = config.customColors[atom.type] ?? DEFAULT_ATOM_COLOR;
      return {
        id: atom.id,
        position: new THREE.Vector3(atom.x, atom.y, atom.z),
        texture: getLabelTexture(symbol, color),
      };
    });
  }, [atoms, config.showLabels, config.customColors, maxLabels]);

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

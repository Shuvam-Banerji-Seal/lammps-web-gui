import React, { useMemo } from 'react';
import * as THREE from 'three';
import { Atom, VisualizationConfig } from '../types';
import { MeasurementResult } from '../services/measure';
import { atomDisplayRadius } from '../services/atomStyle';

interface MeasurementOverlayProps {
  selected: Atom[];
  config: VisualizationConfig;
  /** Computed by App via measureSelection(); null until enough atoms are picked. */
  result: MeasurementResult | null;
}

const RING_KEY = '__m3d_ring__';
const textCache = new Map<string, THREE.Texture>();

const getRingTexture = (): THREE.Texture => {
  const cached = textureCacheGet(RING_KEY);
  if (cached) return cached;
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 10, 0, Math.PI * 2);
  ctx.lineWidth = 12;
  ctx.strokeStyle = '#38bdf8'; // sky-400 accent
  ctx.stroke();
  // soft outer glow ring for visibility on any background
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 4, 0, Math.PI * 2);
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(56,189,248,0.45)';
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  textureCacheSet(RING_KEY, tex);
  return tex;
};

const getTextTexture = (text: string): THREE.Texture => {
  const cached = textCache.get(text);
  if (cached) return cached;
  const pad = 24;
  const font = 'bold 44px ui-sans-serif, system-ui, Arial, sans-serif';
  const measure = document.createElement('canvas').getContext('2d')!;
  measure.font = font;
  const w = Math.ceil(measure.measureText(text).width) + pad * 2;
  const h = 76;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = 'rgba(13,17,23,0.88)';
  roundRect(ctx, 2, 2, w - 4, h - 4, 18);
  ctx.fill();
  ctx.strokeStyle = 'rgba(56,189,248,0.9)';
  ctx.lineWidth = 3;
  roundRect(ctx, 2, 2, w - 4, h - 4, 18);
  ctx.stroke();
  ctx.font = font;
  ctx.fillStyle = '#e6f6ff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, h / 2 + 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  textCache.set(text, tex);
  return tex;
};

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// tiny indirection so both caches live in one place
const _cache = new Map<string, THREE.Texture>();
function textureCacheGet(k: string) { return _cache.get(k); }
function textureCacheSet(k: string, t: THREE.Texture) { _cache.set(k, t); }

/**
 * Selection highlights + bond-path lines + floating value label for the
 * measurement tool. Rendered inside the molecule group (raw coordinates).
 */
const MeasurementOverlay: React.FC<MeasurementOverlayProps> = ({
  selected,
  config,
  result,
}) => {
  const ring = useMemo(getRingTexture, []);

  const lineGeometry = useMemo(() => {
    if (selected.length < 2) return null;
    const positions = new Float32Array((selected.length - 1) * 6);
    for (let i = 1; i < selected.length; i++) {
      const p = selected[i - 1];
      const q = selected[i];
      positions.set([p.x, p.y, p.z], (i - 1) * 6);
      positions.set([q.x, q.y, q.z], (i - 1) * 6 + 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [selected]);

  const labelInfo = useMemo(() => {
    if (!selected.length) return null;
    const cx = selected.reduce((s, a) => s + a.x, 0) / selected.length;
    const cy = selected.reduce((s, a) => s + a.y, 0) / selected.length;
    const cz = selected.reduce((s, a) => s + a.z, 0) / selected.length;
    const maxR = Math.max(...selected.map(a => atomDisplayRadius(a, config)));
    return { pos: new THREE.Vector3(cx, cy, cz + maxR * 2.2), size: maxR };
  }, [selected, config]);

  if (!selected.length) return null;

  return (
    <group raycast={() => null}>
      {selected.map(atom => {
        const s = atomDisplayRadius(atom, config) * 3.2;
        return (
          <sprite key={atom.id} position={[atom.x, atom.y, atom.z]} scale={[s, s, s]}>
            <spriteMaterial map={ring} transparent depthTest={false} depthWrite={false} />
          </sprite>
        );
      })}

      {lineGeometry && (
        <lineSegments geometry={lineGeometry}>
          <lineBasicMaterial color="#38bdf8" transparent opacity={0.95} depthTest={false} />
        </lineSegments>
      )}

      {labelInfo && result && (
        <sprite position={labelInfo.pos} scale={[labelInfo.size * 4, labelInfo.size * 4 * 0.28, 1]}>
          <spriteMaterial
            map={getTextTexture(result.label)}
            transparent
            depthTest={false}
            depthWrite={false}
          />
        </sprite>
      )}
    </group>
  );
};

export default MeasurementOverlay;

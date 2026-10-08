import React from 'react';
import { LightingPreset } from '../types';

interface LightingRigProps {
  preset: LightingPreset;
  /** Enable shadow casting on the key light. */
  shadows: boolean;
  /** Scene bounding radius (world units) — sizes the key light's shadow camera. */
  radius?: number;
}

export interface LightConfig {
  ambient: number;
  hemiSky?: string;
  hemiGround?: string;
  hemiIntensity?: number;
  keyIntensity: number;
  keyColor: string;
  keyPosition: [number, number, number];
  fillIntensity: number;
  fillColor: string;
  rimIntensity: number;
  rimColor: string;
}

/**
 * Hand-tuned lighting presets. Deliberately NO <Environment> HDR fetch:
 * the old studio preset pulled an HDR file from a CDN at runtime, adding
 * latency and a network dependency to first render. These rigs are pure
 * three.js lights — instant, offline-safe, deterministic.
 */
export const LIGHT_PRESETS: Record<LightingPreset, LightConfig> = {
  studio: {
    ambient: 0.35,
    keyIntensity: 1.7,
    keyColor: '#ffffff',
    keyPosition: [8, 12, 6],
    fillIntensity: 0.45,
    fillColor: '#dfe8ff',
    rimIntensity: 0.9,
    rimColor: '#8fb3ff',
  },
  lab: {
    ambient: 0.55,
    keyIntensity: 1.15,
    keyColor: '#f4f7ff',
    keyPosition: [5, 10, 8],
    fillIntensity: 0.6,
    fillColor: '#e8f0ff',
    rimIntensity: 0.35,
    rimColor: '#ffffff',
  },
  outdoor: {
    ambient: 0.4,
    hemiSky: '#bcd8ff',
    hemiGround: '#4a4033',
    hemiIntensity: 0.7,
    keyIntensity: 2.1,
    keyColor: '#fff1d6',
    keyPosition: [-10, 14, 4],
    fillIntensity: 0.25,
    fillColor: '#cfe2ff',
    rimIntensity: 0.5,
    rimColor: '#ffe9c4',
  },
  space: {
    ambient: 0.06,
    keyIntensity: 2.4,
    keyColor: '#ffffff',
    keyPosition: [10, 6, -8],
    fillIntensity: 0.05,
    fillColor: '#223',
    rimIntensity: 0.25,
    rimColor: '#6688ff',
  },
  soft: {
    ambient: 0.85,
    keyIntensity: 0.65,
    keyColor: '#ffffff',
    keyPosition: [6, 10, 10],
    fillIntensity: 0.35,
    fillColor: '#ffffff',
    rimIntensity: 0.15,
    rimColor: '#ffffff',
  },
};

const LightingRig: React.FC<LightingRigProps> = ({ preset, shadows, radius = 10 }) => {
  const cfg = LIGHT_PRESETS[preset] ?? LIGHT_PRESETS.studio;
  // A directional light's shadow camera is an orthographic box (three.js
  // default: +-5 units, near 0.5). Sized here to the scene so shadows are not
  // clipped for systems larger than 10 units: the light sits outside the
  // bounding sphere along its preset direction and its frustum covers it.
  const [kx, ky, kz] = cfg.keyPosition;
  const kn = Math.hypot(kx, ky, kz) || 1;
  const keyDist = Math.max(kn, 2 * radius + 5);
  const keyPos: [number, number, number] = [kx / kn * keyDist, ky / kn * keyDist, kz / kn * keyDist];
  const half = radius * 1.25 + 1;

  return (
    <>
      <ambientLight intensity={cfg.ambient} />
      {cfg.hemiSky && cfg.hemiGround && (
        <hemisphereLight
          color={cfg.hemiSky}
          groundColor={cfg.hemiGround}
          intensity={cfg.hemiIntensity ?? 0.5}
        />
      )}
      <directionalLight
        position={keyPos}
        intensity={cfg.keyIntensity}
        color={cfg.keyColor}
        castShadow={shadows}
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.0002}
        shadow-camera-left={-half}
        shadow-camera-right={half}
        shadow-camera-top={half}
        shadow-camera-bottom={-half}
        shadow-camera-near={Math.max(0.1, keyDist - half - 1)}
        shadow-camera-far={keyDist + half + 1}
      />
      <directionalLight position={[-6, 3, -8]} intensity={cfg.fillIntensity} color={cfg.fillColor} />
      <spotLight
        position={[-8, 4, -10]}
        intensity={cfg.rimIntensity}
        angle={0.6}
        penumbra={1}
        color={cfg.rimColor}
      />
    </>
  );
};

export default LightingRig;

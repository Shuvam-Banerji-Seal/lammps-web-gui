import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import type { ToneMappingMode } from '../types';

const TONE: Record<ToneMappingMode, THREE.ToneMapping> = {
  aces: THREE.ACESFilmicToneMapping,
  agx: THREE.AgXToneMapping,
  neutral: THREE.NeutralToneMapping,
  none: THREE.NoToneMapping,
};

interface SceneEffectsProps {
  toneMapping: ToneMappingMode;
  exposure: number;
  /** Image-based lighting from three's procedural RoomEnvironment (no file fetched; its module loads on first use). */
  envLighting: boolean;
  /** Fog toward the background, 0 (off) to 1. */
  depthCue: number;
  background: string;
  /** Scene bounding radius around the origin (the structure is centred there). */
  radius: number;
}

/**
 * Renderer-wide look: tone mapping and exposure, an environment map that the PBR materials reflect (their
 * envMapIntensity), and depth cueing by linear fog that follows the camera distance so zooming keeps the front of the
 * structure clear and fades its back toward the background colour.
 */
const SceneEffects: React.FC<SceneEffectsProps> = ({ toneMapping, exposure, envLighting, depthCue, background, radius }) => {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const invalidate = useThree((s) => s.invalidate);

  useEffect(() => {
    gl.toneMapping = TONE[toneMapping] ?? THREE.ACESFilmicToneMapping;
    gl.toneMappingExposure = exposure;
    invalidate();
  }, [gl, toneMapping, exposure, invalidate]);

  useEffect(() => {
    if (!envLighting) return;
    let target: THREE.WebGLRenderTarget | null = null;
    let live = true;
    void import('three/examples/jsm/environments/RoomEnvironment.js').then(({ RoomEnvironment }) => {
      if (!live) return;
      const pmrem = new THREE.PMREMGenerator(gl);
      const room = new RoomEnvironment();
      target = pmrem.fromScene(room, 0.04);
      room.dispose();
      pmrem.dispose();
      scene.environment = target.texture;
      // reflections on top of the lighting rig, not a second key light: the room is bright everywhere
      scene.environmentIntensity = 0.4;
      invalidate();
    });
    return () => {
      live = false;
      scene.environment = null;
      scene.environmentIntensity = 1;
      target?.dispose();
      invalidate();
    };
  }, [gl, scene, envLighting, invalidate]);

  const fog = useRef<THREE.Fog | null>(null);
  const fogOn = depthCue > 0;
  useEffect(() => {
    if (!fogOn) return;
    const f = new THREE.Fog(background, 1, 2);
    scene.fog = f;
    fog.current = f;
    invalidate();
    return () => {
      scene.fog = null;
      fog.current = null;
      invalidate();
    };
  }, [scene, fogOn, background, invalidate]);

  useFrame(({ camera }) => {
    const f = fog.current;
    if (!f) return;
    // the structure spans d - R .. d + R from the camera: the fog starts at its front and, at full strength, has its
    // back 90% of the way to the background colour
    const d = camera.position.length();
    const k = 1 - depthCue;
    f.near = Math.max(0, d - radius);
    f.far = d + radius * (1.2 + 4 * k * k);
  });

  return null;
};

export default SceneEffects;

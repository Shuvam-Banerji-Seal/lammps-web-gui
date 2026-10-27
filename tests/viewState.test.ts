import { describe, it, expect } from 'vitest';
import {
  encodeViewState,
  decodeViewState,
  viewStateFromSearch,
} from '../src/services/viewState';
import { VisualizationConfig } from '../src/types';

const baseConfig: VisualizationConfig = {
  atomScale: 1.25,
  bondScale: 0.8,
  materialType: 'metallic',
  backgroundColor: '#10141c',
  showBonds: true,
  customColors: { 6: '#ff8800', 1: '#123456' },
  visualizationMode: 'space-fill',
  lightingPreset: 'outdoor',
  showBox: true,
  showAxes: false,
  showLabels: true,
  shadowsEnabled: false,
  autoRotateSpeed: 2.5,
  fov: 55,
};

describe('viewState share links', () => {
  it('round-trips a full config', () => {
    const token = encodeViewState(baseConfig);
    const out = decodeViewState(token);
    expect(out).toMatchObject({
      visualizationMode: 'space-fill',
      materialType: 'metallic',
      lightingPreset: 'outdoor',
      showBonds: true,
      showBox: true,
      showLabels: true,
      showAxes: false,
      shadowsEnabled: false,
      atomScale: 1.25,
      bondScale: 0.8,
      fov: 55,
      autoRotateSpeed: 2.5,
      backgroundColor: '#10141c',
      customColors: { 6: '#ff8800', 1: '#123456' },
    });
  });

  it('produces URL-safe tokens (no +, /, =)', () => {
    const token = encodeViewState(baseConfig);
    expect(token).not.toMatch(/[+/=]/);
  });

  it('rejects garbage and hostile input safely', () => {
    expect(decodeViewState('!!!not-base64!!!')).toBeNull();
    expect(decodeViewState('')).toBeNull();
    // valid b64 of "null" / non-object JSON
    expect(decodeViewState(btoa('42'))).toBeNull();
    // oversized input
    expect(decodeViewState('a'.repeat(5000))).toBeNull();
  });

  it('maps all-invalid payloads to null ("no usable state"), never throws', () => {
    const json = btoa(JSON.stringify({ v: 'DROP TABLE;--', m: 'hax' }));
    const out = decodeViewState(json.replace(/\+/g, '-').replace(/\//g, '_'));
    expect(out).toBeNull();
  });

  it('clamps numeric fields into safe ranges', () => {
    const json = btoa(JSON.stringify({ as: 9999, f: -50, r: 100000 }));
    const out = decodeViewState(json.replace(/\//g, '_'));
    expect(out!.atomScale).toBe(3);
    expect(out!.fov).toBe(15);
    expect(out!.autoRotateSpeed).toBe(6);
  });

  it('extracts token from a search string', () => {
    const token = encodeViewState(baseConfig);
    const out = viewStateFromSearch(`?s=${token}&other=1`);
    expect(out?.visualizationMode).toBe('space-fill');
    expect(viewStateFromSearch('?x=1')).toBeNull();
  });

  it('round-trips the colour, rendering and size settings, and drops or clamps bad values', () => {
    const token = encodeViewState({
      ...baseConfig, visualizationMode: 'points', colorBy: 'speed', colorMap: 'turbo', colorReverse: true, colorAuto: false,
      colorMin: -1.5, colorMax: 4, quality: 'ultra', toneMapping: 'agx', exposure: 1.35, envLighting: true, depthCue: 0.4,
      bondColor: '#aabbcc', boxColor: '#112233', typeRadii: { 1: 1.5, 8: 0.5 },
    });
    expect(decodeViewState(token)).toMatchObject({
      visualizationMode: 'points', colorBy: 'speed', colorMap: 'turbo', colorReverse: true, colorAuto: false, colorMin: -1.5,
      colorMax: 4, quality: 'ultra', toneMapping: 'agx', exposure: 1.35, envLighting: true, depthCue: 0.4,
      bondColor: '#aabbcc', boxColor: '#112233', typeRadii: { 1: 1.5, 8: 0.5 },
    });
    const bad = Buffer.from(JSON.stringify({ cb: 'mass', cm: 'jet', q: 'max', tm: 'filmic', ex: 9999, dc: -5, bc: 'red', tr: '1:9999,x:5' })).toString('base64url');
    const out = decodeViewState(bad)!;
    expect(out.colorBy).toBeUndefined();
    expect(out.colorMap).toBeUndefined();
    expect(out.quality).toBeUndefined();
    expect(out.toneMapping).toBeUndefined();
    expect(out.exposure).toBe(3);
    expect(out.depthCue).toBe(0);
    expect(out.bondColor).toBeUndefined();
    expect(out.typeRadii).toEqual({ 1: 3 });
  });
});

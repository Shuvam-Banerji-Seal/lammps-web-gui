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
});

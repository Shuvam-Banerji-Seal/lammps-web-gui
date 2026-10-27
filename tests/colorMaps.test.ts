import { describe, expect, it } from 'vitest';
import {
  atomScalar, categoricalColor, colorKeyOf, colorMapGradient, COLOR_MAP_NAMES, hexToRgb, isContinuous, makeAtomColorer,
  sampleColorMap, scalarRange,
} from '../src/services/colorMaps';
import { atomDisplayRadius, bondSegments, qualityDpr, radiusKeyOf, shadowLimit, shadowMapSize, sphereSegments } from '../src/services/atomStyle';
import type { Atom, VisualizationConfig } from '../src/types';

const atom = (o: Partial<Atom>): Atom => ({ id: 1, molId: 0, type: 1, charge: 0, x: 0, y: 0, z: 0, ...o });
const view = (o: Partial<VisualizationConfig>): VisualizationConfig => ({
  atomScale: 1, bondScale: 1, materialType: 'realistic', backgroundColor: '#000000', showBonds: true,
  customColors: {}, visualizationMode: 'ball-and-stick', lightingPreset: 'studio', showBox: false, showAxes: false,
  showLabels: false, shadowsEnabled: true, autoRotateSpeed: 0.5, fov: 40, ...o,
});
const rgb = (): Float32Array => new Float32Array(3);

describe('colour maps', () => {
  it('sample the end stops at 0 and 1 and clamp outside, reversed when asked', () => {
    const out = rgb();
    sampleColorMap('viridis', 0, out);
    expect([...out].map((v) => Math.round(v * 255))).toEqual([0x44, 0x01, 0x54]);
    sampleColorMap('viridis', 1, out);
    expect([...out].map((v) => Math.round(v * 255))).toEqual([0xfd, 0xe7, 0x25]);
    sampleColorMap('viridis', 7, out);
    expect([...out].map((v) => Math.round(v * 255))).toEqual([0xfd, 0xe7, 0x25]);
    sampleColorMap('viridis', 0, out, true);
    expect([...out].map((v) => Math.round(v * 255))).toEqual([0xfd, 0xe7, 0x25]);
    sampleColorMap('grayscale', 0.5, out);
    expect(out[0]).toBeCloseTo(0x80 / 255, 6);
  });

  it('interpolate linearly between neighbouring stops', () => {
    const a = rgb(), b = rgb(), m = rgb();
    sampleColorMap('turbo', 0.125, a);
    sampleColorMap('turbo', 0.25, b);
    sampleColorMap('turbo', 0.1875, m);
    for (let c = 0; c < 3; c++) expect(m[c]).toBeCloseTo((a[c] + b[c]) / 2, 6);
  });

  it('have nine stops each and a CSS gradient', () => {
    for (const name of COLOR_MAP_NAMES) {
      const g = colorMapGradient(name);
      expect(g.startsWith('linear-gradient(to right, rgb(')).toBe(true);
      expect(g.match(/rgb\(/g)?.length).toBe(9);
    }
  });
});

describe('atom quantities', () => {
  it('read charge, speed, id, molecule and position; speed is NaN without velocities', () => {
    const a = atom({ id: 7, molId: 3, charge: -0.5, x: 1, y: 2, z: 3, vx: 3, vy: 4, vz: 0 });
    expect(atomScalar(a, 'charge')).toBe(-0.5);
    expect(atomScalar(a, 'speed')).toBe(5);
    expect(atomScalar(a, 'id')).toBe(7);
    expect(atomScalar(a, 'molecule')).toBe(3);
    expect([atomScalar(a, 'x'), atomScalar(a, 'y'), atomScalar(a, 'z')]).toEqual([1, 2, 3]);
    expect(atomScalar(atom({}), 'speed')).toBeNaN();
    expect(isContinuous('type')).toBe(false);
    expect(isContinuous('molecule')).toBe(false);
    expect(isContinuous('z')).toBe(true);
  });

  it('take the range over every frame, skipping missing values', () => {
    const f1 = [atom({ x: -2 }), atom({ x: 5 })], f2 = [atom({ x: 9 })];
    expect(scalarRange([f1, f2], 'x')).toEqual([-2, 9]);
    expect(scalarRange([f1], 'speed')).toBeNull();
  });

  it('give distinct molecule hues', () => {
    const a = rgb(), b = rgb();
    categoricalColor(1, a);
    categoricalColor(2, b);
    expect([...a]).not.toEqual([...b]);
    for (const v of [...a, ...b]) expect(v >= 0 && v <= 1).toBe(true);
  });
});

describe('atom colourer', () => {
  it('colours by type from the type colours, magenta when unset', () => {
    const out = rgb();
    makeAtomColorer(view({ customColors: { 1: '#ff0000' } }), '#ff00ff')(atom({ type: 1 }), out);
    expect([...out]).toEqual([1, 0, 0]);
    makeAtomColorer(view({}), '#ff00ff')(atom({ type: 2 }), out);
    expect([...out]).toEqual([1, 0, 1]);
  });

  it('maps a quantity over colorMin..colorMax and greys a missing one', () => {
    const c = makeAtomColorer(view({ colorBy: 'x', colorMap: 'grayscale', colorMin: 0, colorMax: 10 }), '#ffffff');
    const out = rgb();
    c(atom({ x: 10 }), out);
    expect(out[0]).toBeCloseTo(1, 6);
    c(atom({ x: 0 }), out);
    expect(out[0]).toBeCloseTo(0, 6);
    const s = makeAtomColorer(view({ colorBy: 'speed', colorMin: 0, colorMax: 1 }), '#ffffff');
    s(atom({}), out);
    expect([...out]).toEqual([0.5, 0.5, 0.5]);
  });

  it('keys change with what the colours depend on', () => {
    expect(colorKeyOf(view({ customColors: { 1: '#ff0000' } }))).not.toBe(colorKeyOf(view({ customColors: { 1: '#00ff00' } })));
    expect(colorKeyOf(view({ colorBy: 'x', colorMin: 0, colorMax: 1 }))).not.toBe(colorKeyOf(view({ colorBy: 'x', colorMin: 0, colorMax: 2 })));
    expect(colorKeyOf(view({ colorBy: 'x', colorMap: 'viridis' }))).not.toBe(colorKeyOf(view({ colorBy: 'x', colorMap: 'turbo' })));
  });

  it('parses hex colours', () => {
    const out = rgb();
    hexToRgb('#336699', out);
    expect([...out].map((v) => Math.round(v * 255))).toEqual([0x33, 0x66, 0x99]);
  });
});


describe('render quality and radii', () => {
  it('tessellate by quality and system size', () => {
    expect(sphereSegments('auto', 500)).toBe(32);
    expect(sphereSegments(undefined, 50000)).toBe(8);
    expect(sphereSegments('ultra', 500)).toBe(64);
    expect(sphereSegments('low', 500)).toBe(16);
    expect(bondSegments('auto', 100)).toBe(16);
    expect(bondSegments('ultra', 100)).toBe(32);
    expect(bondSegments('low', 100)).toBe(8);
  });

  it('set the pixel ratio and shadows by quality', () => {
    expect(qualityDpr('auto', 30000)).toEqual([0.75, 1.25]);
    expect(qualityDpr('ultra', 30000)).toEqual([2, 3]);
    expect(shadowLimit('auto')).toBe(8000);
    expect(shadowLimit('ultra')).toBe(50000);
    expect(shadowMapSize('high')).toBe(4096);
  });

  it('scale radii per type and by covalent radius for element balls', () => {
    const c = view({ typeRadii: { 6: 2 } });
    expect(atomDisplayRadius(atom({ type: 6 }), c)).toBeCloseTo(0.9, 12);
    expect(atomDisplayRadius(atom({ type: 1 }), c)).toBeCloseTo(0.45, 12);
    expect(atomDisplayRadius(atom({ type: 6 }), view({ visualizationMode: 'element-balls' }))).toBeCloseTo(0.76 * 0.6, 12);
    expect(radiusKeyOf(c)).not.toBe(radiusKeyOf(view({})));
  });
});

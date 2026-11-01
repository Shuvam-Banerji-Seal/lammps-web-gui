/**
 * The Structure Viewer's Lines, Cubes, Glass and Flat discs representations (checked in Chromium with SwiftShader
 * WebGL on 1728 bonded atoms; jsdom has no WebGL): which renderer each mode uses, the atom sizes, the bonds' form,
 * share links that carry the new modes, the V key that cycles through every mode, and the half bonds BondLines draws.
 */
import { describe, it, expect, afterEach } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { atomDisplayRadius, cylinderBondMode, meshMode } from '../src/services/atomStyle';
import { decodeViewState, encodeViewState } from '../src/services/viewState';
import { useKeyboardShortcuts, SHORTCUT_CATALOG, type ShortcutActionMap } from '../src/hooks/useKeyboardShortcuts';
import { halfBonds } from '../src/components/InstancedBondMesh';
import type { Atom, VisualizationConfig, VisualizationMode } from '../src/types';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const config = (mode: VisualizationMode): VisualizationConfig => ({
  atomScale: 1, bondScale: 1, materialType: 'realistic', backgroundColor: '#000000', showBonds: true, customColors: {},
  visualizationMode: mode, lightingPreset: 'studio', showBox: true, showAxes: false, showLabels: false,
  shadowsEnabled: false, autoRotateSpeed: 1, fov: 40,
});
const atom = (id: number, type: number, x: number, y = 0, z = 0): Atom => ({ id, molId: 1, type, charge: 0, x, y, z });

describe('viewer representations', () => {
  it('draws Lines and Flat discs as point sprites, Cubes and Glass with the instanced mesh', () => {
    expect(meshMode('lines')).toBe(false);
    expect(meshMode('flat')).toBe(false);
    expect(meshMode('points')).toBe(false);
    expect(meshMode('cubes')).toBe(true);
    expect(meshMode('glass')).toBe(true);
    expect(meshMode('ball-and-stick')).toBe(true);
  });

  it('draws cylinder bonds for Cubes only among the new modes (Lines draws lines; Glass and Flat discs none)', () => {
    expect(cylinderBondMode('cubes')).toBe(true);
    for (const m of ['lines', 'glass', 'flat', 'space-fill', 'points'] as const) expect(cylinderBondMode(m)).toBe(false);
    expect(cylinderBondMode('ball-and-stick')).toBe(true);
  });

  it('sizes atoms: Glass as Space Fill, Flat discs touching at unit spacing, Lines a small dot, a Cube as Ball & Stick', () => {
    const c = atom(1, 6, 0);
    expect(atomDisplayRadius(c, config('glass'))).toBe(atomDisplayRadius(c, config('space-fill')));
    expect(atomDisplayRadius(c, config('flat'))).toBe(0.5);
    expect(atomDisplayRadius(c, config('lines'))).toBeLessThan(atomDisplayRadius(c, config('ball-and-stick')) / 2);
    expect(atomDisplayRadius(c, config('cubes'))).toBe(atomDisplayRadius(c, config('ball-and-stick')));
  });

  it('keeps the new modes in a share link', () => {
    for (const m of ['lines', 'cubes', 'glass', 'flat'] as const) {
      expect(decodeViewState(encodeViewState(config(m)))?.visualizationMode).toBe(m);
    }
  });

  it('splits each bond at its midpoint, each half for its own atom, and skips wrapped bonds', () => {
    const a = atom(1, 1, 0), b = atom(2, 8, 2), far = atom(3, 1, 9);
    const map = new Map([a, b, far].map((x) => [x.id, x]));
    const h = halfBonds([{ atom1Id: 1, atom2Id: 2 }, { atom1Id: 2, atom2Id: 3 }] as never, map, 5);
    expect(h.count).toBe(2);
    expect([h.ax[0], h.bx[0], h.ax[1], h.bx[1]]).toEqual([0, 1, 1, 2]);
    expect(h.atom.slice(0, 2).map((x) => x.id)).toEqual([1, 2]);
  });
});

describe('the V key', () => {
  let root: Root | null = null;
  let host: HTMLDivElement | null = null;
  afterEach(async () => {
    if (root) await act(async () => { root!.unmount(); });
    host?.remove();
    root = null; host = null;
  });

  it('cycles the representation (the modes past the digit keys too) and is in the shortcut list', async () => {
    let cycles = 0;
    const noop = () => {};
    const actions: ShortcutActionMap = {
      onToggleRotate: noop, onFit: noop, onViewPreset: noop, onZoom: noop, onOrbit: noop, onVisualizationMode: noop,
      onToggleBonds: noop, onToggleBox: noop, onToggleLabels: noop, onCycleLighting: noop, onToggleTheme: noop,
      onToggleSidebar: noop, onScreenshot: noop, onHelp: noop, onEscape: noop,
      onCycleVisualizationMode: () => { cycles++; },
    };
    const Probe = () => { useKeyboardShortcuts(actions); return null; };
    host = document.createElement('div');
    document.body.appendChild(host);
    root = createRoot(host);
    await act(async () => { root!.render(React.createElement(Probe)); });
    for (const key of ['v', 'V']) window.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
    expect(cycles).toBe(2);
    expect(SHORTCUT_CATALOG.some((s) => s.keys === 'V' && /Lines, Cubes, Glass, Flat discs/.test(s.action))).toBe(true);
  });
});

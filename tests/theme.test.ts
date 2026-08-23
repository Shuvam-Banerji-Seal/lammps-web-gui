import { describe, it, expect } from 'vitest';
import { getThemeTokens, PALETTE, type ThemeTokens } from '../src/theme';

const tokenKeys = (t: ThemeTokens) => Object.keys(t) as (keyof ThemeTokens)[];

describe('theme tokens', () => {
  it('defines every token key in both themes', () => {
    const dark = getThemeTokens('dark');
    const light = getThemeTokens('light');
    const reference = tokenKeys(dark);
    expect(reference.length).toBeGreaterThanOrEqual(30);
    for (const key of tokenKeys(light)) {
      expect(reference, key).toContain(key);
    }
    for (const key of reference) {
      expect(typeof dark[key], String(key)).toBe('string');
      expect(dark[key].length, String(key)).toBeGreaterThan(0);
      expect(typeof light[key], String(key)).toBe('string');
      expect(light[key].length, String(key)).toBeGreaterThan(0);
    }
  });

  it('light and dark palettes genuinely differ (no copy-paste themes)', () => {
    const dark = getThemeTokens('dark');
    const light = getThemeTokens('light');
    const differing = tokenKeys(dark).filter(k => dark[k] !== light[k]);
    // Nearly every token should differ; allow a small shared set (e.g. toggles).
    expect(differing.length).toBeGreaterThan(tokenKeys(dark).length * 0.7);
  });

  it('workbench chrome carries no blue-tinted hexes', () => {
    const blueish = /#(?:[0-9a-f]{2})?(?:[2-9a-f][0-9a-f])?(?:[4-9a-f][0-9a-f])f{0,2}\b/i;
    for (const theme of ['dark', 'light'] as const) {
      const t = getThemeTokens(theme);
      for (const key of tokenKeys(t)) {
        const hexes = t[key].match(/#[0-9a-f]{6}/gi) ?? [];
        for (const hex of hexes) {
          const r = parseInt(hex.slice(1, 3), 16);
          const g = parseInt(hex.slice(3, 5), 16);
          const b = parseInt(hex.slice(5, 7), 16);
          // warm palette rule: blue channel must not dominate green+red
          expect(b, `${theme}.${key} ${hex}`).toBeLessThanOrEqual(Math.max(r, g) + 26);
        }
      }
    }
  });

  it('palette constants match the token bases', () => {
    expect(PALETTE.dark.base).toBe('#16130f');
    expect(PALETTE.light.base).toBe('#f4efe6');
  });
});

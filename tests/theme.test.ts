import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  getThemeTokens,
  initialTheme,
  isDarkTheme,
  nextTheme,
  PALETTE,
  THEMES,
  THEME_STORAGE_KEY,
  themeColor,
  type Theme,
  type ThemeTokens,
} from '../src/theme';

const tokenKeys = (t: ThemeTokens) => Object.keys(t) as (keyof ThemeTokens)[];
const ids = THEMES.map(t => t.id);

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

  it('every theme in THEMES carries the complete token set', () => {
    const reference = tokenKeys(getThemeTokens('dark'));
    for (const id of ids) {
      const t = getThemeTokens(id);
      expect(tokenKeys(t).sort(), id).toEqual([...reference].sort());
      for (const key of reference) {
        expect(t[key].length, `${id}.${key}`).toBeGreaterThan(0);
      }
    }
  });

  it('light and dark palettes genuinely differ (no copy-paste themes)', () => {
    const dark = getThemeTokens('dark');
    const light = getThemeTokens('light');
    const differing = tokenKeys(dark).filter(k => dark[k] !== light[k]);
    // Nearly every token should differ; allow a small shared set (e.g. toggles).
    expect(differing.length).toBeGreaterThan(tokenKeys(dark).length * 0.7);
  });

  it('every new theme differs from the original dark and light palettes', () => {
    for (const id of ids.filter(i => !['dark', 'light'].includes(i))) {
      const t = getThemeTokens(id);
      for (const base of ['dark', 'light'] as const) {
        const b = getThemeTokens(base);
        const differing = tokenKeys(t).filter(k => t[k] !== b[k]);
        expect(differing.length, `${id} vs ${base}`).toBeGreaterThan(tokenKeys(t).length * 0.7);
      }
    }
  });

  it('original workbench chrome carries no blue-tinted hexes', () => {
    // Midnight (cool blue-slate), Solarized (blue accents) and High contrast
    // are deliberately outside this warm-palette rule.
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

describe('theme registry', () => {
  it('lists five unique themes in picker order', () => {
    expect(ids).toEqual(['dark', 'light', 'midnight', 'solarized', 'high-contrast']);
    expect(new Set(ids).size).toBe(ids.length);
    for (const t of THEMES) expect(t.label.length, t.id).toBeGreaterThan(0);
  });

  it('isDarkTheme follows the dark flag of each theme', () => {
    expect(isDarkTheme('dark')).toBe(true);
    expect(isDarkTheme('midnight')).toBe(true);
    expect(isDarkTheme('high-contrast')).toBe(true);
    expect(isDarkTheme('light')).toBe(false);
    expect(isDarkTheme('solarized')).toBe(false);
  });

  it('nextTheme cycles through THEMES and wraps', () => {
    let t: Theme = 'dark';
    const seen: Theme[] = [];
    for (let i = 0; i < ids.length; i++) {
      seen.push(t);
      t = nextTheme(t);
    }
    expect(seen).toEqual(ids);
    expect(t).toBe('dark');
  });

  it('themeColor is the page background of each theme', () => {
    for (const id of ids) {
      expect(themeColor(id), id).toBe(getThemeTokens(id).bg.match(/#[0-9a-f]{6}/i)![0]);
    }
    expect(themeColor('dark')).toBe(PALETTE.dark.base);
  });
});

describe('initialTheme', () => {
  const original = globalThis.localStorage;
  let store: Record<string, string>;

  beforeEach(() => {
    store = {};
    vi.stubGlobal('localStorage', {
      getItem: (k: string) => (k in store ? store[k] : null),
      setItem: (k: string, v: string) => { store[k] = v; },
      removeItem: (k: string) => { delete store[k]; },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    if (original) vi.stubGlobal('localStorage', original);
  });

  it('honours every stored theme id, including the new ones', () => {
    for (const id of ids) {
      store[THEME_STORAGE_KEY] = id;
      expect(initialTheme(), id).toBe(id);
    }
  });

  it('restores a stored new theme id such as solarized', () => {
    store[THEME_STORAGE_KEY] = 'solarized';
    expect(initialTheme()).toBe('solarized');
  });

  it('falls back on an unknown stored value', () => {
    store[THEME_STORAGE_KEY] = 'neon-pink';
    expect(['dark', 'light']).toContain(initialTheme());
  });
});

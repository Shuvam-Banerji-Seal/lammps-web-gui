import { describe, it, expect } from 'vitest';
import { getThemeTokens, THEMES, type Theme } from '../src/theme';

/**
 * WCAG 2.1 SC 1.4.3 (Contrast, Minimum): normal-size text needs 4.5:1.
 * https://www.w3.org/TR/WCAG21/#contrast-minimum
 * Relative luminance per https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
 *
 * Text tokens are checked against EVERY surface they are rendered on —
 * including the darker hover/stat surfaces, which is where the old light
 * `muted` (#7c7060) dropped to 3.58:1.
 */
const luminance = (hex: string): number => {
  const h = hex.replace('#', '');
  const [r, g, b] = [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16) / 255)
    .map(c => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a: string, b: string): number => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
/** First `text-[#rrggbb]` (or `placeholder:text-[#…]`) in a token's class list. */
const textHex = (cls: string, prefix = 'text-'): string => {
  const m = cls.match(new RegExp(`(?:^|\\s)${prefix.replace(':', '\\:')}\\[(#[0-9a-fA-F]{6})\\]`));
  if (!m) throw new Error(`no ${prefix}[#hex] in "${cls}"`);
  return m[1];
};

const SURFACES: Record<Theme, string[]> = {
  // bg, surface, raised/input, white, stat/button, hover, accentSoft
  light: ['#f4efe6', '#fbf8f1', '#fffdf8', '#ffffff', '#efe9dc', '#e5ddcb', '#e7efdf'],
  // input, bg, surface
  dark: ['#14110c', '#16130f', '#1e1913'],
  // input, bg, surface, raised, button, hover, accentSoft
  midnight: ['#0a1020', '#0c1322', '#121b2d', '#1a2539', '#1d2a42', '#22314b', '#16353d'],
  // bg, surface, card, input, button, hover, accentSoft, warn
  solarized: ['#fdf6e3', '#eee8d5', '#fffbf0', '#e4dcc8', '#dbd2bb', '#dfeaf0', '#f6e3cf'],
  // bg, surface, card, button, hover, accentSoft, warn
  'high-contrast': ['#000000', '#0a0a0a', '#111111', '#1a1a1a', '#262626', '#1a1a00', '#1a1600'],
};

/** Minimum text contrast per theme: AAA (7:1) for high contrast, AA (4.5:1) otherwise. */
const MIN_RATIO: Record<Theme, number> = {
  dark: 4.5,
  light: 4.5,
  midnight: 4.5,
  solarized: 4.5,
  'high-contrast': 7,
};

/** Text tokens checked on every surface of a theme. */
const TEXT_KEYS = ['muted', 'accentText', 'text', 'headerText', 'accentCode'] as const;
/** The high-contrast theme also checks the status and error text tokens. */
const STRICT_TEXT_KEYS = [...TEXT_KEYS, 'danger', 'invalidText'] as const;

describe('theme text contrast (WCAG AA 4.5:1; high contrast AAA 7:1)', () => {
  for (const theme of THEMES.map(x => x.id)) {
    const t = getThemeTokens(theme);
    const min = MIN_RATIO[theme];
    const keys = theme === 'high-contrast' ? STRICT_TEXT_KEYS : TEXT_KEYS;
    for (const key of keys) {
      it(`${theme} ${key} clears ${min}:1 on every surface`, () => {
        const fg = textHex(t[key]);
        for (const bg of SURFACES[theme]) {
          expect(ratio(fg, bg), `${fg} on ${bg}`).toBeGreaterThanOrEqual(min);
        }
      });
    }
    it(`${theme} input placeholder clears ${min}:1 on the input background`, () => {
      const fg = textHex(t.input, 'placeholder:text-');
      const inputBg = t.input.match(/bg-\[(#[0-9a-fA-F]{6})\]/)![1];
      expect(ratio(fg, inputBg), `${fg} on ${inputBg}`).toBeGreaterThanOrEqual(min);
    });
  }

  it('the ratio helper matches a published reference value', () => {
    // black on white is exactly 21:1 by definition
    expect(ratio('#000000', '#ffffff')).toBeCloseTo(21, 6);
    // and the old light muted on the light bg — the value that failed
    expect(ratio('#7c7060', '#f4efe6')).toBeCloseTo(4.22, 2);
  });
});

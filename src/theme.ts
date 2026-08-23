/**
 * Molecule3D Workbench — shared flat-design theme tokens.
 *
 * Policy: NO gradients anywhere. Solid surfaces, 1px borders, restrained
 * accents. Dark mode is a warm "coffee & sage" palette (deep brown-black
 * surfaces, sage-green primary accent, caramel secondary highlight) —
 * deliberately not blue-tinted. Light mode is warm paper.
 */

export type Theme = 'dark' | 'light';

export const THEME_STORAGE_KEY = 'm3d.theme';

export const prefersLightTheme = (): boolean =>
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-color-scheme: light)').matches;

/** Core palette values (also used for canvas defaults / meta tags). */
export const PALETTE = {
  dark: {
    base: '#16130f',
    surface: '#1e1913',
    raised: '#241e16',
    border: '#332a1f',
    text: '#ede5d8',
    muted: '#a3937f',
    accentGreen: '#7fa66b',
    accentGreenDeep: '#567a46',
    amber: '#d9a05b',
  },
  light: {
    base: '#f4efe6',
    surface: '#fbf8f1',
    raised: '#ffffff',
    border: '#e0d7c6',
    text: '#2e2920',
    muted: '#7c7060',
    accentGreen: '#4e7a41',
    accentGreenDeep: '#40663a',
    amber: '#b97f3e',
  },
} as const;

export interface ThemeTokens {
  /** Page background classes. */
  bg: string;
  text: string;
  /** Sidebar / large fixed panels. */
  panel: string;
  headerText: string;
  muted: string;
  card: string;
  input: string;
  button: string;
  chip: string;
  /** Primary action (sage green). */
  accent: string;
  /** Confirmation / run action (deeper green). */
  go: string;
  /** Selected-item treatment. */
  active: string;
  /** Amber highlight for warnings / secondary emphasis. */
  warn: string;
  danger: string;
  divider: string;
  stat: string;
}

const DARK: ThemeTokens = {
  bg: 'bg-[#16130f]',
  text: 'text-[#ede5d8]',
  panel: 'bg-[#1e1913] border-[#332a1f]',
  headerText: 'text-[#e5dccd]',
  muted: 'text-[#a3937f]',
  card: 'bg-[#241e16] border-[#332a1f]',
  input:
    'bg-[#14110c] border-[#453a2b] text-[#ece4d6] placeholder:text-[#6f6353] focus:border-[#7fa66b]',
  button: 'bg-[#2a2318] hover:bg-[#342b1d] border border-[#3f3526]',
  chip: 'bg-[#241e16] border border-[#3f3526]',
  accent: 'bg-[#567a46] hover:bg-[#659054] text-[#f2f6ee]',
  go: 'bg-[#47693b] hover:bg-[#557c47] text-white',
  active: 'bg-[#31402a] border border-[#7fa66b] text-[#c4ddb2]',
  warn: 'border-[#6b5124]/60 bg-[#332612]/50 text-[#e4b877]',
  danger: 'text-[#cf8b76] hover:bg-[#3a241c]',
  divider: 'border-[#332a1f]',
  stat: 'bg-[#241e16]',
};

const LIGHT: ThemeTokens = {
  bg: 'bg-[#f4efe6]',
  text: 'text-[#2e2920]',
  panel: 'bg-[#fbf8f1] border-[#e0d7c6]',
  headerText: 'text-[#2e2920]',
  muted: 'text-[#7c7060]',
  card: 'bg-white border-[#e6ddcc]',
  input:
    'bg-[#fffdf8] border-[#d8cdb8] text-[#2e2920] placeholder:text-[#a2937c] focus:border-[#4e7a41]',
  button: 'bg-[#efe9dc] hover:bg-[#e5ddcb] border border-[#ddd2bd]',
  chip: 'bg-[#f3eee2] border border-[#ddd2bd]',
  accent: 'bg-[#4e7a41] hover:bg-[#5b8c4c] text-white',
  go: 'bg-[#40663a] hover:bg-[#4c7842] text-white',
  active: 'bg-[#e7efdf] border border-[#4e7a41] text-[#3c5c32]',
  warn: 'border-[#caa15c] bg-[#f7ecd7] text-[#7a5716]',
  danger: 'text-[#a4502f] hover:bg-[#f3e0d8]',
  divider: 'border-[#e4dbc9]',
  stat: 'bg-[#efe9dc]',
};

export const getThemeTokens = (theme: Theme): ThemeTokens =>
  theme === 'light' ? LIGHT : DARK;

/** Initial theme honoring the persisted choice, then the OS preference. */
export const initialTheme = (): Theme => {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    /* storage unavailable */
  }
  return prefersLightTheme() ? 'light' : 'dark';
};

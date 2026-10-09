/**
 * Molecule3D Workbench — shared flat-design theme tokens.
 *
 * Policy: NO gradients anywhere. Solid surfaces, 1px borders, restrained
 * accents. Dark mode is a warm "coffee & sage" palette (deep brown-black
 * surfaces, sage-green primary accent, caramel secondary highlight) —
 * deliberately not blue-tinted. Light mode is warm paper.
 */

export type Theme = 'dark' | 'light' | 'midnight' | 'solarized' | 'high-contrast';

/** Every selectable theme, in picker order. */
export const THEMES: { id: Theme; label: string; dark: boolean }[] = [
  { id: 'dark', label: 'Coffee & sage', dark: true },
  { id: 'light', label: 'Warm paper', dark: false },
  { id: 'midnight', label: 'Midnight', dark: true },
  { id: 'solarized', label: 'Solarized light', dark: false },
  { id: 'high-contrast', label: 'High contrast', dark: true },
];

/** Dark-flavoured themes take the dark branch of every colour decision. */
export const isDarkTheme = (t: Theme): boolean =>
  THEMES.find(x => x.id === t)?.dark ?? true;

/** The theme after `t` in THEMES order (wraps around). */
export const nextTheme = (t: Theme): Theme => {
  const i = THEMES.findIndex(x => x.id === t);
  return THEMES[(i + 1) % THEMES.length].id;
};

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
    muted: '#6b6053',
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
  /* ---- semantic accents shared by the workbench modules ---- */
  /** Sage text for icons/labels/active states. */
  accentText: string;
  /** Monospace command/code accent. */
  accentCode: string;
  /** Soft-selected chip: bg + text + ring. */
  accentSoft: string;
  /** Hover treatment for list rows / palette items. */
  hoverSurface: string;
  /** Strong border (inputs, idle toggles, dividers that need weight). */
  borderStrong: string;
  /** Focus ring color class. */
  focusRing: string;
  toggleOn: string;
  toggleOff: string;
  /** Idle flag/command chip. */
  chipIdle: string;
  /** Flowchart node card (idle / disabled). */
  nodeCard: string;
  nodeDisabled: string;
  /** Flowchart connector line + connection pill (idle / active). */
  edgeLine: string;
  edgePill: string;
  edgeActive: string;
  startBadge: string;
  endBadge: string;
  /** Destructive menu/button items. */
  dangerItem: string;
  /** Amber action button (manual-mode exit). */
  warnAction: string;
  /** Step-editor enabled/disabled toggle chip. */
  enabledBtn: string;
  disabledBtn: string;
  removeBtn: string;
  /** Border of a parameter field that LAMMPS needs but is blank. */
  invalidField: string;
  /** Text of the note under such a field. */
  invalidText: string;
  /** Left accent of a notebook cell whose run failed. */
  errorAccent: string;
  /** Progress bar: the empty track and the filled part. */
  track: string;
  trackFill: string;
  /** Keyboard focus ring (focus-visible, 2 px) of the script tabs. */
  tabFocusRing: string;
  /** Loading spinner ring. */
  loader: string;
  /** Inline error alert. */
  errorBox: string;
}

const DARK: ThemeTokens = {
  bg: 'bg-[#16130f]',
  text: 'text-[#ede5d8]',
  panel: 'bg-[#1e1913] border-[#332a1f]',
  headerText: 'text-[#e5dccd]',
  muted: 'text-[#a3937f]',
  card: 'bg-[#241e16] border-[#332a1f]',
  input:
    'bg-[#14110c] border-[#453a2b] text-[#ece4d6] placeholder:text-[#8a8174] focus:border-[#7fa66b]',
  button: 'bg-[#2a2318] hover:bg-[#342b1d] border border-[#3f3526]',
  chip: 'bg-[#241e16] border border-[#3f3526]',
  accent: 'bg-[#567a46] hover:bg-[#659054] text-[#f2f6ee]',
  go: 'bg-[#47693b] hover:bg-[#557c47] text-white',
  active: 'bg-[#31402a] border border-[#7fa66b] text-[#c4ddb2]',
  warn: 'border-[#6b5124]/60 bg-[#332612]/50 text-[#e4b877]',
  danger: 'text-[#cf8b76] hover:bg-[#3a241c]',
  divider: 'border-[#332a1f]',
  stat: 'bg-[#241e16]',
  accentText: 'text-[#9dc487]',
  accentCode: 'text-[#a9cba0]',
  accentSoft: 'bg-[#31402a] text-[#c4ddb2] ring-1 ring-[#7fa66b]',
  hoverSurface: 'hover:bg-[#342b1d]/60 hover:text-[#ede5d8]',
  borderStrong: 'border-[#453a2b]',
  focusRing: 'focus:ring-[#7fa66b]',
  toggleOn: 'bg-[#567a46]',
  toggleOff: 'bg-[#453a2b]',
  chipIdle: 'bg-[#342b1d]/60 text-[#a9cba0] hover:bg-[#342b1d] hover:text-[#c4ddb2]',
  nodeCard: 'border-[#3f3526] bg-[#1e1913]/60 hover:border-[#659054]',
  nodeDisabled: 'border-[#332a1f] bg-[#1e1913]/20',
  edgeLine: 'bg-[#453a2b]',
  edgePill:
    'border-[#3f3526] bg-[#241e16] text-[#a3937f] hover:border-[#7fa66b] hover:text-[#c4ddb2]',
  edgeActive: 'border-[#7fa66b] bg-[#31402a] text-[#c4ddb2]',
  startBadge: 'border border-[#47693b]/60 bg-[#22301c]/40 text-[#9dc487]',
  endBadge: 'border border-[#6b5124]/60 bg-[#332612]/40 text-[#e4b877]',
  dangerItem: 'text-[#cf8b76] hover:bg-[#3a241c]/60',
  warnAction: 'text-[#e4b877] bg-[#332612]/60 hover:bg-[#332612]',
  enabledBtn: 'text-[#9dc487] bg-[#342b1d]/60 hover:bg-[#342b1d]',
  disabledBtn: 'text-[#a3937f] bg-[#342b1d]/60 hover:bg-[#342b1d]',
  removeBtn: 'text-[#cf8b76] hover:bg-[#3a241c]/40',
  invalidField: 'border-[#cf8b76] focus:border-[#cf8b76]',
  invalidText: 'text-[#e8a68f]',
  errorAccent: 'border-l-4 border-l-[#cf8b76]',
  track: 'bg-[#342b1d]',
  trackFill: 'bg-[#7fa66b]',
  tabFocusRing: 'focus-visible:ring-2 focus-visible:ring-[#9dc487]',
  loader: 'border-[#453a2b] border-t-[#7fa66b]',
  errorBox: 'bg-[#3a1f16]/40 border border-[#6b3a2a]/60 text-[#e8a68f]',
};

const LIGHT: ThemeTokens = {
  bg: 'bg-[#f4efe6]',
  text: 'text-[#2e2920]',
  panel: 'bg-[#fbf8f1] border-[#e0d7c6]',
  headerText: 'text-[#2e2920]',
  muted: 'text-[#6b6053]',
  card: 'bg-white border-[#e6ddcc]',
  input:
    'bg-[#fffdf8] border-[#d8cdb8] text-[#2e2920] placeholder:text-[#6b6053] focus:border-[#4e7a41]',
  button: 'bg-[#efe9dc] hover:bg-[#e5ddcb] border border-[#ddd2bd]',
  chip: 'bg-[#f3eee2] border border-[#ddd2bd]',
  accent: 'bg-[#4e7a41] hover:bg-[#5b8c4c] text-white',
  go: 'bg-[#40663a] hover:bg-[#4c7842] text-white',
  active: 'bg-[#e7efdf] border border-[#4e7a41] text-[#3c5c32]',
  warn: 'border-[#caa15c] bg-[#f7ecd7] text-[#7a5716]',
  danger: 'text-[#a4502f] hover:bg-[#f3e0d8]',
  divider: 'border-[#e4dbc9]',
  stat: 'bg-[#efe9dc]',
  accentText: 'text-[#456b39]',
  accentCode: 'text-[#3c5c32]',
  accentSoft: 'bg-[#e7efdf] text-[#3c5c32] ring-1 ring-[#4e7a41]',
  hoverSurface: 'hover:bg-[#e5ddcb]/70 hover:text-[#2e2920]',
  borderStrong: 'border-[#d8cdb8]',
  focusRing: 'focus:ring-[#4e7a41]',
  toggleOn: 'bg-[#4e7a41]',
  toggleOff: 'bg-[#c9bda6]',
  chipIdle: 'bg-[#efe9dc] text-[#3c5c32] hover:bg-[#e5ddcb] hover:text-[#2e2920]',
  nodeCard: 'border-[#ddd2bd] bg-white hover:border-[#5b8c4c]',
  nodeDisabled: 'border-[#e4dbc9] bg-[#f4efe6]/60',
  edgeLine: 'bg-[#d8cdb8]',
  edgePill:
    'border-[#ddd2bd] bg-[#fbf8f1] text-[#6b6053] hover:border-[#4e7a41] hover:text-[#3c5c32]',
  edgeActive: 'border-[#4e7a41] bg-[#e7efdf] text-[#3c5c32]',
  startBadge: 'border border-[#4e7a41]/50 bg-[#e7efdf] text-[#3c5c32]',
  endBadge: 'border border-[#caa15c]/70 bg-[#f7ecd7] text-[#7a5716]',
  dangerItem: 'text-[#a4502f] hover:bg-[#f3e0d8]',
  warnAction: 'text-[#7a5716] bg-[#f7ecd7] hover:bg-[#f0e2c2]',
  enabledBtn: 'text-[#456b39] bg-[#efe9dc] hover:bg-[#e5ddcb]',
  disabledBtn: 'text-[#6b6053] bg-[#efe9dc] hover:bg-[#e5ddcb]',
  removeBtn: 'text-[#a4502f] hover:bg-[#f3e0d8]',
  invalidField: 'border-[#a4502f] focus:border-[#a4502f]',
  invalidText: 'text-[#8f3b1f]',
  errorAccent: 'border-l-4 border-l-[#a4502f]',
  track: 'bg-[#e5ddcb]',
  trackFill: 'bg-[#4e7a41]',
  tabFocusRing: 'focus-visible:ring-2 focus-visible:ring-[#456b39]',
  loader: 'border-[#d8cdb8] border-t-[#4e7a41]',
  errorBox: 'bg-[#f7e3dd] border border-[#d9a08c] text-[#8f3b1f]',
};

/*
 * Midnight: cool blue-slate surfaces, teal accent, amber highlight. This is
 * the one deliberately blue-tinted set; the warm-palette test skips it.
 */
const MIDNIGHT: ThemeTokens = {
  bg: 'bg-[#0c1322]',
  text: 'text-[#e2eaf4]',
  panel: 'bg-[#121b2d] border-[#243450]',
  headerText: 'text-[#d6e0ee]',
  muted: 'text-[#93a4bb]',
  card: 'bg-[#1a2539] border-[#243450]',
  input:
    'bg-[#0a1020] border-[#34466a] text-[#dde6f1] placeholder:text-[#7d8ea6] focus:border-[#2fb3b0]',
  button: 'bg-[#1d2a42] hover:bg-[#22314b] border border-[#2e3f5c]',
  chip: 'bg-[#1a2539] border border-[#2e3f5c]',
  accent: 'bg-[#1d6b70] hover:bg-[#24807f] text-[#eefbfb]',
  go: 'bg-[#1a5a5e] hover:bg-[#226c70] text-white',
  active: 'bg-[#16353d] border border-[#2fb3b0] text-[#bdeeed]',
  warn: 'border-[#7a5a22]/60 bg-[#33280f]/50 text-[#f0c57f]',
  danger: 'text-[#e8977f] hover:bg-[#3a2024]',
  divider: 'border-[#243450]',
  stat: 'bg-[#1a2539]',
  accentText: 'text-[#7fd6d6]',
  accentCode: 'text-[#a6e3e0]',
  accentSoft: 'bg-[#16353d] text-[#bdeeed] ring-1 ring-[#2fb3b0]',
  hoverSurface: 'hover:bg-[#22314b]/60 hover:text-[#e2eaf4]',
  borderStrong: 'border-[#34466a]',
  focusRing: 'focus:ring-[#2fb3b0]',
  toggleOn: 'bg-[#1d6b70]',
  toggleOff: 'bg-[#34466a]',
  chipIdle: 'bg-[#22314b]/60 text-[#a6e3e0] hover:bg-[#22314b] hover:text-[#bdeeed]',
  nodeCard: 'border-[#2e3f5c] bg-[#121b2d]/60 hover:border-[#24807f]',
  nodeDisabled: 'border-[#243450] bg-[#121b2d]/20',
  edgeLine: 'bg-[#34466a]',
  edgePill:
    'border-[#2e3f5c] bg-[#1a2539] text-[#93a4bb] hover:border-[#2fb3b0] hover:text-[#bdeeed]',
  edgeActive: 'border-[#2fb3b0] bg-[#16353d] text-[#bdeeed]',
  startBadge: 'border border-[#1d6b70]/60 bg-[#10282e]/40 text-[#7fd6d6]',
  endBadge: 'border border-[#7a5a22]/60 bg-[#33280f]/40 text-[#f0c57f]',
  dangerItem: 'text-[#e8977f] hover:bg-[#3a2024]/60',
  warnAction: 'text-[#f0c57f] bg-[#33280f]/60 hover:bg-[#33280f]',
  enabledBtn: 'text-[#7fd6d6] bg-[#22314b]/60 hover:bg-[#22314b]',
  disabledBtn: 'text-[#93a4bb] bg-[#22314b]/60 hover:bg-[#22314b]',
  removeBtn: 'text-[#e8977f] hover:bg-[#3a2024]/40',
  invalidField: 'border-[#e8977f] focus:border-[#e8977f]',
  invalidText: 'text-[#f0ae9c]',
  errorAccent: 'border-l-4 border-l-[#e8977f]',
  track: 'bg-[#22314b]',
  trackFill: 'bg-[#2fb3b0]',
  tabFocusRing: 'focus-visible:ring-2 focus-visible:ring-[#7fd6d6]',
  loader: 'border-[#34466a] border-t-[#2fb3b0]',
  errorBox: 'bg-[#3a1f24]/40 border border-[#6b3a40]/60 text-[#f0ae9c]',
};

/*
 * Solarized light: base3 page, base2 surfaces, base02 text, base01 muted.
 * Blue and teal accents are darkened from the canonical Solarized values so
 * accent text clears 4.5:1 on the surfaces; orange (darkened) is the highlight.
 */
const SOLARIZED: ThemeTokens = {
  bg: 'bg-[#fdf6e3]',
  text: 'text-[#073642]',
  panel: 'bg-[#eee8d5] border-[#d8d0ba]',
  headerText: 'text-[#073642]',
  muted: 'text-[#43575f]',
  card: 'bg-[#fffbf0] border-[#d8d0ba]',
  input:
    'bg-[#fffbf0] border-[#c9bfa6] text-[#073642] placeholder:text-[#43575f] focus:border-[#174f80]',
  button: 'bg-[#e4dcc8] hover:bg-[#dbd2bb] border border-[#cfc5ab]',
  chip: 'bg-[#eee8d5] border border-[#cfc5ab]',
  accent: 'bg-[#1f6fa8] hover:bg-[#1a639a] text-white',
  go: 'bg-[#1d6b66] hover:bg-[#185c57] text-white',
  active: 'bg-[#dfeaf0] border border-[#174f80] text-[#174f80]',
  warn: 'border-[#cb9a4b] bg-[#f6e3cf] text-[#8a3d10]',
  danger: 'text-[#a8321a] hover:bg-[#f6dcd0]',
  divider: 'border-[#ddd5bf]',
  stat: 'bg-[#eee8d5]',
  accentText: 'text-[#174f80]',
  accentCode: 'text-[#1d5f58]',
  accentSoft: 'bg-[#dfeaf0] text-[#174f80] ring-1 ring-[#174f80]',
  hoverSurface: 'hover:bg-[#e4dcc8]/70 hover:text-[#073642]',
  borderStrong: 'border-[#c9bfa6]',
  focusRing: 'focus:ring-[#174f80]',
  toggleOn: 'bg-[#1f6fa8]',
  toggleOff: 'bg-[#c9bfa6]',
  chipIdle: 'bg-[#e4dcc8] text-[#1d5f58] hover:bg-[#dbd2bb] hover:text-[#073642]',
  nodeCard: 'border-[#cfc5ab] bg-[#fffbf0] hover:border-[#174f80]',
  nodeDisabled: 'border-[#ddd5bf] bg-[#fdf6e3]/60',
  edgeLine: 'bg-[#c9bfa6]',
  edgePill:
    'border-[#cfc5ab] bg-[#fffbf0] text-[#43575f] hover:border-[#174f80] hover:text-[#174f80]',
  edgeActive: 'border-[#174f80] bg-[#dfeaf0] text-[#174f80]',
  startBadge: 'border border-[#1d6b66]/50 bg-[#dcebe6] text-[#1d5f58]',
  endBadge: 'border border-[#cb9a4b]/70 bg-[#f6e3cf] text-[#8a3d10]',
  dangerItem: 'text-[#a8321a] hover:bg-[#f6dcd0]',
  warnAction: 'text-[#8a3d10] bg-[#f6e3cf] hover:bg-[#f0d6b8]',
  enabledBtn: 'text-[#174f80] bg-[#e4dcc8] hover:bg-[#dbd2bb]',
  disabledBtn: 'text-[#43575f] bg-[#e4dcc8] hover:bg-[#dbd2bb]',
  removeBtn: 'text-[#a8321a] hover:bg-[#f6dcd0]',
  invalidField: 'border-[#a8321a] focus:border-[#a8321a]',
  invalidText: 'text-[#8f2a12]',
  errorAccent: 'border-l-4 border-l-[#a8321a]',
  track: 'bg-[#dbd2bb]',
  trackFill: 'bg-[#1f6fa8]',
  tabFocusRing: 'focus-visible:ring-2 focus-visible:ring-[#174f80]',
  loader: 'border-[#cfc5ab] border-t-[#1f6fa8]',
  errorBox: 'bg-[#f7e0d6] border border-[#d9a08c] text-[#8f2a12]',
};

/*
 * High contrast (low vision): pure black surfaces, white text, yellow
 * accent and focus. Every text token clears 7:1 (WCAG AAA) on its surfaces;
 * borders are strong enough to read as non-text UI boundaries (3:1+).
 */
const HIGH_CONTRAST: ThemeTokens = {
  bg: 'bg-[#000000]',
  text: 'text-[#ffffff]',
  panel: 'bg-[#0a0a0a] border-[#7a7a7a]',
  headerText: 'text-[#ffffff]',
  muted: 'text-[#d4d4d4]',
  card: 'bg-[#111111] border-[#7a7a7a]',
  input:
    'bg-[#000000] border-[#bdbdbd] text-[#ffffff] placeholder:text-[#b3b3b3] focus:border-[#ffd60a]',
  button: 'bg-[#1a1a1a] hover:bg-[#262626] border border-[#7a7a7a]',
  chip: 'bg-[#111111] border border-[#7a7a7a]',
  accent: 'bg-[#ffd60a] hover:bg-[#ffe14d] text-[#000000]',
  go: 'bg-[#ffffff] hover:bg-[#e6e6e6] text-[#000000]',
  active: 'bg-[#1a1a00] border-2 border-[#ffd60a] text-[#ffe14d]',
  warn: 'border-[#ffd60a]/70 bg-[#1a1600] text-[#ffd60a]',
  danger: 'text-[#ff9e8f] hover:bg-[#2a0f0c]',
  divider: 'border-[#4d4d4d]',
  stat: 'bg-[#111111]',
  accentText: 'text-[#ffd60a]',
  accentCode: 'text-[#ffe14d]',
  accentSoft: 'bg-[#1a1a00] text-[#ffe14d] ring-1 ring-[#ffd60a]',
  hoverSurface: 'hover:bg-[#262626] hover:text-[#ffffff]',
  borderStrong: 'border-[#bdbdbd]',
  focusRing: 'focus:ring-[#ffd60a]',
  toggleOn: 'bg-[#ffd60a]',
  toggleOff: 'bg-[#5c5c5c]',
  chipIdle: 'bg-[#1a1a1a] text-[#ffe14d] hover:bg-[#262626] hover:text-[#ffffff]',
  nodeCard: 'border-[#7a7a7a] bg-[#0a0a0a] hover:border-[#ffd60a]',
  nodeDisabled: 'border-[#4d4d4d] bg-[#0a0a0a]',
  edgeLine: 'bg-[#7a7a7a]',
  edgePill:
    'border-[#7a7a7a] bg-[#111111] text-[#d4d4d4] hover:border-[#ffd60a] hover:text-[#ffe14d]',
  edgeActive: 'border-[#ffd60a] bg-[#1a1a00] text-[#ffe14d]',
  startBadge: 'border border-[#ffd60a] bg-[#000000] text-[#ffd60a]',
  endBadge: 'border border-[#ffffff] bg-[#000000] text-[#ffffff]',
  dangerItem: 'text-[#ff9e8f] hover:bg-[#2a0f0c]',
  warnAction: 'text-[#000000] bg-[#ffd60a] hover:bg-[#ffe14d]',
  enabledBtn: 'text-[#ffe14d] bg-[#1a1a1a] hover:bg-[#262626]',
  disabledBtn: 'text-[#d4d4d4] bg-[#1a1a1a] hover:bg-[#262626]',
  removeBtn: 'text-[#ff9e8f] hover:bg-[#2a0f0c]',
  invalidField: 'border-[#ff9e8f] focus:border-[#ff9e8f]',
  invalidText: 'text-[#ffb8ac]',
  errorAccent: 'border-l-4 border-l-[#ff9e8f]',
  track: 'bg-[#262626]',
  trackFill: 'bg-[#ffd60a]',
  tabFocusRing: 'focus-visible:ring-2 focus-visible:ring-[#ffd60a]',
  loader: 'border-[#5c5c5c] border-t-[#ffd60a]',
  errorBox: 'bg-[#2a0f0c] border border-[#ff9e8f] text-[#ffb8ac]',
};

const TOKEN_SETS: Record<Theme, ThemeTokens> = {
  dark: DARK,
  light: LIGHT,
  midnight: MIDNIGHT,
  solarized: SOLARIZED,
  'high-contrast': HIGH_CONTRAST,
};

export const getThemeTokens = (theme: Theme): ThemeTokens => TOKEN_SETS[theme] ?? DARK;

/** The page background hex of a theme (for <meta name="theme-color">). */
export const themeColor = (theme: Theme): string => {
  const m = getThemeTokens(theme).bg.match(/#[0-9a-fA-F]{6}/);
  return m ? m[0] : PALETTE.dark.base;
};

const isThemeId = (v: unknown): v is Theme => THEMES.some(t => t.id === v);

/** Initial theme honoring the persisted choice, then the OS preference. */
export const initialTheme = (): Theme => {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (isThemeId(stored)) return stored;
  } catch {
    /* storage unavailable */
  }
  return prefersLightTheme() ? 'light' : 'dark';
};

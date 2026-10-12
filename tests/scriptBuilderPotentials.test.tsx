/**
 * The Script Builder's library-file picker and the library-aware lint:
 *  - which library files a pair style offers (filterByStyles, isChoosable),
 *  - the pair_coeff text a choice writes (coeffsForPick: file, then element names in the file's order),
 *  - the units warning of validateScript (units/potential-file-units), quoted from the pair style doc,
 *  - the picker component itself (rendered in jsdom against a stubbed index.json).
 */
import { describe, it, expect, beforeAll, beforeEach, afterEach, afterAll, vi } from 'vitest';
import React, { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { join } from 'node:path';
import { buildIndex } from '../scripts/potential-index.mjs';
import {
  coeffsForPick, coeffFormOf, filterByStyles, isChoosable, pairStyleWords, snapParamName, unitsDocFor,
  type PotentialEntry, type PotentialIndex,
} from '../src/lammps/potentialIndex';
import { validateScript, type Diagnostic } from '../src/lammps/validate';
import { PotentialPicker } from '../src/components/workbench/builder/PotentialPicker';
import { getThemeTokens } from '../src/theme';
import { COMMAND_BY_ID } from '../src/lammps/catalog';

(globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const DIR = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const INDEX: PotentialIndex = {
  format: 1,
  source: { upstream: '', commit: 'b332f28', license: 'GPL-2.0', note: '' },
  files: buildIndex(DIR).files as PotentialEntry[],
};
const byName = (name: string): PotentialEntry => {
  const e = INDEX.files.find((f) => f.name === name);
  if (!e) throw new Error(`fixture ${name} missing from the library index`);
  return e;
};

/** A small index for the cases that need exact entries (independent of the bundled files). */
const SYNTH: PotentialIndex = {
  ...INDEX,
  files: [
    { name: 'Fe.snapcoeff', bytes: 10, kind: 'snap', elements: ['Fe'], styles: ['snap'], units: 'metal' },
    { name: 'Fe.snapparam', bytes: 10, kind: 'snap', elements: [], styles: ['snap'], units: 'metal' },
    { name: 'Ni.snapcoeff', bytes: 10, kind: 'snap', elements: ['Ni'], styles: ['snap'] },
    { name: 'Real.reax', bytes: 10, kind: 'reaxff', elements: [], styles: ['reaxff'], units: 'real' },
  ],
};

describe('picker: which library files a pair style offers', () => {
  it('keeps the files that read the style, and every file for an empty style list', () => {
    const names = filterByStyles(INDEX.files, ['eam/alloy']).map((f) => f.name);
    expect(names).toContain('CuNi.eam.alloy');
    expect(names).not.toContain('Cu_u3.eam');
    expect(filterByStyles(INDEX.files, [])).toHaveLength(INDEX.files.length);
  });

  it('a hybrid list matches any of its sub-style words', () => {
    const words = pairStyleWords('hybrid/overlay eam/alloy 10.0 sw');
    const names = filterByStyles(INDEX.files, words).map((f) => f.name);
    expect(names).toContain('CuNi.eam.alloy');
    expect(names).toContain('Si.sw');
  });

  it('does not offer README files or the .snapparam half of a SNAP pair', () => {
    expect(isChoosable(byName('Cu_Zuo_JPCA2020.snapcoeff'))).toBe(true);
    expect(isChoosable(byName('Cu_Zuo_JPCA2020.snapparam'))).toBe(false);
    expect(INDEX.files.filter(isChoosable).every((f) => f.kind !== 'doc')).toBe(true);
  });
});

describe('picker: the pair_coeff text a choice writes', () => {
  it('eam/alloy: file, then the element names in the file order (pair_eam.html: "pair_coeff * * NiAlH_jea.eam.alloy Ni Al Ni Ni")', () => {
    const entry = byName('CuNi.eam.alloy');
    expect(entry.elements).toEqual(['Ni', 'Cu']);
    expect(coeffsForPick(entry, ['eam/alloy'], INDEX)).toBe('CuNi.eam.alloy Ni Cu');
    expect(coeffFormOf('eam/alloy')).toBe('elements');
  });

  it('funcfl eam takes the file alone (pair_eam.html: "pair_coeff * * cuu3")', () => {
    expect(coeffsForPick(byName('Cu_u3.eam'), ['eam'], INDEX)).toBe('Cu_u3.eam');
  });

  it('tersoff and sw take the element names too (pair_tersoff.html: "pair_coeff * * Si.tersoff Si")', () => {
    expect(coeffsForPick(byName('Si.tersoff'), ['tersoff'], INDEX)).toBe('Si.tersoff Si');
    expect(coeffsForPick(byName('Si.sw'), ['sw'], INDEX)).toBe('Si.sw Si');
  });

  it('snap writes the coefficient file, its matching .snapparam, then the elements (pair_snap.html)', () => {
    expect(snapParamName('Fe.snapcoeff')).toBe('Fe.snapparam');
    expect(coeffsForPick(SYNTH.files[0], ['snap'], SYNTH)).toBe('Fe.snapcoeff Fe.snapparam Fe');
    // no matching param in the library: the file and the elements only, the user adds the param
    expect(coeffsForPick(SYNTH.files[2], ['snap'], SYNTH)).toBe('Ni.snapcoeff Ni');
  });

  it('a file whose element list is empty gets no element names (the user writes them)', () => {
    expect(coeffsForPick(SYNTH.files[3], ['reaxff'], SYNTH)).toBe('Real.reax');
  });

  it('inside pair_style hybrid the sub-style name comes before the file (pair_sw.html: "pair_coeff * * sw 1 mW_xL.sw mW NULL")', () => {
    expect(coeffsForPick(byName('CuNi.eam.alloy'), ['hybrid', 'eam/alloy'], INDEX)).toBe('eam/alloy CuNi.eam.alloy Ni Cu');
  });
});

describe('pair_style for a library style: no stale cutoff', () => {
  const build = (style: string, args: string) => COMMAND_BY_ID.pair_style_popular.build({ style, args });

  it('eam/alloy, tersoff, meam and snap take no cutoff (pair_eam.html Syntax: "pair_style eam/alloy")', () => {
    expect(build('eam/alloy', '2.5')).toEqual(['pair_style eam/alloy']);
    expect(build('tersoff', '2.5')).toEqual(['pair_style tersoff']);
    expect(build('meam', '2.5')).toEqual(['pair_style meam']);
    expect(build('snap', '')).toEqual(['pair_style snap']);
  });

  it('a style that takes a cutoff keeps it', () => {
    expect(build('lj/cut', '2.5')).toEqual(['pair_style lj/cut 2.5']);
  });
});

describe('validator: the units rule for library files', () => {
  const units = (d: Diagnostic[]) => d.filter((x) => x.rule === 'units/potential-file-units');
  const script = (u: string | null) => [
    ...(u ? [`units ${u}`] : []),
    'atom_style atomic',
    'lattice fcc 3.6',
    'region box block 0 4 0 4 0 4',
    'create_box 2 box',
    'pair_style eam/alloy',
    'pair_coeff * * CuNi.eam.alloy Ni Cu',
    'mass * 58.7',
    'run 0',
  ].join('\n');

  it('warns when the file is metal and the script runs in real units, citing the pair_eam page', () => {
    const d = units(validateScript(script('real'), { potentials: INDEX }));
    expect(d).toHaveLength(1);
    expect(d[0].level).toBe('warning');
    expect(d[0].doc).toBe('https://docs.lammps.org/pair_eam.html');
    expect(d[0].message).toContain('All of these files are parameterized in terms of LAMMPS metal units.');
  });

  it('no warning when the units match', () => {
    expect(units(validateScript(script('metal'), { potentials: INDEX }))).toEqual([]);
  });

  it('a script without a units command runs in lj units (units.html Default section), so metal files warn', () => {
    expect(units(validateScript(script(null), { potentials: INDEX }))).toHaveLength(1);
  });

  it('is silent without the library index (the lint runs offline)', () => {
    expect(units(validateScript(script('real')))).toEqual([]);
  });

  it('a file the library does not have keeps the current behaviour: no new diagnostic', () => {
    const text = script('real').replace('CuNi.eam.alloy Ni Cu', 'nothere.eam.alloy Ni Cu');
    expect(validateScript(text, { potentials: INDEX })).toEqual(validateScript(text));
  });

  it('a reaxff file in metal units cites the pair_reaxff page', () => {
    const text = [
      'units metal', 'atom_style charge', 'lattice fcc 3.6', 'region box block 0 4 0 4 0 4', 'create_box 4 box',
      'pair_style reaxff NULL', 'pair_coeff * * ffield.reax.cho C H O N', 'fix q all qeq/reaxff 1 0.0 10.0 1e-6 reaxff', 'run 0',
    ].join('\n');
    const d = units(validateScript(text, { potentials: INDEX }));
    expect(d).toHaveLength(1);
    expect(d[0].doc).toBe('https://docs.lammps.org/pair_reaxff.html');
    expect(unitsDocFor('reaxff')?.sentence).toContain('parameterized for real units.');
  });

  it('styles whose doc page states no units get no rule', () => {
    expect(unitsDocFor('morse')).toBeNull();
  });
});

describe('picker component', () => {
  let container: HTMLDivElement;
  let root: Root;
  const theme = getThemeTokens('light');
  const realFetch = globalThis.fetch;

  beforeAll(() => {
    vi.stubGlobal('fetch', async (url: string) => {
      if (!String(url).endsWith('potentials/index.json')) throw new Error(`unexpected fetch ${url}`);
      return { ok: true, json: async () => INDEX } as Response;
    });
  });
  afterAll(() => {
    vi.unstubAllGlobals();
    globalThis.fetch = realFetch;
  });
  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });
  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  const flush = async () => { await act(async () => { await new Promise((r) => setTimeout(r, 0)); }); };
  const open = async (pairWords: string[], onChoose = vi.fn()) => {
    await act(async () => {
      root.render(<PotentialPicker ct={theme} idPrefix="t1" pairWords={pairWords} onChoose={onChoose} />);
    });
    await act(async () => { container.querySelector<HTMLButtonElement>('#t1-potential-open')?.click(); });
    await flush();
    return onChoose;
  };
  const itemNames = () => [...container.querySelectorAll('ul[aria-label="Matching library files"] button')]
    .map((b) => b.getAttribute('aria-label')?.replace(/^Use /, ''));

  it('lists the files of the pair style above the step, and every file with "Show all styles"', async () => {
    await open(['eam/alloy']);
    const matching = itemNames();
    expect(matching).toContain('CuNi.eam.alloy');
    expect(matching).not.toContain('Cu_u3.eam');
    const check = container.querySelector<HTMLInputElement>('#t1-potential-all');
    await act(async () => { check?.click(); });
    expect(itemNames()).toContain('Cu_u3.eam');
  });

  it('searches by element and filters the list', async () => {
    await open(['eam/alloy']);
    const search = container.querySelector<HTMLInputElement>('#t1-potential-search');
    expect(search).not.toBeNull();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(search, 'Cu');
      search?.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(itemNames()).toEqual(expect.arrayContaining(['CuNi.eam.alloy']));
    expect(itemNames().every((n) => n?.toLowerCase().includes('cu') || n?.includes('Cu'))).toBe(true);
  });

  it('a choice hands the coefficient text (file, then elements in file order) to the step', async () => {
    const onChoose = await open(['eam/alloy']);
    await act(async () => {
      container.querySelector<HTMLButtonElement>('#t1-potential-item-CuNi\\.eam\\.alloy')?.click();
    });
    expect(onChoose).toHaveBeenCalledWith('CuNi.eam.alloy Ni Cu', expect.objectContaining({ name: 'CuNi.eam.alloy' }));
  });
});

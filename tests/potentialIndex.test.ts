import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
// the index script is plain ESM (scripts/potential-index.mjs), imported by vite.config.ts too
import { buildIndex, elementsOf, indexEntry, kindOf } from '../scripts/potential-index.mjs';

const DIR = join(__dirname, '..', 'third_party', 'lammps', 'potentials');

describe('potential index: kinds and styles from the file name', () => {
  it('names the families the notebook can run', () => {
    expect(kindOf('Cu_u3.eam')).toBe('eam');
    expect(kindOf('AlCu.eam.alloy')).toBe('eam/alloy');
    expect(kindOf('CuZr_mm.eam.fs')).toBe('eam/fs');
    expect(kindOf('Si.tersoff')).toBe('tersoff');
    expect(kindOf('Si.tersoff.mod')).toBe('tersoff');
    expect(kindOf('Si.sw')).toBe('sw');
    expect(kindOf('Cu.meam')).toBe('meam');
    expect(kindOf('Cu_Zuo_JPCA2020.snapcoeff')).toBe('snap');
    expect(kindOf('ffield.reax.cho')).toBe('reaxff');
    expect(kindOf('GaN_tersoff.poly')).toBe('polymorphic');
    expect(kindOf('README')).toBe('doc');
  });

  it('keeps unknown files listed as unknown', () => {
    expect(kindOf('TMDAu.SAIP')).toBe('unknown');
    expect(indexEntry('TMDAu.SAIP', 'x', 1)).toMatchObject({ kind: 'unknown', styles: [] });
  });

  it('gives the pair styles that read the file', () => {
    expect(indexEntry('Si.tersoff.mod', '', 1).styles).toEqual(['tersoff/mod']);
    expect(indexEntry('Si.tersoff.modc', '', 1).styles).toEqual(['tersoff/mod/c']);
    expect(indexEntry('CH.airebo-m', '', 1).styles).toEqual(['airebo/morse']);
    expect(indexEntry('Cu_u3.eam', '', 1).styles).toEqual(['eam']);
  });
});

describe('potential index: elements the file states', () => {
  it('reads the funcfl element from its atomic number on line 2', () => {
    const text = 'DATE: 2007-06-11 UNITS: metal\n   29     63.550         3.6150    FCC\n';
    expect(elementsOf('eam', text)).toEqual(['Cu']);
  });

  it('reads the setfl element list on line 4', () => {
    const text = '# DATE: 2008-10-01 UNITS: metal\n##Al-Cu##\n####\n2 Al Cu\n1000 1.0 3000 2.2 6.6\n';
    expect(elementsOf('eam/alloy', text)).toEqual(['Al', 'Cu']);
  });

  it('reads the element symbols of three-body rows', () => {
    const text = '# comment\nSi Si Si 3.0 1.0 0.0 1.0e5 16.0 -0.59825 0.78734 1.1e-16 0.5 2.5 0.2 3.0 2.4799 1830.8\n';
    expect(elementsOf('tersoff', text)).toEqual(['Si']);
  });

  it('states no element when the file does not', () => {
    expect(elementsOf('meam', 'rc = 4.0\ndelr = 0.1\n')).toEqual([]);
  });
});

describe('potential index: the shipped directory', () => {
  it('lists every file of the potentials directory once (README files as doc, dotfiles left out)', () => {
    const index = buildIndex(DIR);
    const names = index.files.map((f) => f.name);
    expect(new Set(names).size).toBe(names.length);
    expect(names).toContain('Cu_u3.eam');
    expect(names).toContain('README');
    expect(names.some((n) => n.startsWith('.'))).toBe(false);
    expect(index.source.commit).toBe('b332f28');
    expect(index.source.license).toBe('GPL-2.0');
    expect(index.files.find((f) => f.name === 'README')?.kind).toBe('doc');
  });

  it('bytes and elements match the file on disk', () => {
    const e = buildIndex(DIR).files.find((f) => f.name === 'AlCu.eam.alloy');
    expect(e?.bytes).toBe(readFileSync(join(DIR, 'AlCu.eam.alloy')).length);
    expect(e?.elements).toEqual(['Al', 'Cu']);
  });

  it('is on disk after the build (dist/potentials) when the build has run', () => {
    const built = join(__dirname, '..', 'dist', 'potentials', 'index.json');
    if (!existsSync(built)) return;
    const onDisk = JSON.parse(readFileSync(built, 'utf8')) as { files: unknown[] };
    expect(onDisk.files.length).toBe(buildIndex(DIR).files.length);
  });
});

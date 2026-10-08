import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parsePDBFile } from '../src/services/pdbParser';
import type { MoleculeData } from '../src/types';

// Repo root — the suite is always run from there (`npm test` / npx vitest).
const repoRoot = process.cwd();

/**
 * Fixed-width ATOM/HETATM line builder (wwPDB format 3.3, sect9):
 * cols 1-6 record, 7-11 serial, 13-16 name, 17 altLoc, 18-20 resName,
 * 22 chainID, 23-26 resSeq, 31-54 xyz, 77-78 element (right-justified).
 */
const atomLine = (opts: {
  record?: 'ATOM' | 'HETATM';
  serial?: number;
  name: string;
  altLoc?: string;
  x: number;
  y?: number;
  z?: number;
  element?: string;
}): string => {
  const serial = String(opts.serial ?? 1).padStart(5);
  const name4 = opts.name.padEnd(4, ' ').slice(0, 4);
  const altLoc = opts.altLoc ?? ' ';
  const resSeq = '   1';
  const xyz = [opts.x, opts.y ?? 0, opts.z ?? 0].map(v => v.toFixed(3).padStart(8)).join('');
  const element = opts.element !== undefined ? ' '.repeat(10) + opts.element.padStart(2) : '';
  return (
    (opts.record ?? 'ATOM').padEnd(6) + serial + ' ' + name4 + altLoc + 'ALA A' + resSeq +
    ' ' + '   ' + xyz + '  1.00  0.00' + element
  );
};

/** Resolved element symbol per parsed atom. */
const elementSymbols = (d: MoleculeData): string[] =>
  d.atoms.map(a => d.atomTypes[a.type]?.element ?? '?');

// Real RCSB entries (wwPDB archive data is CC0 1.0), see
// tests/fixtures/pdb/README.md. 1L2Y is trimmed to its first three models.
const FIXTURES = join(repoRoot, 'tests/fixtures/pdb');
const pdbText = (name: string) => readFileSync(join(FIXTURES, name), 'utf8');
/** The same entry with the element columns (77-78) removed, as old files have. */
const withoutElements = (text: string) =>
  text.split('\n').map(l => (/^(ATOM  |HETATM)/.test(l) ? l.slice(0, 76) : l)).join('\n');

describe('element-from-name heuristic (wwPDB sect9 alignment)', () => {
  // One-letter element names start at column 14 (column 13 blank/digit);
  // two-letter element names start at column 13; ATOM hydrogens fill all
  // four columns.
  const cases: Array<[string, 'ATOM' | 'HETATM', string]> = [
    [' CA ', 'HETATM', 'C'],
    ['CA  ', 'HETATM', 'Ca'],
    ['FE  ', 'HETATM', 'Fe'],
    ['CL  ', 'HETATM', 'Cl'],
    ['1HG1', 'HETATM', 'H'],
    ['HG21', 'ATOM', 'H'],
    [' OXT', 'ATOM', 'O'],
  ];
  for (const [name, record, expected] of cases) {
    it(`"${name}" in an ${record} record -> ${expected}`, () => {
      const data = parsePDBFile(atomLine({ record, name, x: 0 }) + '\nEND\n');
      expect(data.atoms).toHaveLength(1);
      expect(elementSymbols(data)[0]).toBe(expected);
    });
  }

  it('HETATM "HG21" is mercury (the hydrogen rule is ATOM-only)', () => {
    const data = parsePDBFile(atomLine({ record: 'HETATM', name: 'HG21', x: 0 }) + '\nEND\n');
    expect(elementSymbols(data)[0]).toBe('Hg');
  });
});

describe('alternate locations (column 17)', () => {
  it('an altLoc A/B pair keeps only A', () => {
    const pdb =
      atomLine({ serial: 1, name: ' CA ', altLoc: 'A', x: 1 }) + '\n' +
      atomLine({ serial: 2, name: ' CA ', altLoc: 'B', x: 2 }) + '\nEND\n';
    const data = parsePDBFile(pdb);
    expect(data.atoms).toHaveLength(1);
    expect(data.atoms[0].x).toBeCloseTo(1);
  });

  it('blank altLoc plus alternates keeps blank and the first letter', () => {
    const pdb =
      atomLine({ serial: 1, name: ' CA ', x: 1 }) + '\n' +
      atomLine({ serial: 2, name: ' CA ', altLoc: 'A', x: 2 }) + '\n' +
      atomLine({ serial: 3, name: ' CA ', altLoc: 'B', x: 3 }) + '\nEND\n';
    const data = parsePDBFile(pdb);
    expect(data.atoms).toHaveLength(2);
    expect(data.atoms.map(a => a.x)).toEqual([1, 2]);
  });

  it('atoms existing only as a later conformer are dropped too (3NIR semantics)', () => {
    const pdb =
      atomLine({ serial: 1, name: ' N  ', altLoc: 'A', x: 1 }) + '\n' +
      atomLine({ serial: 2, name: ' HG1', altLoc: 'B', x: 2 }) + '\nEND\n';
    const data = parsePDBFile(pdb);
    expect(data.atoms).toHaveLength(1);
    expect(data.atoms[0].x).toBeCloseTo(1);
  });
});

describe('MODEL/ENDMDL multi-model entries', () => {
  const model = (n: number, xs: number[]) =>
    `MODEL     ${n}` + '\n' + xs.map((x, i) => atomLine({ serial: i + 1, name: ' CA ', x })).join('\n') + '\nENDMDL\n';

  it('a two-model inline file gives 2 frames and atoms = model 1', () => {
    const data = parsePDBFile(model(1, [1]) + model(2, [5]) + 'END\n');
    expect(data.frames).toHaveLength(2);
    expect(data.atoms).toHaveLength(1);
    expect(data.frames![0].atoms[0].x).toBeCloseTo(1);
    expect(data.frames![1].atoms[0].x).toBeCloseTo(5);
    expect(data.frames![0].comment).toBe('model 1');
  });

  it('a single-model file sets no frames', () => {
    const data = parsePDBFile(atomLine({ name: ' CA ', x: 1 }) + '\nEND\n');
    expect(data.atoms).toHaveLength(1);
    expect(data.frames).toBeUndefined();
  });

  it('a later model with a different atom count stops the trajectory (no throw)', () => {
    const data = parsePDBFile(model(1, [1, 2]) + model(2, [9]) + model(3, [3, 4]) + 'END\n');
    expect(data.atoms).toHaveLength(2);
    expect(data.frames).toBeUndefined();
  });
});

describe('CONECT fixed-width fields (wwPDB sect10)', () => {
  it('two touching 5-digit serials parse as one bond', () => {
    const pdb =
      atomLine({ serial: 10001, name: ' C  ', x: 0 }) + '\n' +
      atomLine({ serial: 10002, name: ' C  ', x: 1.5 }) + '\n' +
      'CONECT1000110002\nEND\n';
    const data = parsePDBFile(pdb);
    expect(data.bonds).toHaveLength(1);
    expect(data.bonds[0].atom1Id).toBe(1);
    expect(data.bonds[0].atom2Id).toBe(2);
  });

  it('multi-bond CONECT lines still yield one bond per serial', () => {
    const pdb =
      atomLine({ serial: 1, name: ' C  ', x: 0 }) + '\n' +
      atomLine({ serial: 2, name: ' C  ', x: 1.5 }) + '\n' +
      atomLine({ serial: 3, name: ' C  ', x: 3 }) + '\n' +
      'CONECT    1    2    3\nEND\n';
    const data = parsePDBFile(pdb);
    expect(data.bonds).toHaveLength(2);
  });
});

describe('real RCSB entries (tests/fixtures/pdb)', () => {
  it('1L2Y (NMR, first 3 models): 3 frames of 304 atoms, not stacked', () => {
    const data = parsePDBFile(pdbText('1L2Y-models1-3.pdb'));
    expect(data.atoms).toHaveLength(304);
    expect(data.frames).toHaveLength(3);
    const a = data.frames![0].atoms[0];
    const b = data.frames![1].atoms[0];
    expect([a.x, a.y, a.z]).toEqual([-8.901, 4.127, -0.555]);
    expect([b.x, b.y, b.z]).not.toEqual([a.x, a.y, a.z]);
  });

  it('3NIR: alternate locations are not duplicated (750 atoms)', () => {
    expect(parsePDBFile(pdbText('3NIR.pdb')).atoms).toHaveLength(750);
  });

  it('1MBN without element columns: heme iron is Fe, not F', () => {
    const elems = elementSymbols(parsePDBFile(withoutElements(pdbText('1MBN.pdb'))));
    expect(elems.filter(e => e === 'Fe')).toHaveLength(1);
    expect(elems.filter(e => e === 'F')).toHaveLength(0);
  });

  it('1MBN with element columns: one Fe, 1260 atoms', () => {
    const data = parsePDBFile(pdbText('1MBN.pdb'));
    expect(data.atoms).toHaveLength(1260);
    expect(elementSymbols(data).filter(e => e === 'Fe')).toHaveLength(1);
  });

  it('public/examples/benzene.pdb still gives 12 atoms and 12 bonds', () => {
    const p = join(repoRoot, 'public/examples/benzene.pdb');
    expect(existsSync(p)).toBe(true);
    const data = parsePDBFile(readFileSync(p, 'utf8'));
    expect(data.atoms).toHaveLength(12);
    expect(data.bonds).toHaveLength(12);
  });
});

describe('left-justified atom names in ATOM records', () => {
  it('"CA  " and "OG1 " in an ATOM record are carbon and oxygen, not Ca / Og', () => {
    const text = [
      atomLine({ serial: 1, name: 'CA', x: 0, y: 0, z: 0 }),
      atomLine({ serial: 2, name: 'OG1', x: 1.4, y: 0, z: 0 }),
      atomLine({ record: 'HETATM', serial: 3, name: 'CA', x: 6, y: 0, z: 0 }),
    ].join('\n');
    expect(elementSymbols(parsePDBFile(text))).toEqual(['C', 'O', 'Ca']);
  });
});

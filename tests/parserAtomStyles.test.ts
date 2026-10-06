import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { parseDataFile } from '../src/services/parser';

/**
 * Row formats transcribed from docs.lammps.org/read_data.html ("Atoms"
 * section, verified 2026-10-07). Every style gets one canonical row that
 * puts the atom at (1,2,3) with type 2; y = x+1 and z = x+2 in every style.
 *
 * Note on "without the hint": per docs.lammps.org/read_data.html the
 * hint-less layout can only be one of full/molecular/charge/atomic (the
 * unambiguous prefixes). For every other style the documented row is
 * inherently ambiguous against those four, so the spec-correct behaviour is
 * either a coincidentally-correct read (when the columns coincide) or the
 * row being SKIPPED — never a plausible-but-wrong re-read. Both behaviours
 * are asserted below.
 */
const STYLE_CASES = [
  { style: 'angle',      row: '1 7 2 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'atomic',     row: '1 2 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'body',       row: '1 2 1 6.5 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'bond',       row: '1 7 2 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'bpm/sphere', row: '1 7 2 1.2 3.4 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'charge',     row: '1 2 -0.5 1.0 2.0 3.0', type: 2, molId: 1, charge: -0.5 },
  { style: 'dielectric', row: '1 2 0.3 1.0 2.0 3.0 0.1 0.2 0.3 4.0 1.0 0.5 2.0 0.1', type: 2, molId: 1, charge: 0.3 },
  { style: 'dipole',     row: '1 2 0.3 1.0 2.0 3.0 0.1 0.2 0.3', type: 2, molId: 1, charge: 0.3 },
  { style: 'dpd',        row: '1 2 0.75 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'edpd',       row: '1 2 300 1.5 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'electron',   row: '1 2 -1.0 0.5 2.0 1.0 2.0 3.0', type: 2, molId: 1, charge: -1.0 },
  { style: 'ellipsoid',  row: '1 2 1 2.5 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'full',       row: '1 7 2 -0.5 1.0 2.0 3.0', type: 2, molId: 7, charge: -0.5 },
  { style: 'line',       row: '1 7 2 1 2.5 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'mdpd',       row: '1 2 1.1 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'molecular',  row: '1 7 2 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'peri',       row: '1 2 0.9 8.0 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'rheo',       row: '1 2 1 1.05 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'sphere',     row: '1 2 1.2 3.4 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
  { style: 'spin',       row: '1 2 1.0 2.0 3.0 0.0 0.0 1.0 0.5', type: 2, molId: 1, charge: 0 },
  { style: 'template',   row: '1 2 7 1 3 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'tri',        row: '1 7 2 1 2.5 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'hybrid',     row: '1 2 1.0 2.0 3.0 0.5 0.6', type: 2, molId: 1, charge: 0 },
];

/** The four hint-less candidate styles, in priority order. */
const AUTO_CASES = [
  { style: 'full', row: '1 7 2 -0.5 1.0 2.0 3.0', type: 2, molId: 7, charge: -0.5 },
  { style: 'molecular', row: '1 7 2 1.0 2.0 3.0', type: 2, molId: 7, charge: 0 },
  { style: 'charge', row: '1 2 -0.5 1.0 2.0 3.0', type: 2, molId: 1, charge: -0.5 },
  { style: 'atomic', row: '1 2 1.0 2.0 3.0', type: 2, molId: 1, charge: 0 },
];

/**
 * Styles whose documented rows fit NONE of the four hint-less candidates.
 * The parser must skip them, never re-read them under another layout.
 */
const SKIP_CASES = [
  'electron:   1 2 -1.0 0.5 2.0 1.0 2.0 3.0',
  'dipole:     1 2 0.3 1.0 2.0 3.0 0.1 0.2 0.3',
  'template:   1 2 7 1 3 1.0 2.0 3.0',
  'bpm/sphere: 1 7 2 1.2 3.4 1.0 2.0 3.0',
  'line:       1 7 2 1 2.5 1.0 2.0 3.0',
  'tri:        1 7 2 1 2.5 1.0 2.0 3.0',
  'dielectric: 1 2 0.3 1.0 2.0 3.0 0.1 0.2 0.3 4.0 1.0 0.5 2.0 0.1',
  'sphere:     1 2 2.0 1.5 0.0 0.0 0.0',
  'peri:       1 2 0.9 8.0 1.0 2.0 3.0',
  'edpd:       1 2 300.5 1.5 1.0 2.0 3.0',
  'spin:       1 2 1.0 2.0 3.0 0.0 0.0 1.0 0.5',
];

const dataFile = (atoms: string, opts: { header?: string; masses?: string; pre?: string } = {}) =>
  `# generated test file
${opts.header ?? ''}
${opts.pre ?? ''}
${opts.masses ? `Masses\n\n${opts.masses}\n` : ''}
${atoms}
`;

// The real files are LAMMPS examples (GPL-2.0), so they are NOT in this
// repo: point LAMMPS_REALDATA_DIR at a folder holding data.peptide,
// data.micelle and data.body (from lammps/examples) to run these checks.
const REALDATA_DIR = process.env.LAMMPS_REALDATA_DIR ?? '';
const realFile = (name: string): string | null => {
  const p = `${REALDATA_DIR}/${name}`;
  return existsSync(p) ? readFileSync(p, 'utf8') : null;
};
const haveRealFiles = realFile('data.peptide') !== null;

describe('atom style layouts (hinted)', () => {
  for (const c of STYLE_CASES) {
    for (const withFlags of [false, true]) {
      it(`${c.style}${withFlags ? ' + image flags' : ''}: atom at (1,2,3) type 2`, () => {
        const row = c.row + (withFlags ? ' 2 -1 3' : '');
        const result = parseDataFile(dataFile(`Atoms # ${c.style}\n\n${row}`));
        expect(result.atoms).toHaveLength(1);
        const a = result.atoms[0];
        expect(a.id).toBe(1);
        expect(a.type).toBe(c.type);
        expect(a.molId).toBe(c.molId);
        expect(a.charge).toBe(c.charge);
        expect(a.x).toBe(1.0);
        expect(a.y).toBe(2.0);
        expect(a.z).toBe(3.0);
        if (withFlags) {
          if (c.style === 'hybrid') {
            // hybrid rows continue with sub-style values after z, so trailing
            // image flags cannot be told apart from them — never claimed.
            expect(a.ix).toBeUndefined();
          } else {
            expect(a.ix).toBe(2);
            expect(a.iy).toBe(-1);
            expect(a.iz).toBe(3);
          }
        } else {
          expect(a).not.toHaveProperty('ix');
        }
      });
    }
  }
});

describe('atom style layouts (hint-less, whole-section choice)', () => {
  for (const c of AUTO_CASES) {
    for (const withFlags of [false, true]) {
      it(`${c.style}${withFlags ? ' + image flags' : ''} without hint`, () => {
        const row = c.row + (withFlags ? ' 0 0 0' : '');
        const result = parseDataFile(dataFile(`6 atom types\n\nAtoms\n\n${row}`));
        expect(result.atoms).toHaveLength(1);
        const a = result.atoms[0];
        expect(a.type).toBe(c.type);
        expect(a.molId).toBe(c.molId);
        expect(a.charge).toBe(c.charge);
        expect(a.x).toBe(1.0);
        expect(a.y).toBe(2.0);
        expect(a.z).toBe(3.0);
        if (withFlags) {
          expect(a.ix).toBe(0);
          expect(a.iy).toBe(0);
          expect(a.iz).toBe(0);
        }
      });
    }
  }

  it('chooses ONE layout for the whole section; molecular outranks charge when both fit', () => {
    // Both rows fit molecular (id mol type x y z [nx ny nz]) and charge
    // (id type q x y z [nx ny nz]); molecular is the earlier candidate.
    const result = parseDataFile(dataFile(`6 atom types

Atoms

1 2 3 1.0 2.0 3.0
2 2 3 2.0 3.0 4.0 0 0 0
`));
    expect(result.atoms).toHaveLength(2);
    expect(result.atoms[0]).toMatchObject({ molId: 2, type: 3, charge: 0, x: 1.0, y: 2.0, z: 3.0 });
    expect(result.atoms[1]).toMatchObject({ molId: 2, type: 3, x: 2.0, y: 3.0, z: 4.0, ix: 0, iy: 0, iz: 0 });
  });

  it('falls back to the candidate that fits the most rows; misfits are skipped', () => {
    // Two molecular rows + one atomic row: molecular and charge each fit 2,
    // molecular comes first -> the atomic row must be skipped, not re-read.
    const result = parseDataFile(dataFile(`6 atom types

Atoms

1 1 2 1.0 2.0 3.0
2 1 2 2.0 3.0 4.0
3 2 5.0 6.0 7.0
`));
    expect(result.atoms).toHaveLength(2);
    expect(result.atoms[0]).toMatchObject({ type: 2, molId: 1, x: 1.0 });
    expect(result.atoms[1]).toMatchObject({ type: 2, molId: 1, x: 2.0 });
  });

  it('skips rows of other documented styles instead of re-reading them (no hint)', () => {
    for (const line of SKIP_CASES) {
      const idx = line.indexOf(':');
      const name = line.slice(0, idx).trim();
      const row = line.slice(idx + 1).trim();
      const result = parseDataFile(dataFile(`Atoms\n\n${row}\n`));
      expect(result.atoms, `${name} row must be skipped`).toHaveLength(0);
    }
  });
});

describe('regressions for the six verified misreads', () => {
  it('(1) hint-less molecular row with trailing image flags reads x,y,z — not full', () => {
    // Row `2 1 2 0.8 0.6 0.0 0 0 0` is molecular (id mol type x y z nx ny nz);
    // the old guesser tested `full` first (isFloat('0') is true) and read x=0.6.
    const result = parseDataFile(dataFile(`2 atom types

Atoms

2 1 2 0.8 0.6 0.0 0 0 0
`));
    expect(result.atoms).toHaveLength(1);
    expect(result.atoms[0]).toMatchObject({ id: 2, molId: 1, type: 2, x: 0.8, y: 0.6, z: 0.0, ix: 0, iy: 0, iz: 0 });
  });

  it('(2) `Atoms # sphere` reads id type diameter density x y z — x is not the density', () => {
    // Row `1 1 2.0 1.5 0.0 0.0 0.0`: the old code only knew atomic/charge/
    // molecular/full and read it as charge -> x=1.5 (the density).
    const result = parseDataFile(dataFile(`Atoms # sphere\n\n1 1 2.0 1.5 0.0 0.0 0.0\n`));
    expect(result.atoms).toHaveLength(1);
    expect(result.atoms[0]).toMatchObject({ id: 1, type: 1, charge: 0, x: 0.0, y: 0.0, z: 0.0 });
  });

  it('(3) `Atoms # full` with type labels resolves the label, keeps positions', () => {
    // Row `1 1 C 0.5 1.0 2.0 3.0`: the old parser failed the full layout and
    // the retry loop re-read it as charge -> x=0.5 (the charge), charge=NaN.
    const result = parseDataFile(dataFile(`2 atom types

Atom Type Labels

1 C
2 O

Atoms # full

1 1 C 0.5 1.0 2.0 3.0
`));
    expect(result.atoms).toHaveLength(1);
    // label C -> type 1 per the "Atom Type Labels" section
    expect(result.atoms[0]).toMatchObject({ id: 1, molId: 1, type: 1, charge: 0.5, x: 1.0, y: 2.0, z: 3.0 });
  });

  it('(4) rows that do not fit are skipped — no cross-style retry, extents stay finite', () => {
    // Old code re-read the 6-token row as molecular and pushed a bogus atom.
    const result = parseDataFile(dataFile(`Atoms # full

1 1 2 0.1 1.0 2.0 3.0
2 2 2 0.2 5.0 6.0
`));
    expect(result.atoms).toHaveLength(1);
    expect(result.atoms[0]).toMatchObject({ id: 1, type: 2, x: 1.0, y: 2.0, z: 3.0 });
    expect(result.min).toEqual({ x: 1.0, y: 2.0, z: 3.0 });
    expect(result.max).toEqual({ x: 1.0, y: 2.0, z: 3.0 });
    expect(Number.isFinite(result.min.x)).toBe(true);
  });

  it('(5) `Atom Type Labels` is its own section and never feeds the Atoms parser', () => {
    const result = parseDataFile(dataFile(`2 atom types

Atom Type Labels

1 C
2 O

Atoms # full

1 1 C 0.5 1.0 2.0 3.0
2 2 O -0.5 4.0 5.0 6.0
`));
    expect(result.atoms).toHaveLength(2);
    // label C -> type 1, label O -> type 2
    expect(result.atoms[0]).toMatchObject({ type: 1, x: 1.0, y: 2.0, z: 3.0 });
    expect(result.atoms[1]).toMatchObject({ type: 2, x: 4.0, y: 5.0, z: 6.0, charge: -0.5 });
  });

  it('(6) hint-less 6-column rows prefer charge when the type column is 0 (q=0)', () => {
    // Row `1 1 0 1.0 2.0 3.0` is charge style with q=0; the old guesser read
    // it as molecular -> type 0. molecular cannot fit: type column 0 < 1.
    const result = parseDataFile(dataFile(`1 atom types

Atoms

1 1 0 1.0 2.0 3.0
`));
    expect(result.atoms).toHaveLength(1);
    expect(result.atoms[0]).toMatchObject({ id: 1, type: 1, charge: 0, x: 1.0, y: 2.0, z: 3.0 });
  });
});

describe('type labels', () => {
  it('resolves an element-symbol type label to the type element (no Masses)', () => {
    const result = parseDataFile(dataFile(`2 atom types

Atom Type Labels

1 C
2 O

Atoms # full

1 1 C 0.0 1.0 2.0 3.0
2 2 O 0.0 4.0 5.0 6.0
`));
    expect(result.atomTypes[1].element).toBe('C');
    expect(result.atomTypes[2].element).toBe('O');
  });

  it('uses the label element only when Masses gives no better element', () => {
    // Mass 7.5 is within 0.5 amu of no element -> the label "O" decides.
    const result = parseDataFile(dataFile(`1 atom types

Atom Type Labels

1 O

Masses

1 7.5

Atoms # full

1 1 O 0.0 1.0 2.0 3.0
`));
    expect(result.atomTypes[1].element).toBe('O');
  });

  it('resolves a type label in the Masses first column', () => {
    const result = parseDataFile(dataFile(`2 atom types

Atom Type Labels

1 C
2 O

Masses

O 15.999 # Oxygen

Atoms # full

1 1 C 0.0 1.0 2.0 3.0
2 2 O 0.0 4.0 5.0 6.0
`));
    expect(result.atomTypes[2].mass).toBe(15.999);
    expect(result.atomTypes[2].element).toBe('O');
    expect(result.atomTypes[1].element).toBe('C'); // from label, no Masses entry
  });

  it('keeps the type-as-atomic-number fallback for non-element labels', () => {
    const result = parseDataFile(dataFile(`1 atom types

Atom Type Labels

1 OW

Atoms # full

1 1 OW 0.0 1.0 2.0 3.0
`));
    expect(result.atomTypes[1].element).toBe('H'); // unchanged fallback for type 1
  });
});

describe('robustness and extents', () => {
  it('never throws on malformed rows; good rows are kept', () => {
    const result = parseDataFile(dataFile(`Atoms # full

1 1 2 0.0 1.0 2.0 3.0
2 1 2 0.0 1.0 2.0 3.0 0 0 0 0 0
1 x y z
@@@ nonsense
`));
    expect(result.atoms).toHaveLength(1);
    expect(result.atoms[0]).toMatchObject({ id: 1, x: 1.0, y: 2.0, z: 3.0 });
  });

  it('extents are never Infinity when atoms exist', () => {
    const result = parseDataFile(dataFile(`Atoms # full

1 1 2 0.0 1.0 2.0 3.0
2 1 2 0.0 4.0 5.0 6.0
`));
    expect(result.min).toEqual({ x: 1.0, y: 2.0, z: 3.0 });
    expect(result.max).toEqual({ x: 4.0, y: 5.0, z: 6.0 });
    expect(result.center).toEqual({ x: 2.5, y: 3.5, z: 4.5 });
  });

  it('stores trailing image flags as ix/iy/iz', () => {
    const result = parseDataFile(dataFile(`Atoms # molecular

1 3 2 1.0 2.0 3.0 3 -2 1
`));
    expect(result.atoms[0]).toMatchObject({ x: 1.0, y: 2.0, z: 3.0, ix: 3, iy: -2, iz: 1 });
  });
});

describe.skipIf(!haveRealFiles)('real LAMMPS data files (read from disk)', () => {
  it('data.peptide: 2004 atoms, full style with image flags', () => {
    const result = parseDataFile(realFile('data.peptide')!);
    expect(result.atoms).toHaveLength(2004);
    const a1 = result.atoms.find(a => a.id === 1)!;
    expect(a1.type).toBe(1);
    expect(a1.charge).toBe(0.510);
    expect(a1.x).toBe(43.99993);
    expect(a1.y).toBe(58.52678);
    expect(a1.z).toBe(36.78550);
    const a2004 = result.atoms.find(a => a.id === 2004)!;
    expect(a2004.type).toBe(14);
    expect(a2004.x).toBe(56.55074);
    expect(a2004.y).toBe(49.75049);
    expect(a2004.z).toBe(48.61854);
    expect(a2004.ix).toBe(1);
    expect(a2004.iy).toBe(1);
    expect(a2004.iz).toBe(1);
  });

  it('data.micelle: 1200 atoms, hint-less molecular style', () => {
    const result = parseDataFile(realFile('data.micelle')!);
    expect(result.atoms).toHaveLength(1200);
    const a1 = result.atoms.find(a => a.id === 1)!;
    expect(a1.type).toBe(2);
    expect(a1.molId).toBe(139);
    expect(a1.x).toBe(0);
    expect(a1.y).toBe(0);
    expect(a1.z).toBe(0);
    const a1200 = result.atoms.find(a => a.id === 1200)!;
    expect(a1200.type).toBe(4);
    expect(a1200.molId).toBe(150);
    expect(a1200.x).toBe(2.516);
    expect(a1200.y).toBe(32.754);
    expect(a1200.z).toBe(0);
  });

  it('data.body: 100 atoms, body-style rows at x index 4', () => {
    const result = parseDataFile(realFile('data.body')!);
    expect(result.atoms).toHaveLength(100);
    const a1 = result.atoms.find(a => a.id === 1)!;
    expect(a1.type).toBe(1);
    expect(a1.x).toBe(-15.5322);
    expect(a1.y).toBe(-15.5322);
    expect(a1.z).toBe(0);
    const a100 = result.atoms.find(a => a.id === 100)!;
    expect(a100.x).toBe(12.4258);
    expect(a100.y).toBe(12.4258);
    expect(a100.z).toBe(0);
  });
});

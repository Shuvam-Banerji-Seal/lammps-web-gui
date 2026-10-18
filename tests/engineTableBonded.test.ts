import { describe, expect, it } from 'vitest';
import { BondTable } from '../src/engine-gpl/table_bonded/bond_table';
import { AngleTable } from '../src/engine-gpl/table_bonded/angle_table';
import { DihedralTable } from '../src/engine-gpl/table_bonded/dihedral_table';
import { DihedralTableCut } from '../src/engine-gpl/table_bonded/dihedral_table_cut';
import { TableFileReader, fmtG } from '../src/engine-gpl/table_bonded/tablefile';
import { BONDS, ANGLES, DIHEDRALS } from '../src/engine-gpl/registry/table_bonded';
import { StyleError, type StyleContext } from '../src/engine/force/types';

// Stub of the notebook's files: the table text is served by name; logged warnings are collected.
const ctxWith = (files: Record<string, string>, logged: string[] = []): StyleContext => ({
  s: null,
  readFile: (name: string) => {
    if (!(name in files)) throw new StyleError(`cannot open file ${name}`);
    return files[name];
  },
  log: (t: string) => { logged.push(t); },
}) as unknown as StyleContext;

const err = (fn: () => unknown): string => {
  try { fn(); } catch (e) { if (e instanceof StyleError) return e.message; throw e; }
  return 'no error';
};

// A bond table: section BLIN (no FP, no EQ), 4 points; BERR (bad number), BMISS (too few lines), BONE (N 1).
const bondFile = [
  '# comment line',
  'BLIN',
  'N 4',
  '',
  '1 0.9 4.0 -2.0',
  '2 1.0 1.0 -0.5',
  '3 1.1 0.5 0.5',
  '4 1.2 2.0 2.0',
  '',
  'BERR',
  'N 2',
  'skipped line',
  '1 0.9 x 1.0',
  '2 1.0 1.0 1.0',
  '',
  'BMISS',
  'N 3',
  'skipped line',
  '1 0.9 1.0 1.0',
  '',
  'BONE',
  'N 1',
  'skipped line',
  '1 0.9 1.0 1.0',
  '',
  'BFP',
  'N 4 FP 1.0',
  'skipped line',
  '1 0.9 1.0 1.0',
  '2 1.0 1.0 1.0',
  '3 1.1 1.0 1.0',
  '4 1.2 1.0 1.0',
  '',
  'BKW',
  'N 4 FOO 1',
  'skipped line',
  '1 0.9 1.0 1.0',
  '',
  'BNOTN',
  'EQ 1.0',
  'skipped line',
  '1 0.9 1.0 1.0',
  '',
  'BUNSORT',
  'N 3',
  'x',
  '1 1.1 1.0 1.0',
  '2 1.0 1.0 1.0',
  '3 0.9 1.0 1.0',
].join('\n');

describe('bond_style table: arguments and table file', () => {
  const style = (): BondTable => { const b = new BondTable(); b.allocate(2); return b; };

  it('checks the style arguments', () => {
    expect(err(() => new BondTable().settings(['linear']))).toBe('Illegal bond_style command: must have 2 arguments');
    expect(err(() => new BondTable().settings(['cubic', '10']))).toBe('Unknown table style cubic in bond style table');
    expect(err(() => new BondTable().settings(['linear', '1']))).toBe('Illegal number of bond table entries: 1');
    expect(err(() => new BondTable().settings(['linear', 'ten']))).toContain('Expected integer parameter instead of');
  });

  it('checks the coeff arguments and the file', () => {
    const b = style();
    b.settings(['linear', '50']);
    expect(err(() => b.coeff(['1', 'f.table'], ctxWith({})))).toBe('Illegal bond_coeff command: must have 3 arguments');
    expect(err(() => b.coeff(['1', 'missing.table', 'BLIN'], ctxWith({})))).toBe(
      'cannot open bond table potential file missing.table: No such file or directory');
    expect(err(() => b.coeff(['1', 't.table', 'NOPE'], ctxWith({ 't.table': bondFile })))).toBe(
      'Did not find keyword NOPE in table file');
  });

  it('reports the parameter-line and data-line errors of the native reader', () => {
    const b = style();
    b.settings(['linear', '50']);
    const files = { 't.table': bondFile };
    expect(err(() => b.coeff(['1', 't.table', 'BNOTN'], ctxWith(files)))).toBe('Bond table parameters did not set N');
    expect(err(() => b.coeff(['1', 't.table', 'BKW'], ctxWith(files)))).toBe('Unknown keyword FOO in bond table parameters');
    expect(err(() => b.coeff(['1', 't.table', 'BFP'], ctxWith(files)))).toBe('No more tokens');
    expect(err(() => b.coeff(['1', 't.table', 'BERR'], ctxWith(files)))).toContain(
      "Error parsing bond table 'BERR' line 1 of 2. Not a valid floating-point number: 'x'");
    expect(err(() => b.coeff(['1', 't.table', 'BMISS'], ctxWith(files)))).toBe(
      "Error parsing bond table 'BMISS' line 2 of 3. Not a valid integer number: 'BONE'\nLine was: BONE");
    expect(err(() => b.coeff(['1', 't.table', 'BONE'], ctxWith(files)))).toBe('Invalid bond table length: 1');
    expect(err(() => b.coeff(['1', 't.table', 'BUNSORT'], ctxWith(files)))).toBe('Bond table values are not increasing');
  });

  it('skips one raw line after the parameter line (the native skip_line)', () => {
    // BERR: the first data line after the parameters is skipped, so the parse error is on the line after it
    const r = new TableFileReader(bondFile, 't.table', 'bond');
    expect(r.findSectionStart('BERR')).not.toBeNull();
    expect(r.nextLine()).toBe('N 2');
    r.skipLine();
    expect(r.nextLine()).toBe('1 0.9 x 1.0');
  });

  it('keeps the bond table registry name and an equilibrium from EQ', () => {
    expect(Object.keys(BONDS)).toEqual(['table']);
    const b = style();
    b.settings(['spline', '40']);
    b.coeff(['1', 't.table', 'BFP'], ctxWith({ 't.table': bondFile.replace('N 4 FP 1.0', 'N 4 FP 1.0 2.0 EQ 1.05') }));
    expect(b.equilibrium(1)).toBe(1.05);
  });
});

describe('angle_style table: arguments and table file', () => {
  const angleFile = [
    'A0', 'N 3', 'comment', '1 0.0 1.0 0.0', '2 90.0 0.5 0.1', '3 180.0 1.0 0.0', '',
    'ANEG', 'N 3', 'comment', '1 0.0 1.0 0.0', 'x 90.0 0.5 0.1', '3 180.0 1.0 0.0', '',
    'ARANGE', 'N 3', 'comment', '1 10.0 1.0 0.0', '2 90.0 0.5 0.1', '3 170.0 1.0 0.0', '',
    'AKW', 'N 2 BAD 3', 'comment', '1 0.0 1.0 0.0', '2 180.0 1.0 0.0', '',
    'AEQ', 'N 2 EQ 90.0', 'comment', '1 0.0 1.0 0.0', '2 180.0 1.0 0.0', '',
  ].join('\n');

  it('checks the style and the table range', () => {
    expect(err(() => new AngleTable().settings(['linear']))).toBe('Illegal angle_style command: must have 2 arguments');
    expect(err(() => new AngleTable().settings(['cubic', '10']))).toBe('Unknown table style cubic in angle style table');
    expect(err(() => new AngleTable().settings(['linear', '1']))).toBe('Illegal number of angle table entries: 1');
    const a = new AngleTable();
    a.allocate(2);
    a.settings(['linear', '20']);
    expect(err(() => a.coeff(['1', 'a.table'], ctxWith({})))).toBe('Illegal angle_coeff command: must have 3 arguments');
    const files = { 'a.table': angleFile };
    expect(err(() => a.coeff(['1', 'a.table', 'ARANGE'], ctxWith(files)))).toBe('Angle table must range from 0 to 180 degrees');
    expect(err(() => a.coeff(['1', 'a.table', 'ANEG'], ctxWith(files)))).toContain("Error parsing angle table 'ANEG' line 2 of 3. Not a valid integer number: 'x'");
    expect(err(() => a.coeff(['1', 'a.table', 'AKW'], ctxWith(files)))).toBe('Unknown keyword BAD in angle table parameters');
  });

  it('defaults theta0 to pi and converts EQ from degrees', () => {
    const a = new AngleTable();
    a.allocate(2);
    a.settings(['linear', '20']);
    const files = { 'a.table': angleFile };
    a.coeff(['1', 'a.table', 'A0'], ctxWith(files));
    expect(a.equilibriumRadians(1)).toBeCloseTo(Math.PI, 12);
    expect(a.equilibrium(1)).toBeCloseTo(180, 12);
    a.coeff(['2', 'a.table', 'AEQ'], ctxWith(files));
    expect(a.equilibrium(2)).toBeCloseTo(90, 12);
    expect(Object.keys(ANGLES)).toEqual(['table']);
  });
});

describe('dihedral_style table and table/cut: arguments, keywords and table file', () => {
  const dihFile = [
    'DDEG', 'N 4 DEGREES', '1 -180.0 1.0 0.5', '2 -90.0 0.0 0.0', '3 0.0 -1.0 0.0', '4 90.0 0.0 0.0', '',
    'DRAD', 'N 4 radians', '1 -3.0 1.0 0.5', '2 -1.0 0.0 0.0', '3 0.0 -1.0 0.0', '4 1.0 0.0 0.0', '',
    'DNOF', 'N 4 NOF', '1 0.0 1.0', '2 90.0 0.0', '3 180.0 -1.0', '4 270.0 0.0', '',
    'DNOFBIG', 'N 4 NOF', '1 0.0 1.0', '2 90.0 0.0', '3 180.0 -1.0', '4 360.0 0.0', '',
    'DORDER', 'N 3', '1 90.0 1.0 0.0', '2 0.0 0.0 0.0', '3 180.0 -1.0 0.0', '',
    'DONE', 'N 1', '1 0.0 1.0 0.0', '',
    'DTWO', 'N 2', '1 0.0 1.0 0.0', '2 90.0 1.0 0.0', '',
    'DCHK', 'N 4 NOF CHECKU out.dat', '1 0.0 1.0', '2 90.0 0.0', '3 180.0 -1.0', '4 270.0 0.0', '',
    'DUNK', 'N 4 PARTY', '1 0.0 1.0 0.0', '',
    'DNON', 'N 4 DEGREES', '1 0.0 1.0', '',
    'DNOTN', 'DEGREES', '',
  ].join('\n');

  it('checks the style arguments', () => {
    expect(err(() => new DihedralTable().settings(['linear']))).toBe('Illegal dihedral_style command: must have 2 arguments');
    expect(err(() => new DihedralTable().settings(['cubic', '10']))).toBe('Unknown table style cubic in dihedral style table');
    expect(err(() => new DihedralTable().settings(['linear', '2']))).toBe('Illegal number of dihedral table entries: 2');
  });

  it('parses N, NOF, DEGREES and RADIANS and rejects other keywords', () => {
    const d = new DihedralTable();
    d.allocate(2);
    d.settings(['linear', '36']);
    const files = { 'd.table': dihFile };
    expect(err(() => d.coeff(['1', 'd.table', 'DUNK'], ctxWith(files)))).toBe('Unknown keyword PARTY in dihedral table parameters');
    expect(err(() => d.coeff(['1', 'd.table', 'DNOTN'], ctxWith(files)))).toBe('Dihedral table parameters did not set N');
    expect(err(() => d.coeff(['1', 'd.table', 'DNON'], ctxWith(files)))).toBe("Error parsing dihedral table 'DNON' line 1 of 4. No more tokens\nLine was: 1 0.0 1.0");
    // DRAD (RADIANS) and DDEG (DEGREES) are accepted: the angle range is below 2 pi and below 360 degrees
    d.coeff(['1', 'd.table', 'DRAD'], ctxWith(files));
    d.coeff(['2', 'd.table', 'DDEG'], ctxWith(files));
    expect(d.init()).toBeUndefined();
  });

  it('refuses the CHECKU and CHECKF keywords by name (they write files)', () => {
    const d = new DihedralTable();
    d.allocate(2);
    d.settings(['linear', '36']);
    expect(err(() => d.coeff(['1', 'd.table', 'DCHK'], ctxWith({ 'd.table': dihFile })))).toBe(
      'dihedral table keyword CHECKU out.dat is not supported: it writes a check file, and the notebook\'s styles write no files');
  });

  it('checks the table length, the spline minimum, the order and the 2 pi range', () => {
    const d = new DihedralTable();
    d.allocate(2);
    d.settings(['linear', '36']);
    const files = { 'd.table': dihFile };
    expect(err(() => d.coeff(['1', 'd.table', 'DONE'], ctxWith(files)))).toBe('Invalid dihedral table length: DONE');
    const s = new DihedralTable();
    s.allocate(2);
    s.settings(['spline', '36']);
    expect(err(() => s.coeff(['1', 'd.table', 'DTWO'], ctxWith(files)))).toBe('Invalid dihedral spline table length: DTWO (Try linear)');
    expect(err(() => d.coeff(['1', 'd.table', 'DORDER'], ctxWith(files)))).toContain(
      'Dihedral table values are not increasing (DORDER, entry 1)\n(This is probably a mistake with your table format.)\n');
    expect(err(() => d.coeff(['1', 'd.table', 'DNOFBIG'], ctxWith(files)))).toBe(
      'Dihedral table angle range must be < 360 degrees (DNOFBIG).');
  });

  it('refuses a table whose forces disagree with its energies unless NOF is given', () => {
    const bad = [
      'DBAD', 'N 12', ...Array.from({ length: 12 }, (_, i) => `${i + 1} ${i * 30.0} ${Math.sin(i)} ${i % 2 ? 50.0 : -50.0}`), '',
    ].join('\n');
    const d = new DihedralTable();
    d.allocate(2);
    d.settings(['linear', '36']);
    expect(err(() => d.coeff(['1', 'b.table', 'DBAD'], ctxWith({ 'b.table': bad })))).toContain('Dihedral table has inconsistent forces and energies');
  });

  it('table/cut: needs the aat style word, seven arguments, and the same table checks', () => {
    const c = new DihedralTableCut();
    c.allocate(2);
    c.settings(['linear', '36']);
    const files = { 'd.table': dihFile };
    expect(err(() => c.coeff(['1', 'd.table', 'DDEG'], ctxWith(files)))).toBe(
      'Incorrect args for dihedral coefficients\nFor more information see https://docs.lammps.org/err0021');
    expect(err(() => c.coeff(['1', 'bend', '2.0', '100', '120', 'd.table', 'DDEG'], ctxWith(files)))).toBe(
      "dihedral_coeff table/cut: the style word after the type must be aat, got 'bend'");
    expect(err(() => c.coeff(['1', 'aat', '2.0', '100', '120', 'd.table', 'DORDER'], ctxWith(files)))).toContain(
      'Dihedral table values are not increasing (d.table, entry 1)');
    c.coeff(['1', 'aat', '2.0', '100', '120', 'd.table', 'DDEG'], ctxWith(files));
    expect(Object.keys(DIHEDRALS).sort()).toEqual(['table', 'table/cut']);
  });
});

describe('table file reader and number formatting', () => {
  it('cuts comments, skips blank lines and keeps the first matching section', () => {
    const r = new TableFileReader('# head\n\nX  1  # trailing\n   \nY\n', 't', 'bond');
    expect(r.findSectionStart('X')).toBe('X  1  ');
    expect(r.nextLine()).toBe('Y');
    expect(r.nextLine()).toBeNull();
  });

  it('formats like fmt {:.8}', () => {
    expect(fmtG(0.5)).toBe('0.5');
    expect(fmtG(1234.5678)).toBe('1234.5678');
    expect(fmtG(1.5e-7)).toBe('1.5e-07');
    expect(fmtG(12345678901)).toBe('1.2345679e+10');
    expect(fmtG(0)).toBe('0');
  });
});

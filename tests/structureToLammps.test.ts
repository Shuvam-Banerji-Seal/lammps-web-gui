import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { parseFile } from '../src/services/fileParser';
import { ELEMENT_DATA } from '../src/constants';
import type { MoleculeData } from '../src/types';
import {
  MAX_ATOMS,
  StructureConvertError,
  convertStructure,
  pdbAtomNames,
  pdbBox,
  type ConvertOptions,
  type ConvertResult,
} from '../src/lammps/structureToLammps';

/*
 * Structure -> LAMMPS converter (src/lammps/structureToLammps.ts). The generated read_data files and
 * molecule templates must be accepted by the engine (the notebook runs the suggested cell), and the
 * layout is the one read_data.html and molecule.html describe.
 */

// --- fixtures -------------------------------------------------------------------------------

/** One PDB ATOM/HETATM record in the wwPDB columns (serial 7-11, name 13-16, element 77-78). */
const atomRec = (serial: number, name: string, x: number, y: number, z: number, element: string): string =>
  'ATOM  ' + String(serial).padStart(5) + ' ' + (' ' + name).padEnd(4) + ' ' + 'MOL' + ' A' + '   1' + '    ' +
  x.toFixed(3).padStart(8) + y.toFixed(3).padStart(8) + z.toFixed(3).padStart(8) + '  1.00  0.00' +
  '          ' + element.padStart(2);

/** Six atoms: C1-C2-O3 chain with hydrogens on C1 (H4, H5) and C2 (H6). */
const ATOMS = [
  ['C1', 1.0, 1.0, 1.0, 'C'],
  ['C2', 2.5, 1.0, 1.0, 'C'],
  ['O3', 3.2, 2.2, 1.0, 'O'],
  ['H4', 0.5, 1.9, 1.0, 'H'],
  ['H5', 0.5, 0.1, 1.0, 'H'],
  ['H6', 2.9, 0.1, 1.9, 'H'],
] as const;

const atomLines = (): string[] =>
  ATOMS.map(([name, x, y, z, el], i) => atomRec(i + 1, name, x, y, z, el));

const CONECT = ['CONECT    1    2    4    5', 'CONECT    2    1    3    6', 'CONECT    3    2'];

const PDB_CELL = [
  'CRYST1   12.000   12.000   12.000  90.00  90.00  90.00 P 1           1',
  ...atomLines(),
  ...CONECT,
  'END',
].join('\n');

const PDB_NOCELL = [...atomLines(), 'END'].join('\n');

/** P1 CIF with a non-orthogonal (gamma = 110 deg) cell; fractional coordinates. */
const CIF_TRICLINIC = `data_test
_cell_length_a 10.000
_cell_length_b 8.000
_cell_length_c 9.000
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 110
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.10 0.20 0.30
O1 O 0.55 0.40 0.62
H1 H 0.25 0.25 0.25
H2 H 0.90 0.65 0.10
`;

const XYZ = `3
water
O 0.0000 0.0000 0.0000
H 0.9572 0.0000 0.0000
H -0.2400 0.9266 0.0000
`;

const opts = (over: Partial<ConvertOptions> = {}): ConvertOptions => ({
  target: 'data',
  source: 'elements',
  atomStyle: 'full',
  typeBy: 'element',
  bonds: true,
  padding: 10,
  fileName: 'x.data',
  title: 'test structure',
  ...over,
});

const pdbMol = (text: string): MoleculeData => parseFile(text, 'pdb');

/** Header line value: the first number of the line that ends with the keyword. */
const headerValue = (text: string, keyword: string): string | undefined => {
  const line = text.split('\n').find((l) => l.trim().endsWith(keyword));
  return line?.trim().split(/\s+/)[0];
};

/** Atom lines of an Atoms section: the lines between "Atoms # <style>" and the next section. */
const atomsSection = (text: string): string[] => {
  const lines = text.split('\n');
  const start = lines.findIndex((l) => l.startsWith('Atoms #'));
  const out: string[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const t = lines[i].trim();
    if (t === '') continue;
    if (/^[A-Z][A-Za-z ]+$/.test(t) || /^[A-Z][A-Za-z ]+ #/.test(t)) break;
    out.push(lines[i].replace(/#.*$/, '').trim());
  }
  return out;
};

const massOf = (symbol: string): number => ELEMENT_DATA.find((e) => e.symbol === symbol)!.mass;

/** Runs a script on a fresh engine session with the given input files; returns the session. */
const engine = async (files: Record<string, string>, script: string): Promise<Session> => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  await session.execute(script);
  return session;
};

/** Fractional coordinates of a point in the restricted triclinic box (read_data layout). */
const fractional = (p: [number, number, number], box: NonNullable<ConvertResult['box']>): [number, number, number] => {
  const lx = box.xhi - box.xlo, ly = box.yhi - box.ylo, lz = box.zhi - box.zlo;
  const xy = box.xy ?? 0, xz = box.xz ?? 0, yz = box.yz ?? 0;
  const dz = p[2] - box.zlo;
  const u = dz / lz;
  const t = (p[1] - box.ylo - u * yz) / ly;
  const s = (p[0] - box.xlo - t * xy - u * xz) / lx;
  return [s, t, u];
};

const dataAtomCoords = (text: string): [number, number, number][] =>
  atomsSection(text).map((l) => {
    const c = l.split(/\s+/).map(Number);
    return [c[c.length - 3], c[c.length - 2], c[c.length - 1]] as [number, number, number];
  });

/** The header box of a data file. */
const headerBox = (text: string): NonNullable<ConvertResult['box']> => {
  const get = (kw: string) => {
    const l = text.split('\n').find((x) => x.trim().endsWith(kw))!.trim().split(/\s+/).map(Number);
    return l;
  };
  const [xlo, xhi] = get('xlo xhi');
  const [ylo, yhi] = get('ylo yhi');
  const [zlo, zhi] = get('zlo zhi');
  const tilt = text.split('\n').find((x) => x.trim().endsWith('xy xz yz'));
  const [xy, xz, yz] = tilt ? tilt.trim().split(/\s+/).slice(0, 3).map(Number) : [0, 0, 0];
  return { xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz };
};

// --- PDB with CRYST1 and CONECT ----------------------------------------------------------------

describe('PDB with a cell and CONECT records', () => {
  const mol = pdbMol(PDB_CELL);
  const box = pdbBox(PDB_CELL);
  const names = pdbAtomNames(PDB_CELL, mol.atoms)!;

  it('reads the cell and the atom names of the PDB', () => {
    expect(box).toMatchObject({ xlo: 0, xhi: 12, yhi: 12, zhi: 12 });
    expect(Math.abs(box!.xy ?? 0)).toBeLessThan(1e-9);
    expect(names).toEqual(['C1', 'C2', 'O3', 'H4', 'H5', 'H6']);
  });

  const res = convertStructure(mol, opts({ box, title: 'PDB cell', fileName: 'mol.data' }));

  it('writes the header counts and the orthogonal box', () => {
    expect(headerValue(res.text, 'atoms')).toBe('6');
    expect(headerValue(res.text, 'bonds')).toBe('5');
    expect(headerValue(res.text, 'atom types')).toBe('3');
    expect(headerValue(res.text, 'bond types')).toBe('3');
    expect(res.text).toContain('0.000000 12.000000 xlo xhi');
    expect(res.text.split('\n').some((l) => l.endsWith('xy xz yz'))).toBe(false);
    expect(res.triclinic).toBe(false);
    expect(res.text.split('\n')[0]).toBe('PDB cell');
  });

  it('gives masses from the element table, one type per element in order of appearance', () => {
    expect(res.types.map((t) => [t.element, t.mass, t.count])).toEqual([
      ['C', massOf('C'), 2],
      ['O', massOf('O'), 1],
      ['H', massOf('H'), 3],
    ]);
    expect(res.text).toContain(`1 ${massOf('C')} # C`);
    expect(res.text).toContain(`3 ${massOf('H')} # H`);
  });

  it('writes the full style with molecule IDs, types and bonds', () => {
    const atoms = atomsSection(res.text);
    expect(res.text).toContain('Atoms # full');
    expect(atoms).toHaveLength(6);
    expect(atoms[0].split(/\s+/)).toEqual(['1', '1', '1', '0', '1.000000', '1.000000', '1.000000']);
    const bonds = res.text.split('\nBonds\n\n')[1].trim().split('\n');
    // bond types by element pair, numbered in order of first appearance: C-C 1, C-H 2, C-O 3
    expect(bonds).toEqual(['1 1 1 2', '2 2 1 4', '3 2 1 5', '4 3 2 3', '5 2 2 6']);
  });

  it('suggests a cell that reads the file and runs', async () => {
    expect(res.suggestion).toContain('read_data mol.data');
    expect(res.suggestion).toContain('boundary p p p');
    const s = await engine({ 'mol.data': res.text }, res.suggestion);
    expect(s.system!.n).toBe(6);
  });

  it('round-trips through read_data for atomic, charge and full styles', async () => {
    for (const style of ['atomic', 'charge', 'full'] as const) {
      const r = convertStructure(mol, opts({ box, atomStyle: style, fileName: `${style}.data` }));
      const s = await engine({ [`${style}.data`]: r.text }, r.suggestion);
      expect(s.system!.n).toBe(6);
      expect(Array.from(s.system!.massByType).slice(1, 4)).toEqual([massOf('C'), massOf('O'), massOf('H')]);
      const lines = atomsSection(r.text);
      expect(lines[0].split(/\s+/)).toHaveLength(style === 'atomic' ? 5 : style === 'charge' ? 6 : 7);
    }
  });

  it('atomic style writes no bonds and no charges', () => {
    const r = convertStructure(mol, opts({ box, atomStyle: 'atomic' }));
    expect(r.text).toContain('Atoms # atomic');
    expect(r.text).not.toContain('bonds\n');
    expect(r.text).not.toContain('\nBonds\n');
    expect(r.nbonds).toBe(0);
  });

  it('uses PDB atom names as types when asked, with the element as the mass', async () => {
    const r = convertStructure(mol, opts({ box, typeBy: 'pdbName', atomNames: names }));
    expect(r.types.map((t) => t.label)).toEqual(['C1 (C)', 'C2 (C)', 'O3 (O)', 'H4 (H)', 'H5 (H)', 'H6 (H)']);
    expect(r.ntypes).toBe(6);
    expect(r.types.map((t) => t.mass)).toEqual([massOf('C'), massOf('C'), massOf('O'), massOf('H'), massOf('H'), massOf('H')]);
    const s = await engine({ 'x.data': r.text }, r.suggestion);
    expect(s.system!.n).toBe(6);
  });

  it('writes a molecule template and runs create_atoms mol', async () => {
    const r = convertStructure(mol, opts({ box, target: 'molecule', atomStyle: 'full', fileName: 'x.mol' }));
    expect(r.text).toContain('6 atoms');
    expect(r.text).toContain('5 bonds');
    for (const section of ['Coords', 'Types', 'Charges', 'Bonds']) expect(r.text).toContain(`\n${section}\n`);
    expect(r.suggestion).toContain('molecule m x.mol');
    expect(r.suggestion).toContain('create_atoms 0 single');
    expect(r.suggestion).toContain('mol m 12345');
    const s = await engine({ 'x.mol': r.text }, r.suggestion);
    expect(s.system!.n).toBe(6);
  });
});

// --- PDB without a cell --------------------------------------------------------------------------

describe('PDB without a cell', () => {
  const mol = pdbMol(PDB_NOCELL);
  const res = convertStructure(mol, opts({ fileName: 'nocell.data', padding: 5 }));

  it('uses the atom extent plus the padding as the box and infers bonds', () => {
    const box = headerBox(res.text);
    expect(box.xlo).toBeCloseTo(0.5 - 5, 6);
    expect(box.xhi).toBeCloseTo(3.2 + 5, 6);
    expect(box.yhi).toBeCloseTo(2.2 + 5, 6);
    expect(res.triclinic).toBe(false);
    expect(res.warnings.join(' ')).toMatch(/No cell/);
    expect(res.nbonds).toBeGreaterThanOrEqual(4);
    expect(res.suggestion).toContain('boundary f f f');
  });

  it('runs through read_data with non-periodic boundaries', async () => {
    const s = await engine({ 'nocell.data': res.text }, res.suggestion);
    expect(s.system!.n).toBe(6);
  });
});

// --- CIF with a non-orthogonal cell ------------------------------------------------------------

describe('CIF with a triclinic cell', () => {
  const mol = parseFile(CIF_TRICLINIC, 'cif');
  const res = convertStructure(mol, opts({ box: mol.box, fileName: 'tri.data' }));

  it('writes the tilt factors on the xy xz yz line', () => {
    expect(res.triclinic).toBe(true);
    expect(res.text.split('\n').some((l) => l.endsWith('xy xz yz'))).toBe(true);
    expect(headerValue(res.text, 'atoms')).toBe('4');
  });

  it('wraps every atom into the box', () => {
    const box = headerBox(res.text);
    for (const p of dataAtomCoords(res.text)) {
      const f = fractional(p, box);
      for (const v of f) {
        expect(v).toBeGreaterThanOrEqual(-1e-6);
        expect(v).toBeLessThan(1 + 1e-6);
      }
    }
  });

  it('runs through read_data as a triclinic box', async () => {
    const s = await engine({ 'tri.data': res.text }, res.suggestion);
    expect(s.system!.n).toBe(4);
    expect(s.system!.box.triclinic).toBe(true);
    expect(s.system!.box.tilt[0]).toBeCloseTo(mol.box!.xy!, 6);
  });
});

// --- XYZ ------------------------------------------------------------------------------------------

describe('XYZ', () => {
  const mol = parseFile(XYZ, 'xyz');
  const res = convertStructure(mol, opts({ fileName: 'water.data', padding: 4 }));

  it('infers O-H bonds and writes one bond type per element pair', () => {
    expect(res.nbonds).toBe(2);
    expect(res.nbondTypes).toBe(1);
    expect(res.types.map((t) => t.element)).toEqual(['O', 'H']);
  });

  it('runs in the full style', async () => {
    const s = await engine({ 'water.data': res.text }, res.suggestion);
    expect(s.system!.n).toBe(3);
  });

  it('writes a molecule template that runs', async () => {
    const r = convertStructure(mol, opts({ target: 'molecule', fileName: 'water.mol' }));
    const s = await engine({ 'water.mol': r.text }, r.suggestion);
    expect(s.system!.n).toBe(3);
  });
});

// --- LAMMPS data input ----------------------------------------------------------------------------

describe('LAMMPS data input', () => {
  it('keeps the types and masses of a data file when it is re-emitted', async () => {
    const first = convertStructure(pdbMol(PDB_CELL), opts({ box: pdbBox(PDB_CELL), fileName: 'a.data' }));
    const parsed = parseFile(first.text, 'lammps');
    const second = convertStructure(parsed, opts({ source: 'lammps', atomStyle: 'charge', box: parsed.box, bonds: false, fileName: 'b.data' }));
    expect(second.ntypes).toBe(3);
    expect(second.text).toContain(`1 ${massOf('C')} # type 1`);
    const s = await engine({ 'b.data': second.text }, second.suggestion);
    expect(s.system!.n).toBe(6);
  });
});

// --- refusals and limits -------------------------------------------------------------------------

describe('refusals', () => {
  it('refuses an element that is not in the table', () => {
    const mol = pdbMol(PDB_NOCELL);
    mol.atoms[0] = { ...mol.atoms[0], type: 999 };
    expect(() => convertStructure(mol, opts())).toThrow(StructureConvertError);
    expect(() => convertStructure(mol, opts())).toThrow(/element table/);
  });

  it('refuses an empty structure', () => {
    expect(() => convertStructure({ ...pdbMol(PDB_NOCELL), atoms: [] }, opts())).toThrow(/no atoms/);
  });

  it('refuses a structure above the atom limit', () => {
    const mol = pdbMol(PDB_NOCELL);
    const atoms = Array.from({ length: MAX_ATOMS + 1 }, (_, i) => ({ ...mol.atoms[0], id: i + 1 }));
    expect(() => convertStructure({ ...mol, atoms, bonds: [] }, opts({ bonds: false }))).toThrow(/above the limit/);
  });
});

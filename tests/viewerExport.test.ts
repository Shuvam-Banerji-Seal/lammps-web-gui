/**
 * Structure export of the Structure Viewer (src/lammps/exporter.ts): LAMMPS data with the tilt line of a triclinic
 * cell, extended XYZ (one frame and every frame) and PDB (CRYST1, fixed columns, element or T<type>).
 */
import { describe, it, expect } from 'vitest';
import { framesToExtXyz, toExtXyz, toLammpsDataFile, toPdb } from '../src/lammps/exporter';
import type { Atom, MoleculeData } from '../src/types';

const atom = (id: number, type: number, x: number, y: number, z: number): Atom => ({ id, molId: 1, type, charge: 0, x, y, z });

const mol = (box?: MoleculeData['box']): MoleculeData => ({
  atoms: [atom(1, 1, 1.5, 2.25, 3), atom(2, 2, 4, 5, 6)],
  bonds: [],
  atomTypes: {
    1: { id: 1, mass: 12.011, element: 'C', label: 'C', count: 1 },
    2: { id: 2, mass: 1, element: 'X', label: 'type 2', count: 1 },
  },
  min: { x: 1.5, y: 2.25, z: 3 },
  max: { x: 4, y: 5, z: 6 },
  center: { x: 2.75, y: 3.625, z: 4.5 },
  box,
});

describe('toLammpsDataFile', () => {
  it('writes the xy xz yz line for a triclinic cell and none for an orthogonal one', () => {
    const tri = toLammpsDataFile(mol({ xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10, xy: 1, xz: 0, yz: 0.5 }));
    expect(tri).toContain('1.000000 0.000000 0.500000 xy xz yz');
    const ortho = toLammpsDataFile(mol({ xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10 }));
    expect(ortho).not.toContain('xy xz yz');
    expect(ortho).toContain('2 atoms');
  });

  it('renumbers atom types 1..N (a PDB import numbers types by atomic number)', () => {
    const m = mol({ xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10 });
    m.atoms = [atom(1, 8, 1, 1, 1), atom(2, 1, 2, 1, 1)];
    m.atomTypes = {
      1: { id: 1, mass: 1.008, element: 'H', label: 'H', count: 1 },
      8: { id: 8, mass: 15.999, element: 'O', label: 'O', count: 1 },
    };
    const t = toLammpsDataFile(m);
    expect(t).toContain('2 atom types');
    expect(t).toContain('1 1.0080 # H');
    expect(t).toContain('2 15.9990 # O');
    expect(t).toContain('1 1 2 0 1.000000 1.000000 1.000000');
    expect(t).toContain('2 1 1 0 2.000000 1.000000 1.000000');
  });

  it('uses the box passed in (the shown frame) over the structure box', () => {
    const t = toLammpsDataFile(mol({ xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10 }), { box: { xlo: -1, xhi: 9, ylo: 0, yhi: 10, zlo: 0, zhi: 10 } });
    expect(t).toContain('-1.000000 9.000000 xlo xhi');
  });
});

describe('toExtXyz', () => {
  it('writes the count, Lattice/Origin/Properties and element or T<type> species', () => {
    const lines = toExtXyz(mol({ xlo: -1, xhi: 9, ylo: 0, yhi: 10, zlo: 0, zhi: 5, xy: 2 })).trimEnd().split('\n');
    expect(lines[0]).toBe('2');
    expect(lines[1]).toBe('Lattice="10 0 0 2 10 0 0 0 5" Origin="-1 0 0" Properties=species:S:1:pos:R:3');
    expect(lines[2]).toBe('C 1.5 2.25 3');
    expect(lines[3]).toBe('T2 4 5 6');
  });

  it('writes every frame, each with its own cell', () => {
    const m = mol();
    m.frames = [
      { atoms: m.atoms, box: { xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10 } },
      { atoms: m.atoms.map((a) => ({ ...a, x: a.x + 1 })), box: { xlo: 0, xhi: 11, ylo: 0, yhi: 10, zlo: 0, zhi: 10 } },
    ];
    const lines = framesToExtXyz(m).trimEnd().split('\n');
    expect(lines).toHaveLength(8);
    expect(lines[1]).toContain('Frame=0');
    expect(lines[5]).toContain('Lattice="11 0 0');
    expect(lines[6]).toBe('C 2.5 2.25 3');
  });
});

describe('toPdb', () => {
  it('writes CRYST1 for an orthogonal box and HETATM lines in their columns', () => {
    const lines = toPdb(mol({ xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10 })).trimEnd().split('\n');
    expect(lines[0]).toHaveLength(70);
    expect(lines[0].slice(0, 15)).toBe('CRYST1   10.000');
    const c = lines[1];
    expect(c).toHaveLength(78);
    expect(c.slice(0, 11)).toBe('HETATM    1');
    expect(c.slice(12, 16)).toBe(' C  ');
    expect(c.slice(30, 54)).toBe('   1.500   2.250   3.000');
    expect(c.slice(76, 78)).toBe(' C');
    // an unnamed type: T<type>, blank element
    expect(lines[2].slice(12, 16)).toBe('T2  ');
    expect(lines[2].slice(76, 78)).toBe('  ');
    expect(lines[3]).toBe('END');
  });

  it('omits CRYST1 for a tilted cell', () => {
    expect(toPdb(mol({ xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10, xz: 1 })).startsWith('HETATM')).toBe(true);
  });
});

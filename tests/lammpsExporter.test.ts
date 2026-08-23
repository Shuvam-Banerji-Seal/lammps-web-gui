import { describe, it, expect } from 'vitest';
import { toLammpsDataFile } from '../src/lammps/exporter';
import { MoleculeData, Atom, Bond, AtomTypeInfo } from '../src/types';

const atom = (id: number, type: number, x: number, y: number, z: number, charge = 0): Atom =>
  ({ id, molId: 1, type, charge, x, y, z });

const typeInfo = (id: number, mass: number, element: string): AtomTypeInfo =>
  ({ id, mass, element, label: element, count: 1 });

const sample: MoleculeData = {
  atoms: [
    atom(1, 1, 0, 0, 0, -0.8),
    atom(2, 2, 0.96, 0, 0, 0.4),
    atom(3, 2, -0.24, 0.93, 0, 0.4),
  ],
  bonds: [{ id: 1, type: 1, atom1Id: 1, atom2Id: 2 }],
  atomTypes: { 1: typeInfo(1, 15.999, 'O'), 2: typeInfo(2, 1.008, 'H') },
  min: { x: -0.24, y: 0, z: 0 },
  max: { x: 0.96, y: 0.93, z: 0 },
  center: { x: 0.36, y: 0.465, z: 0 },
};

describe('LAMMPS data-file exporter', () => {
  it('emits header counts, box, masses, Atoms and Bonds sections', () => {
    const text = toLammpsDataFile(sample, { title: 'water test' });
    expect(text).toContain('water test (written by Molecule3D — Shuvam Banerji Seal)');
    expect(text).toContain('3 atoms');
    expect(text).toContain('1 bonds');
    expect(text).toContain('2 atom types');
    expect(text).toContain('1 bond types');
    expect(text).toContain('Atoms # full');
    expect(text).toContain('1 1 1 -0.8 0.000000 0.000000 0.000000');
    expect(text).toContain('2 1 2 0.4 0.960000 0.000000 0.000000');
    expect(text).toContain('Bonds');
    expect(text).toContain('1 1 1 2');
    // Masses section with element comments
    expect(text).toContain('1 15.9990 # O');
    expect(text).toContain('2 1.0080 # H');
    // Box: atom bounds ± 1.0 margin
    expect(text).toMatch(/-1\.240000 1\.960000 xlo xhi/);
  });

  it('uses the parsed simulation box when present instead of atom bounds', () => {
    const withBox: MoleculeData = {
      ...sample,
      box: { xlo: -5, xhi: 5, ylo: -5, yhi: 5, zlo: -5, zhi: 5 },
    };
    const text = toLammpsDataFile(withBox);
    expect(text).toContain('-5.000000 5.000000 xlo xhi');
  });

  it('falls back to carbon mass for unknown types with mass 0', () => {
    const unknown: MoleculeData = {
      ...sample,
      atomTypes: { 1: typeInfo(1, 0, 'X') },
      bonds: [],
    };
    const text = toLammpsDataFile(unknown);
    expect(text).toContain('1 12.0110 # X');
    expect(text).not.toContain('Bonds');
  });
});

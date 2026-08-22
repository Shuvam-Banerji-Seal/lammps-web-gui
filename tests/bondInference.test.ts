import { describe, it, expect } from 'vitest';
import { inferBonds } from '../src/services/bondInference';
import { Atom } from '../src/types';

const mkAtom = (id: number, type: number, x: number, y: number, z: number): Atom =>
  ({ id, molId: 1, type, charge: 0, x, y, z });

describe('Spatial-hash bond inference', () => {
  it('bonds two hydrogens at 0.74 A (H2)', () => {
    const atoms = [mkAtom(1, 1, 0, 0, 0), mkAtom(2, 1, 0.74, 0, 0)];
    const bonds = inferBonds(atoms);
    expect(bonds).toHaveLength(1);
    expect(bonds[0]).toMatchObject({ atom1Id: 1, atom2Id: 2 });
  });

  it('does not bond distant atoms', () => {
    const atoms = [mkAtom(1, 6, 0, 0, 0), mkAtom(2, 6, 5.0, 0, 0)];
    expect(inferBonds(atoms)).toHaveLength(0);
  });

  it('bonds C-H at methane-like distance but not C-C at 3 A', () => {
    const atoms = [
      mkAtom(1, 6, 0, 0, 0),
      mkAtom(2, 1, 1.09, 0, 0),
      mkAtom(3, 1, -0.63, 0.88, 0.62),
      mkAtom(4, 6, 3.0, 0, 0),
    ];
    const bonds = inferBonds(atoms);
    const pairs = bonds.map(b => [b.atom1Id, b.atom2Id].sort().join('-'));
    expect(pairs).toContain('1-2');
    expect(pairs).toContain('1-3');
    expect(pairs).not.toContain('1-4');
  });

  it('deduplicates pairs (each unordered pair once)', () => {
    const atoms = [
      mkAtom(1, 1, 0, 0, 0),
      mkAtom(2, 1, 0.7, 0, 0),
      mkAtom(3, 1, 0, 0.7, 0),
    ];
    const bonds = inferBonds(atoms);
    const keys = bonds.map(b => [b.atom1Id, b.atom2Id].sort((a, c) => a - c).join('-'));
    expect(new Set(keys).size).toBe(keys.length);
  });

  it('handles empty and single-atom inputs', () => {
    expect(inferBonds([])).toHaveLength(0);
    expect(inferBonds([mkAtom(1, 6, 0, 0, 0)])).toHaveLength(0);
  });

  it('respects maxBondLength cap', () => {
    const atoms = [mkAtom(1, 53, 0, 0, 0), mkAtom(2, 53, 2.6, 0, 0)]; // I-I at 2.66 would bond
    expect(inferBonds(atoms, { maxBondLength: 1.0 })).toHaveLength(0);
    expect(inferBonds(atoms).length).toBeGreaterThan(0);
  });

  it('scales linearly enough for large systems (10k atoms under 2s)', () => {
    // water box grid: 10k atoms
    const atoms: Atom[] = [];
    let id = 1;
    for (let i = 0; i < 22; i++)
      for (let j = 0; j < 22; j++)
        for (let k = 0; k < 21; k++) {
          atoms.push(mkAtom(id++, 8, i * 3.1, j * 3.1 + (id % 2) * 1.5, k * 3.1));
        }
    const t0 = performance.now();
    const bonds = inferBonds(atoms);
    const dt = performance.now() - t0;
    expect(dt).toBeLessThan(2000);
    expect(bonds.length).toBeGreaterThanOrEqual(0);
  });
});

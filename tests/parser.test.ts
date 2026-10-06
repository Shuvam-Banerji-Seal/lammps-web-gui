import { describe, it, expect } from 'vitest';
import { parseDataFile } from '../src/services/parser';

const SAMPLE_LAMMPS_DATA = `# Test molecule
4 atoms
3 bonds

2 atom types
1 bond types

-5.0 5.0 xlo xhi
-5.0 5.0 ylo yhi
-5.0 5.0 zlo zhi

Masses

1 12.011 # Carbon
2 1.008 # H

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 2 0.0 1.0 0.0 0.0
3 1 2 0.0 0.0 1.0 0.0
4 1 2 0.0 0.0 0.0 1.0

Bonds

1 1 1 2
2 1 1 3
3 1 1 4
`;

const EMPTY_DATA = `# Empty file
0 atoms
0 bonds
`;

describe('LAMMPS Data Parser', () => {
  it('should parse atoms correctly', () => {
    const result = parseDataFile(SAMPLE_LAMMPS_DATA);
    expect(result.atoms).toHaveLength(4);
    expect(result.atoms[0]).toEqual({
      id: 1, molId: 1, type: 1, charge: 0.0, x: 0.0, y: 0.0, z: 0.0,
    });
  });

  it('should parse bonds correctly', () => {
    const result = parseDataFile(SAMPLE_LAMMPS_DATA);
    expect(result.bonds).toHaveLength(3);
    expect(result.bonds[0]).toEqual({
      id: 1, type: 1, atom1Id: 1, atom2Id: 2,
    });
  });

  it('should compute bounding box and center (box center preferred when box present)', () => {
    const result = parseDataFile(SAMPLE_LAMMPS_DATA);
    expect(result.min).toEqual({ x: 0, y: 0, z: 0 });
    expect(result.max).toEqual({ x: 1, y: 1, z: 1 });
    // Box is [-5,5]^3 -> box center (0,0,0) wins over atom-bounds center (0.5,0.5,0.5)
    expect(result.center).toEqual({ x: 0, y: 0, z: 0 });
    expect(result.box).toEqual({ xlo: -5, xhi: 5, ylo: -5, yhi: 5, zlo: -5, zhi: 5 });
  });

  it('should fall back to atom-bounds center when no box bounds exist', () => {
    const data = `# no box
2 atoms

Masses

1 12.011 # C

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 1 0.0 2.0 4.0 6.0
`;
    const result = parseDataFile(data);
    expect(result.box).toBeUndefined();
    expect(result.center).toEqual({ x: 1, y: 2, z: 3 });
  });

  it('should parse triclinic tilt factors', () => {
    const data = `# triclinic
2 atoms

-5.0 5.0 xlo xhi
-5.0 5.0 ylo yhi
-5.0 5.0 zlo zhi
0.0 1.0 0.5 xy xz yz

Masses

1 12.011 # C

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 1 0.0 1.0 1.0 1.0
`;
    const result = parseDataFile(data);
    expect(result.box).toBeDefined();
    expect(result.box!.xy).toBe(0.0);
    expect(result.box!.xz).toBe(1.0);
    expect(result.box!.yz).toBe(0.5);
  });

  it('should support atomic style (id type x y z)', () => {
    const data = `# atomic style
2 atoms

Masses

1 12.011 # C

Atoms # atomic

1 1 1.0 2.0 3.0
2 1 -1.0 -2.0 -3.0
`;
    const result = parseDataFile(data);
    expect(result.atoms).toHaveLength(2);
    expect(result.atoms[0]).toMatchObject({ id: 1, type: 1, x: 1.0, y: 2.0, z: 3.0 });
    expect(result.atoms[1]).toMatchObject({ id: 2, type: 1, x: -1.0, y: -2.0, z: -3.0 });
  });

  it('should support charge style (id type q x y z)', () => {
    const data = `# charge style
2 atoms

Masses

1 15.999 # O

Atoms # charge

1 1 -0.8 1.0 0.0 0.0
2 1 0.4 2.0 1.0 1.0
`;
    const result = parseDataFile(data);
    expect(result.atoms).toHaveLength(2);
    expect(result.atoms[0]).toMatchObject({ id: 1, type: 1, charge: -0.8, x: 1.0 });
    expect(result.atoms[1]).toMatchObject({ id: 2, type: 1, charge: 0.4 });
  });

  it('should support molecular style (id mol type x y z)', () => {
    const data = `# molecular style
3 atoms

Masses

1 12.011 # C

Atoms # molecular

1 1 1 0.0 0.0 0.0
2 1 1 1.5 0.0 0.0
3 2 1 0.0 1.5 0.0
`;
    const result = parseDataFile(data);
    expect(result.atoms).toHaveLength(3);
    expect(result.atoms[2]).toMatchObject({ id: 3, molId: 2, type: 1, x: 0.0, y: 1.5, z: 0.0 });
  });

  it('should auto-detect style per row when header lacks a style comment', () => {
    const data = `# no style comment
2 atoms

Masses

1 12.011 # C

Atoms

1 1 6 0.0 1.0 2.0 3.0
`;
    const result = parseDataFile(data);
    expect(result.atoms).toHaveLength(1);
    expect(result.atoms[0]).toMatchObject({ id: 1, molId: 1, type: 6, x: 1.0, y: 2.0, z: 3.0 });
  });

  it('should resolve element from full element-name comments', () => {
    const data = `# name comment
1 atoms

Masses

1 26.982 # Aluminium

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
`;
    const result = parseDataFile(data);
    expect(result.atomTypes[1].element).toBe('Al');
  });

  it('should identify atom types from masses', () => {
    const result = parseDataFile(SAMPLE_LAMMPS_DATA);
    expect(result.atomTypes[1]).toBeDefined();
    expect(result.atomTypes[1].element).toBe('C');
    expect(result.atomTypes[1].count).toBe(1);
    expect(result.atomTypes[2]).toBeDefined();
    expect(result.atomTypes[2].element).toBe('H');
    expect(result.atomTypes[2].count).toBe(3);
  });

  it('should handle empty data gracefully', () => {
    const result = parseDataFile(EMPTY_DATA);
    expect(result.atoms).toHaveLength(0);
    expect(result.bonds).toHaveLength(0);
    expect(result.center).toEqual({ x: 0, y: 0, z: 0 });
  });

  it('should handle data without Masses section', () => {
    const data = `# No masses
2 atoms

1 atom types

-5.0 5.0 xlo xhi
-5.0 5.0 ylo yhi
-5.0 5.0 zlo zhi

Atoms # full

1 1 6 0.0 1.0 2.0 3.0
2 1 6 0.0 4.0 5.0 6.0
`;
    const result = parseDataFile(data);
    expect(result.atoms).toHaveLength(2);
    // Without mass, type 6 should match Carbon by atomic number fallback
    expect(result.atomTypes[6].element).toBe('C');
  });

  it('should skip non-atom/bond sections like Velocities', () => {
    const data = `# With velocities
2 atoms

1 atom types

-5.0 5.0 xlo xhi
-5.0 5.0 ylo yhi
-5.0 5.0 zlo zhi

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 1 0.0 1.0 1.0 1.0

Velocities

1 0.0 0.0 0.0
2 0.0 0.0 0.0
`;
    const result = parseDataFile(data);
    expect(result.atoms).toHaveLength(2);
  });
});

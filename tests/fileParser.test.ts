import { describe, it, expect } from 'vitest';
import { detectFileFormat, detectFormatFromContent, parseFile } from '../services/fileParser';

describe('File Format Detection', () => {
  describe('detectFileFormat (by extension)', () => {
    it('should detect LAMMPS data files', () => {
      expect(detectFileFormat('molecule.data')).toBe('lammps');
      expect(detectFileFormat('file.lmp')).toBe('lammps');
      expect(detectFileFormat('file.lammps')).toBe('lammps');
      expect(detectFileFormat('file.txt')).toBe('lammps');
    });

    it('should detect XYZ files', () => {
      expect(detectFileFormat('molecule.xyz')).toBe('xyz');
    });

    it('should detect PDB files', () => {
      expect(detectFileFormat('protein.pdb')).toBe('pdb');
      expect(detectFileFormat('protein.ent')).toBe('pdb');
    });

    it('should default to lammps for unknown extensions', () => {
      expect(detectFileFormat('file.unknown')).toBe('lammps');
    });
  });

  describe('detectFormatFromContent', () => {
    it('should detect XYZ content', () => {
      const content = '3\ncomment\nO 0.0 0.0 0.0\nH 1.0 0.0 0.0\nH 0.0 1.0 0.0\n';
      expect(detectFormatFromContent(content)).toBe('xyz');
    });

    it('should detect PDB content', () => {
      const content = 'ATOM      1  N   ALA A   1       1.0   2.0   3.0  1.00  0.00           N\nEND\n';
      expect(detectFormatFromContent(content)).toBe('pdb');
    });

    it('should detect PDB from HEADER', () => {
      const content = 'HEADER    TEST\nATOM      1  N   ALA A   1       1.0   2.0   3.0\nEND\n';
      expect(detectFormatFromContent(content)).toBe('pdb');
    });

    it('should default to lammps for typical LAMMPS data', () => {
      const content = '# LAMMPS data\n4 atoms\n3 bonds\n';
      expect(detectFormatFromContent(content)).toBe('lammps');
    });
  });

  describe('parseFile (unified parser)', () => {
    it('should parse LAMMPS data with explicit format', () => {
      const data = `# Test
2 atoms
0 bonds

1 atom types

-5.0 5.0 xlo xhi
-5.0 5.0 ylo yhi
-5.0 5.0 zlo zhi

Masses

1 12.011 # C

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 1 0.0 1.0 1.0 1.0
`;
      const result = parseFile(data, 'lammps');
      expect(result.atoms).toHaveLength(2);
    });

    it('should parse XYZ data with explicit format', () => {
      const data = '2\ntest\nC 0.0 0.0 0.0\nC 1.5 0.0 0.0\n';
      const result = parseFile(data, 'xyz');
      expect(result.atoms).toHaveLength(2);
    });

    it('should auto-detect XYZ format', () => {
      const data = '2\ntest\nC 0.0 0.0 0.0\nC 1.5 0.0 0.0\n';
      const result = parseFile(data);
      expect(result.atoms).toHaveLength(2);
    });

    it('should auto-detect PDB format', () => {
      const data = 'ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\nEND\n';
      const result = parseFile(data);
      expect(result.atoms).toHaveLength(1);
    });
  });
});

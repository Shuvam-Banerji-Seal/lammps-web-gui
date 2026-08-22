import { describe, it, expect } from 'vitest';
import { parsePDBFile } from '../src/services/pdbParser';

const SAMPLE_PDB = `HEADER    TEST
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O
CONECT    1    2
CONECT    2    3
CONECT    3    4
END
`;

const SAMPLE_HETATM = `HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
HETATM    2  H1  HOH A   1       0.960   0.000   0.000  1.00  0.00           H
HETATM    3  H2  HOH A   1      -0.240   0.930   0.000  1.00  0.00           H
END
`;

describe('PDB Parser', () => {
  it('should parse ATOM records', () => {
    const result = parsePDBFile(SAMPLE_PDB);
    expect(result.atoms).toHaveLength(4);
    expect(result.atoms[0].x).toBeCloseTo(1.0);
    expect(result.atoms[0].y).toBeCloseTo(2.0);
    expect(result.atoms[0].z).toBeCloseTo(3.0);
  });

  it('should parse element types from PDB columns', () => {
    const result = parsePDBFile(SAMPLE_PDB);
    // Should have N, C, O types
    const elements = Object.values(result.atomTypes).map(t => t.element).sort();
    expect(elements).toContain('N');
    expect(elements).toContain('C');
    expect(elements).toContain('O');
  });

  it('should parse CONECT bonds', () => {
    const result = parsePDBFile(SAMPLE_PDB);
    expect(result.bonds).toHaveLength(3);
    expect(result.bonds[0].atom1Id).toBe(1);
    expect(result.bonds[0].atom2Id).toBe(2);
  });

  it('should parse HETATM records', () => {
    const result = parsePDBFile(SAMPLE_HETATM);
    expect(result.atoms).toHaveLength(3);
  });

  it('should compute bounding box', () => {
    const result = parsePDBFile(SAMPLE_PDB);
    expect(result.min.x).toBeCloseTo(1.0);
    expect(result.max.x).toBeCloseTo(4.0);
    expect(result.center.x).toBeCloseTo(2.5);
  });

  it('should handle empty PDB gracefully', () => {
    const result = parsePDBFile('END\n');
    expect(result.atoms).toHaveLength(0);
    expect(result.center).toEqual({ x: 0, y: 0, z: 0 });
  });

  it('should deduplicate CONECT bonds', () => {
    const pdb = `ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
CONECT    1    2
CONECT    2    1
END
`;
    const result = parsePDBFile(pdb);
    expect(result.bonds).toHaveLength(1);
  });
});

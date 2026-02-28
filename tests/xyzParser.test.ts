import { describe, it, expect } from 'vitest';
import { parseXYZFile } from '../services/xyzParser';

const SAMPLE_XYZ = `3
Water molecule
O 0.0 0.0 0.0
H 0.96 0.0 0.0
H -0.24 0.93 0.0
`;

const SAMPLE_XYZ_MULTI_ELEMENT = `5
Methane
C 0.0 0.0 0.0
H 0.63 0.63 0.63
H -0.63 -0.63 0.63
H -0.63 0.63 -0.63
H 0.63 -0.63 -0.63
`;

describe('XYZ Parser', () => {
  it('should parse atom count and positions', () => {
    const result = parseXYZFile(SAMPLE_XYZ);
    expect(result.atoms).toHaveLength(3);
    expect(result.atoms[0].x).toBeCloseTo(0.0);
    expect(result.atoms[1].x).toBeCloseTo(0.96);
  });

  it('should assign correct element types', () => {
    const result = parseXYZFile(SAMPLE_XYZ);
    // Oxygen should be type 8 (atomic number)
    expect(result.atomTypes[8]).toBeDefined();
    expect(result.atomTypes[8].element).toBe('O');
    expect(result.atomTypes[8].count).toBe(1);
    // Hydrogen should be type 1
    expect(result.atomTypes[1]).toBeDefined();
    expect(result.atomTypes[1].element).toBe('H');
    expect(result.atomTypes[1].count).toBe(2);
  });

  it('should compute center correctly', () => {
    const result = parseXYZFile(SAMPLE_XYZ);
    // Center = (min+max)/2
    expect(result.center.x).toBeCloseTo((-0.24 + 0.96) / 2);
  });

  it('should auto-generate bonds for close atoms', () => {
    const result = parseXYZFile(SAMPLE_XYZ);
    // O-H bonds should be ~0.96 Å which is < 1.8 threshold
    expect(result.bonds.length).toBeGreaterThan(0);
  });

  it('should parse multi-element molecules', () => {
    const result = parseXYZFile(SAMPLE_XYZ_MULTI_ELEMENT);
    expect(result.atoms).toHaveLength(5);
    expect(Object.keys(result.atomTypes)).toHaveLength(2);
  });

  it('should throw on invalid XYZ data', () => {
    expect(() => parseXYZFile('')).toThrow('Invalid XYZ file');
    expect(() => parseXYZFile('abc\ncomment\n')).toThrow('Invalid XYZ file');
  });

  it('should handle XYZ with zero atoms', () => {
    expect(() => parseXYZFile('0\ncomment\n')).toThrow('Invalid XYZ file');
  });
});

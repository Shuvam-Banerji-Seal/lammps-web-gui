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

describe('XYZ trajectory (multi-frame)', () => {
  const TRAJ = `3
frame one
O 0.0 0.0 0.0
H 0.76 0.59 0.0
H -0.76 0.59 0.0
3
frame two — displaced
O 0.2 0.1 0.0
H 0.96 0.69 0.1
H -0.56 0.69 -0.1

3
frame three
O 0.4 0.2 0.5
H 1.16 0.79 0.6
H -0.36 0.79 0.4
`;

  it('parses every frame with stable ids and shared type mapping', () => {
    const r = parseXYZFile(TRAJ);
    expect(r.frames).toHaveLength(3);
    expect(r.atoms).toHaveLength(3);
    expect(r.frames![0].comment).toBe('frame one');
    expect(r.frames![1].comment).toBe('frame two — displaced');
    // ids restart per frame; types identical across frames
    expect(r.frames![1].atoms[0].id).toBe(1);
    expect(r.frames![1].atoms[0].type).toBe(r.atoms[0].type);
  });

  it('bonds derive from the first frame only', () => {
    const r = parseXYZFile(TRAJ);
    expect(r.bonds.length).toBeGreaterThanOrEqual(2); // two O-H per water
  });

  it('bounding box spans ALL frames so playback never re-frames', () => {
    const r = parseXYZFile(TRAJ);
    expect(r.min.x).toBeLessThanOrEqual(-0.76);
    expect(r.max.x).toBeGreaterThanOrEqual(1.16);
    expect(r.max.z).toBeGreaterThanOrEqual(0.6);
  });

  it('single-frame files expose no frames array', () => {
    const single = `2\nc\nHe 0 0 0\nHe 1 0 0\n`;
    expect(parseXYZFile(single).frames).toBeUndefined();
  });
});

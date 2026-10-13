import { describe, expect, it } from 'vitest';
import { parseStl } from '../src/engine-gpl/machdyn/stl_reader';
import { PairSMDUlsph } from '../src/engine-gpl/machdyn/pair_smd_ulsph';
import { StyleError } from '../src/engine/force/types';

// The STL reader of fix smd/wall_surface (ASCII and binary, detected from the file size) and the settings and
// coefficient checks of pair smd/ulsph that the port refuses (native keywords not ported yet).

const ascii = [
  'solid test mesh',
  '  facet normal 0 0 1',
  '    outer loop',
  '      vertex 0 0 0',
  '      vertex 1 0 0',
  '      vertex 0 1 0',
  '    endloop',
  '  endfacet',
  'endsolid test mesh',
  '',
].join('\n');

/** The binary layout of an STL file: 80-byte header, uint32 count, 50 bytes per facet (floats little-endian, attribute). */
const binary = (): string => {
  const buf = new ArrayBuffer(84 + 50);
  const v = new DataView(buf);
  v.setUint32(80, 1, true);
  const floats = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];
  floats.forEach((f, k) => v.setFloat32(84 + 4 * k, f, true));
  return String.fromCharCode(...new Uint8Array(buf));
};

describe('STL reader (fix smd/wall_surface)', () => {
  it('reads an ASCII facet with its normal, vertices and title', () => {
    const mesh = parseStl(ascii);
    expect(mesh.title).toBe('test mesh');
    expect(mesh.triangles).toHaveLength(1);
    expect(mesh.triangles[0].normal).toEqual([0, 0, 1]);
    expect(mesh.triangles[0].vert[1]).toEqual([1, 0, 0]);
  });

  it('reads a binary facet (size 84 + 50 N bytes)', () => {
    const mesh = parseStl(binary());
    expect(mesh.triangles).toHaveLength(1);
    expect(mesh.triangles[0].normal).toEqual([0, 0, 1]);
    expect(mesh.triangles[0].vert[2]).toEqual([0, 1, 0]);
  });

  it('refuses a file that is neither ASCII STL nor binary STL', () => {
    expect(() => parseStl('not a mesh\n')).toThrow(StyleError);
  });
});

describe('pair smd/ulsph settings and coefficients', () => {
  const ctx = { log: () => undefined, s: null } as never;

  it('refuses density summation and gradient correction, naming the keyword', () => {
    const p = new PairSMDUlsph();
    expect(() => p.settings(['*DENSITY_SUMMATION', '*VELOCITY_GRADIENT', '*NO_GRADIENT_CORRECTION'], ctx)).toThrow(/\*DENSITY_SUMMATION/);
    expect(() => p.settings(['*DENSITY_CONTINUITY', '*VELOCITY_GRADIENT', '*GRADIENT_CORRECTION'], ctx)).toThrow(/\*GRADIENT_CORRECTION/);
  });

  it('refuses the material keywords that are not ported, naming them', () => {
    const p = new PairSMDUlsph();
    p.allocate(1);
    expect(() => p.coeff(['1', '1', '*COMMON', '1', '10', '0.06', '1', '0', '*VISCOSITY_NEWTON', '1', '*END'], ctx)).toThrow(/\*VISCOSITY_NEWTON/);
    expect(() => p.coeff(['1', '1', '*COMMON', '1', '10', '0.06', '1', '0', '*ARTIFICIAL_STRESS', '1', '*END'], ctx)).toThrow(/\*ARTIFICIAL_STRESS/);
  });

  it('requires an equation of state', () => {
    const p = new PairSMDUlsph();
    p.allocate(1);
    expect(() => p.coeff(['1', '1', '*COMMON', '1', '10', '0.06', '1', '0', '*END'], ctx)).toThrow(/must specify an EOS/);
  });
});

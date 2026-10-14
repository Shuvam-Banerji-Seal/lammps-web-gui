import { describe, expect, it } from 'vitest';
import { parseStl } from '../src/engine/stl';
import { readInputText } from '../src/services/fileStore';

/*
 * Binary STL (docs.lammps.org/create_atoms.html: "The reader supports both ASCII and binary files conforming to the
 * format on the Wikipedia page"). The binary facets are built here in the documented layout, read back, and compared
 * with the ASCII STL of the same facets. The values are exact in float32, so the comparison is exact.
 */

interface Facet {
  n: [number, number, number];
  v: [[number, number, number], [number, number, number], [number, number, number]];
}

// values exact in binary32 (multiples of 1/8), so the ASCII text and the float32 facets agree bit for bit
const FACETS: Facet[] = [
  { n: [0, 0, 1], v: [[0, 0, 0], [2, 0, 0.5], [1, 3.25, 0.125]] },
  { n: [0.5, -0.5, 0], v: [[-1.5, 2, 4], [0.75, -2.5, 1], [3, 3, -0.625]] },
  { n: [0, 1, 0], v: [[10, 10, 10], [11, 10, 10], [10, 10, 12.5]] },
];

const binaryOf = (facets: Facet[], header = 'COLOR=177 177 177 MATERIAL=177 177 177'): string => {
  const size = 84 + 50 * facets.length;
  const buf = new DataView(new ArrayBuffer(size));
  const bytes = new Uint8Array(buf.buffer);
  for (let k = 0; k < header.length && k < 80; k++) bytes[k] = header.charCodeAt(k);
  buf.setUint32(80, facets.length, true);
  facets.forEach((f, t) => {
    const o = 84 + 50 * t;
    const vals = [...f.n, ...f.v[0], ...f.v[1], ...f.v[2]];
    vals.forEach((x, k) => buf.setFloat32(o + 4 * k, x, true));
    buf.setUint16(o + 48, 0, true);
  });
  let out = '';
  for (let k = 0; k < bytes.length; k++) out += String.fromCharCode(bytes[k]);
  return out;
};

const asciiOf = (facets: Facet[]): string => {
  const lines = ['solid mesh'];
  for (const f of facets) {
    lines.push(`  facet normal ${f.n.join(' ')}`, '    outer loop');
    for (const v of f.v) lines.push(`      vertex ${v.join(' ')}`);
    lines.push('    endloop', '  endfacet');
  }
  lines.push('endsolid mesh', '');
  return lines.join('\n');
};

describe('binary STL reader', () => {
  it('reads the facets of a binary file in file order, the same as its ASCII form', () => {
    const fromBinary = parseStl(binaryOf(FACETS), 'part.stl');
    const fromAscii = parseStl(asciiOf(FACETS), 'part.stl');
    expect(fromBinary).toHaveLength(3);
    expect(fromBinary).toEqual(fromAscii);
    expect(fromBinary[1].v).toEqual(FACETS[1].v);
  });

  it('reads a binary file whose header starts with the word solid (the size decides, not the header)', () => {
    const tris = parseStl(binaryOf(FACETS, 'solid but binary'), 'part.stl');
    expect(tris).toHaveLength(3);
    expect(tris[0].v).toEqual(FACETS[0].v);
  });

  it('reads a binary file with the same bytes as a latin1 string of a file, including high bytes', () => {
    // a normal of -0.5 has its sign bit in the last byte: the high bytes must survive as one character each
    const text = binaryOf([FACETS[1]]);
    expect(Array.from(text).some((c) => c.charCodeAt(0) > 127)).toBe(true);
    expect(parseStl(text, 'x.stl')[0].v).toEqual(FACETS[1].v);
  });

  it('refuses a binary file whose size does not match its triangle count', () => {
    const text = binaryOf(FACETS).slice(0, -1);
    expect(() => parseStl(text, 'part.stl')).toThrow(/neither an ASCII STL/);
  });

  it('refuses a binary file with no triangles', () => {
    expect(() => parseStl(binaryOf([]), 'empty.stl')).toThrow(/no triangles/);
  });

  it('reads a binary .stl file picked in the browser with its bytes intact', async () => {
    const text = binaryOf(FACETS);
    const bytes = new Uint8Array(text.length);
    for (let k = 0; k < text.length; k++) bytes[k] = text.charCodeAt(k);
    const file = new File([bytes], 'mixer.stl');
    const back = await readInputText(file);
    expect(back).toBe(text);
    expect(parseStl(back, 'mixer.stl')).toEqual(parseStl(asciiOf(FACETS), 'mixer.stl'));
  });
});

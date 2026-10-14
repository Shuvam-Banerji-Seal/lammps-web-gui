import { StyleError } from './force/types';

/**
 * One triangle of an ASCII STL mesh: its three vertices in the order they
 * appear under `outer loop` (normals are ignored, as in LAMMPS).
 */
export interface StlTriangle {
  /** Vertices v0, v1, v2, each [x, y, z]. */
  v: [number, number, number][];
}

/** Bytes of a binary STL header (80 bytes of text) and of its uint32 triangle count. */
const BIN_HEADER = 84;
/** Bytes of one binary facet: 12 little-endian float32 values (normal, then three vertices) and a uint16 attribute. */
const BIN_FACET = 50;

/**
 * The triangle count of a binary STL file, or -1 when the text is not one. As native LAMMPS does (src/stl_reader.cpp
 * at b332f28, the size test): a binary file has exactly 84 + 50 N bytes for the N triangles its header claims, which
 * is more robust than looking for the word "solid", since a binary header may start with it too.
 */
const binaryFacetCount = (text: string): number => {
  if (text.length < BIN_HEADER) return -1;
  const n = ((text.charCodeAt(80) & 0xff) | ((text.charCodeAt(81) & 0xff) << 8) |
    ((text.charCodeAt(82) & 0xff) << 16) | ((text.charCodeAt(83) & 0xff) << 24)) >>> 0;
  return BIN_HEADER + BIN_FACET * n === text.length ? n : -1;
};

/**
 * Reads the facets of a binary STL mesh: an 80-byte header, a uint32 triangle count, then per facet the normal and
 * the three vertices as little-endian float32 values and a two-byte attribute (which is skipped). The facets keep
 * their file order; the normals are not used, as for ASCII.
 */
const parseBinaryStl = (text: string, count: number): StlTriangle[] => {
  const bytes = new Uint8Array(text.length);
  for (let k = 0; k < text.length; k++) bytes[k] = text.charCodeAt(k) & 0xff;
  const view = new DataView(bytes.buffer);
  const tris: StlTriangle[] = [];
  for (let t = 0; t < count; t++) {
    const o = BIN_HEADER + t * BIN_FACET;
    const vert = (c: number): [number, number, number] => {
      // float 0..2 is the normal, the vertices are floats 3..11
      const b = o + 12 + 12 * c;
      return [view.getFloat32(b, true), view.getFloat32(b + 4, true), view.getFloat32(b + 8, true)];
    };
    tris.push({ v: [vert(0), vert(1), vert(2)] });
  }
  return tris;
};

/**
 * Reads an STL mesh, ASCII or binary. docs.lammps.org/create_atoms.html: "The reader supports both ASCII and binary
 * files conforming to the format on the Wikipedia page." The engine keeps files as strings with one character per
 * byte, so a binary file arrives as its bytes (session input reads such files as latin1, see readInputText).
 */
export const parseStl = (text: string, name: string): StlTriangle[] => {
  const count = binaryFacetCount(text);
  if (count >= 0) {
    if (count === 0) throw new StyleError(`create_atoms mesh: ${name} is a binary STL with no triangles`);
    return parseBinaryStl(text, count);
  }
  if (!/^\s*solid\b/i.test(text)) {
    throw new StyleError(`create_atoms mesh: ${name} is neither an ASCII STL (starting with 'solid') nor a binary STL of 84 + 50 N bytes`);
  }
  const verts: [number, number, number][] = [];
  for (const line of text.split('\n')) {
    // facet normal ... / outer loop / vertex x y z (three times) / endloop / endfacet
    const m = /^\s*vertex\s+(\S+)\s+(\S+)\s+(\S+)\s*$/.exec(line);
    if (!m) continue;
    const x = Number(m[1]), y = Number(m[2]), z = Number(m[3]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new StyleError(`create_atoms mesh: ${name} has a malformed vertex line '${line.trim()}'`);
    }
    verts.push([x, y, z]);
  }
  if (verts.length === 0 || verts.length % 3 !== 0) {
    throw new StyleError(`create_atoms mesh: ${name} has ${verts.length} vertices, not a multiple of 3 (one triangle needs three)`);
  }
  const tris: StlTriangle[] = [];
  for (let i = 0; i < verts.length; i += 3) tris.push({ v: [verts[i], verts[i + 1], verts[i + 2]] });
  return tris;
};

import { StyleError } from './force/types';

/**
 * One triangle of an ASCII STL mesh: its three vertices in the order they
 * appear under `outer loop` (normals are ignored, as in LAMMPS).
 */
export interface StlTriangle {
  /** Vertices v0, v1, v2, each [x, y, z]. */
  v: [number, number, number][];
}

/**
 * Reads an ASCII STL mesh. docs.lammps.org/create_atoms.html: "a file with a
 * triangle mesh in" STL format "is read"; "The reader supports both ASCII and
 * binary files conforming to the format on the Wikipedia page. Binary STL
 * files (e.g. as frequently offered for 3d-printing) can also be first
 * converted to ASCII for editing with the" stl_bin2txt tool. A binary STL (an
 * 80-byte header and a uint32 triangle count, no leading solid keyword) is
 * rejected with a StyleError pointing at that conversion.
 */
export const parseStl = (text: string, name: string): StlTriangle[] => {
  if (!/^\s*solid\b/i.test(text)) {
    throw new StyleError(`create_atoms mesh: ${name} is a binary STL, which the browser engine cannot read; convert it to ASCII first with the stl_bin2txt tool (docs.lammps.org/create_atoms.html)`);
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

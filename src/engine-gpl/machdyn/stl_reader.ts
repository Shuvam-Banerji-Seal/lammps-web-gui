/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/stl_reader.h and src/stl_reader.cpp (STLReader::parse, parse_text, parse_binary) at commit b332f28, used by
 * src/MACHDYN/fix_smd_wall_surface.cpp. Copyright (2003) Sandia Corporation; LAMMPS is distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * An STL file (stereolithography, https://en.wikipedia.org/wiki/STL_(file_format)) is read as ASCII or binary; the format
 * is detected from the size of the file: a binary file has exactly 84 + 50 N bytes for N triangles. The text is given
 * as a string whose characters are the bytes of the file (the engine keeps files as text; a binary file arrives as one
 * character per byte).
 */
import { StyleError } from '../../engine/force/types';

export interface StlTriangle {
  normal: [number, number, number];
  vert: [[number, number, number], [number, number, number], [number, number, number]];
}

export interface StlMesh {
  title: string;
  triangles: StlTriangle[];
}

const BIN_HEADER = 80 + 4;
const BIN_PER_TRI = 12 * 4 + 2;

/** Binary STL: 80-byte header, uint32 triangle count, then per triangle 12 little-endian floats and a 2-byte attribute. */
const parseBinary = (bytes: string): StlMesh => {
  const u8 = (k: number) => bytes.charCodeAt(k) & 0xff;
  let len = 0;
  while (len < 80 && u8(len) !== 0) len++;
  const title = bytes.slice(0, len).trim();
  const view = new DataView(new ArrayBuffer(bytes.length));
  for (let k = 0; k < bytes.length; k++) view.setUint8(k, u8(k));
  const ntri = view.getUint32(80, true);
  const triangles: StlTriangle[] = [];
  for (let t = 0; t < ntri; t++) {
    const o = BIN_HEADER + t * BIN_PER_TRI;
    const f = (k: number) => view.getFloat32(o + 4 * k, true);
    triangles.push({
      normal: [f(0), f(1), f(2)],
      vert: [[f(3), f(4), f(5)], [f(6), f(7), f(8)], [f(9), f(10), f(11)]],
    });
  }
  return { title, triangles };
};

/** ASCII STL: "solid [name]", then facet / outer loop / 3 vertex lines / endloop / endfacet, until endsolid. */
const parseText = (text: string): StlMesh => {
  const lines = text.split(/\r?\n/);
  let k = 0;
  const nextLine = (): string | null => (k < lines.length ? lines[k++] : null);
  const first = nextLine();
  if (first === null || !/^ *solid/.test(first)) throw new StyleError('File is not a valid ASCII STL file');
  const title = first.slice(first.indexOf('solid') + 5).trim();
  const triangles: StlTriangle[] = [];
  for (let line = nextLine(); line !== null; line = nextLine()) {
    const words = line.trim().split(/\s+/).filter((w) => w.length > 0);
    if (words.length === 0) continue;
    if (/^endsolid/.test(words[0])) break;
    if (!/^facet/.test(words[0])) throw new StyleError(`Expected 'facet' or 'endsolid' in STL file, got: ${line.trim()}`);
    const normal: [number, number, number] = [0, 0, 0];
    if (words.length >= 5 && /^normal/.test(words[1])) {
      const v = [Number(words[2]), Number(words[3]), Number(words[4])];
      if (v.every(Number.isFinite)) {
        normal[0] = v[0];
        normal[1] = v[1];
        normal[2] = v[2];
      }
    }
    const outer = nextLine();
    if (outer === null || !/^ *outer *loop/.test(outer)) throw new StyleError("Error reading 'outer loop' in STL file");
    const vert: StlTriangle['vert'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let q = 0; q < 3; q++) {
      let vl = nextLine();
      while (vl !== null && vl.trim() === '') vl = nextLine();
      const w = (vl ?? '').trim().split(/\s+/);
      if (w[0] !== 'vertex' || w.length < 4) throw new StyleError(`Error reading vertex ${q + 1} of facet in STL file`);
      const c = [Number(w[1]), Number(w[2]), Number(w[3])];
      if (!c.every(Number.isFinite)) throw new StyleError(`Error parsing STL file: vertex ${q + 1} of facet is not a number`);
      vert[q] = [c[0], c[1], c[2]];
    }
    const endloop = nextLine();
    if (endloop === null || !/^ *endloop/.test(endloop)) throw new StyleError("Error reading 'endloop' in STL file");
    const endfacet = nextLine();
    if (endfacet === null || !/^ *endfacet/.test(endfacet)) throw new StyleError("Error reading 'endfacet' in STL file");
    triangles.push({ normal, vert });
  }
  return { title, triangles };
};

/**
 * STLReader::parse: the triangles of an STL file (a file with no facets gives an empty list). The binary format is used
 * when the size of the file equals 84 + 50 N with N the triangle count in its header.
 */
export const parseStl = (bytes: string): StlMesh => {
  if (bytes.length >= BIN_HEADER) {
    const b = (k: number) => bytes.charCodeAt(k) & 0xff;
    const ntri = (b(80) | (b(81) << 8) | (b(82) << 16) | (b(83) << 24)) >>> 0;
    if (BIN_HEADER + ntri * BIN_PER_TRI === bytes.length) return parseBinary(bytes);
  }
  return parseText(bytes);
};

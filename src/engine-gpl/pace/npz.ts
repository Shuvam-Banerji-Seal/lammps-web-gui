/*
 * Ported from LAMMPS (commit b332f28): the cnpy library bundled with the ML-PACE library v.2025.12.4.p1 that LAMMPS
 * (https://github.com/lammps/lammps, commit b332f28) builds for src/ML-PACE: cnpy/cnpy.cpp (npz_load, parse_npy_header,
 * load_the_npy_file; MIT licence, Copyright (C) 2011 Carl Rogers). This file is GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * The .npz reader of the active-set file (.asi) of pair_style pace/extrapolation: the zip's local file headers one after
 * another, each stored (uncompressed) entry an .npy array named by its file name without ".npy". As in the bundled
 * cnpy, a deflated entry is not read (its inflate code is commented out there).
 */
import { StyleError } from '../../engine/force/types';

/** One array: its shape, word size and the data as doubles in file order. */
export interface NpyArray {
  shape: number[];
  wordSize: number;
  fortranOrder: boolean;
  data: Float64Array;
}

/**
 * The bytes of a notebook file: a binary upload is stored one character per byte (latin1), a file that decoded as UTF-8
 * as its text, which is encoded back.
 */
export const fileBytes = (text: string): Uint8Array => {
  for (let k = 0; k < text.length; k++) {
    if (text.charCodeAt(k) > 0xff) return new TextEncoder().encode(text);
  }
  const bytes = new Uint8Array(text.length);
  for (let k = 0; k < text.length; k++) bytes[k] = text.charCodeAt(k);
  return bytes;
};

/** cnpy::npz_load(fname): the arrays of an .npz file by name (npz_t is a std::map, iterated in name order). */
export const npzLoad = (bytes: Uint8Array): Map<string, NpyArray> => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const arrays = new Map<string, NpyArray>();
  let pos = 0;
  const fail = (): never => { throw new StyleError('npz_load: failed fread'); };
  for (;;) {
    if (pos + 30 > bytes.length) fail();
    const hdr = pos;
    pos += 30;
    // the global header ends the local entries
    if (bytes[hdr + 2] !== 0x03 || bytes[hdr + 3] !== 0x04) break;
    const nameLen = view.getUint16(hdr + 26, true);
    if (pos + nameLen > bytes.length) fail();
    let varname = '';
    for (let k = 0; k < nameLen; k++) varname += String.fromCharCode(bytes[pos + k]);
    pos += nameLen;
    varname = varname.slice(0, Math.max(0, varname.length - 4));
    const extraLen = view.getUint16(hdr + 28, true);
    if (pos + extraLen > bytes.length) fail();
    pos += extraLen;
    const comprMethod = view.getUint16(hdr + 8, true);
    if (comprMethod !== 0) {
      // the bundled cnpy skips a deflated entry without moving past its data: the next "header" is compressed data,
      // which ends the loop (or the read)
      continue;
    }
    // parse_npy_header: 11 bytes, then the rest of the header line through '\n' (fgets, at most 255 characters)
    if (pos + 11 > bytes.length) throw new StyleError('parse_npy_header: failed fread');
    pos += 11;
    let header = '';
    while (pos < bytes.length && header.length < 255) {
      const ch = String.fromCharCode(bytes[pos++]);
      header += ch;
      if (ch === '\n') break;
    }
    let loc1 = header.indexOf('fortran_order');
    if (loc1 < 0) throw new StyleError("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    const fortranOrder = header.substr(loc1, 4) === 'True';
    loc1 = header.indexOf('(');
    const loc2 = header.indexOf(')');
    if (loc1 < 0 || loc2 < 0) throw new StyleError("parse_npy_header: failed to find header keyword: '(' or ')'");
    const shape = (header.substring(loc1 + 1, loc2).match(/[0-9][0-9]*/g) ?? []).map((s) => parseInt(s, 10));
    let d = header.indexOf('descr');
    if (d < 0) throw new StyleError("parse_npy_header: failed to find header keyword: 'descr'");
    d += 9;
    const ws = header.substring(d + 2);
    const wordSize = parseInt(ws.substring(0, ws.indexOf("'")), 10) || 0;
    let count = 1;
    for (const s of shape) count *= s;
    const nbytes = count * wordSize;
    if (pos + nbytes > bytes.length) throw new StyleError('load_the_npy_file: failed fread');
    // as_vec<double>: the bytes read as little-endian doubles
    const data = new Float64Array(Math.floor(nbytes / 8));
    for (let k = 0; k < data.length; k++) data[k] = view.getFloat64(pos + 8 * k, true);
    pos += nbytes;
    arrays.set(varname, { shape, wordSize, fortranOrder, data });
  }
  return arrays;
};

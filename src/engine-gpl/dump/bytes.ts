/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the binary writes of src/EXTRA-DUMP/
 * dump_dcd.cpp (fwrite_int32 and fwrite of floats and doubles, in the byte order of the machine: little-endian on x86) and
 * of src/EXTRA-DUMP/xdr_compat.cpp (xdr_int, xdr_float and xdr_opaque: big-endian 4-byte units, opaque data padded with
 * zeros to a multiple of 4). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

/** A growable byte buffer for the binary dump styles. */
export class ByteBuffer {
  private buf = new Uint8Array(1024);
  private view = new DataView(this.buf.buffer);
  length = 0;

  private reserve(n: number): void {
    if (this.length + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.length + n) cap *= 2;
    const next = new Uint8Array(cap);
    next.set(this.buf.subarray(0, this.length));
    this.buf = next;
    this.view = new DataView(next.buffer);
  }

  /** A 32-bit integer (wrapped as C's conversion to uint32_t/int32_t does). */
  int32(v: number, little: boolean): void {
    this.reserve(4);
    this.view.setUint32(this.length, v >>> 0, little);
    this.length += 4;
  }

  /** A float (the value rounded to single precision). */
  float32(v: number, little: boolean): void {
    this.reserve(4);
    this.view.setFloat32(this.length, v, little);
    this.length += 4;
  }

  float64(v: number, little: boolean): void {
    this.reserve(8);
    this.view.setFloat64(this.length, v, little);
    this.length += 8;
  }

  /** Raw bytes; with `pad`, zeros up to a multiple of 4 bytes after them (xdr_opaque). */
  bytes(b: Uint8Array, pad = false): void {
    const rnd = pad ? (4 - (b.length % 4)) % 4 : 0;
    this.reserve(b.length + rnd);
    this.buf.set(b, this.length);
    this.length += b.length;
    for (let i = 0; i < rnd; i++) this.buf[this.length++] = 0;
  }

  /** The bytes as a string of one character per byte, as the session file store keeps binary files. */
  latin1(): string {
    return latin1(this.buf.subarray(0, this.length));
  }
}

/** Bytes as one character per byte (in chunks: String.fromCharCode takes its arguments on the stack). */
export const latin1 = (b: Uint8Array): string => {
  let s = '';
  for (let i = 0; i < b.length; i += 8192) s += String.fromCharCode(...b.subarray(i, Math.min(b.length, i + 8192)));
  return s;
};

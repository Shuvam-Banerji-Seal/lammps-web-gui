/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-DUMP/dump_xtc.cpp and
 * dump_xtc.h (class DumpXTC, dump style xtc; contributing authors Naveen Michaud-Agrawal (Johns Hopkins U) and Axel
 * Kohlmeyer (Temple U), with the compression routine xdr3dfcoord and its helpers sendbits, sizeofint, sizeofints and
 * sendints, (c) 1995 Frans van Hoesel; copyright Sandia Corporation), and the XDR units of src/EXTRA-DUMP/xdr_compat.cpp
 * (bytes.ts). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump.html (verbatim): "The xtc style writes XTC files, a compressed trajectory format used by the
 * GROMACS molecular dynamics package, and described here. The precision used in XTC files can be adjusted via the
 * dump_modify command. The default value of 1000 means that coordinates are stored to 1/1000 nanometer accuracy. XTC
 * files are portable binary files written in the NFS XDR data format, so that any machine which supports XDR should be
 * able to read them. The number of atoms per snapshot can change with the xtc style. The unit cell dimension information
 * is also written to the file."
 *
 * The core dump (src/engine/output/dump.ts) keeps the schedule and the dump_modify keywords every style takes, and gives
 * the group's atoms in ID order; this helper does the constructor, modify_param, init_style, write_header, pack's scaling
 * (sfactor), write_data and write_frame of DumpXTC.
 */
import { StyleError } from '../../engine/force/types';
import { logical, numeric } from '../../engine/force/util';
import type { DumpTrajFactory, DumpTrajFrame, DumpTrajHost, DumpTrajSettings, DumpTrajStyle } from '../../engine/output/dump';
import { ByteBuffer } from './bytes';
import { invalidTrajName } from './dump_dcd';

/** dump_xtc.cpp: static constexpr double EPS = 1.0e-5; static constexpr int XTC_MAGIC = 1995. */
const EPS = 1.0e-5;
const XTC_MAGIC = 1995;
const MAXSMALLINT = 0x7fffffff;
const INT_MAX = 0x7fffffff;
const INT_MIN = -0x80000000;
/** XDR is big-endian. */
const BE = false;
/** force->angstrom and force->femtosecond of each unit style (src/update.cpp set_units at b332f28). */
const ANGSTROM: Record<string, number> = { lj: 1.0, real: 1.0, metal: 1.0, si: 1.0e-10, cgs: 1.0e-8, electron: 1.88972612, micro: 1.0e-4, nano: 1.0e-1 };
const FEMTOSECOND: Record<string, number> = { lj: 1.0, real: 1.0, metal: 1.0e-3, si: 1.0e-15, cgs: 1.0e-15, electron: 1.0, micro: 1.0e-9, nano: 1.0e-6 };

class DumpXtc implements DumpTrajStyle {
  unwrap = false;
  private natoms: number;
  private neverySave = 0;
  /** float precision: the user-adjustable precision setting. */
  private precision = Math.fround(1000.0);
  /** Scaling factors for positions and time unit. */
  private sfactor: number;
  private tfactor: number;

  constructor(private readonly host: DumpTrajHost, args: string[]) {
    if (args.length) throw new StyleError('Illegal dump xtc command');
    if (invalidTrajName(host.file)) throw new StyleError('Invalid dump xtc filename');
    const n = host.groupCount();
    if (n > Math.floor(MAXSMALLINT / 3 / 4)) throw new StyleError('Too many atoms for dump xtc');
    this.natoms = n;
    // sfactor = conversion of coords to XTC units, tfactor = conversion of simulation time to XTC units: GROMACS standard
    // is nanometers and picoseconds
    const units = host.sys.state.units.style;
    this.sfactor = 0.1 / ANGSTROM[units];
    this.tfactor = 0.001 / FEMTOSECOND[units];
    // in reduced units we do not scale anything
    if (units === 'lj') {
      this.sfactor = this.tfactor = 1.0;
      host.sys.warn('No automatic unit conversion to XTC file format conventions possible for units lj');
    }
    // openfile: xdropen(xd, filename, "w") at the dump command
    host.create();
  }

  modify(args: string[], k: number): number {
    const key = args[k];
    if (key !== 'unwrap' && key !== 'precision' && key !== 'sfactor' && key !== 'tfactor') return 0;
    if (k + 2 > args.length) throw new StyleError('Illegal dump_modify command');
    const v = args[k + 1];
    if (key === 'unwrap') this.unwrap = logical(v);
    else if (key === 'precision') {
      this.precision = Math.fround(numeric(v));
      const p = this.precision;
      if (Math.abs(p - 10.0) > EPS && Math.abs(p - 100.0) > EPS && Math.abs(p - 1000.0) > EPS && Math.abs(p - 10000.0) > EPS
        && Math.abs(p - 100000.0) > EPS && Math.abs(p - 1000000.0) > EPS) throw new StyleError('Illegal dump_modify command');
    } else if (key === 'sfactor') {
      this.sfactor = numeric(v);
      if (this.sfactor <= 0.0) throw new StyleError('Illegal dump_modify sfactor value (must be > 0.0)');
    } else {
      this.tfactor = numeric(v);
      if (this.tfactor <= 0.0) throw new StyleError('Illegal dump_modify tfactor value (must be > 0.0)');
    }
    return 2;
  }

  init(st: DumpTrajSettings): void {
    if (!st.sortById) throw new StyleError('Dump xtc requires sorting by atom ID');
    // check that flush_flag is not set since dump::write() will use it (the constructor sets it to 0)
    if (st.flush === true) throw new StyleError('Cannot set dump_modify flush for dump xtc');
    // check that dump modify settings are compatible with xtc, but only when not being called from write_dump
    if (st.writeDump) return;
    if (st.everyTime) throw new StyleError('Cannot use every/time setting for dump xtc');
    if (st.every === 0) throw new StyleError('Cannot use every variable setting for dump xtc');
    if (this.neverySave === 0) this.neverySave = st.every;
    else if (this.neverySave !== st.every) throw new StyleError('Cannot change dump_modify every for dump xtc');
  }

  write(f: DumpTrajFrame): void {
    const out = new ByteBuffer();
    if (f.header) this.writeHeader(out, f);
    if (f.natoms !== this.natoms) throw new StyleError(`dump xtc: a snapshot of ${f.natoms} atoms with dump_modify header no after one of ${this.natoms} atoms is not supported by the browser engine`);
    // pack: buf = sfactor * x (double); write_data: coords = buf (float)
    const coords = new Float32Array(3 * this.natoms);
    for (let k = 0; k < coords.length; k++) coords[k] = this.sfactor * f.coords[k];
    xdr3dfcoord(out, coords, this.natoms, this.precision);
    this.host.writeBytes(out.latin1());
  }

  private writeHeader(out: ByteBuffer, f: DumpTrajFrame): void {
    const n = f.natoms;
    if (n > MAXSMALLINT) throw new StyleError('Too many atoms for dump xtc');
    if (f.step > MAXSMALLINT) throw new StyleError('Too big a timestep for dump xtc');
    // all procs realloc coords if total count grew
    if (n !== this.natoms) this.natoms = n;
    out.int32(XTC_MAGIC, BE);
    out.int32(n, BE);
    out.int32(f.step, BE);
    out.float32(f.step * this.tfactor * f.dt, BE);
    // cell basis vectors
    const b = f.box;
    const sf = this.sfactor;
    const xdim = sf * (b.hi[0] - b.lo[0]), ydim = sf * (b.hi[1] - b.lo[1]), zdim = sf * (b.hi[2] - b.lo[2]);
    const [xy, xz, yz] = b.triclinic ? [sf * b.xy, sf * b.xz, sf * b.yz] : [0.0, 0.0, 0.0];
    for (const v of [xdim, 0.0, 0.0, xy, ydim, 0.0, xz, yz, zdim]) out.float32(v, BE);
  }
}

export const createDumpXtc: DumpTrajFactory = (host, args) => new DumpXtc(host, args);

// ----------------------------------------------------------------------
// xdr3dfcoord: compressed 3d coordinates, (c) 1995 Frans van Hoesel, hoesel@chem.rug.nl
// ----------------------------------------------------------------------

/** MAXABS = (float)(INT_MAX-2). */
const MAXABS = Math.fround(INT_MAX - 2);
const MAGICINTS = Int32Array.of(
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
  80, 101, 128, 161, 203, 256, 322, 406, 512, 645,
  812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501,
  8192, 10321, 13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536,
  82570, 104031, 131072, 165140, 208063, 262144, 330280, 416127,
  524287, 660561,
  832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
  4194304, 5284491, 6658042,
  8388607, 10568983, 13316085, 16777216);
const FIRSTIDX = 9;
/** note that magicints[FIRSTIDX-1] == 0 */
const LASTIDX = MAGICINTS.length - 1;

/** The bit buffer of xdr3dfcoord: buf[0] = cnt, buf[1] = lastbits, buf[2] = lastbyte, then the bytes (cbuf). */
interface BitBuf { cnt: number; lastbits: number; lastbyte: number; cbuf: Uint8Array }

/** The (int) conversion of a float the native build makes (cvttss2si: INT_MIN when out of range). */
const toInt = (f: number): number => (f >= 2147483648 || f < -2147483648 || Number.isNaN(f) ? INT_MIN : Math.trunc(f));

/** sendbits: append num to the bits already in buf, using num_of_bits bits (num must be positive). */
const sendbits = (buf: BitBuf, numOfBits: number, num: number): void => {
  const cbuf = buf.cbuf;
  let cnt = buf.cnt;
  let lastbits = buf.lastbits;
  let lastbyte = buf.lastbyte;
  num |= 0;
  while (numOfBits >= 8) {
    lastbyte = ((lastbyte << 8) | (num >> (numOfBits - 8))) >>> 0;
    cbuf[cnt++] = lastbyte >>> lastbits;
    numOfBits -= 8;
  }
  if (numOfBits > 0) {
    lastbyte = ((lastbyte << numOfBits) | num) >>> 0;
    lastbits += numOfBits;
    if (lastbits >= 8) {
      lastbits -= 8;
      cbuf[cnt++] = lastbyte >>> lastbits;
    }
  }
  buf.cnt = cnt;
  buf.lastbits = lastbits;
  buf.lastbyte = lastbyte;
  if (lastbits > 0) cbuf[cnt] = lastbyte << (8 - lastbits);
};

/** sizeofint: the number of bits needed to store an integer with given max size. */
const sizeofint = (size: number): number => {
  let num = 1;
  let numOfBits = 0;
  size |= 0;
  while (size >= (num | 0) && numOfBits < 32) {
    numOfBits++;
    num = (num << 1) >>> 0;
  }
  return numOfBits;
};

/** sizeofints: the 'bitsize' of compressed ints, given the number of small unsigned integers and their maximum values. */
const sizeofints = (numOfInts: number, sizes: Uint32Array): number => {
  const bytes = new Uint32Array(32);
  let numOfBytes = 1;
  bytes[0] = 1;
  let numOfBits = 0;
  for (let i = 0; i < numOfInts; i++) {
    let tmp = 0;
    let bytecnt;
    for (bytecnt = 0; bytecnt < numOfBytes; bytecnt++) {
      tmp = (bytes[bytecnt] * sizes[i] + tmp) >>> 0;
      bytes[bytecnt] = tmp & 0xff;
      tmp >>>= 8;
    }
    while (tmp !== 0) {
      bytes[bytecnt++] = tmp & 0xff;
      tmp >>>= 8;
    }
    numOfBytes = bytecnt;
  }
  let num = 1;
  numOfBytes--;
  while ((bytes[numOfBytes] | 0) >= num) {
    numOfBits++;
    num *= 2;
  }
  return numOfBits + numOfBytes * 8;
};

/** sendints: send a small set of small integers, compressed, to the buffer. */
const sendints = (buf: BitBuf, numOfInts: number, numOfBits: number, sizes: Uint32Array, nums: Uint32Array, off: number): void => {
  const bytes = new Uint32Array(32);
  let tmp = nums[off];
  let numOfBytes = 0;
  do {
    bytes[numOfBytes++] = tmp & 0xff;
    tmp >>>= 8;
  } while (tmp !== 0);
  for (let i = 1; i < numOfInts; i++) {
    if (nums[off + i] >= sizes[i]) throw new StyleError(`major breakdown in sendints num ${nums[off + i]} doesn't match size ${sizes[i]}`);
    // use one step multiply
    tmp = nums[off + i];
    let bytecnt;
    for (bytecnt = 0; bytecnt < numOfBytes; bytecnt++) {
      tmp = (bytes[bytecnt] * sizes[i] + tmp) >>> 0;
      bytes[bytecnt] = tmp & 0xff;
      tmp >>>= 8;
    }
    while (tmp !== 0) {
      bytes[bytecnt++] = tmp & 0xff;
      tmp >>>= 8;
    }
    numOfBytes = bytecnt;
  }
  if (numOfBits >= numOfBytes * 8) {
    for (let i = 0; i < numOfBytes; i++) sendbits(buf, 8, bytes[i]);
    sendbits(buf, numOfBits - numOfBytes * 8, 0);
  } else {
    let i;
    for (i = 0; i < numOfBytes - 1; i++) sendbits(buf, 8, bytes[i]);
    sendbits(buf, numOfBits - (numOfBytes - 1) * 8, bytes[i]);
  }
};

/** The nearest integer of a coordinate times the precision (float arithmetic, then (int) truncation). */
const scaled = (x: number, precision: number): { lint: number; over: boolean } => {
  const prod = Math.fround(x * precision);
  const lf = Math.fround(x >= 0.0 ? prod + 0.5 : prod - 0.5);
  return { lint: toInt(lf), over: Math.abs(lf) > MAXABS };
};

/**
 * xdr3dfcoord, open for writing: the coordinates (fp, *size triplets) as integers of 1/precision, the minimum and maximum
 * of each, and the differences between successive atoms packed in the fewest bits ("small" differences are run-length
 * coded; the first two atoms of a close pair are swapped, for water molecules). Fewer than 10 atoms are written as plain
 * floats.
 */
export const xdr3dfcoord = (out: ByteBuffer, fp: Float32Array, size: number, precision: number): boolean => {
  out.int32(size, BE);
  const size3 = size * 3;
  // when the number of coordinates is small, don't try to compress; just write them as floats using xdr_vector
  if (size <= 9) {
    for (let k = 0; k < size3; k++) out.float32(fp[k], BE);
    return true;
  }
  out.float32(precision, BE);
  let errval = true;
  const ip = new Int32Array(size3);
  // buf[0-2] are special and do not contain actual data
  const buf: BitBuf = { cnt: 0, lastbits: 0, lastbyte: 0, cbuf: new Uint8Array(Math.ceil(size3 * 4.8) + 64) };
  const minint = [INT_MAX, INT_MAX, INT_MAX];
  const maxint = [INT_MIN, INT_MIN, INT_MIN];
  let prevrun = -1;
  let mindiff = INT_MAX;
  let oldlint1 = 0, oldlint2 = 0, oldlint3 = 0;
  for (let a = 0; a < size; a++) {
    const lints = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      // find nearest integer
      const { lint, over } = scaled(fp[3 * a + d], precision);
      if (over) errval = false; // scaling would cause overflow
      if (lint < minint[d]) minint[d] = lint;
      if (lint > maxint[d]) maxint[d] = lint;
      ip[3 * a + d] = lint;
      lints[d] = lint;
    }
    const diff = (Math.abs((oldlint1 - lints[0]) | 0) + Math.abs((oldlint2 - lints[1]) | 0) + Math.abs((oldlint3 - lints[2]) | 0)) | 0;
    if (diff < mindiff && a > 0) mindiff = diff;
    oldlint1 = lints[0];
    oldlint2 = lints[1];
    oldlint3 = lints[2];
  }
  for (let d = 0; d < 3; d++) out.int32(minint[d], BE);
  for (let d = 0; d < 3; d++) out.int32(maxint[d], BE);
  for (let d = 0; d < 3; d++) {
    // turning value in unsigned by subtracting minint would cause overflow
    if (Math.fround(maxint[d]) - Math.fround(minint[d]) >= MAXABS) errval = false;
  }
  const sizeint = Uint32Array.of(maxint[0] - minint[0] + 1, maxint[1] - minint[1] + 1, maxint[2] - minint[2] + 1);
  const bitsizeint = new Uint32Array(3);
  let bitsize: number;
  // check if one of the sizes is to big to be multiplied
  if (((sizeint[0] | sizeint[1] | sizeint[2]) >>> 0) > 0xffffff) {
    bitsizeint[0] = sizeofint(sizeint[0]);
    bitsizeint[1] = sizeofint(sizeint[1]);
    bitsizeint[2] = sizeofint(sizeint[2]);
    bitsize = 0; // flag the use of large sizes
  } else {
    bitsize = sizeofints(3, sizeint);
  }
  let smallidx = FIRSTIDX;
  while (smallidx < LASTIDX && MAGICINTS[smallidx] < mindiff) smallidx++;
  out.int32(smallidx, BE);
  const maxidx = Math.min(LASTIDX, smallidx + 8);
  const minidx = maxidx - 8; // often this equal smallidx
  let smaller = (MAGICINTS[Math.max(FIRSTIDX, smallidx - 1)] / 2) | 0;
  let small = (MAGICINTS[smallidx] / 2) | 0;
  const sizesmall = Uint32Array.of(MAGICINTS[smallidx], MAGICINTS[smallidx], MAGICINTS[smallidx]);
  const larger = (MAGICINTS[maxidx] / 2) | 0;
  const tmpcoord = new Uint32Array(30);
  const prevcoord = [0, 0, 0];
  const sq = (v: number) => Math.imul(v, v);
  let i = 0;
  while (i < size) {
    let isSmall = false;
    let t = i * 3; // thiscoord
    let isSmaller: number;
    if (smallidx < maxidx && i >= 1
      && Math.abs((ip[t] - prevcoord[0]) | 0) < larger
      && Math.abs((ip[t + 1] - prevcoord[1]) | 0) < larger
      && Math.abs((ip[t + 2] - prevcoord[2]) | 0) < larger) {
      isSmaller = 1;
    } else if (smallidx > minidx) {
      isSmaller = -1;
    } else {
      isSmaller = 0;
    }
    if (i + 1 < size) {
      if (Math.abs((ip[t] - ip[t + 3]) | 0) < small && Math.abs((ip[t + 1] - ip[t + 4]) | 0) < small
        && Math.abs((ip[t + 2] - ip[t + 5]) | 0) < small) {
        // interchange first with second atom for better compression of water molecules
        for (let d = 0; d < 3; d++) {
          const tmp = ip[t + d];
          ip[t + d] = ip[t + 3 + d];
          ip[t + 3 + d] = tmp;
        }
        isSmall = true;
      }
    }
    tmpcoord[0] = ip[t] - minint[0];
    tmpcoord[1] = ip[t + 1] - minint[1];
    tmpcoord[2] = ip[t + 2] - minint[2];
    if (bitsize === 0) {
      sendbits(buf, bitsizeint[0], tmpcoord[0]);
      sendbits(buf, bitsizeint[1], tmpcoord[1]);
      sendbits(buf, bitsizeint[2], tmpcoord[2]);
    } else {
      sendints(buf, 3, bitsize, sizeint, tmpcoord, 0);
    }
    prevcoord[0] = ip[t];
    prevcoord[1] = ip[t + 1];
    prevcoord[2] = ip[t + 2];
    t += 3;
    i++;

    let run = 0;
    if (!isSmall && isSmaller === -1) isSmaller = 0;
    while (isSmall && run < 8 * 3) {
      if (isSmaller === -1 && ((sq((ip[t] - prevcoord[0]) | 0) + sq((ip[t + 1] - prevcoord[1]) | 0)
        + sq((ip[t + 2] - prevcoord[2]) | 0)) | 0) >= Math.imul(smaller, smaller)) {
        isSmaller = 0;
      }
      tmpcoord[run++] = ip[t] - prevcoord[0] + small;
      tmpcoord[run++] = ip[t + 1] - prevcoord[1] + small;
      tmpcoord[run++] = ip[t + 2] - prevcoord[2] + small;

      prevcoord[0] = ip[t];
      prevcoord[1] = ip[t + 1];
      prevcoord[2] = ip[t + 2];

      i++;
      t += 3;
      isSmall = false;
      if (i < size && Math.abs((ip[t] - prevcoord[0]) | 0) < small && Math.abs((ip[t + 1] - prevcoord[1]) | 0) < small
        && Math.abs((ip[t + 2] - prevcoord[2]) | 0) < small) {
        isSmall = true;
      }
    }
    if (run !== prevrun || isSmaller !== 0) {
      prevrun = run;
      sendbits(buf, 1, 1); // flag the change in run-length
      sendbits(buf, 5, run + isSmaller + 1);
    } else {
      sendbits(buf, 1, 0); // flag the fact that runlength did not change
    }
    for (let k = 0; k < run; k += 3) sendints(buf, 3, smallidx, sizesmall, tmpcoord, k);
    if (isSmaller !== 0) {
      smallidx += isSmaller;
      if (isSmaller < 0) {
        small = smaller;
        smaller = (MAGICINTS[smallidx - 1] / 2) | 0;
      } else {
        smaller = small;
        small = (MAGICINTS[smallidx] / 2) | 0;
      }
      sizesmall[0] = sizesmall[1] = sizesmall[2] = MAGICINTS[smallidx];
    }
  }
  if (buf.lastbits !== 0) buf.cnt++;
  out.int32(buf.cnt, BE); // buf[0] holds the length in bytes
  out.bytes(buf.cbuf.subarray(0, buf.cnt), true);
  return errval;
};

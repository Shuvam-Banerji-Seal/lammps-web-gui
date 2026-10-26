/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-DUMP/dump_dcd.cpp and
 * dump_dcd.h (class DumpDCD, dump style dcd; contributing author Naveen Michaud-Agrawal (Johns Hopkins U), open-source
 * XDR routines from Frans van Hoesel; copyright Sandia Corporation). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump.html (verbatim): "The dcd style writes DCD files, a standard atomic trajectory format used by the
 * CHARMM, NAMD, and XPLOR molecular dynamics packages. DCD files are binary and thus may not be portable to different
 * machines. The number of atoms per snapshot cannot change with the dcd style. The unit cell dimension information is
 * also written to the file."
 *
 * The core dump (src/engine/output/dump.ts) keeps the schedule and the dump_modify keywords every style takes, and gives
 * the group's atoms in ID order (Dump::write with sort by ID, then pack); this helper does the constructor, modify_param,
 * init_style, write_header, write_data and write_frame of DumpDCD. The file is little-endian, as native LAMMPS writes it
 * on x86.
 */
import { StyleError } from '../../engine/force/types';
import { logical, missingArgs } from '../../engine/force/util';
import type { DumpTrajFactory, DumpTrajFrame, DumpTrajHost, DumpTrajSettings, DumpTrajStyle } from '../../engine/output/dump';
import { ByteBuffer } from './bytes';

/** dump_dcd.cpp: #define NFILE_POS 8L, NSTEP_POS 20L (the header fields write_frame updates). */
const NFILE_POS = 8;
const NSTEP_POS = 20;
/** lmptype.h: MAXSMALLINT = 0x7FFFFFFF. */
const MAXSMALLINT = 0x7fffffff;
/** The byte order of the native binary's fwrite (x86). */
const LE = true;
/** strftime %B in the C locale. */
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

/** The extensions of platform::has_compress_extension (src/platform.cpp compress_styles at b332f28). */
const COMPRESS_EXT = /\.(gz|bz2|zst|xz|lzma|lz4|br|7z)$/;

/**
 * The Dump base class's file-name flags that dump dcd and xtc refuse (src/dump.cpp Dump::Dump at b332f28): "%" (one file
 * per processor), "*" (one file per snapshot), a .bin or .lammpsbin name (binary) and a compression extension.
 */
export const invalidTrajName = (file: string): boolean =>
  file.includes('%') || file.includes('*') || /\.bin$/.test(file) || /\.lammpsbin$/.test(file) || COMPRESS_EXT.test(file);

/** strftime "%d %B,%Y at %H:%M" of the local time (dump_dcd.cpp write_dcd_header). */
const createdStamp = (d: Date): string => {
  const two = (v: number) => String(v).padStart(2, '0');
  return `REMARKS Created ${two(d.getDate())} ${MONTHS[d.getMonth()]},${d.getFullYear()} at ${two(d.getHours())}:${two(d.getMinutes())}`;
};

/** An 80-byte DCD title record: the text padded with blanks and cut at 80 characters. */
const title80 = (text: string): string => (text + ' '.repeat(80)).slice(0, 80);

class DumpDcd implements DumpTrajStyle {
  unwrap = false;
  private readonly natoms: number;
  private headerflag = false;
  private neverySave = 0;
  // dump_dcd.h leaves nframes unset until the first header; dump_modify header no writes frames without one
  private nframes = 0;

  constructor(private readonly host: DumpTrajHost, args: string[]) {
    if (args.length) throw new StyleError('Illegal dump dcd command');
    if (invalidTrajName(host.file)) throw new StyleError(`Invalid dump dcd filename ${host.file}`);
    const n = host.groupCount();
    if (n > Math.floor(MAXSMALLINT / 3 / 4)) throw new StyleError('Too many atoms for dump dcd');
    this.natoms = n;
    // openfile: fopen(filename, "wb") at the dump command
    host.create();
  }

  modify(args: string[], k: number): number {
    if (args[k] !== 'unwrap') return 0;
    if (k + 2 > args.length) throw missingArgs('dump_modify unwrap');
    this.unwrap = logical(args[k + 1]);
    return 2;
  }

  init(st: DumpTrajSettings): void {
    if (!st.sortById) throw new StyleError('Dump dcd requires sorting by atom ID');
    // check that dump modify settings are compatible with dcd, but only when not being called from write_dump
    if (st.writeDump) return;
    if (st.everyTime) throw new StyleError('Cannot use every/time setting for dump dcd');
    if (st.every === 0) throw new StyleError('Cannot use variable every setting for dump dcd');
    if (this.neverySave === 0) this.neverySave = st.every;
    else if (this.neverySave !== st.every) throw new StyleError('Cannot change dump_modify every for dump dcd');
  }

  write(f: DumpTrajFrame): void {
    const out = new ByteBuffer();
    if (f.header) this.writeHeader(out, f);
    // write_data copies the atoms into the global arrays and writes the frame once all natoms are there
    if (f.natoms !== this.natoms) throw new StyleError(`dump dcd: a snapshot of ${f.natoms} atoms with dump_modify header no in a file of ${this.natoms} atoms is not supported by the browser engine`);
    const n = this.natoms;
    const rec = 4 * n;
    for (let d = 0; d < 3; d++) {
      out.int32(rec, LE);
      for (let i = 0; i < n; i++) out.float32(f.coords[3 * i + d], LE);
      out.int32(rec, LE);
    }
    this.host.writeBytes(out.latin1());
    // update NFILE and NSTEP fields in DCD header
    this.nframes++;
    const patch = (v: number) => { const b = new ByteBuffer(); b.int32(v, LE); return b.latin1(); };
    this.host.writeBytes(patch(this.nframes), NFILE_POS);
    this.host.writeBytes(patch(f.step), NSTEP_POS);
  }

  private writeHeader(out: ByteBuffer, f: DumpTrajFrame): void {
    if (f.natoms !== this.natoms) throw new StyleError('Dump dcd of non-matching # of atoms');
    if (f.step > MAXSMALLINT) throw new StyleError('Too big a timestep for dump dcd');
    // first time, write header for entire file
    if (!this.headerflag) {
      this.writeDcdHeader(out, 'Written by LAMMPS', f.step, f.dt);
      this.headerflag = true;
      this.nframes = 0;
    }
    // dim[] = size and angle cosines of orthogonal or triclinic box: a, cos(gamma), b, cos(beta), cos(alpha), c
    const b = f.box;
    const dim = [0, 0, 0, 0, 0, 0];
    if (b.triclinic) {
      // domain->h = xprd, yprd, zprd, yz, xz, xy
      const h = [b.xprd, b.yprd, b.zprd, b.yz, b.xz, b.xy];
      const alen = h[0];
      const blen = Math.sqrt(h[5] * h[5] + h[1] * h[1]);
      const clen = Math.sqrt(h[4] * h[4] + h[3] * h[3] + h[2] * h[2]);
      dim[0] = alen;
      dim[2] = blen;
      dim[5] = clen;
      dim[4] = (h[5] * h[4] + h[1] * h[3]) / blen / clen; // alpha
      dim[3] = (h[0] * h[4]) / alen / clen; // beta
      dim[1] = (h[0] * h[5]) / alen / blen; // gamma
    } else {
      dim[0] = b.xprd;
      dim[2] = b.yprd;
      dim[5] = b.zprd;
    }
    out.int32(48, LE);
    for (const v of dim) out.float64(v, LE);
    out.int32(48, LE);
  }

  private writeDcdHeader(out: ByteBuffer, remarks: string, ntimestep: number, dt: number): void {
    out.int32(84, LE);
    for (const c of 'CORD') out.bytes(Uint8Array.of(c.charCodeAt(0)));
    out.int32(0, LE); // NFILE = # of snapshots in file
    out.int32(ntimestep, LE); // START = timestep of first snapshot
    out.int32(this.neverySave, LE); // SKIP = interval between snapshots
    out.int32(ntimestep, LE); // NSTEP = timestep of last snapshot
    for (let k = 0; k < 5; k++) out.int32(0, LE); // NAMD writes NSTEP or ISTART
    out.float32(dt, LE);
    out.int32(1, LE);
    for (let k = 0; k < 8; k++) out.int32(0, LE);
    out.int32(24, LE); // pretend to be Charmm version 24
    out.int32(84, LE);
    out.int32(164, LE);
    out.int32(2, LE);
    const text = (s: string) => { for (let i = 0; i < s.length; i++) out.bytes(Uint8Array.of(s.charCodeAt(i) & 0xff)); };
    // pad string with blanks and truncate at 80 chars
    text(title80(remarks));
    text(title80(createdStamp(new Date())));
    out.int32(164, LE);
    out.int32(4, LE);
    out.int32(this.natoms, LE); // number of atoms in each snapshot
    out.int32(4, LE);
  }
}

export const createDumpDcd: DumpTrajFactory = (host, args) => new DumpDcd(host, args);

/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/H5MD/dump_h5md.cpp and
 * dump_h5md.h (class DumpH5MD, dump style h5md; contributing author Pierre de Buyl (KU Leuven)), with the calls of the
 * ch5md library bundled with LAMMPS (lib/h5md/src/ch5md.c, Copyright (C) 2013-2014 Pierre de Buyl, BSD three-clause
 * licence, kept in LICENSE.ch5md here). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump_h5md.html (verbatim): "Dump a snapshot of atom coordinates every N timesteps in the HDF5 based
 * H5MD file format"; "Each data element is written every N*N_element steps. For image, no sub-interval is needed as it
 * must be present at the same interval as position. image must be given after position in any case. The box
 * information (edges in each dimension) is stored at the same interval than the position element, if present."
 * "Only orthogonal domains are currently supported."
 *
 * The core dump (src/engine/output/dump.ts) keeps the schedule and the dump_modify keywords every style takes, and gives
 * the group's atoms in ID order; this helper does the constructor, modify_param, init_style, openfile, pack and
 * write_frame of DumpH5MD, and writes the HDF5 file with ./hdf5.ts. As native does, the file is created at the dump
 * command; the elements are time series, extended by one record per output (native's write_fixed_frame is reached only
 * with a dump interval of 0, which Output::add_dump refuses: "Invalid dump frequency 0").
 */
import { hasCharge } from '../../engine/atoms';
import { StyleError } from '../../engine/force/types';
import { inumeric, logical } from '../../engine/force/util';
import type { DumpTrajFactory, DumpTrajFrame, DumpTrajHost, DumpTrajSettings, DumpTrajStyle } from '../../engine/output/dump';
import { latin1 } from '../dump/bytes';
import { invalidTrajName } from '../dump/dump_dcd';
import { H5File, f64Bytes, i32Bytes, intAttr, stringArrayAttr, stringAttr, type H5Dataset, type H5Group, type H5Type } from './hdf5';

/** ch5md.c: #define MAX_CHUNK 256 (the largest chunk along the atoms). */
const MAX_CHUNK = 256;
/** lmptype.h: MAXSMALLINT = 0x7FFFFFFF. */
const MAXSMALLINT = 0x7fffffff;
/** LAMMPS_VERSION of the native binary (version.h), the creator version of the files it writes. */
const LAMMPS_VERSION = '2 Sep 2026';

/** h5md_element of a time-dependent element: its group, its step and time (its own or linked) and its values. */
interface Element { group: H5Group; step: H5Dataset; time: H5Dataset; value: H5Dataset; type: H5Type }

/** The file and the particles group of a dump (shared with the dumps that name it with file_from). */
interface H5mdData { file: H5File; h5md: H5Group; particles: H5Group }

const I32: H5Type = { kind: 'i32' };
const F64: H5Type = { kind: 'f64' };

/**
 * h5md_create_time_data: the group `name` with a step and a time dataset (or hard links to those of `link`) and a value
 * dataset of `dims` per record, in chunks of one record by at most MAX_CHUNK along the first of `dims`.
 */
const createTimeData = (file: H5File, loc: H5Group, name: string, dims: number[], type: H5Type, link: Element | null, attrs: ReturnType<typeof stringAttr>[] = []): Element => {
  const chunk = [1, MAX_CHUNK < Math.trunc(dims[0] / 4) ? MAX_CHUNK : dims[0], ...dims.slice(1)];
  const group = file.createGroup(loc, name, attrs);
  let step: H5Dataset, time: H5Dataset;
  if (!link) {
    time = file.createChunked(group, 'time', F64, [0], [1]);
    step = file.createChunked(group, 'step', I32, [0], [1]);
  } else {
    file.hardLink(group, 'step', link.step);
    file.hardLink(group, 'time', link.time);
    step = link.step;
    time = link.time;
  }
  const value = file.createChunked(group, 'value', type, [0, ...dims], chunk);
  return { group, step, time, value, type };
};

/** h5md_append: one record of step, time (unless linked) and the values. */
const append = (file: H5File, e: Element, values: Uint8Array, step: number, time: number, linked: boolean): void => {
  if (!linked) {
    file.appendRecord(e.step, i32Bytes([step]));
    file.appendRecord(e.time, f64Bytes([time]));
  }
  file.appendRecord(e.value, values);
};

class DumpH5md implements DumpTrajStyle {
  unwrap = false;
  readonly wantsImages = true;
  /** the file and particles group, for a later dump's file_from */
  data!: H5mdData;
  /** this dump's particles group (its own, or the other dump's with create_group no) */
  private group!: H5Group;
  private readonly natoms: number;
  private readonly dim: number;
  private everyDump: number;
  private everyPosition = -1;
  private everyImage = -1;
  private everyVelocity = -1;
  private everyForce = -1;
  private everySpecies = -1;
  private position: Element | null = null;
  private boxEdges: Element | null = null;
  private image: Element | null = null;
  private velocity: Element | null = null;
  private force: Element | null = null;
  private species: Element | null = null;

  get needsForces(): boolean { return this.everyForce > 0; }

  constructor(private readonly host: DumpTrajHost, args: string[]) {
    const sys = host.sys;
    // DumpH5MD::DumpH5MD: "if (narg<6)" (the dump command's words: ID, group, style, N, file and at least one element)
    if (args.length < 1) throw new StyleError('Illegal dump h5md command');
    if (invalidTrajName(host.file)) throw new StyleError('Invalid dump h5md filename');
    if (sys.state.box.triclinic) throw new StyleError('Invalid domain for dump h5md. Only triclinic domains supported.');
    this.dim = sys.dimension;
    this.everyDump = host.nevery;
    const defaultEvery = this.everyDump === 0 ? 0 : 1;
    let other: DumpH5md | null = null;
    let doBox = true, createGroup = true, boxIsSet = false, createGroupIsSet = false;
    let author: string | null = null;
    // element_args: "every N_element" after an element (native checks the count of the remaining words, narg < 2)
    const elementArgs = (k: number, set: (v: number) => void): number => {
      const rem = args.length - k;
      let j = k;
      while (j < args.length && args[j] === 'every') {
        if (rem < 2 || j + 1 >= args.length) throw new StyleError('Illegal dump h5md command');
        set(inumeric(args[j + 1]));
        j += 2;
      }
      return j;
    };
    let k = 0;
    while (k < args.length) {
      const w = args[k];
      if (w === 'position') {
        this.everyPosition = defaultEvery;
        k = elementArgs(k + 1, (v) => { this.everyPosition = v; });
      } else if (w === 'image') {
        if (this.everyPosition < 0) throw new StyleError('Illegal dump h5md command');
        k += 1;
        this.everyImage = this.everyPosition;
      } else if (w === 'velocity') {
        this.everyVelocity = defaultEvery;
        k = elementArgs(k + 1, (v) => { this.everyVelocity = v; });
      } else if (w === 'force') {
        this.everyForce = defaultEvery;
        k = elementArgs(k + 1, (v) => { this.everyForce = v; });
      } else if (w === 'species') {
        this.everySpecies = defaultEvery;
        k = elementArgs(k + 1, (v) => { this.everySpecies = v; });
      } else if (w === 'charge') {
        if (!hasCharge(sys.state)) throw new StyleError('Requesting non-allocated quantity q in dump_h5md');
        // native keeps the charges in an int array (dump_h5md.h: int *dump_charge) and writes it as H5T_NATIVE_DOUBLE
        // records (h5md_append with the element's datatype), reading twice the bytes the array holds: the values are
        // undefined, so the browser engine does not write this element
        throw new StyleError('dump h5md charge is not supported by the browser engine: native LAMMPS stores the charges in an integer array and writes it as doubles (src/H5MD/dump_h5md.cpp), so the values it writes are undefined');
      } else if (w === 'file_from') {
        if (k + 1 >= args.length) throw new StyleError('Invalid number of arguments in dump h5md');
        if (boxIsSet || createGroupIsSet) throw new StyleError('Cannot set file_from in dump h5md after box or create_group');
        const o = host.otherTraj(args[k + 1]);
        if (!(o instanceof DumpH5md)) throw new StyleError(`Cound not find dump_modify H5MD dump ID ${args[k + 1]}`);
        other = o;
        doBox = false;
        createGroup = false;
        k += 2;
      } else if (w === 'box') {
        if (k + 1 >= args.length) throw new StyleError('Invalid number of arguments in dump h5md');
        boxIsSet = true;
        doBox = logical(args[k + 1]);
        k += 2;
      } else if (w === 'create_group') {
        if (k + 1 >= args.length) throw new StyleError('Invalid number of arguments in dump h5md');
        createGroupIsSet = true;
        createGroup = logical(args[k + 1]);
        k += 2;
      } else if (w === 'author') {
        if (k + 1 >= args.length) throw new StyleError('Invalid number of arguments in dump h5md');
        if (author !== null) throw new StyleError('Illegal dump h5md command: author argument repeated');
        author = args[k + 1];
        k += 2;
      } else {
        throw new StyleError('Invalid argument to dump h5md');
      }
    }
    // the box keyword only matters without a position element (box edges are written with the positions), where
    // native's box element is never created: doBox has no effect on the file
    void doBox;
    const n = host.groupCount();
    if (this.dim * n > MAXSMALLINT) throw new StyleError('Too many atoms for dump h5md');
    if (n === 0) throw new StyleError('dump h5md: the dump group has no atoms (HDF5 needs chunks of at least one atom)');
    this.natoms = n;
    this.openfile(other, createGroup, author);
  }

  /** DumpH5MD::openfile: the file (or the other dump's) and the time-data elements. */
  private openfile(other: DumpH5md | null, createGroup: boolean, author: string | null): void {
    const host = this.host;
    const groupName = host.groupName;
    if (!other) {
      host.create();
      const file = new H5File((bytes, at) => host.writeBytes(latin1(bytes), at));
      // h5md_create_file: h5md (version 1 0, author, creator), particles, observables, parameters
      const h5md = file.createGroup(file.root, 'h5md', [intAttr('version', [1, 0])]);
      file.createGroup(h5md, 'author', [stringAttr('name', author ?? 'N/A')]);
      file.createGroup(h5md, 'creator', [stringAttr('name', 'lammps'), stringAttr('version', LAMMPS_VERSION)]);
      const particles = file.createGroup(file.root, 'particles');
      file.createGroup(file.root, 'observables');
      file.createGroup(file.root, 'parameters');
      this.data = { file, h5md, particles };
      if (file.has(particles, groupName)) throw new StyleError(`dump h5md: the file already has the particles group ${groupName}`);
      this.group = createGroup ? file.createGroup(particles, groupName) : this.openGroup(particles, groupName);
    } else {
      this.data = other.data;
      const { file, particles } = this.data;
      if (createGroup) {
        if (file.has(particles, groupName)) throw new StyleError(`dump h5md: the file already has the particles group ${groupName}`);
        this.group = file.createGroup(particles, groupName);
      } else {
        this.group = other.group;
      }
    }
    const file = this.data.file;
    const g = this.group;
    const dims = [this.natoms, this.dim];
    const fresh = (name: string): void => {
      if (file.has(g, name)) throw new StyleError(`dump h5md: the particles group already has the element ${name}`);
    };
    if (this.everyPosition > 0) {
      fresh('position'); fresh('box');
      this.position = createTimeData(file, g, 'position', dims, F64, null);
      // h5md_create_box: dimension, boundary (each name in an element of the longest name's length plus one), edges
      const periodic = this.host.sys.state.box.periodic;
      const boundary = [0, 1, 2].slice(0, this.dim).map((d) => (periodic[d] ? 'periodic' : 'none'));
      const len = boundary.reduce((m, b) => Math.max(m, b.length + 1), 0);
      const box = file.createGroup(g, 'box', [intAttr('dimension', [this.dim], true), stringArrayAttr('boundary', boundary, len)]);
      this.boxEdges = createTimeData(file, box, 'edges', [this.dim], F64, this.position);
    }
    if (this.everyImage > 0) { fresh('image'); this.image = createTimeData(file, g, 'image', dims, I32, this.position); }
    if (this.everyVelocity > 0) { fresh('velocity'); this.velocity = createTimeData(file, g, 'velocity', dims, F64, null); }
    if (this.everyForce > 0) { fresh('force'); this.force = createTimeData(file, g, 'force', dims, F64, null); }
    if (this.everySpecies > 0) { fresh('species'); this.species = createTimeData(file, g, 'species', [this.natoms], I32, null); }
    file.flush();
  }

  /** h5md_open_particles_group: create_group no without file_from names a group this new file does not have. */
  private openGroup(particles: H5Group, name: string): H5Group {
    const g = this.data.file.findGroup(particles, name);
    if (!g) throw new StyleError(`dump h5md: create_group no, but the file has no particles group ${name}`);
    return g;
  }

  modify(args: string[], k: number): number {
    if (args[k] !== 'unwrap') return 0;
    if (k + 2 > args.length) throw new StyleError('Illegal dump_modify command');
    this.unwrap = logical(args[k + 1]);
    return 2;
  }

  init(st: DumpTrajSettings): void {
    if (!st.sortById) throw new StyleError('Dump h5md requires sorting by atom ID');
  }

  /** DumpH5MD::pack and write_frame. */
  write(frame: DumpTrajFrame): void {
    const s = this.host.sys.state;
    const rows = frame.rows;
    const n = rows.length;
    const dim = this.dim;
    const step = frame.step;
    // local_step is an int, local_time = local_step * update->dt
    const time = step * frame.dt;
    const file = this.data.file;
    const due = (every: number): boolean => every > 0 && step % (every * this.everyDump) === 0;
    const vec = (src: (k: number, d: number) => number): Float64Array => {
      const out = new Float64Array(n * dim);
      for (let k = 0; k < n; k++) for (let d = 0; d < dim; d++) out[dim * k + d] = src(k, d);
      return out;
    };
    if (this.position && due(this.everyPosition)) {
      append(file, this.position, f64Bytes(vec((k, d) => frame.coords[3 * k + d])), step, time, false);
      // the edges are boxhi - boxlo of each dimension the element holds
      const edges = [0, 1, 2].slice(0, dim).map((d) => frame.box.hi[d] - frame.box.lo[d]);
      append(file, this.boxEdges!, f64Bytes(edges), step, time, true);
      if (this.image) {
        const im = new Int32Array(n * dim);
        for (let k = 0; k < n; k++) for (let d = 0; d < dim; d++) im[dim * k + d] = frame.images![3 * k + d];
        append(file, this.image, i32Bytes(im), step, time, true);
      }
    }
    if (this.velocity && due(this.everyVelocity)) append(file, this.velocity, f64Bytes(vec((k, d) => s.v[3 * rows[k] + d])), step, time, false);
    if (this.force && due(this.everyForce)) append(file, this.force, f64Bytes(vec((k, d) => s.f[3 * rows[k] + d])), step, time, false);
    if (this.species && due(this.everySpecies)) {
      const t = new Int32Array(n);
      for (let k = 0; k < n; k++) t[k] = s.type[rows[k]];
      append(file, this.species, i32Bytes(t), step, time, false);
    }
    file.flush();
  }
}

export const createDumpH5md: DumpTrajFactory = (host, args) => new DumpH5md(host, args);

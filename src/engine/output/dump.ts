import { isSmdStyle } from '../smd';
import type { System } from '../system';
import { StyleError } from '../force/types';
import { compileCFormat, checkCFormat, formatNumber, type ValueKind } from '../cformat';
import { localDumpColumns, localColumnSource, localOutputOf } from '../compute/local_dump';
import { generalBoxFromRestricted, toGeneralPoint, unrotateVector, type V3 } from '../triclinic_general';
import { fillWildcard } from '../commands/restart';
import { hasChargeStyle, hasDipoleStyle, isMolecularStyle, isSphereStyle, hasRmassStyle, hasQuatStyle, isTriStyle, massOf, CUSTOM_ATTR, customAttr, hasCharge, hasMolecule, nativeOrder } from '../atoms';
import { ImageDump } from './dump_image';
import type { ColorMap } from './image_colormap';
import { gplHook } from '../gplHooks';
import { expandGridRefs } from '../grid';
import { gzipText } from '../gzip';

/**
 * The compression extension of a dump file name (the extensions of native platform::find_compress_type,
 * src/platform.cpp at b332f28: gz, bz2, zst, xz, lzma, lz4, br, 7z), or '' for a plain file.
 */
export const compressExtension = (file: string): string => {
  const m = /\.([^.]+)$/.exec(file);
  return m && ['gz', 'bz2', 'zst', 'xz', 'lzma', 'lz4', 'br', '7z'].includes(m[1]) ? m[1] : '';
};

/*
 * Per-atom snapshots — docs.lammps.org/dump.html and dump_modify.html.
 *
 * "N = dump on timesteps which are multiples of N"; "If a "*" character
 * appears in the filename, then one file per snapshot is written and the "*"
 * character is replaced with the timestep value." atom style writes
 * "id type xs ys zs" (scaled; "scale = yes"); custom writes the listed
 * attributes; xyz writes "element x y z" with "element = "C" for every atom
 * type" unless dump_modify element is used (native LAMMPS writes the numeric
 * type by default for xyz, as observed: line "1 0 0 0"); defaults "format =
 * %d and %g for each integer or floating point value", "sort = off for dump
 * styles atom, custom, cfg, and local", "sort = id for dump styles dcd, xtc,
 * and xyz", "append = no", "first = no", "pbc = no", "scale = yes",
 * "time = no", "units = no", "unwrap = no".
 * Box bounds (native LAMMPS output): "ITEM: BOX BOUNDS pp pp pp" then
 * "%-1.16e %-1.16e" per dimension; for a triclinic box "ITEM: BOX BOUNDS xy
 * xz yz pp pp pp" with xlo_bound = xlo + MIN(0.0,xy,xz,xy+xz) etc.
 * (Howto_triclinic.html). xyz comment line: " Atoms. Timestep: N".
 * "ITEM: TIME" lines with dump_modify time yes; "ITEM: UNITS" with units yes.
 */

const INT_COLS = new Set(['id', 'mol', 'proc', 'procp1', 'type', 'ix', 'iy', 'iz']);
/** The dump styles of native's DumpCustom family (custom, cfg, image, yaml and their compressed forms): dump_modify region and thresh. */
const THRESH_STYLES = new Set<string>(['custom', 'cfg', 'image', 'yaml']);
/** The per-atom attributes a dump_modify thresh can test (src/dump_custom.cpp at b332f28, modify_param "thresh"). */
const THRESH_ATTRS = new Set([
  'id', 'mol', 'proc', 'procp1', 'type', 'mass', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu', 'xsu', 'ysu', 'zsu',
  'ix', 'iy', 'iz', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'q', 'mux', 'muy', 'muz', 'mu', 'radius', 'diameter',
  'omegax', 'omegay', 'omegaz', 'angmomx', 'angmomy', 'angmomz', 'tqx', 'tqy', 'tqz',
]);
/** ... or a compute, fix, variable, or custom per-atom vector/array reference (ArgInfo COMPUTE FIX VARIABLE DNAME INAME). */
const THRESH_REF = /^(?:[cfv]|[di]2?)_[A-Za-z0-9_]+(?:\[\d+\])?$/;

const ATOM_COLS = new Set([
  'id', 'mol', 'proc', 'procp1', 'type', 'element', 'mass', 'x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu',
  'xsu', 'ysu', 'zsu', 'ix', 'iy', 'iz', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'q',
  // dump.html: "radius,diameter = radius, diameter of spherical particle", "omegax,omegay,omegaz =
  // angular velocity of spherical particle", "tqx,tqy,tqz = torque on finite-size particles"
  'radius', 'diameter', 'omegax', 'omegay', 'omegaz', 'tqx', 'tqy', 'tqz',
  // dump.html: "mux,muy,muz = orientation of dipole moment of atom", "mu = magnitude of dipole moment of atom"
  'mux', 'muy', 'muz', 'mu',
  // dump.html: "The *angmomx*, *angmomy*, and *angmomz* attributes are specific to finite-size aspherical
  // particles that have an angular momentum.  Only the *ellipsoid* atom style defines this quantity."
  'angmomx', 'angmomy', 'angmomz',
]);
const SPHERE_COLS = new Set(['radius', 'diameter', 'omegax', 'omegay', 'omegaz']);
// dump.html: "The *tqx*, *tqy*, and *tqz* attributes are for finite-size particles that can sustain a rotational
// torque due to interactions with other particles."
const TORQUE_COLS = new Set(['tqx', 'tqy', 'tqz']);
const ANGMOM_COLS = new Set(['angmomx', 'angmomy', 'angmomz']);
const DIPOLE_COLS = new Set(['mux', 'muy', 'muz', 'mu']);

/**
 * A compiled format for one value kind. dump_modify.html: "Format strings are checked when they are set and LAMMPS
 * will stop with an error if a format string does not match the kind of value it is applied to". The compile
 * cache keeps this cheap for large dumps.
 */
export const fmt = (f: string): ((v: number) => string) => compileCFormat(f, 'float', 'Invalid dump format');

export type DumpStyle = 'atom' | 'custom' | 'cfg' | 'xyz' | 'extxyz' | 'yaml' | 'local' | 'image' | 'vtk' | 'grid' | 'grid/vtk';

/**
 * The core side of the lazily loaded dump vtk helper (engine-gpl/dump/dump_vtk.ts, HOOKS 'vtk' of group gpl_dump_vtk):
 * the core Dump passes itself through this narrow host, and the helper does the attributes, the dump_modify keywords and
 * the files. "dump vtk" is documented at docs.lammps.org/dump_vtk.html.
 */
export interface DumpVtkHost {
  readonly sys: System;
  readonly groupBit: number;
  readonly file: string;
  /** dump_modify pad: the timestep digits of a "*" file name (live value). */
  readonly pad: number;
  /** The atoms in native storage order (nativeOrder) and the "*" file name substitution (fillWildcard), from the core. */
  nativeRows(): Int32Array;
  fillWildcard(file: string, text: string): string;
  validateColumn(c: string): void;
  columnValues(c: string): Float64Array;
  isIntColumn(c: string): boolean;
}

/** What the helper gives the core dump: its dump_modify keywords (returns the words taken, 0 for the core's) and the write. */
export interface DumpVtkStyle {
  modify(args: string[], k: number): number;
  write(step: number): void;
}

export type DumpVtkFactory = (host: DumpVtkHost, cols: string[]) => DumpVtkStyle;

/**
 * The core side of the lazily loaded dump cfg helper (engine-gpl/dump/dump_cfg.ts, HOOKS 'cfg' of group gpl_dump_cfg): the
 * core parses the attributes as for dump custom (dump.html: "Style cfg has the same command syntax as style custom"),
 * selects, orders and formats the atoms; the helper checks the cfg attributes and writes the snapshot text.
 */
export interface DumpCfgSnapshot {
  /** The atoms to write (indices into the state), in output order. */
  rows: ArrayLike<number>;
  /** Column values per atom index, one array per column. */
  values: Float64Array[];
  /** The compiled format of each column. */
  formatters: ((v: number) => string)[];
  /** dump_modify element: one name per atom type (empty: the default name of every type). */
  element: string[];
  /** dump_modify colname, by column index. */
  colnames: Map<number, string>;
  /** dump_modify header. */
  header: boolean;
}

export interface DumpCfgStyle {
  text(snap: DumpCfgSnapshot): string;
}

export type DumpCfgFactory = (sys: System, columns: string[]) => DumpCfgStyle;

/**
 * The core side of the lazily loaded dump grid and grid/vtk helper (engine-gpl/dump/dump_grid.ts, HOOKS 'grid' and
 * 'grid/vtk' of group gpl_dump_grid): the helper resolves the per-grid references (grid.ts) and writes the snapshots;
 * the core keeps the schedule, the files and the dump_modify keywords it shares with the other styles. "dump grid" is
 * documented at docs.lammps.org/dump.html: "The grid styles output one or more values per grid cell, which are
 * produced by other commands which overlay the simulation domain with a regular grid."
 */
export interface DumpGridHost {
  readonly sys: System;
  readonly id: string;
  readonly file: string;
  /** dump every N: a fix's per-grid data must be valid on those steps. */
  readonly every: number;
  readonly pad: number;
  readonly units: boolean;
  readonly time: boolean;
  readonly header: boolean;
  readonly sort: 'off' | 'id' | { col: number; desc: boolean };
  readonly colnames: Map<number, string>;
  /** The file has been written to (ITEM: UNITS goes into the first snapshot only). */
  readonly opened: boolean;
  /** One formatter per field, from dump_modify format (every field is a double, default %g). */
  formatters(): ((v: number) => string)[];
  fillWildcard(file: string, text: string): string;
  /** Writes one snapshot of a text dump with the core's file rules ("*" names, append, gz, maxfiles). */
  writeText(text: string): void;
}

export interface DumpGridStyle {
  /** dump_modify keywords of this style: the words taken, 0 for the core's. */
  modify(args: string[], k: number): number;
  write(step: number): void;
}

export type DumpGridFactory = (host: DumpGridHost, fields: string[]) => DumpGridStyle;

/** The dump_modify keywords of every dump style (the base list of Dump::modify_params, src/dump.cpp at b332f28). */
const GRID_DUMP_KEYWORDS = new Set(['append', 'balance', 'buffer', 'colname', 'delay', 'every', 'every/time', 'fileper', 'first', 'flush', 'format', 'header', 'maxfiles', 'nfile', 'pad', 'pbc', 'skip', 'sort', 'time', 'units']);

/** The image-count offset of the native image field (src/lmptype.h at b332f28, the default smallbig build: IMGMAX 512). */
const IMGMAX = 512;

/** Columns dump_modify triclinic/general rotates (dump.html: "vx,vy,vz = atom velocities" and the others listed there). */
const GENERAL_ROTATED = /^(x|y|z|xu|yu|zu|vx|vy|vz|fx|fy|fz)$/;
/** Per-atom vector columns the browser engine does not rotate; they stop the dump rather than write restricted values. */
const GENERAL_UNSUPPORTED = /^(mu[xyz]|omega[xyz]|angmom[xyz]|tq[xyz]|sp[xyz]|quat[wijk]|shape[xyz])$/;

export class Dump {
  readonly groupBit: number;
  every: number;
  everyVar: string | null = null;
  append = false;
  first = false;
  delay = 0;
  pbc = false;
  scale = true;
  image = false;
  unwrap = false;
  /** dump_modify triclinic/general: write the box and per-atom vectors in the general triclinic frame. */
  triclinicGeneral = false;
  time = false;
  units = false;
  header = true;
  sort: 'off' | 'id' | { col: number; desc: boolean };
  element: string[] = [];
  formatLine: string | null = null;
  /** dump_modify format int / float; null keeps the default ("%d" for integers, "%g" for floats). */
  formatInt: string | null = null;
  formatFloat: string | null = null;
  formatCol = new Map<number, string>();
  colnames = new Map<number, string>();
  region: string | null = null;
  thresh: { col: string; op: string; value: number | 'LAST'; last?: Map<number, number> }[] = [];
  pad = 0;
  /** dump_modify label (dump local): the word in the header lines, default ENTRIES. */
  label = 'ENTRIES';
  lastStep = -1;
  /** dump_modify maxfiles Fmax: with a "*" file name, the number of snapshot files kept (0 keeps all). */
  maxfiles = 0;
  private written: string[] = [];
  private opened = false;
  /** The file is written gzipped (a .gz name, or a COMPRESS /gz style). */
  private gz = false;
  private nextStep = -1;
  /** dump image: the image renderer (dump_image.ts); null for every other style. */
  private imageDump: ImageDump | null = null;
  /** dump vtk: the lazily loaded helper (null for every other style). */
  private vtkStyle: DumpVtkStyle | null = null;
  /** dump grid and grid/vtk: the lazily loaded helper (null for every other style). */
  private gridStyle: DumpGridStyle | null = null;
  /** dump cfg: the lazily loaded helper (null for every other style). */
  private cfgStyle: DumpCfgStyle | null = null;

  /** The atom colour map of a dump image as the previous output left it (fix graphics/labels colorscale); null for other styles. */
  colorScaleMap(which: 'atom' | 'grid' | 'bond' = 'atom'): ColorMap | null {
    if (!this.imageDump) return null;
    return which === 'bond' ? this.imageDump.bondColorScaleMap() : which === 'grid' ? this.imageDump.gridColorScaleMap() : this.imageDump.colorScaleMap();
  }
  readonly columns: string[];

  /**
   * The single file's name, resolved from the working directory (shell cd) when it is first opened. As native LAMMPS
   * does (src/dump_custom.cpp init_style at b332f28: "open single file, one time only"; src/dump.cpp write: "if file per
   * timestep, open new file"), a single file is opened once, at the first run after the dump command, and a multi-file
   * dump opens each snapshot's file when it is written. The session opens it at the first init command (Session
   * dispatch, INIT_COMMANDS); a write before any of them opens it then.
   */
  private singleKey: string | null = null;

  /** Opens the single file (resolves its name from the working directory) if it is not open yet; see singleKey. */
  openFile(): void {
    // As native LAMMPS does (src/dump_cfg.cpp init_style at b332f28): "Dump cfg requires one snapshot per file" unless
    // the name has a "*" (write_dump sets multifile_override)
    if (this.cfgStyle && !this.file.includes('*') && !this.multifileOverride) throw new StyleError('Dump cfg requires one snapshot per file');
    if (!this.file.includes('*') && this.singleKey === null) this.singleKey = this.sys.resolveFile(this.file);
  }

  constructor(private sys: System, readonly id: string, readonly group: string, readonly style: DumpStyle, every: number | string, readonly file: string, cols: string[], private readonly multifileOverride = false,
    /** A COMPRESS style (atom/gz, custom/gz, ...): its name, for dump_modify compression_level and the messages. */
    private readonly gzStyle: string | null = null) {
    // dump.html: "If the filename ends with ".gz", the dump file (or files, if "*" or "%" is also used) is written in
    // gzipped format." As native LAMMPS does (src/dump.cpp at b332f28, Dump::openfile): a name with a compression
    // extension is written through the compressor, and dump_modify append is then ignored (the file is rewritten).
    // The browser engine writes gzip only: each snapshot is one gzip member (stored blocks), and the members of a
    // file decompress to the whole dump. A COMPRESS /gz style always writes gzip and needs a compressed name.
    const ext = compressExtension(file);
    if (gzStyle) {
      if (!ext) throw new StyleError(`Dump ${gzStyle} only writes compressed files`);
      this.gz = true;
    } else if (ext) {
      if (style === 'image' || style === 'vtk' || style === 'grid/vtk' || ext !== 'gz') {
        throw new StyleError(`dump ${id}: a compressed (.${ext}) ${style} dump file is not supported by the browser engine; ${style === 'image' || style === 'vtk' || style === 'grid/vtk' ? 'use a plain file name' : 'use .gz or a plain file name'}`);
      }
      this.gz = true;
    }
    this.groupBit = sys.groupBit(group);
    if (typeof every === 'string') this.everyVar = every;
    this.every = typeof every === 'number' ? every : 0;
    // dump.html: "For the dcd, extxyz, grid/vtk, xtc, xyz, and molfile styles, sorting by atom ID or grid ID is on by
    // default."
    this.sort = style === 'xyz' || style === 'extxyz' || style === 'grid/vtk' ? 'id' : 'off';
    if (style === 'image') {
      this.imageDump = new ImageDump(sys, id, group, file, cols, multifileOverride, typeof every === 'number' ? every : 0);
      this.columns = [];
    } else if (style === 'atom') {
      if (cols.length) throw new StyleError('dump atom takes no attributes (use dump custom)');
      this.columns = ['id', 'type', 'xs', 'ys', 'zs'];
    } else if (style === 'xyz' || style === 'extxyz') {
      if (cols.length) throw new StyleError(`dump ${style} takes no attributes`);
      this.columns = ['type', 'x', 'y', 'z'];
    } else if (style === 'local') {
      this.columns = localDumpColumns(sys, id, cols);
    } else if (style === 'grid' || style === 'grid/vtk') {
      // dump grid: the fields are per-grid references (c_ID:gname:dname[I] with wildcards, grid.ts), resolved by the
      // helper; N must be an integer (a fix's per-grid data is valid on multiples of its frequency)
      if (typeof every !== 'number') throw new StyleError(`Expected integer parameter instead of 'v_${every}' in input script or data file`);
      this.columns = expandGridRefs(sys, cols);
      const dump = this;
      this.gridStyle = gplHook<DumpGridFactory>(style)({
        sys,
        id,
        file,
        every,
        get pad() { return dump.pad; },
        get units() { return dump.units; },
        get time() { return dump.time; },
        get header() { return dump.header; },
        get sort() { return dump.sort; },
        get colnames() { return dump.colnames; },
        get opened() { return dump.opened; },
        formatters: () => this.columnFormatters(),
        fillWildcard,
        writeText: (text) => this.writeText(text),
      }, this.columns);
    } else if (style === 'vtk') {
      // dump vtk: the attributes are parsed by the helper, which reads the live pad and the core's column access
      this.columns = [];
      const dump = this;
      this.vtkStyle = gplHook<DumpVtkFactory>('vtk')({
        sys,
        groupBit: this.groupBit,
        file,
        get pad() { return dump.pad; },
        nativeRows: () => nativeOrder(sys.state),
        fillWildcard,
        validateColumn: (c) => this.validate(c),
        columnValues: (c) => this.columnValues(c),
        isIntColumn: (c) => this.isInt(c),
      }, cols);
    } else {
      // As native LAMMPS does (src/dump_custom.cpp at b332f28): "No dump {} arguments specified"
      if (!cols.length) throw new StyleError(`No dump ${style} arguments specified`);
      const out: string[] = [];
      for (const c of cols) out.push(...this.expand(c));
      for (const c of out) this.validate(c);
      this.columns = out;
      // dump cfg: the attribute checks of the lazily loaded helper (lazyStyles.ts group gpl_dump_cfg)
      if (style === 'cfg') this.cfgStyle = gplHook<DumpCfgFactory>('cfg')(sys, out);
    }
  }

  private expand(c: string): string[] {
    // i2_name[*] / d2_name[*]: every column of a custom array (dump.html "I can include wildcard")
    const cm = /^([id])2_([A-Za-z0-9_]+)\[\*\]$/.exec(c);
    if (cm) {
      const cp = this.sys.state.custom.get(cm[2]);
      if (!cp || cp.cols === 0) throw new StyleError(`dump ${this.id}: custom per-atom array ${c} does not exist`);
      return Array.from({ length: cp.cols }, (_, k) => `${cm[1]}2_${cm[2]}[${k + 1}]`);
    }
    const m = /^([cf])_([A-Za-z0-9_]+)\[(\d*)\*(\d*)\]$/.exec(c);
    if (!m) return [c];
    const [, kind, id, lo, hi] = m;
    const n = kind === 'c' ? this.sys.compute(id).sizePeratomCols : this.sys.fix(id).sizePeratomCols;
    if (!n) throw new StyleError(`dump ${this.id}: ${c} has no per-atom array to expand`);
    const out: string[] = [];
    for (let k = lo ? Number(lo) : 1; k <= (hi ? Number(hi) : n); k++) out.push(`${kind}_${id}[${k}]`);
    return out;
  }

  private validate(c: string): void {
    if (ATOM_COLS.has(c)) {
      const st = this.sys.atomStyle;
      const sh = this.sys.hasBox ? this.sys.state : null;
      if (c === 'q' && !(sh ? hasCharge(sh) : hasChargeStyle(st))) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (q needs atom_style charge or full)`);
      // atom_style.html, smd: "atomic + molecule, radius, rmass + smd data", so radius is allocated for it
      if (SPHERE_COLS.has(c) && !isSphereStyle(st) && !(c === 'radius' && isSmdStyle(st))) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs atom_style sphere)`);
      if (TORQUE_COLS.has(c) && !hasRmassStyle(st)) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs atom_style sphere or ellipsoid)`);
      // As native LAMMPS does (src/dump_custom.cpp at b332f28): angmomx, angmomy and angmomz need atom->angmom_flag, set
      // by the styles with an angular momentum (ellipsoid, body and tri)
      if (ANGMOM_COLS.has(c) && !(sh ? sh.angmom : hasQuatStyle(st) || isTriStyle(st))) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs an atom style with angular momentum: ellipsoid, body or tri)`);
      if (DIPOLE_COLS.has(c) && !hasDipoleStyle(st)) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (${c} needs atom_style dipole)`);
      if (c === 'mol' && !(sh ? hasMolecule(sh) : isMolecularStyle(st))) throw new StyleError(`dump ${this.id}: dumping an atom property that isn't allocated (mol needs a molecular atom_style)`);
      return;
    }
    if (CUSTOM_ATTR.test(c)) {
      customAttr(this.sys.state, c);
      return;
    }
    const m = /^([cfv])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(c);
    if (!m) throw new StyleError(`invalid dump custom attribute '${c}'`);
    if (m[1] === 'c') {
      const cp = this.sys.compute(m[2]);
      if (!cp.peratomFlag) throw new StyleError(`dump ${this.id}: compute ${m[2]} does not compute per-atom info`);
    } else if (m[1] === 'f') {
      const f = this.sys.fix(m[2]);
      if (!f.peratomFlag) throw new StyleError(`dump ${this.id}: fix ${m[2]} does not compute per-atom info`);
    } else {
      const v = this.sys.vars.get(m[2]);
      if (v && v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`dump ${this.id}: variable ${m[2]} is not atom-style`);
    }
  }

  /** dump_modify keywords. */
  modify(args: string[]): void {
    const yesno = (w: string | undefined, key: string) => {
      if (w !== 'yes' && w !== 'no') throw new StyleError(`dump_modify ${key} must be yes or no`);
      return w === 'yes';
    };
    for (let k = 0; k < args.length;) {
      const key = args[k];
      const v = args[k + 1];
      // dump vtk: its own keywords (label, double, element, region, thresh ...); the common ones stay with the core
      if (this.vtkStyle) {
        const n = this.vtkStyle.modify(args, k);
        if (n > 0) { k += n; continue; }
      }
      // dump grid and grid/vtk: the helper's keywords, then only the keywords every dump style takes (the base
      // Dump::modify_params list of src/dump.cpp at b332f28)
      if (this.gridStyle) {
        const n = this.gridStyle.modify(args, k);
        if (n > 0) { k += n; continue; }
        if (!GRID_DUMP_KEYWORDS.has(key)) throw new StyleError('Illegal dump_modify command');
      }
      // dump image/movie-only dump_modify keywords (dump_image.html, "dump_modify options for dump
      // image/movie"): acolor, adiam, bcolor, bdiam, backcolor, boxcolor, color.
      if (this.imageDump) {
        const n = this.imageDump.modifyKeyword(args, k);
        if (n > 0) { k += n; continue; }
      }
      switch (key) {
        case 'append': this.append = yesno(v, key); k += 2; break;
        case 'first': this.first = yesno(v, key); k += 2; break;
        case 'pbc': this.pbc = yesno(v, key); k += 2; break;
        case 'scale': this.scale = yesno(v, key); this.applyScaleImage(); k += 2; break;
        case 'image': this.image = yesno(v, key); this.applyScaleImage(); k += 2; break;
        case 'unwrap': this.unwrap = yesno(v, key); k += 2; break;
        case 'time': this.time = yesno(v, key); k += 2; break;
        case 'units': this.units = yesno(v, key); k += 2; break;
        case 'header': this.header = yesno(v, key); k += 2; break;
        case 'flush': case 'buffer': case 'balance': case 'thermo': yesno(v, key); k += 2; break;
        case 'delay': this.delay = intArg(v, key); k += 2; break;
        case 'compression_level': {
          // As native LAMMPS does (src/COMPRESS/dump_custom_gz.cpp and gz_file_writer.cpp at b332f28): a /gz style takes a
          // level in [Z_DEFAULT_COMPRESSION, Z_BEST_COMPRESSION] = [-1, 9] while its file is not open. The level changes
          // the bytes, not the decompressed dump; the engine writes stored blocks at every level.
          if (!this.gzStyle) throw new StyleError(`unsupported dump_modify keyword '${key}'`);
          if (v === undefined) throw new StyleError('Illegal dump_modify command');
          const level = intArg(v, key);
          if (this.opened) throw new StyleError('Illegal dump_modify command: Compression level can not be changed while file is open');
          if (level < -1 || level > 9) throw new StyleError('Illegal dump_modify command: Compression level must in the range of [-1, 9]');
          k += 2;
          break;
        }
        case 'pad': this.pad = intArg(v, key); k += 2; break;
        case 'every': {
          if (v?.startsWith('v_')) { this.everyVar = v.slice(2); this.every = 0; } else {
            this.every = intArg(v, key);
            if (this.every < 1) throw new StyleError('dump_modify every must be > 0');
            this.everyVar = null;
          }
          this.nextStep = -1;
          k += 2;
          break;
        }
        case 'sort': {
          if (v === 'off') this.sort = 'off';
          else if (v === 'id') this.sort = 'id';
          else {
            const n = intArg(v, key);
            if (n === 0 || Math.abs(n) > this.columns.length) throw new StyleError('dump_modify sort column out of range');
            this.sort = { col: Math.abs(n) - 1, desc: n < 0 };
          }
          k += 2;
          break;
        }
        case 'element': {
          const n = this.sys.state.ntypes;
          const el = args.slice(k + 1, k + 1 + n);
          if (el.length !== n) throw new StyleError(`dump_modify element needs ${n} element names (one per atom type)`);
          this.element = el;
          k += 1 + n;
          break;
        }
        case 'format': {
          if (v === 'none') { this.formatLine = null; this.formatInt = null; this.formatFloat = null; this.formatCol.clear(); k += 2; break; }
          const f = args[k + 2];
          if (f === undefined) throw new StyleError('dump_modify format needs a format string');
          if (v === 'line') this.formatLine = f;
          else if (v === 'int') { checkCFormat(f, 'int', 'Invalid dump_modify int format'); this.formatInt = f; }
          else if (v === 'float') { checkCFormat(f, 'float', 'Invalid dump_modify float format'); this.formatFloat = f; }
          else {
            // dump_modify.rst: the format keyword takes an integer column M; a column name is not accepted.
            // Measured with native LAMMPS (black box): a column name stops with the message Expected integer parameter.
            if (!/^-?\d+$/.test(v ?? '')) throw new StyleError(`Expected integer parameter instead of '${v ?? ''}' in input script or data file`);
            const col = this.columnIndex(v ?? '');
            checkCFormat(f, this.kindOfCol(this.columns[col]), `Invalid dump_modify format for column ${col + 1}`);
            this.formatCol.set(col, f);
          }
          k += 3;
          break;
        }
        case 'colname': {
          if (v === 'default') { this.colnames.clear(); k += 2; break; }
          const name = args[k + 2];
          if (name === undefined) throw new StyleError('dump_modify colname needs a column and a name');
          this.colnames.set(this.columnIndex(v ?? ''), name);
          k += 3;
          break;
        }
        case 'region': {
          // As native LAMMPS does (src/dump_custom.cpp at b332f28, modify_param): region and thresh belong to the custom
          // dump family; the other styles reach Dump::modify_params, which does not know them
          if (!THRESH_STYLES.has(this.style)) throw new StyleError(`Unknown dump_modify keyword: ${key}`);
          if (v === undefined) throw new StyleError('Illegal dump_modify region command: missing argument(s)');
          if (v !== 'none' && !this.sys.regions.has(v)) throw new StyleError(`Dump_modify region ${v} does not exist`);
          this.region = v === 'none' ? null : v;
          k += 2;
          break;
        }
        case 'thresh': {
          if (!THRESH_STYLES.has(this.style)) throw new StyleError(`Unknown dump_modify keyword: ${key}`);
          if (v === 'none') { this.thresh = []; k += 2; break; }
          if (args.length < k + 4) throw new StyleError('Illegal dump_modify thresh command: missing argument(s)');
          if (!THRESH_ATTRS.has(v!) && !THRESH_REF.test(v!)) throw new StyleError(`Invalid dump_modify thresh attribute: ${v}`);
          const op = args[k + 2], val = args[k + 3];
          if (!['<', '<=', '>', '>=', '==', '!=', '|^'].includes(op ?? '')) throw new StyleError('Invalid dump_modify thresh operator');
          this.thresh.push({ col: v!, op: op!, value: val === 'LAST' ? 'LAST' : numArg(val, key) });
          k += 4;
          break;
        }
        case 'label': if (!v) throw new StyleError('dump_modify label needs a string'); this.label = v; k += 2; break;
        case 'maxfiles': {
          // dump_modify.html: "The specified Fmax is how many snapshots will be kept. Once this number is reached, the
          // file(s) containing the oldest snapshot is deleted before a new dump file is written." Measured with native
          // LAMMPS (black box): with a name without * the keyword stops with "Cannot use dump_modify maxfiles without *
          // in dump file name"; 5 snapshots of "snap.*.dump" with maxfiles 2 leave snap.3.dump and snap.4.dump.
          if (!this.file.includes('*')) throw new StyleError('Cannot use dump_modify maxfiles without * in dump file name');
          if (this.imageDump) throw new StyleError('dump_modify maxfiles is not supported for dump image by the browser engine');
          this.maxfiles = intArg(v, key);
          k += 2;
          break;
        }
        case 'nfile': case 'fileper': {
          // Measured with native LAMMPS (black box): without % in the file name the keyword stops with
          // "Cannot use dump_modify nfile without % in dump file name"; the % split is not written by the engine.
          if (!this.file.includes('%')) throw new StyleError(`Cannot use dump_modify ${key} without % in dump file name`);
          throw new StyleError(`dump_modify ${key} with a % file name is not supported by the browser engine`);
        }
        // dump_modify precision / sfactor / tfactor / at belong to other dump styles (xtc, netcdf). Measured with native
        // LAMMPS (black box): for dump atom, custom, xyz and extxyz each stops with "Unknown dump_modify keyword".
        case 'precision': case 'sfactor': case 'tfactor': case 'at':
          throw new StyleError(`Unknown dump_modify keyword: ${key}`);
        case 'triclinic/general': this.triclinicGeneral = yesno(v, key); k += 2; break;
        case 'types': if (v !== 'numeric') throw new StyleError('dump_modify types labels needs type labels (not supported)'); k += 2; break;
        default:
          throw new StyleError(`unsupported dump_modify keyword '${key}'`);
      }
    }
  }

  private applyScaleImage(): void {
    if (this.style !== 'atom') return;
    const c = this.scale ? ['id', 'type', 'xs', 'ys', 'zs'] : ['id', 'type', 'x', 'y', 'z'];
    if (this.image) c.push('ix', 'iy', 'iz');
    (this as unknown as { columns: string[] }).columns = c;
  }

  /** Kind of a dump column for format checks: integer columns take an integer conversion. */
  private kindOfCol(c: string): ValueKind {
    // the per-grid fields of dump grid are doubles (DumpGrid::parse_fields: vtype = Dump::DOUBLE)
    if (this.gridStyle) return 'float';
    const isInt = this.style === 'local' ? c === 'index' : this.isInt(c);
    return isInt ? 'int' : 'float';
  }

  /**
   * One formatter per column. Precedence (dump_modify.html): the column's own format, then the int or float
   * setting, then the line. Measured with native LAMMPS (black box): a line with fewer formats than columns is an
   * error, reported as "too short"; a format that does not match its column is an error when the line is written.
   */
  private columnFormatters(): ((v: number) => string)[] {
    const toks = this.formatLine ? this.formatLine.split(/\s+/).filter(Boolean) : null;
    if (toks && toks.length < this.columns.length) throw new StyleError(`Dump_modify format line is too short: ${this.formatLine}`);
    return this.columns.map((c, k) => {
      const kind = this.kindOfCol(c);
      const spec = this.formatCol.get(k) ?? (kind === 'int' ? this.formatInt : this.formatFloat) ?? toks?.[k] ?? (kind === 'int' ? '%d' : '%g');
      return compileCFormat(spec, kind, `Invalid dump ${this.id} format for column ${k + 1}`);
    });
  }

  private columnIndex(w: string): number {
    if (/^-?\d+$/.test(w)) {
      const n = Number(w);
      const k = n > 0 ? n - 1 : this.columns.length + n;
      if (k < 0 || k >= this.columns.length) throw new StyleError(`dump_modify: column ${w} out of range`);
      return k;
    }
    const k = this.columns.indexOf(w);
    if (k < 0) throw new StyleError(`dump_modify: '${w}' is not a column of dump ${this.id}`);
    return k;
  }

  /**
   * Writes one snapshot file. With dump_modify maxfiles, a "*" dump keeps only the last Fmax files: the oldest
   * are removed from the file store once a newer one is written (dump_modify.html).
   */
  private writeSnapshot(name: string, text: string, append: boolean): void {
    this.sys.writeFileAt(name, this.gz ? gzipText(text) : text, append);
    if (this.maxfiles <= 0) return;
    if (!this.written.includes(name)) this.written.push(name);
    while (this.written.length > this.maxfiles) {
      const old = this.written.shift();
      if (old !== undefined) this.sys.files.delete(old);
    }
  }

  /** Should a snapshot be written on this step? */
  due(step: number, firstStepOfRun: boolean): boolean {
    if (step < this.delay) return false;
    if (this.lastStep === step) return false;
    if (firstStepOfRun && this.first) return true;
    if (this.everyVar) {
      if (this.nextStep < 0) this.nextStep = Math.trunc(this.sys.equalVariable(this.everyVar));
      return step === this.nextStep;
    }
    return step % this.every === 0;
  }

  write(): void {
    const sys = this.sys;
    const s = sys.state;
    const step = s.step;
    this.lastStep = step;
    if (this.everyVar) this.nextStep = Math.trunc(sys.equalVariable(this.everyVar));
    if (this.imageDump) {
      const stepText = this.pad > 0 ? String(step).padStart(this.pad, '0') : String(step);
      const name = this.file.includes('*') ? fillWildcard(this.file, stepText) : this.file;
      // dump_image.cpp create_image: the drawn atoms are DumpCustom::count()'s choice (group, dump_modify region and
      // thresh), which also bounds a dynamic colour map and selects both atoms of a drawn bond
      if (this.thresh.length) sys.forces();
      const chosen = new Uint8Array(s.n);
      for (const i of this.chooseRows()) chosen[i] = 1;
      this.imageDump.write(s, name, this, chosen);
      return;
    }
    if (this.vtkStyle) {
      this.vtkStyle.write(step);
      return;
    }
    if (this.gridStyle) {
      this.gridStyle.write(step);
      return;
    }
    sys.forces();
    if (this.style === 'local') {
      this.writeLocal(step);
      return;
    }
    const g = sys.geom;
    // dump_modify.html: "It can only be used with a value of *yes* if the" simulation box was created as a general
    // triclinic box. Only the atom and custom styles take it (Howto_triclinic.html lists dump atom, dump custom).
    if (this.triclinicGeneral) {
      if (!s.box.general) throw new StyleError(`dump ${this.id}: dump_modify triclinic/general yes needs a general triclinic box (create_box NULL, or a general triclinic data file)`);
      if (this.style !== 'atom' && this.style !== 'custom') throw new StyleError(`dump ${this.id}: dump_modify triclinic/general applies only to the atom and custom styles`);
    }
    const keep = this.chooseRows();
    const values = this.columns.map((c) => this.columnValues(c));
    if (this.sort === 'id') keep.sort((a, b) => s.id[a] - s.id[b]);
    else if (this.sort !== 'off') {
      const { col, desc } = this.sort;
      const v = values[col];
      keep.sort((a, b) => (desc ? v[b] - v[a] : v[a] - v[b]));
    }
    const multi = this.file.includes('*');
    const stepText = this.pad > 0 ? String(step).padStart(this.pad, '0') : String(step);
    if (this.cfgStyle) {
      // dump cfg: one snapshot per file (the "*" name, or write_dump's single file)
      const text = this.cfgStyle.text({ rows: keep, values, formatters: this.columnFormatters(), element: this.element, colnames: this.colnames, header: this.header });
      const name = this.sys.resolveFile(multi ? fillWildcard(this.file, stepText) : this.file);
      this.writeSnapshot(name, text, false);
      this.opened = true;
      return;
    }
    const lines: string[] = [];
    const time = s.time + (s.step - s.timeStep) * s.dt;
    if (this.style === 'xyz' || this.style === 'extxyz') {
      lines.push(String(keep.length));
      if (this.style === 'extxyz') {
        lines.push(`Lattice="${g.lx} 0 0 ${g.xy} ${g.ly} 0 ${g.xz} ${g.yz} ${g.lz}" Origin="${s.box.lo.join(' ')}" Properties=species:S:1:pos:R:3 Timestep=${step}${this.time ? ` Time=${time}` : ''}`);
      } else lines.push(` Atoms. Timestep: ${step}${this.time ? ` Time: ${formatNumber(time, '%g')}` : ''}`);
      const fx = fmt(this.formatFloat ?? '%g');
      for (const i of keep) {
        const t = s.type[i];
        const el = this.element.length ? this.element[t - 1] : String(t);
        lines.push(`${el} ${fx(s.x[3 * i])} ${fx(s.x[3 * i + 1])} ${fx(s.x[3 * i + 2])}`);
      }
    } else if (this.style === 'yaml') {
      lines.push('---', `creator: LAMMPS web notebook`, `timestep: ${step}`, `natoms: ${keep.length}`);
      lines.push(`boundary: [ ${s.box.boundary.flat().join(', ')} ]`);
      lines.push('box:', ...[0, 1, 2].map((d) => `  - [ ${s.box.lo[d]}, ${s.box.hi[d]} ]`));
      lines.push(`keywords: [ ${this.columns.map((c, k) => this.colnames.get(k) ?? c).join(', ')} ]`, 'data:');
      for (const i of keep) lines.push(`  - [ ${values.map((v) => String(v[i])).join(', ')} ]`);
      lines.push('...');
    } else {
      if (this.units && !this.opened) lines.push('ITEM: UNITS', s.units.style);
      if (this.time) lines.push('ITEM: TIME', formatNumber(time, '%.16g'));
      if (this.header) {
        lines.push('ITEM: TIMESTEP', String(step), 'ITEM: NUMBER OF ATOMS', String(keep.length));
        const bnd = s.box.boundary.map((f) => f[0] + f[1]).join(' ');
        const e = (v: number) => formatNumber(v, '%-1.16e');
        if (this.triclinicGeneral && s.box.general) {
          // dump.html: "ITEM: BOX BOUNDS abc origin" then "ax ay az originx", "bx by bz originy", "cx cy cz originz"
          const gb = generalBoxFromRestricted(s.box.general.Q, s.box.lo, s.box.hi, s.box.tilt);
          const o = s.box.lo;
          lines.push(`ITEM: BOX BOUNDS abc origin ${bnd}`, `${e(gb.A[0])} ${e(gb.A[1])} ${e(gb.A[2])} ${e(o[0])}`, `${e(gb.B[0])} ${e(gb.B[1])} ${e(gb.B[2])} ${e(o[1])}`, `${e(gb.C[0])} ${e(gb.C[1])} ${e(gb.C[2])} ${e(o[2])}`);
        } else if (g.triclinic) {
          const [xy, xz, yz] = s.box.tilt;
          const xlob = s.box.lo[0] + Math.min(0, xy, xz, xy + xz), xhib = s.box.hi[0] + Math.max(0, xy, xz, xy + xz);
          const ylob = s.box.lo[1] + Math.min(0, yz), yhib = s.box.hi[1] + Math.max(0, yz);
          lines.push(`ITEM: BOX BOUNDS xy xz yz ${bnd}`, `${e(xlob)} ${e(xhib)} ${e(xy)}`, `${e(ylob)} ${e(yhib)} ${e(xz)}`, `${e(s.box.lo[2])} ${e(s.box.hi[2])} ${e(yz)}`);
        } else {
          lines.push(`ITEM: BOX BOUNDS ${bnd}`, ...[0, 1, 2].map((d) => `${e(s.box.lo[d])} ${e(s.box.hi[d])}`));
        }
        lines.push(`ITEM: ATOMS ${this.columns.map((c, k) => this.colnames.get(k) ?? c).join(' ')}`);
      }
      const formatters = this.columnFormatters();
      // dump_modify.html, Default section: the element name is C for every atom type without dump_modify element
      // (measured the same with native LAMMPS for dump custom ... element); the column is the name of the atom's type
      const elementName = (i: number): string => (this.element.length ? this.element[s.type[i] - 1] : 'C');
      for (const i of keep) {
        lines.push(this.columns.map((c, k) => (c === 'element' ? elementName(i) : formatters[k](values[k][i]))).join(' '));
      }
    }
    const name = multi ? this.sys.resolveFile(fillWildcard(this.file, stepText)) : (this.singleKey ??= this.sys.resolveFile(this.file));
    const appendNow = multi ? false : this.opened || (this.append && !this.gz);
    this.writeSnapshot(name, lines.join('\n') + '\n', appendNow);
    this.opened = true;
  }

  /** One snapshot of a text dump (dump grid) into the dump's file: a "*" name per snapshot, else the single file. */
  private writeText(text: string): void {
    const step = this.sys.state.step;
    const multi = this.file.includes('*');
    const stepText = this.pad > 0 ? String(step).padStart(this.pad, '0') : String(step);
    const name = multi ? this.sys.resolveFile(fillWildcard(this.file, stepText)) : (this.singleKey ??= this.sys.resolveFile(this.file));
    const appendNow = multi ? false : this.opened || (this.append && !this.gz);
    this.writeSnapshot(name, text, appendNow);
    this.opened = true;
  }

  /**
   * dump local: "ITEM: TIMESTEP", "ITEM: NUMBER OF ENTRIES", the box, "ITEM: ENTRIES ..." (the word
   * ENTRIES is replaced by dump_modify label), then one line per local datum. Measured with native
   * LAMMPS (black box): every value, the index included, is followed by one space; dump_modify header
   * no leaves out all header lines.
   */
  private writeLocal(step: number): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    if (this.thresh.length || this.region) throw new StyleError(`dump ${this.id} local: thresh and region are not supported by the browser engine`);
    const srcs = this.columns.map((c) => localColumnSource(sys, c));
    let nrows = -1;
    for (const c of this.columns) {
      if (c === 'index') continue;
      const m = /^([cf])_([A-Za-z0-9_]+)/.exec(c)!;
      const r = localOutputOf(sys, m[1] as 'c' | 'f', m[2]).localRows;
      if (nrows >= 0 && r !== nrows) throw new StyleError(`dump ${this.id} local: the computes do not have the same number of local entries (${nrows} and ${r})`);
      nrows = r;
    }
    if (nrows < 0) nrows = 0;
    const lines: string[] = [];
    if (this.header) {
      lines.push('ITEM: TIMESTEP', String(step), `ITEM: NUMBER OF ${this.label}`, String(nrows));
      const bnd = s.box.boundary.map((f) => f[0] + f[1]).join(' ');
      const e = (v: number) => formatNumber(v, '%-1.16e');
      if (g.triclinic) {
        const [xy, xz, yz] = s.box.tilt;
        const xlob = s.box.lo[0] + Math.min(0, xy, xz, xy + xz), xhib = s.box.hi[0] + Math.max(0, xy, xz, xy + xz);
        const ylob = s.box.lo[1] + Math.min(0, yz), yhib = s.box.hi[1] + Math.max(0, yz);
        lines.push(`ITEM: BOX BOUNDS xy xz yz ${bnd}`, `${e(xlob)} ${e(xhib)} ${e(xy)}`, `${e(ylob)} ${e(yhib)} ${e(xz)}`, `${e(s.box.lo[2])} ${e(s.box.hi[2])} ${e(yz)}`);
      } else {
        lines.push(`ITEM: BOX BOUNDS ${bnd}`, ...[0, 1, 2].map((d) => `${e(s.box.lo[d])} ${e(s.box.hi[d])}`));
      }
      lines.push(`ITEM: ${this.label} ${this.columns.map((c, k) => this.colnames.get(k) ?? c).join(' ')}`);
    }
    const formatters = this.columnFormatters();
    for (let r = 0; r < nrows; r++) {
      let text = '';
      this.columns.forEach((_, k) => {
        const src = srcs[k];
        const v = src ? src.data[r * src.ncol + src.col] : r + 1;
        text += formatters[k](v) + ' ';
      });
      lines.push(text);
    }
    const multi = this.file.includes('*');
    const stepText = this.pad > 0 ? String(step).padStart(this.pad, '0') : String(step);
    const name = multi ? this.sys.resolveFile(fillWildcard(this.file, stepText)) : (this.singleKey ??= this.sys.resolveFile(this.file));
    const appendNow = multi ? false : this.opened || (this.append && !this.gz);
    this.writeSnapshot(name, lines.join('\n') + '\n', appendNow);
    this.opened = true;
  }

  private isInt(c: string): boolean {
    return INT_COLS.has(c) || /^i2?_/.test(c);
  }

  /**
   * The atoms a snapshot writes, in native storage order: the dump group, then dump_modify region, then each
   * dump_modify thresh in turn (DumpCustom::count).
   */
  private chooseRows(): number[] {
    const sys = this.sys;
    const s = sys.state;
    const reg = this.region ? sys.region(this.region) : null;
    const rows: number[] = [];
    // unsorted dumps list atoms in native LAMMPS's storage order (SimState.order)
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (reg && !reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      rows.push(i);
    }
    let keep = rows;
    for (const t of this.thresh) {
      const col = this.columnValues(t.col);
      keep = keep.filter((i) => {
        const ref = t.value === 'LAST' ? (t.last?.get(s.id[i]) ?? col[i]) : t.value;
        const x = col[i];
        switch (t.op) {
          case '<': return x < ref;
          case '<=': return x <= ref;
          case '>': return x > ref;
          case '>=': return x >= ref;
          case '==': return x === ref;
          case '!=': return x !== ref;
          default: return (x !== 0) !== (ref !== 0);
        }
      });
      // As native LAMMPS does (src/dump_custom.cpp at b332f28, DumpCustom::count): the LAST values are stored for every
      // atom after each comparison, chosen or not
      if (t.value === 'LAST') {
        t.last = new Map();
        for (let i = 0; i < s.n; i++) t.last.set(s.id[i], col[i]);
      }
    }
    return keep;
  }

  /** Values of one column for every atom index (length n). */
  private columnValues(c: string): Float64Array {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const out = new Float64Array(s.n);
    const lam = [0, 0, 0];
    const u = [0, 0, 0];
    if (this.triclinicGeneral && GENERAL_UNSUPPORTED.test(c)) {
      throw new StyleError(`dump ${this.id}: dump_modify triclinic/general does not rotate column ${c} (the browser engine rotates x, y, z, xu, yu, zu, vx, vy, vz, fx, fy, fz)`);
    }
    const pos = (i: number, d: number) => {
      if (!this.pbc) return s.x[3 * i + d];
      const x = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
      const im = [s.image[3 * i], s.image[3 * i + 1], s.image[3 * i + 2]];
      const xx = Float64Array.from(x);
      const ii = Int32Array.from(im);
      g.remap(xx, ii, 0);
      return xx[d];
    };
    // dump_modify triclinic/general: positions (x, xu: about the box origin), velocities and forces are rotated
    // into the general frame (dump.html; Howto_triclinic.html). The rotation is the stored one (SimBox.general).
    if (this.triclinicGeneral && s.box.general && GENERAL_ROTATED.test(c)) {
      const Q = s.box.general.Q;
      const o: V3 = [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
      const d = 'xyz'.indexOf(c[c.length - 1]);
      for (let i = 0; i < s.n; i++) {
        let w: V3;
        if (c[0] === 'v') w = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
        else if (c[0] === 'f') w = [s.f[3 * i], s.f[3 * i + 1], s.f[3 * i + 2]];
        else if (c.endsWith('u') || this.unwrap) { g.unwrap(s.x, s.image, i, u); w = [u[0], u[1], u[2]]; } else w = [pos(i, 0), pos(i, 1), pos(i, 2)];
        out[i] = c[0] === 'v' || c[0] === 'f' ? unrotateVector(Q, w)[d] : toGeneralPoint(Q, o, w)[d];
      }
      return out;
    }
    switch (c) {
      case 'id': for (let i = 0; i < s.n; i++) out[i] = s.id[i]; return out;
      case 'mol': for (let i = 0; i < s.n; i++) out[i] = s.molecule[i]; return out;
      case 'proc': return out;
      case 'procp1': return out.fill(1);
      case 'type': for (let i = 0; i < s.n; i++) out[i] = s.type[i]; return out;
      // the element column is written as text (writeCustom); its numeric stand-in is the type
      case 'element': for (let i = 0; i < s.n; i++) out[i] = s.type[i]; return out;
      case 'mass': for (let i = 0; i < s.n; i++) out[i] = massOf(s, i); return out;
      case 'q': for (let i = 0; i < s.n; i++) out[i] = s.q[i]; return out;
      case 'x': case 'y': case 'z': {
        const d = 'xyz'.indexOf(c);
        for (let i = 0; i < s.n; i++) {
          if (this.unwrap) { g.unwrap(s.x, s.image, i, u); out[i] = u[d]; } else out[i] = pos(i, d);
        }
        return out;
      }
      case 'xs': case 'ys': case 'zs': {
        const d = 'xyz'.indexOf(c[0]);
        for (let i = 0; i < s.n; i++) { g.toLamda(pos(i, 0), pos(i, 1), pos(i, 2), lam); out[i] = lam[d]; }
        return out;
      }
      case 'xu': case 'yu': case 'zu': {
        const d = 'xyz'.indexOf(c[0]);
        for (let i = 0; i < s.n; i++) { g.unwrap(s.x, s.image, i, u); out[i] = u[d]; }
        return out;
      }
      case 'xsu': case 'ysu': case 'zsu': {
        // As native LAMMPS does (src/dump_custom.cpp pack_xsu and pack_xsu_triclinic at b332f28): "lamda + (image[j] &
        // IMGMASK) - IMGMAX" adds the stored image field, the count plus IMGMAX = 512, and subtracts IMGMAX after, so
        // the sum is rounded at the magnitude of 512 (to about 1e-13)
        const d = 'xyz'.indexOf(c[0]);
        for (let i = 0; i < s.n; i++) {
          g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
          out[i] = (lam[d] + (s.image[3 * i + d] + IMGMAX)) - IMGMAX;
        }
        return out;
      }
      case 'ix': case 'iy': case 'iz': {
        const d = 'xyz'.indexOf(c[1]);
        for (let i = 0; i < s.n; i++) out[i] = s.image[3 * i + d];
        return out;
      }
      case 'vx': case 'vy': case 'vz': {
        const d = 'xyz'.indexOf(c[1]);
        for (let i = 0; i < s.n; i++) out[i] = s.v[3 * i + d];
        return out;
      }
      case 'fx': case 'fy': case 'fz': {
        const d = 'xyz'.indexOf(c[1]);
        for (let i = 0; i < s.n; i++) out[i] = s.f[3 * i + d];
        return out;
      }
      case 'radius': case 'diameter': {
        const f = c === 'radius' ? 1 : 2;
        for (let i = 0; i < s.n; i++) out[i] = f * s.radius![i];
        return out;
      }
      case 'omegax': case 'omegay': case 'omegaz': {
        const d = 'xyz'.indexOf(c[5]);
        for (let i = 0; i < s.n; i++) out[i] = s.omega![3 * i + d];
        return out;
      }
      case 'tqx': case 'tqy': case 'tqz': {
        const d = 'xyz'.indexOf(c[2]);
        for (let i = 0; i < s.n; i++) out[i] = s.torque![3 * i + d];
        return out;
      }
      case 'angmomx': case 'angmomy': case 'angmomz': {
        const d = 'xyz'.indexOf(c[6]);
        for (let i = 0; i < s.n; i++) out[i] = s.angmom![3 * i + d];
        return out;
      }
      case 'mux': case 'muy': case 'muz': case 'mu': {
        const d = c === 'mu' ? 3 : 'xyz'.indexOf(c[2]);
        for (let i = 0; i < s.n; i++) out[i] = s.mu![4 * i + d];
        return out;
      }
    }
    const getter = customAttr(s, c);
    if (getter) { for (let i = 0; i < s.n; i++) out[i] = getter(i); return out; }
    const m = /^([cfv])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(c);
    if (!m) throw new StyleError(`invalid dump attribute '${c}'`);
    const col = m[3] ? Number(m[3]) : 0;
    if (m[1] === 'v') return sys.atomVariable(m[2]);
    const cols = m[1] === 'c' ? sys.compute(m[2]).sizePeratomCols : sys.fix(m[2]).sizePeratomCols;
    const vals = m[1] === 'c' ? sys.compute(m[2]).peratomValues() : (cols ? sys.fix(m[2]).arrayAtom : sys.fix(m[2]).vectorAtom);
    if (cols === 0) { out.set(vals.subarray(0, s.n)); return out; }
    if (col < 1 || col > cols) throw new StyleError(`dump attribute ${c}: column out of range 1..${cols}`);
    for (let i = 0; i < s.n; i++) out[i] = vals[i * cols + col - 1];
    return out;
  }
}

const intArg = (w: string | undefined, key: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isInteger(n)) throw new StyleError(`dump_modify ${key} needs an integer`);
  return n;
};

const numArg = (w: string | undefined, key: string): number => {
  const n = Number(w);
  if (w === undefined || !Number.isFinite(n)) throw new StyleError(`dump_modify ${key} needs a number`);
  return n;
};

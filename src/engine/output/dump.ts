import { isSmdStyle } from '../smd';
import type { System } from '../system';
import { StyleError } from '../force/types';
import { errorUrl } from '../formula';
import { inumeric, missingArgs, numeric, logical } from '../force/util';
import { compileCFormat, checkCFormat, formatNumber, fmtShortest, type ValueKind } from '../cformat';
import { localDumpColumns, localColumnSource, localOutputOf } from '../compute/local_dump';
import { generalBoxFromRestricted, toGeneralPoint, unrotateVector, type V3 } from '../triclinic_general';
import { fillWildcard } from '../commands/restart';
import { hasChargeStyle, hasDipoleStyle, isMolecularStyle, isSphereStyle, hasRmassStyle, hasQuatStyle, isTriStyle, massOf, CUSTOM_ATTR, customAttr, hasCharge, hasMolecule, nativeOrder } from '../atoms';
import type { ImageDump } from './dump_image';
import type { ColorMap } from './image_colormap';
import { gplHook } from '../gplHooks';
import { expandGridRefs } from '../grid';
import { gzipText } from '../gzip';
import { zstdText } from '../zstd';

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
/**
 * Which dump_modify keywords each dump style takes, as native LAMMPS splits them (b332f28): Dump::modify_params
 * (src/dump.cpp) takes the base list for every style, then the style's modify_param takes its own: DumpAtom image, scale
 * and triclinic/general; DumpCustom (custom, cfg, image, yaml) element, format N|int|float, refresh, region, thresh and
 * triclinic/general, DumpYAML thermo too; DumpXYZ (xyz, extxyz) element and types; DumpLocal format N|int|float and
 * label; the compressed styles compression_level. Any other keyword stops with "Unknown dump_modify keyword: X".
 */
const BASE_KEYWORDS = new Set(['append', 'balance', 'buffer', 'colname', 'delay', 'every', 'every/time', 'fileper', 'first', 'flush',
  'format', 'header', 'maxfiles', 'nfile', 'pad', 'pbc', 'skip', 'sort', 'time', 'units']);
const STYLE_KEYWORDS: Record<string, ReadonlySet<string>> = {
  atom: new Set(['image', 'scale', 'triclinic/general']),
  custom: new Set(['element', 'refresh', 'region', 'thresh', 'triclinic/general']),
  cfg: new Set(['element', 'refresh', 'region', 'thresh', 'triclinic/general']),
  'cfg/uef': new Set(['element', 'refresh', 'region', 'thresh', 'triclinic/general']),
  image: new Set(['element', 'refresh', 'region', 'thresh', 'triclinic/general']),
  yaml: new Set(['element', 'refresh', 'region', 'thresh', 'triclinic/general', 'thermo']),
  xyz: new Set(['element', 'types']),
  extxyz: new Set(['element', 'types']),
  local: new Set(['label']),
};
/** DumpExtXYZ::init_style without element names or type labels (native's spelling). */
const EXTXYZ_NO_NAMES = 'Must use either type lables or dump_modify element with dump style extxyz';

/** The styles whose modify_param takes the column formats (format N, int, float): DumpCustom, DumpLocal and DumpGrid. */
const COLUMN_FORMAT_STYLES = new Set(['custom', 'cfg', 'cfg/uef', 'image', 'yaml', 'local', 'grid', 'grid/vtk']);

/** The dump styles of native's DumpCustom family (custom, cfg, image, yaml and their compressed forms): dump_modify region and thresh. */
const THRESH_STYLES = new Set<string>(['custom', 'cfg', 'cfg/uef', 'image', 'yaml']);
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

/** Creates the dump image writer (registry/dump_image.ts, HOOKS image.createDump). */
export type ImageDumpFactory = (sys: System, id: string, group: string, file: string, cols: string[], multifileOverride: boolean | undefined, every: number) => ImageDump;

export type DumpStyle = 'atom' | 'custom' | 'cfg' | 'cfg/uef' | 'xyz' | 'extxyz' | 'yaml' | 'local' | 'image' | 'vtk' | 'grid' | 'grid/vtk' | 'dcd' | 'xtc';

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
  /** The style's own init checks, after the custom ones (dump cfg/uef: its fix nvt/uef or npt/uef). */
  init?(): void;
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

/**
 * The core side of the lazily loaded dump dcd and xtc helpers (engine-gpl/dump/dump_dcd.ts and dump_xtc.ts, HOOKS 'dcd'
 * and 'xtc' of group gpl_dump_traj): binary trajectory files, one frame per snapshot. dump.html: "The dcd style writes DCD
 * files, a standard atomic trajectory format used by the CHARMM, NAMD, and XPLOR molecular dynamics packages"; "The xtc
 * style writes XTC files, a compressed trajectory format used by the GROMACS molecular dynamics package". The core keeps
 * the schedule and the dump_modify keywords every style takes and gives the group's atoms in ID order; the helper checks
 * its settings and writes the bytes. A binary file is kept in the file store as one character per byte.
 */
export interface DumpTrajHost {
  readonly sys: System;
  /** The file name as given in the dump command. */
  readonly file: string;
  /** The number of atoms in the dump group now. */
  groupCount(): number;
  /** Creates the file empty (native opens it at the dump command). */
  create(): void;
  /** Appends bytes (one character per byte) to the file, or with `at` overwrites the bytes from that offset on. */
  writeBytes(bytes: string, at?: number): void;
}

/** What the helper's init checks (Dump::init then the style's init_style, at each run and at write_dump). */
export interface DumpTrajSettings {
  /** write_dump's snapshot: native's dump ID WRITE_DUMP, whose init leaves out the checks of the output interval. */
  writeDump: boolean;
  /** dump_modify every/time (Output mode_dump 1). */
  everyTime: boolean;
  /** The output interval (Output every_dump): 0 with dump_modify every v_name. */
  every: number;
  /** dump_modify sort id (the default of these styles). */
  sortById: boolean;
  /** dump_modify flush as set; null when it was not set (each style has its own default). */
  flush: boolean | null;
}

export interface DumpTrajFrame {
  step: number;
  dt: number;
  /** The atoms of the snapshot (Dump::count: the group's atoms). */
  natoms: number;
  /** dump_modify header (write_header_flag). */
  header: boolean;
  box: { triclinic: boolean; lo: readonly number[]; hi: readonly number[]; xprd: number; yprd: number; zprd: number; xy: number; xz: number; yz: number };
  /** x, y, z of each atom in ID order (dump_modify pbc applied to a copy), unwrapped when the helper's unwrap is set. */
  coords: Float64Array;
}

export interface DumpTrajStyle {
  /** dump_modify unwrap. */
  readonly unwrap: boolean;
  /** The helper's dump_modify keywords: the words taken, 0 for the core's. */
  modify(args: string[], k: number): number;
  init(settings: DumpTrajSettings): void;
  write(frame: DumpTrajFrame): void;
}

export type DumpTrajFactory = (host: DumpTrajHost, args: string[]) => DumpTrajStyle;

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
  /** The N of the dump command, which init checks the fix intervals against (dump_modify every leaves it). */
  private readonly nevery: number;
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
  /** dump_modify skip v_name: the equal-style variable that skips a snapshot when it is not 0 (null = none). */
  private skipVar: string | null = null;
  /** dump_modify refresh c_ID: the compute refreshed after each snapshot (null = none). */
  private refreshId: string | null = null;
  private written: string[] = [];
  private opened = false;
  /** The file is written compressed (a .gz or .zst name, or a COMPRESS /gz or /zstd style). */
  private gz = false;
  /** Zstandard rather than gzip (a .zst name or a /zstd style), and its content checksum (dump_modify checksum). */
  private zst = false;
  private zstChecksum = true;
  private nextStep = -1;
  /** dump image: the image renderer (dump_image.ts); null for every other style. */
  private imageDump: ImageDump | null = null;
  /** dump vtk: the lazily loaded helper (null for every other style). */
  private vtkStyle: DumpVtkStyle | null = null;
  /** dump grid and grid/vtk: the lazily loaded helper (null for every other style). */
  private gridStyle: DumpGridStyle | null = null;
  /** dump cfg: the lazily loaded helper (null for every other style). */
  private cfgStyle: DumpCfgStyle | null = null;
  /** dump dcd and xtc: the lazily loaded helper (null for every other style). */
  private trajStyle: DumpTrajStyle | null = null;
  /** dump_modify flush as set (null: not set). */
  private flush: boolean | null = null;
  /** dump_modify every/time was given (dump dcd and xtc only, whose init then stops; other styles refuse it). */
  private everyTime = false;

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
    // As native LAMMPS does (src/EXTRA-DUMP/dump_extxyz.cpp at b332f28, init_style, at the init of a run)
    if (this.style === 'extxyz' && !this.element.length) throw new StyleError(EXTXYZ_NO_NAMES);
    if (!this.file.includes('*') && this.singleKey === null) this.singleKey = this.sys.resolveFile(this.file);
  }

  /**
   * The dump's init at each run (Output::init, after the thermo init). As native LAMMPS does (src/dump_custom.cpp and
   * src/dump_local.cpp init_style at b332f28): the computes, then the fixes, that the columns (and a dump_modify thresh)
   * read must still exist, and the dump's interval must be a multiple of each fix's output interval ("if (nevery %
   * fix[i]->peratom_freq)", local_freq for dump local; an interval of -1, not set, divides any).
   */
  init(): void {
    if (this.trajStyle) { this.initTraj(false); return; }
    const local = this.style === 'local';
    if (!local && (!THRESH_STYLES.has(this.style) || this.style === 'image')) return;
    const st = this.gzStyle ?? this.style;
    const ids = { c: [] as string[], f: [] as string[] };
    for (const w of [...this.columns, ...this.thresh.map((t) => t.col)]) {
      const m = /^([cf])_([A-Za-z0-9_]+)/.exec(w);
      if (m && !ids[m[1] as 'c' | 'f'].includes(m[2])) ids[m[1] as 'c' | 'f'].push(m[2]);
    }
    for (const id of ids.c) {
      const c = this.sys.computes.find((x) => x.id === id);
      if (!c) throw new StyleError(`Could not find dump ${st} compute ID ${id}`);
      if (!local && !c.peratomFlag) throw new StyleError(`Compute ID ${id} for dump ${st} does not compute per-atom data`);
    }
    for (const id of ids.f) {
      const f = this.sys.fixes.find((x) => x.id === id);
      if (!f) throw new StyleError(`Could not find dump ${st} fix ID ${id}`);
      if (local) {
        if (f.localFreq > 0 && this.nevery % f.localFreq) throw new StyleError(`Dump local and fix ${id} not computed at compatible times${errorUrl(7)}`);
        continue;
      }
      if (!f.peratomFlag) throw new StyleError(`Fix ID ${id} for dump ${st} does not compute per-atom data`);
      if (f.peratomFreq > 0 && this.nevery % f.peratomFreq) throw new StyleError(`Dump ${st} and fix not computed at compatible times${errorUrl(7)}`);
    }
    this.cfgStyle?.init?.();
  }

  /**
   * dump dcd and xtc: the style's init_style (Dump::init, src/dump.cpp at b332f28: init_style comes first). write_dump's
   * snapshot (native's dump ID WRITE_DUMP) leaves out the checks of the output interval.
   */
  /**
   * write_dump's init before its one snapshot (src/write_dump.cpp at b332f28 calls dump->init()): the file checks of
   * openFile (dump cfg: one snapshot per file unless write_dump overrides it), the cfg helper's own (dump cfg/uef: its
   * fix) and dump dcd and xtc's init_style.
   */
  initWriteDump(): void {
    this.openFile();
    this.cfgStyle?.init?.();
    if (this.trajStyle) this.initTraj(true);
  }

  initTraj(writeDump: boolean): void {
    this.trajStyle?.init({ writeDump, everyTime: this.everyTime, every: this.everyVar ? 0 : this.every, sortById: this.sort === 'id', flush: this.flush });
  }

  constructor(private sys: System, readonly id: string, readonly group: string, readonly style: DumpStyle, every: number | string, readonly file: string, cols: string[], private readonly multifileOverride = false,
    /** A COMPRESS style (atom/gz, custom/zstd, ...): its name, for dump_modify compression_level and the messages. */
    private readonly gzStyle: string | null = null) {
    // dump.html: "If the filename ends with ".gz", the dump file (or files, if "*" or "%" is also used) is written in
    // gzipped format." As native LAMMPS does (src/dump.cpp at b332f28, Dump::openfile): a name with a compression
    // extension is written through the compressor, and dump_modify append is then ignored (the file is rewritten).
    // The browser engine writes gzip and Zstandard: each snapshot is one gzip member (stored blocks) or one Zstandard
    // frame (raw blocks, zstd.ts), and the members or frames of a file decompress to the whole dump. A COMPRESS style
    // writes its own format (src/COMPRESS/dump_atom_zstd.cpp at b332f28: "if (!compressed) error->all(FLERR, "Dump
    // atom/zstd only writes compressed files")") whatever the compressed name; a .zst name with any other style is
    // written through the zstd tool by native, with its default content checksum.
    const ext = compressExtension(file);
    if (gzStyle) {
      if (!ext) throw new StyleError(`Dump ${gzStyle} only writes compressed files`);
      this.gz = true;
      this.zst = gzStyle.endsWith('/zstd');
    } else if (ext && style !== 'dcd' && style !== 'xtc') {
      // (dump dcd and xtc stop with their own "Invalid dump ... filename")
      if (style === 'image' || style === 'vtk' || style === 'grid/vtk' || (ext !== 'gz' && ext !== 'zst')) {
        throw new StyleError(`dump ${id}: a compressed (.${ext}) ${style} dump file is not supported by the browser engine; ${style === 'image' || style === 'vtk' || style === 'grid/vtk' ? 'use a plain file name' : 'use .gz, .zst or a plain file name'}`);
      }
      this.gz = true;
      this.zst = ext === 'zst';
    }
    this.groupBit = sys.groupBit(group);
    if (typeof every === 'string') this.everyVar = every;
    this.every = typeof every === 'number' ? every : 0;
    this.nevery = this.every;
    // dump.html: "For the dcd, extxyz, grid/vtk, xtc, xyz, and molfile styles, sorting by atom ID or grid ID is on by
    // default."
    this.sort = style === 'xyz' || style === 'extxyz' || style === 'grid/vtk' || style === 'dcd' || style === 'xtc' ? 'id' : 'off';
    if (style === 'image') {
      // dump image is loaded on first use (lazyStyles.ts group dump_image, slot 'image'), out of the engine's first chunk
      this.imageDump = gplHook<{ createDump: ImageDumpFactory }>('image').createDump(sys, id, group, file, cols, multifileOverride, typeof every === 'number' ? every : 0);
      this.columns = [];
    } else if (style === 'atom') {
      // As native LAMMPS does (src/dump_atom.cpp at b332f28, DumpAtom::DumpAtom): narg must be 5, so any extra word is illegal
      if (cols.length) throw new StyleError('Illegal dump atom command');
      this.columns = ['id', 'type', 'xs', 'ys', 'zs'];
    } else if (style === 'xyz' || style === 'extxyz') {
      // As native LAMMPS does (src/dump_xyz.cpp at b332f28, DumpXYZ::DumpXYZ): narg must be 5
      if (cols.length) throw new StyleError(`Illegal dump ${style} command`);
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
    } else if (style === 'dcd' || style === 'xtc') {
      // dump dcd and xtc: the helper checks the arguments and the file name, and creates the file (as native LAMMPS does,
      // src/EXTRA-DUMP/dump_dcd.cpp and dump_xtc.cpp at b332f28: the constructor opens it)
      this.columns = [];
      const key = sys.resolveFile(file);
      this.trajStyle = gplHook<DumpTrajFactory>(style)({
        sys,
        file,
        groupCount: () => {
          let n = 0;
          for (let i = 0; i < sys.state.n; i++) if (sys.state.mask[i] & this.groupBit) n++;
          return n;
        },
        create: () => sys.writeFileAt(key, '', false),
        writeBytes: (bytes, at) => sys.writeFileAt(key, bytes, at === undefined, at),
      }, cols);
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
      // (dump cfg/uef: the same helper with the box of fix nvt/uef or npt/uef, HOOKS 'cfg/uef')
      if (style === 'cfg' || style === 'cfg/uef') this.cfgStyle = gplHook<DumpCfgFactory>(style)(sys, out);
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
    // As native utils::expand_args does (src/utils.cpp at b332f28): a wildcard of a compute or fix that does not exist is
    // left as it is, and the column check then finds no valid attribute
    const src = kind === 'c' ? this.sys.computes.find((x) => x.id === id) : this.sys.fixes.find((x) => x.id === id);
    if (!src) return [c];
    const n = src.sizePeratomCols;
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
    if (!m) throw new StyleError(`Invalid attribute ${c} in dump ${this.style} command`);
    // As native LAMMPS does (src/dump_custom.cpp DumpCustom::parse_fields at b332f28): the reference exists, then its
    // per-atom data, its vector or array shape and the column range
    const [, kind, name, idx] = m;
    const st = this.style;
    if (kind === 'c' || kind === 'f') {
      const what = kind === 'c' ? 'compute' : 'fix';
      const src = kind === 'c' ? this.sys.compute(name, `Could not find dump ${st} compute ID: ${name}`) : this.sys.fix(name, `Could not find dump ${st} fix ID: ${name}`);
      const verb = kind === 'c' ? 'calculate' : 'compute';
      if (!src.peratomFlag) throw new StyleError(`Dump ${st} ${what} ${name} does not compute per-atom info`);
      if (idx === undefined && src.sizePeratomCols > 0) throw new StyleError(`Dump ${st} ${what} ${name} does not ${verb} per-atom vector`);
      if (idx !== undefined && src.sizePeratomCols === 0) throw new StyleError(`Dump ${st} ${what} ${name} does not ${verb} per-atom array`);
      if (idx !== undefined && Number(idx) > src.sizePeratomCols) throw new StyleError(`Dump ${st} ${what} ${name} vector is accessed out-of-range${errorUrl(20)}`);
    } else {
      const v = this.sys.vars.get(name);
      if (!v) throw new StyleError(`Could not find dump ${st} variable name ${name}`);
      if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Dump ${st} variable ${name} is not atom-style variable`);
    }
  }

  /**
   * A c_ID, f_ID or v_name attribute of dump_modify thresh, checked as native LAMMPS does (src/dump_custom.cpp
   * DumpCustom::modify_param at b332f28): it exists, has per-atom data of the shape the word asks for, and the column
   * is in range.
   */
  private threshRefCheck(word: string): void {
    const m = /^([cfv])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(word);
    if (!m) return;
    const [, kind, name, idx] = m;
    if (kind === 'v') {
      const v = this.sys.vars.get(name);
      if (!v) throw new StyleError(`Could not find dump modify variable name: ${name}`);
      if (v.style !== 'atom' && v.style !== 'atomfile') throw new StyleError(`Dump modify variable ${name} is not atom-style variable`);
      return;
    }
    const what = kind === 'c' ? 'compute' : 'fix';
    const src = kind === 'c' ? this.sys.compute(name, `Could not find dump modify compute ID ${name}`) : this.sys.fix(name, `Could not find dump modify fix ID: ${name}`);
    const k = idx === undefined ? 0 : Number(idx);
    if (!src.peratomFlag) throw new StyleError(`Dump modify ${what} ID ${name} does not compute per-atom info`);
    if (k === 0 && src.sizePeratomCols > 0) throw new StyleError(`Dump modify ${what} ID ${name} does not compute per-atom vector`);
    if (k > 0 && src.sizePeratomCols === 0) throw new StyleError(`Dump modify ${what} ID ${name} does not compute per-atom array`);
    if (k > 0 && k > src.sizePeratomCols) throw new StyleError(`Dump modify ${what} ID ${name} vector is not large enough`);
  }

  /** dump_modify keywords. */
  modify(args: string[]): void {
    const yesno = (w: string | undefined, key: string) => {
      // As native LAMMPS does (src/dump.cpp and src/dump_custom.cpp at b332f28, modify_param): "if (iarg+2 > narg)
      // utils::missing_cmd_args(FLERR, "dump_modify append", error)", and "if (narg < 2) utils::missing_cmd_args(FLERR,
      // "dump_modify triclinic/general",error)" for the yes/no keywords with no value
      if (w === undefined) throw new StyleError(`Illegal dump_modify ${key} command: missing argument(s)`);
      return logical(w);
    };
    for (let k = 0; k < args.length;) {
      const key = args[k];
      const v = args[k + 1];
      // dump dcd and xtc: their own keywords (unwrap; xtc precision, sfactor, tfactor)
      if (this.trajStyle) {
        const n = this.trajStyle.modify(args, k);
        if (n > 0) { k += n; continue; }
      }
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
        // DumpGrid::modify_param returns 0 for a word it does not read (src/dump_grid.cpp at b332f28)
        if (!GRID_DUMP_KEYWORDS.has(key)) throw new StyleError(`Unknown dump_modify keyword: ${key}`);
      }
      // dump image/movie-only dump_modify keywords (dump_image.html, "dump_modify options for dump
      // image/movie"): acolor, adiam, bcolor, bdiam, backcolor, boxcolor, color.
      if (this.imageDump) {
        const n = this.imageDump.modifyKeyword(args, k);
        if (n > 0) { k += n; continue; }
      }
      if (!this.vtkStyle && !this.gridStyle && !BASE_KEYWORDS.has(key) && !STYLE_KEYWORDS[this.style]?.has(key)
        && !(key === 'compression_level' && this.gzStyle) && !(key === 'checksum' && this.gzStyle?.endsWith('/zstd'))) {
        throw new StyleError(`Unknown dump_modify keyword: ${key}`);
      }
      // As native LAMMPS does (src/dump.cpp Dump::modify_params at b332f28): each base keyword without its value is
      // "Illegal dump_modify {keyword} command: missing argument(s)" (skip: "Illegal dump_modify command")
      if (v === undefined && BASE_KEYWORDS.has(key)) throw key === 'skip' ? new StyleError('Illegal dump_modify command') : missingArgs(`dump_modify ${key}`);
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
        case 'flush': this.flush = yesno(v, key); k += 2; break;
        // As native LAMMPS does (src/dump.cpp Dump::modify_params at b332f28): "if (buffer_flag && buffer_allow == 0)", and
        // dump dcd and xtc leave buffer_allow 0
        case 'buffer':
          if (yesno(v, key) && this.trajStyle) throw new StyleError('Dump_modify buffer yes not allowed for this style');
          k += 2;
          break;
        case 'balance': case 'thermo': yesno(v, key); k += 2; break;
        case 'delay': this.delay = intArg(v, key); k += 2; break;
        case 'compression_level': {
          // As native LAMMPS does (src/COMPRESS/dump_custom_gz.cpp and gz_file_writer.cpp at b332f28): a /gz style takes a
          // level in [Z_DEFAULT_COMPRESSION, Z_BEST_COMPRESSION] = [-1, 9] while its file is not open. The level changes
          // the bytes, not the decompressed dump; the engine writes stored blocks at every level.
          if (!this.gzStyle) throw new StyleError(`unsupported dump_modify keyword '${key}'`);
          if (v === undefined) throw new StyleError('Illegal dump_modify command');
          const level = intArg(v, key);
          if (this.opened) throw new StyleError('Illegal dump_modify command: Compression level can not be changed while file is open');
          // src/COMPRESS/zstd_file_writer.cpp setCompressionLevel: [ZSTD_minCLevel(), ZSTD_maxCLevel()] = [-131072, 22]
          const [lo, hi] = this.zst ? [-131072, 22] : [-1, 9];
          if (level < lo || level > hi) throw new StyleError(`Illegal dump_modify command: Compression level must in the range of [${lo}, ${hi}]`);
          k += 2;
          break;
        }
        // As native LAMMPS does (src/COMPRESS/dump_atom_zstd.cpp modify_param and zstd_file_writer.cpp setChecksum at
        // b332f28): checksum yes/no for a /zstd style while its file is not open (default yes)
        case 'checksum': {
          if (v === undefined) throw new StyleError('Illegal dump_modify command');
          const on = yesno(v, key);
          if (this.opened) throw new StyleError('Illegal dump_modify command: Checksum flag can not be changed while file is open');
          this.zstChecksum = on;
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
          this.everyTime = false;
          this.nextStep = -1;
          k += 2;
          break;
        }
        case 'every/time': {
          // As native LAMMPS does (src/dump.cpp Dump::modify_params at b332f28): a v_name or a delta > 0 ("Invalid dump_modify
          // every/time argument: {}"). The browser engine writes no snapshots by simulation time; dump dcd and xtc keep the
          // setting so that their init stops as native's does ("Cannot use every/time setting for dump dcd").
          if (!this.trajStyle) throw new StyleError(`unsupported dump_modify keyword '${key}'`);
          if (!v!.startsWith('v_')) {
            const delta = numArg(v, key);
            if (delta <= 0.0) throw new StyleError(`Invalid dump_modify every/time argument: ${fmtShortest(delta, false)}`);
          }
          this.everyTime = true;
          k += 2;
          break;
        }
        case 'sort': {
          if (v === 'off') this.sort = 'off';
          else if (v === 'id') {
            // As native LAMMPS does (src/dump_local.cpp at b332f28, init_style)
            if (this.style === 'local') throw new StyleError('Dump local cannot sort by atom ID');
            this.sort = 'id';
          }
          else {
            const n = intArg(v, key);
            if (n === 0) throw new StyleError('dump_modify sort column out of range');
            // As native LAMMPS does (src/dump.cpp at b332f28, Dump::init; dump dcd and xtc stop at their init_style first)
            if (!this.trajStyle && Math.abs(n) > this.columns.length) throw new StyleError(`Dump sort column index ${Math.abs(n)} is invalid`);
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
          // Dump::modify_params: the format string is missing as the keyword's value would be
          if (f === undefined) throw missingArgs('dump_modify format');
          // the column formats are DumpCustom's and DumpLocal's (Dump::modify_params hands them to modify_param)
          if (v !== 'line' && !COLUMN_FORMAT_STYLES.has(this.style)) throw new StyleError(`Unknown dump_modify format keyword: ${v ?? ''}`);
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
          // As native LAMMPS does (src/dump.cpp Dump::modify_params at b332f28): default, or a column number (negative
          // counts from the end) or keyword and a name; a column that is not there (xyz and extxyz have none) is
          // "Incorrect dump_modify arguments"
          if (k + 2 > args.length) throw missingArgs('dump_modify colname');
          if (v === 'default') { this.colnames.clear(); k += 2; break; }
          if (k + 3 > args.length) throw missingArgs('dump_modify colname');
          const name = args[k + 2];
          const ncol = this.style === 'xyz' || this.style === 'extxyz' ? 0 : this.columns.length;
          let icol = -1;
          if (/^[+-]?\d+$/.test(v!)) {
            icol = inumeric(v);
            if (icol < 0) icol = ncol + icol + 1;
            icol--;
          } else if (ncol) icol = this.columns.indexOf(v!);
          if (icol < 0 || icol >= ncol) throw new StyleError(`Incorrect dump_modify arguments: colname ${v} ${name}`);
          this.colnames.set(icol, name);
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
          this.threshRefCheck(v!);
          const op = args[k + 2], val = args[k + 3];
          if (!['<', '<=', '>', '>=', '==', '!=', '|^'].includes(op ?? '')) throw new StyleError('Invalid dump_modify thresh operator');
          this.thresh.push({ col: v!, op: op!, value: val === 'LAST' ? 'LAST' : numArg(val, key) });
          k += 4;
          break;
        }
        // src/dump_local.cpp DumpLocal::modify_param at b332f28
        case 'label': if (!v) throw missingArgs('dump_modify label'); this.label = v; k += 2; break;
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
        case 'types': {
          // As native LAMMPS does (src/dump_xyz.cpp at b332f28, modify_param): numeric drops the type names (element or
          // labels), labels takes them from the label map
          if (v === undefined) throw new StyleError('Illegal dump_modify types command: missing argument(s)');
          if (v === 'numeric') this.element = [];
          else if (v === 'labels') {
            const lm = this.sys.labels;
            if (!lm.defined) throw new StyleError("Label map must be defined when using 'types labels'");
            const n = this.sys.state.ntypes;
            this.element = Array.from({ length: n }, (_, t) => lm.labelOf('atom', t + 1) ?? '');
          } else throw new StyleError(`Unknown option ${v} for dump_modify 'types' keyword`);
          k += 2;
          break;
        }
        case 'skip': {
          // dump_modify.html: "skip arg = v_name"; Dump::write skips the snapshot when the equal-style variable is not 0
          if (v === undefined || !v.startsWith('v_')) throw new StyleError('Illegal dump_modify command');
          this.skipVar = v.slice(2);
          k += 2;
          break;
        }
        case 'refresh': {
          // dump_modify.html: "refresh arg = c_ID = compute ID that supports a refresh operation"; called after each
          // snapshot is written (Dump::write). As native: one refresh per dump, a c_ID without index.
          if (v === undefined) throw new StyleError('Illegal dump_modify refresh command: missing argument(s)');
          if (!/^c_[^[\]]+$/.test(v)) throw new StyleError('Illegal dump_modify command');
          if (this.refreshId !== null) throw new StyleError('Dump_modify can only have one refresh');
          this.refreshId = v.slice(2);
          k += 2;
          break;
        }
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
    this.sys.writeFileAt(name, !this.gz ? text : this.zst ? zstdText(text, this.zstChecksum) : gzipText(text), append);
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

  /**
   * dump_modify skip (src/dump.cpp at b332f28, Dump::init and Dump::write): the variable must exist and be equal-style,
   * and a snapshot is not written when it evaluates to non-zero (after the atoms are chosen, before any file is opened).
   */
  private skipSnapshot(): boolean {
    if (this.skipVar === null) return false;
    const v = this.sys.vars.get(this.skipVar);
    if (!v) throw new StyleError(`Dump skip variable ${this.skipVar} not found`);
    if (v.style !== 'equal' && v.style !== 'internal' && v.style !== 'timer') throw new StyleError(`Variable ${this.skipVar} for dump skip is invalid style`);
    return this.sys.equalVariable(this.skipVar) !== 0;
  }

  /** dump_modify refresh (Dump::write): the compute's refresh after a snapshot ("Dump could not find refresh compute ID"). */
  private refreshAfter(): void {
    if (this.refreshId === null) return;
    const c = this.sys.computes.find((x) => x.id === this.refreshId) as { refresh?: () => void } | undefined;
    if (!c) throw new StyleError(`Dump could not find refresh compute ID ${this.refreshId}`);
    c.refresh?.();
  }

  write(): void {
    if (this.writeSnapshot_()) this.refreshAfter();
  }

  /** One snapshot; false when dump_modify skip skipped it. */
  private writeSnapshot_(): boolean {
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
      if (this.skipSnapshot()) return false;
      this.imageDump.write(s, name, this, chosen);
      return true;
    }
    if (this.vtkStyle) {
      if (this.skipSnapshot()) return false;
      this.vtkStyle.write(step);
      return true;
    }
    if (this.gridStyle) {
      if (this.skipSnapshot()) return false;
      this.gridStyle.write(step);
      return true;
    }
    if (this.trajStyle) {
      // Dump::write: count (the group's atoms), the skip test, then the atoms in ID order and the style's header and frame
      const rows = this.chooseRows();
      if (this.skipSnapshot()) return false;
      rows.sort((a, b) => s.id[a] - s.id[b]);
      const g = sys.geom;
      this.trajStyle.write({
        step, dt: s.dt, natoms: rows.length, header: this.header,
        box: { triclinic: g.triclinic, lo: s.box.lo, hi: s.box.hi, xprd: g.lx, yprd: g.ly, zprd: g.lz, xy: g.xy, xz: g.xz, yz: g.yz },
        coords: this.trajCoords(rows, this.trajStyle.unwrap),
      });
      this.opened = true;
      return true;
    }
    sys.forces();
    if (this.style === 'local') {
      if (this.skipSnapshot()) return false;
      this.writeLocal(step);
      return true;
    }
    const g = sys.geom;
    // dump_modify.html: "It can only be used with a value of *yes* if the" simulation box was created as a general
    // triclinic box. Only the atom and custom styles take it (Howto_triclinic.html lists dump atom, dump custom).
    if (this.triclinicGeneral) {
      if (!s.box.general) throw new StyleError(`dump ${this.id}: dump_modify triclinic/general yes needs a general triclinic box (create_box NULL, or a general triclinic data file)`);
      if (this.style !== 'atom' && this.style !== 'custom') throw new StyleError(`dump ${this.id}: dump_modify triclinic/general applies only to the atom and custom styles`);
    }
    const keep = this.chooseRows();
    if (this.skipSnapshot()) return false;
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
      return true;
    }
    const lines: string[] = [];
    const time = s.time + (s.step - s.timeStep) * s.dt;
    if (this.style === 'xyz' || this.style === 'extxyz') {
      if (this.style === 'extxyz' && !this.element.length) throw new StyleError(EXTXYZ_NO_NAMES);
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
    return true;
  }


  /**
   * The coordinates dump dcd and xtc pack (src/EXTRA-DUMP/dump_dcd.cpp and dump_xtc.cpp pack at b332f28): with dump_modify
   * pbc, Dump::write first remaps a copy of x and the image counts into the box; with unwrap, "x[i][0] + ix * xprd + iy *
   * xy + iz * xz", "x[i][1] + iy * yprd + iz * yz", "x[i][2] + iz * zprd" (the xy, xz and yz terms for a triclinic box only).
   */
  private trajCoords(rows: number[], unwrap: boolean): Float64Array {
    const s = this.sys.state, g = this.sys.geom;
    const out = new Float64Array(3 * rows.length);
    const x = new Float64Array(3), im = new Int32Array(3);
    for (let k = 0; k < rows.length; k++) {
      const i = rows[k];
      for (let d = 0; d < 3; d++) { x[d] = s.x[3 * i + d]; im[d] = s.image[3 * i + d]; }
      if (this.pbc) g.remap(x, im, 0);
      if (!unwrap) {
        out[3 * k] = x[0]; out[3 * k + 1] = x[1]; out[3 * k + 2] = x[2];
      } else if (g.triclinic) {
        out[3 * k] = x[0] + im[0] * g.lx + im[1] * g.xy + im[2] * g.xz;
        out[3 * k + 1] = x[1] + im[1] * g.ly + im[2] * g.yz;
        out[3 * k + 2] = x[2] + im[2] * g.lz;
      } else {
        out[3 * k] = x[0] + im[0] * g.lx;
        out[3 * k + 1] = x[1] + im[1] * g.ly;
        out[3 * k + 2] = x[2] + im[2] * g.lz;
      }
    }
    return out;
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

// the dump_modify values, read as native's utils::inumeric and utils::numeric (src/dump.cpp modify_params at b332f28)
const intArg = (w: string | undefined, _key: string): number => inumeric(w);

const numArg = (w: string | undefined, _key: string): number => numeric(w);

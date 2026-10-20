/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the binary restart reader of src/read_restart.cpp
 * (ReadRestart::command, header, type_arrays, force_fields, file_layout, magic_string, endian, format_revision,
 * check_eof_magic, read_int, read_string, read_double_vec), src/write_restart.cpp and src/lmprestart.h (the flags,
 * ENDIAN and FORMAT_REVISION), src/group.cpp (Group::read_restart), src/modify.cpp (Modify::read_restart: the fix
 * entries), src/domain.cpp (Domain::remap), src/atom_vec.cpp (AtomVec::unpack_restart) with the fields of
 * src/atom_vec_*.cpp, MOLECULE/atom_vec_molecular.cpp and MOLECULE/atom_vec_full.cpp, and the per-style read_restart of
 * src/KSPACE/pair_lj_cut_coul_long.cpp, src/pair_lj_cut.cpp, src/SPIN/pair_spin_exchange.cpp, src/MOLECULE/bond_harmonic.cpp
 * and src/MOLECULE/angle_harmonic.cpp, and the write_restart layouts of the other styles listed below (each cites its file).
 * LAMMPS is Copyright (2003) Sandia Corporation and the LAMMPS development team,
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * What it reads: a serial (one-file, no "%") binary restart file of revision 3, written by native LAMMPS on this
 * platform (little-endian, 32-bit imageint and tagint, 64-bit bigint), from one or more processes. The box (orthogonal
 * and triclinic), the header settings, the masses, the groups, the per-atom attributes of atom_style atomic, charge,
 * molecular, full, sphere and spin, and the topology (bonds, angles, dihedrals, impropers) of the molecular styles.
 * Pair styles: lj/cut, lj/cut/coul/long, lj/cut/coul/cut, lj/cut/coul/debye, lj/cut/coul/dsf, coul/cut, coul/long,
 * lj/charmm/coul/long, lj/charmm/coul/charmm, morse, buck, buck/coul/long, born, soft, yukawa, lj96/cut, lj/expand,
 * lj/smooth/linear, table (settings only: its tables are not stored), spin/exchange, the hybrid styles (their sub-style settings only: pair_coeff must be re-issued,
 * as native LAMMPS stores no sub-style coefficients), and the NO_PAIR marker. Bond styles: harmonic, fene, morse,
 * class2, gromos; angle styles: harmonic, charmm, cosine, cosine/squared, class2; dihedral styles: harmonic, charmm,
 * opls, multi/harmonic; improper styles: harmonic, cvff (umbrella is read too, but has no oracle case: the engine's
 * umbrella energy differs from native on a fresh-run geometry, see tests/oracle).
 * The fix state of fix nvt (its thermostat chain). Any other fix state is a marker that refuses the definition of that
 * fix after read_restart. Anything else is refused by name (StyleError).
 *
 * The file's values go into the engine's own state (SimState) and style objects, through the public engine API.
 */
import { emptyState, emptyTopology, pushTopo } from '../../engine/atoms';
import { makeBox } from '../../engine/domain';
import { UNIT_SYSTEMS, isUnitStyle } from '../../engine/units';
import { PAIR_STYLES, BOND_STYLES, ANGLE_STYLES, DIHEDRAL_STYLES, IMPROPER_STYLES } from '../../engine/styles';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import { StyleError, type Bonded, type Pair, type StyleContext } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { FixRecord } from '../../engine/fix/restartRecord';
import type { NHRestart } from '../../engine/fix/nh';
import type { AtomStyle, BoundaryStyle, SimBox, SimState, Topology } from '../../engine/types';

/** The first 16 bytes of every binary restart file (lmprestart.h MAGIC_STRING, with its terminating NUL). */
export const NATIVE_RESTART_MAGIC = 'LammpS RestartT\u0000';

const ENDIAN = 0x0001;
const FORMAT_REVISION = 3;
const MAX_GROUP = 32;
/** lmprestart.h enum order (the flag numbers of the file). */
const FLAG_NAMES = [
  'VERSION', 'SMALLINT', 'TAGINT', 'BIGINT', 'UNITS', 'NTIMESTEP', 'DIMENSION', 'NPROCS', 'PROCGRID',
  'NEWTON_PAIR', 'NEWTON_BOND', 'XPERIODIC', 'YPERIODIC', 'ZPERIODIC', 'BOUNDARY', 'ATOM_STYLE', 'NATOMS', 'NTYPES',
  'NBONDS', 'NBONDTYPES', 'BOND_PER_ATOM', 'NANGLES', 'NANGLETYPES', 'ANGLE_PER_ATOM', 'NDIHEDRALS', 'NDIHEDRALTYPES',
  'DIHEDRAL_PER_ATOM', 'NIMPROPERS', 'NIMPROPERTYPES', 'IMPROPER_PER_ATOM', 'TRICLINIC', 'BOXLO', 'BOXHI', 'XY', 'XZ',
  'YZ', 'SPECIAL_LJ', 'SPECIAL_COUL', 'MASS', 'PAIR', 'BOND', 'ANGLE', 'DIHEDRAL', 'IMPROPER', 'MULTIPROC', 'MPIIO',
  'PROCSPERFILE', 'PERPROC', 'IMAGEINT', 'BOUNDMIN', 'TIMESTEP', 'ATOM_ID', 'ATOM_MAP_STYLE', 'ATOM_MAP_USER',
  'ATOM_SORTFREQ', 'ATOM_SORTBIN', 'COMM_MODE', 'COMM_CUTOFF', 'COMM_VEL', 'NO_PAIR', 'EXTRA_BOND_PER_ATOM',
  'EXTRA_ANGLE_PER_ATOM', 'EXTRA_DIHEDRAL_PER_ATOM', 'EXTRA_IMPROPER_PER_ATOM', 'EXTRA_SPECIAL_PER_ATOM',
  'ATOM_MAXSPECIAL', 'NELLIPSOIDS', 'NLINES', 'NTRIS', 'NBODIES', 'ATIME', 'ATIMESTEP', 'LABELMAP',
  'TRICLINIC_GENERAL', 'ROTATE_G2R', 'ATOM_MAXEXCHANGE',
] as const;
const F = Object.fromEntries(FLAG_NAMES.map((n, i) => [n, i])) as Record<(typeof FLAG_NAMES)[number], number>;

/** The atom fields each readable atom_style stores after its base attributes (atom_vec_*.cpp fields_restart). */
const ATOM_FIELDS: Record<string, readonly string[]> = {
  atomic: [],
  charge: ['q'],
  molecular: ['molecule', 'topo'],
  full: ['q', 'molecule', 'topo'],
  sphere: ['radius', 'rmass', 'omega'],
  spin: ['sp'],
};

/** lmprestart.h / lmptype.h: 32-bit smallint, imageint and tagint, 64-bit bigint (the default LAMMPS build). */
const IMGBITS = 10;
const IMGMAX = 512;
const IMGMASK = 1023;

const MIX_RULE = ['geometric', 'arithmetic', 'sixthpower'] as const;
const BOUNDARY_CODE: BoundaryStyle[] = ['p', 'f', 's', 'm'];

/** A binary cursor over the file (one byte per character of the latin1 text). */
class Bin {
  readonly b: Uint8Array;
  readonly dv: DataView;
  pos = 0;

  constructor(text: string, private readonly name: string) {
    this.b = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) this.b[i] = text.charCodeAt(i) & 0xff;
    this.dv = new DataView(this.b.buffer);
  }

  fail(msg: string): never {
    throw new StyleError(`native restart ${this.name}: ${msg}`);
  }

  take(n: number): number {
    if (n < 0 || this.pos + n > this.b.length) this.fail('the file ends early (truncated or corrupt)');
    const p = this.pos;
    this.pos += n;
    return p;
  }

  int(): number {
    return this.dv.getInt32(this.take(4), true);
  }

  /** A 64-bit integer (bigint, or an int stored in a double slot as its bit pattern: ubuf in lmptype.h). */
  i64(): number {
    return Number(this.dv.getBigInt64(this.take(8), true));
  }

  dbl(): number {
    return this.dv.getFloat64(this.take(8), true);
  }

  ints(n: number): number[] {
    return Array.from({ length: n }, () => this.int());
  }

  dbls(n: number): number[] {
    return Array.from({ length: n }, () => this.dbl());
  }

  /** n characters (a C string: the text ends at its first NUL). */
  chars(n: number): string {
    if (n < 0) this.fail('illegal string size (corrupt file)');
    const p = this.take(n);
    let s = '';
    for (let i = 0; i < n; i++) {
      const c = this.b[p + i];
      if (c === 0) break;
      s += String.fromCharCode(c);
    }
    return s;
  }

  /** write_string: an int size (the characters including the terminating NUL), then the characters. */
  str(): string {
    return this.chars(this.int());
  }
}

/** The header values the reader keeps (lmprestart.h flags; a missing flag keeps its default). */
interface Header {
  version: string;
  units: string;
  ntimestep: number;
  dimension: number;
  nprocs: number;
  newtonBond: number;
  periodic: [number, number, number];
  boundary: number[];
  boundmin: number[] | null;
  atomStyle: string;
  atomStyleArgs: string[];
  natoms: number;
  ntypes: number;
  nbonds: number;
  nbondtypes: number;
  nangles: number;
  nangletypes: number;
  ndihedrals: number;
  ndihedraltypes: number;
  nimpropers: number;
  nimpropertypes: number;
  triclinic: number;
  boxlo: number[];
  boxhi: number[];
  xy: number;
  xz: number;
  yz: number;
  specialLJ: number[];
  specialCoul: number[];
  dt: number;
  atomId: number;
  commMode: number;
  commCutoff: number;
  commVel: number;
  atime: number;
  atimestep: number;
}

/** One pair style read from the file: the arguments that rebuild it, its coefficients, and its pair_modify flags. */
export interface NativePairStyle {
  name: string;
  settings: string[];
  coeffs: string[][];
  modify: PairFlags;
  /** hybrid styles: the sub-styles in order, each with its pair_modify flags */
  subs?: HybridSub[];
}

/** One bonded style (bond or angle) read from the file: its class and its per-type coefficient arguments. */
export interface NativeBondedStyle {
  name: string;
  coeffs: string[][];
}

const need = (cond: boolean, bin: Bin, msg: string): void => {
  if (!cond) bin.fail(msg);
};

/** Reads the header flags until the -1 terminator (ReadRestart::header). */
const readHeader = (b: Bin): Header => {
  const h: Header = {
    version: '', units: 'lj', ntimestep: 0, dimension: 3, nprocs: 1, newtonBond: 1, periodic: [1, 1, 1],
    boundary: [0, 0, 0, 0, 0, 0], boundmin: null, atomStyle: 'atomic', atomStyleArgs: [], natoms: 0, ntypes: 0,
    nbonds: 0, nbondtypes: 0, nangles: 0, nangletypes: 0, ndihedrals: 0, ndihedraltypes: 0, nimpropers: 0, nimpropertypes: 0, triclinic: 0,
    boxlo: [0, 0, 0], boxhi: [1, 1, 1], xy: 0, xz: 0, yz: 0, specialLJ: [0, 0, 0], specialCoul: [0, 0, 0],
    dt: 0, atomId: 1, commMode: 0, commCutoff: 0, commVel: 0, atime: 0, atimestep: 0,
  };
  let flag = b.int();
  while (flag >= 0) {
    switch (flag) {
      case F.VERSION: h.version = b.str(); break;
      case F.SMALLINT: need(b.int() === 4, b, 'smallint size differs from this build of LAMMPS (32 bits expected)'); break;
      case F.IMAGEINT: need(b.int() === 4, b, 'imageint size differs from this build of LAMMPS (32 bits expected)'); break;
      case F.TAGINT: need(b.int() === 4, b, 'tagint size differs from this build of LAMMPS (32 bits expected)'); break;
      case F.BIGINT: need(b.int() === 8, b, 'bigint size differs from this build of LAMMPS (64 bits expected)'); break;
      case F.UNITS: h.units = b.str(); break;
      case F.NTIMESTEP: h.ntimestep = b.i64(); break;
      case F.DIMENSION: h.dimension = b.int(); break;
      case F.NPROCS: h.nprocs = b.int(); break;
      case F.PROCGRID: b.int(); b.ints(3); break;
      case F.NEWTON_PAIR: b.int(); break;
      case F.NEWTON_BOND: h.newtonBond = b.int(); break;
      case F.XPERIODIC: h.periodic[0] = b.int(); break;
      case F.YPERIODIC: h.periodic[1] = b.int(); break;
      case F.ZPERIODIC: h.periodic[2] = b.int(); break;
      case F.BOUNDARY: b.int(); h.boundary = b.ints(6); break;
      case F.BOUNDMIN: b.int(); h.boundmin = b.dbls(6); break;
      case F.ATOM_STYLE: {
        h.atomStyle = b.str();
        const nargs = b.int();
        for (let i = 0; i < nargs; i++) h.atomStyleArgs.push(b.str());
        break;
      }
      case F.NATOMS: h.natoms = b.i64(); break;
      case F.NTYPES: h.ntypes = b.int(); break;
      case F.NBONDS: h.nbonds = b.i64(); break;
      case F.NBONDTYPES: h.nbondtypes = b.int(); break;
      case F.BOND_PER_ATOM: b.int(); break;
      case F.NANGLES: h.nangles = b.i64(); break;
      case F.NANGLETYPES: h.nangletypes = b.int(); break;
      case F.ANGLE_PER_ATOM: b.int(); break;
      case F.NDIHEDRALS: h.ndihedrals = b.i64(); break;
      case F.NDIHEDRALTYPES: h.ndihedraltypes = b.int(); break;
      case F.DIHEDRAL_PER_ATOM: b.int(); break;
      case F.NIMPROPERS: h.nimpropers = b.i64(); break;
      case F.NIMPROPERTYPES: h.nimpropertypes = b.int(); break;
      case F.IMPROPER_PER_ATOM: b.int(); break;
      case F.TRICLINIC: h.triclinic = b.int(); break;
      case F.BOXLO: b.int(); h.boxlo = b.dbls(3); break;
      case F.BOXHI: b.int(); h.boxhi = b.dbls(3); break;
      case F.XY: h.xy = b.dbl(); break;
      case F.XZ: h.xz = b.dbl(); break;
      case F.YZ: h.yz = b.dbl(); break;
      case F.TRICLINIC_GENERAL: need(b.int() === 0, b, 'general triclinic boxes are not readable by the browser engine yet'); break;
      case F.ROTATE_G2R: b.fail('general triclinic boxes are not readable by the browser engine yet');
      case F.SPECIAL_LJ: b.int(); h.specialLJ = b.dbls(3); break;
      case F.SPECIAL_COUL: b.int(); h.specialCoul = b.dbls(3); break;
      case F.TIMESTEP: h.dt = b.dbl(); break;
      case F.ATOM_ID: h.atomId = b.int(); break;
      case F.ATOM_MAP_STYLE: b.int(); break;
      case F.ATOM_MAP_USER: b.int(); break;
      case F.ATOM_SORTFREQ: b.int(); break;
      case F.ATOM_SORTBIN: b.dbl(); break;
      case F.COMM_MODE: h.commMode = b.int(); break;
      case F.COMM_CUTOFF: h.commCutoff = b.dbl(); break;
      case F.COMM_VEL: h.commVel = b.int(); break;
      case F.EXTRA_BOND_PER_ATOM: case F.EXTRA_ANGLE_PER_ATOM: case F.EXTRA_DIHEDRAL_PER_ATOM:
      case F.EXTRA_IMPROPER_PER_ATOM: case F.EXTRA_SPECIAL_PER_ATOM: case F.ATOM_MAXSPECIAL: case F.ATOM_MAXEXCHANGE:
        b.int(); break;
      case F.NELLIPSOIDS: case F.NLINES: case F.NTRIS: case F.NBODIES:
        need(b.i64() === 0, b, 'ellipsoids, lines, triangles and bodies are not readable by the browser engine yet');
        break;
      case F.ATIMESTEP: h.atimestep = b.i64(); break;
      case F.ATIME: h.atime = b.dbl(); break;
      default: b.fail('Invalid flag in header section of restart file');
    }
    flag = b.int();
  }
  return h;
};

/** Group::read_restart: MAX_GROUP slots; a name (or an empty slot) per group, in slot order. */
const readGroups = (b: Bin): (string | null)[] => {
  const ngroup = b.int();
  const names: (string | null)[] = Array.from({ length: MAX_GROUP }, () => null);
  let count = 0;
  for (let i = 0; i < MAX_GROUP; i++) {
    if (count === ngroup) continue;
    const n = b.int();
    if (n < 0 || n > 65536) b.fail('Invalid group name length in restart file');
    if (n) {
      names[i] = b.chars(n);
      count++;
    }
  }
  return names;
};

/** ReadRestart::type_arrays: the per-type masses (label maps are refused). */
const readTypeArrays = (b: Bin, h: Header): number[] | null => {
  let masses: number[] | null = null;
  let flag = b.int();
  while (flag >= 0) {
    if (flag === F.MASS) {
      b.int();
      masses = b.dbls(h.ntypes);
    } else if (flag === F.LABELMAP) {
      b.fail('type label maps are not readable by the browser engine yet');
    } else b.fail('Invalid flag in type arrays section of restart file');
    flag = b.int();
  }
  return masses;
};

/** Flags of a pair style the engine stores per style (pair_modify shift, mix, tail, table and tabinner). */
interface PairFlags {
  mix?: (typeof MIX_RULE)[number];
  shift?: boolean;
  tail?: boolean;
  table?: number;
  tabinnerRsq?: number;
}

/** The settings block of one pair style: its pair_style arguments and its pair_modify flags. */
interface PairSettings {
  args: string[];
  modify: PairFlags;
  /** spin/exchange: the per-pair block needs the global offset flag */
  eOffset?: number;
}

/**
 * One pair style of the file (write_restart_settings, then write_restart). settings reads the settings block;
 * pairs reads the per-pair block (null when the style stores none, as the sub-styles of pair hybrid do). The per-pair
 * values become pair_coeff arguments (type I, type J, then the values in the order of the style's pair_coeff command).
 */
interface PairLayout {
  settings: (b: Bin, ntypes: number) => PairSettings;
  pairs: ((b: Bin, ntypes: number, st: PairSettings) => string[][]) | null;
}

const num = (v: number): string => String(v);

/** The pair_modify mix, shift and tail flags of a settings block (offset_flag, mix_flag, tail_flag). */
const mixOf = (b: Bin, k: number): (typeof MIX_RULE)[number] => {
  if (k < 0 || k > 2) b.fail(`unknown pair mixing rule ${k}`);
  return MIX_RULE[k];
};

/**
 * The (i <= j) loop of a write_restart: a set flag per pair (setflag), then the values of a set pair. read() gets the
 * type pair and returns the pair_coeff arguments after it (the values it reads from the file).
 */
const eachSetPair = (b: Bin, ntypes: number, read: (i: number, j: number) => string[]): string[][] => {
  const out: string[][] = [];
  for (let i = 1; i <= ntypes; i++) {
    for (let j = i; j <= ntypes; j++) {
      if (b.int()) out.push([num(i), num(j), ...read(i, j)]);
    }
  }
  return out;
};

const dbls = (b: Bin, n: number): string[] => Array.from({ length: n }, () => num(b.dbl()));

/** The pair_modify flags of the offset, mix and tail ints that most settings blocks store. */
const flags3 = (b: Bin): PairFlags => {
  const offset = b.int();
  const mix = mixOf(b, b.int());
  const tail = b.int();
  return { shift: offset !== 0, mix, tail: tail !== 0 };
};
const flags2 = (b: Bin): PairFlags => {
  const offset = b.int();
  const mix = mixOf(b, b.int());
  return { shift: offset !== 0, mix };
};

/** The coulomb table of the long-range styles: ncoultablebits (table) and tabinner (its squared value is the inner cutoff). */
const tableFlags = (b: Bin): PairFlags => {
  const bits = b.int();
  const tabinner = b.dbl();
  return { table: bits, tabinnerRsq: tabinner * tabinner };
};

/** A scale factor that the engine cannot store per pair (only fix adapt sets one): refused unless it is 1. */
const unitScale = (b: Bin, name: string, v: number): void => {
  if (v !== 1) b.fail(`pair style ${name} with a per-pair scale factor ${v} is not readable by the browser engine yet`);
};

/** The pair styles whose settings and per-pair blocks are read. Each cites its write_restart in the C++ source. */
const PAIR_LAYOUTS: Record<string, PairLayout> = {
  // pair_lj_cut.cpp: cut_global, offset_flag, mix_flag, tail_flag; per pair eps, sigma, cut
  'lj/cut': {
    settings: (b) => {
      const cut = b.dbl();
      return { args: [num(cut)], modify: flags3(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 3)),
  },
  // KSPACE/pair_lj_cut_coul_long.cpp: cut_lj_global, cut_coul, offset, mix, tail, ncoultablebits, tabinner; per pair eps, sigma, cut_lj
  'lj/cut/coul/long': {
    settings: (b) => {
      const cutLJ = b.dbl(), cutCoul = b.dbl();
      const f = flags3(b);
      return { args: [num(cutLJ), num(cutCoul)], modify: { ...f, ...tableFlags(b) } };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 3)),
  },
  // pair_lj_cut_coul_cut.cpp: cut_lj_global, cut_coul_global, offset, mix, tail; per pair eps, sigma, cut_lj, cut_coul
  'lj/cut/coul/cut': {
    settings: (b) => {
      const cutLJ = b.dbl(), cutCoul = b.dbl();
      return { args: [num(cutLJ), num(cutCoul)], modify: flags3(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // EXTRA-PAIR/pair_lj_cut_coul_debye.cpp (write_restart inherited from lj/cut/coul/cut): cut_lj_global, cut_coul_global,
  // kappa, offset, mix (no tail); the per-pair block of lj/cut/coul/cut
  'lj/cut/coul/debye': {
    settings: (b) => {
      const cutLJ = b.dbl(), cutCoul = b.dbl(), kappa = b.dbl();
      return { args: [num(kappa), num(cutLJ), num(cutCoul)], modify: flags2(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // EXTRA-PAIR/pair_lj_cut_coul_dsf.cpp: alpha, cut_lj_global, cut_coul, offset, mix, tail; per pair eps, sigma, cut_lj
  'lj/cut/coul/dsf': {
    settings: (b) => {
      const alpha = b.dbl(), cutLJ = b.dbl(), cutCoul = b.dbl();
      return { args: [num(alpha), num(cutLJ), num(cutCoul)], modify: flags3(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 3)),
  },
  // pair_coul_cut.cpp: cut_global, offset, mix; per pair scale (always stored), setflag, cut when set
  'coul/cut': {
    settings: (b) => {
      const cut = b.dbl();
      return { args: [num(cut)], modify: flags2(b) };
    },
    pairs: (b, nt) => {
      const out: string[][] = [];
      for (let i = 1; i <= nt; i++) {
        for (let j = i; j <= nt; j++) {
          unitScale(b, 'coul/cut', b.dbl());
          if (b.int()) out.push([num(i), num(j), num(b.dbl())]);
        }
      }
      return out;
    },
  },
  // KSPACE/pair_coul_long.cpp: cut_coul, offset, mix, ncoultablebits, tabinner; per pair setflag, then scale when set
  'coul/long': {
    settings: (b) => {
      const cutCoul = b.dbl();
      const f = flags2(b);
      return { args: [num(cutCoul)], modify: { ...f, ...tableFlags(b) } };
    },
    // a set pair is a pair_coeff I J (the engine's coul/long takes no coefficients but marks the pair as set)
    pairs: (b, nt) => {
      const out: string[][] = [];
      for (let i = 1; i <= nt; i++) {
        for (let j = i; j <= nt; j++) {
          if (b.int()) {
            unitScale(b, 'coul/long', b.dbl());
            out.push([num(i), num(j)]);
          }
        }
      }
      return out;
    },
  },
  // KSPACE/pair_lj_charmm_coul_long.cpp: cut_lj_inner, cut_lj, cut_coul, offset, mix, ncoultablebits, tabinner;
  // per pair eps, sigma, eps14, sigma14
  'lj/charmm/coul/long': {
    settings: (b) => {
      const inner = b.dbl(), outer = b.dbl(), cutCoul = b.dbl();
      const f = flags2(b);
      return { args: [num(inner), num(outer), num(cutCoul)], modify: { ...f, ...tableFlags(b) } };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // MOLECULE/pair_lj_charmm_coul_charmm.cpp: cut_lj_inner, cut_lj, cut_coul_inner, cut_coul, offset, mix;
  // per pair eps, sigma, eps14, sigma14
  'lj/charmm/coul/charmm': {
    settings: (b) => {
      const v = dbls(b, 4);
      return { args: v, modify: flags2(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // pair_morse.cpp: cut_global, offset, mix; per pair d0, alpha, r0, cut
  morse: {
    settings: (b) => {
      const cut = b.dbl();
      return { args: [num(cut)], modify: flags2(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // pair_buck.cpp: cut_global, offset, mix, tail; per pair a, rho, c, cut
  buck: {
    settings: (b) => {
      const cut = b.dbl();
      return { args: [num(cut)], modify: flags3(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // KSPACE/pair_buck_coul_long.cpp: cut_lj_global, cut_coul, offset, mix, tail, ncoultablebits, tabinner; per pair a, rho, c, cut_lj
  'buck/coul/long': {
    settings: (b) => {
      const cutLJ = b.dbl(), cutCoul = b.dbl();
      const f = flags3(b);
      return { args: [num(cutLJ), num(cutCoul)], modify: { ...f, ...tableFlags(b) } };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // pair_born.cpp: cut_global, offset, mix, tail; per pair a, rho, sigma, c, d, cut
  born: {
    settings: (b) => {
      const cut = b.dbl();
      return { args: [num(cut)], modify: flags3(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 6)),
  },
  // pair_soft.cpp: cut_global, mix; per pair prefactor, cut
  soft: {
    settings: (b) => {
      const cut = b.dbl();
      const mix = mixOf(b, b.int());
      return { args: [num(cut)], modify: { mix } };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 2)),
  },
  // pair_yukawa.cpp: kappa, cut_global, offset, mix; per pair a, cut
  yukawa: {
    settings: (b) => {
      const kappa = b.dbl(), cut = b.dbl();
      return { args: [num(kappa), num(cut)], modify: flags2(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 2)),
  },
  // EXTRA-PAIR/pair_lj96_cut.cpp: cut_global, offset, mix, tail; per pair epsilon, sigma, cut
  'lj96/cut': {
    settings: (b) => {
      const cut = b.dbl();
      return { args: [num(cut)], modify: flags3(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 3)),
  },
  // pair_lj_expand.cpp: cut_global, offset, mix, tail; per pair epsilon, sigma, shift, cut
  'lj/expand': {
    settings: (b) => {
      const cut = b.dbl();
      return { args: [num(cut)], modify: flags3(b) };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 4)),
  },
  // EXTRA-PAIR/pair_lj_smooth_linear.cpp: cut_global, mix; per pair epsilon, sigma, cut
  'lj/smooth/linear': {
    settings: (b) => {
      const cut = b.dbl();
      const mix = mixOf(b, b.int());
      return { args: [num(cut)], modify: { mix } };
    },
    pairs: (b, nt) => eachSetPair(b, nt, () => dbls(b, 3)),
  },
  // pair_table.cpp write_restart: its settings only (tabstyle, tablength, the ewald, pppm, msm, dispersion and tip4p
  // flags). The tables are not stored: pair_coeff must be re-issued after read_restart (the engine refuses a run before).
  // The flags are set by the kspace style that follows, so they are not kept here.
  table: {
    settings: (b) => {
      const style = b.int(), n = b.int();
      b.ints(5);
      const names = ['lookup', 'linear', 'spline', 'bitmap'];
      const name = names[style];
      if (name === undefined || name === 'bitmap') b.fail(`pair style table with tabstyle ${style} is not readable by the browser engine yet (bitmap tables are not supported)`);
      return { args: [name, num(n)], modify: {} };
    },
    pairs: null,
  },
  // pair_spin_exchange.cpp (SPIN): cut_spin_exchange_global, e_offset, offset_flag, mix_flag; per pair J1_mag,
  // J1_mech, J2, J3, cut. The pair_coeff command takes J1_mech (J1_mag = J1_mech / hbar is derived from it, as coeff()
  // does), then J2 and J3, with the pair's cutoff.
  'spin/exchange': {
    settings: (b) => {
      const cut = b.dbl();
      const eOffset = b.int(), offset = b.int();
      b.int();
      if (offset !== 0) b.fail('pair style spin/exchange with offset_flag set is not readable by the browser engine yet');
      return { args: [num(cut)], modify: {}, eOffset };
    },
    pairs: (b, nt, st) => eachSetPair(b, nt, () => {
      b.dbl(); // J1_mag (derived from J1_mech)
      const j1mech = b.dbl(), j2 = b.dbl(), j3 = b.dbl(), rc = b.dbl();
      return ['exchange', num(rc), num(j1mech), num(j2), num(j3), ...(st.eOffset ? ['offset', 'yes'] : [])];
    }),
  },
};

/** The hybrid styles: pair_hybrid.cpp write_restart (nstyles, compute_tally, then per sub-style its keyword, its settings and its special weights, NO per-pair block). */
const HYBRID_STYLES = ['hybrid', 'hybrid/overlay', 'hybrid/scaled'];

/** One sub-style of a hybrid pair style: its keyword, settings and pair_modify flags, and its scale factor (hybrid/scaled). */
interface HybridSub {
  name: string;
  st: PairSettings;
  scale: number | null;
}

/**
 * PairHybrid::write_restart (and PairHybridScaled::write_restart): the sub-style settings, then for hybrid/scaled the
 * scale factor of each sub-style (scaleval) with its variable indices (scaleidx, atomvar) and names. Variable scale
 * factors (any atomvar, or a scale variable name) are refused. A sub-style's special weights are refused (its
 * pair_modify special is not restored).
 */
const readHybrid = (b: Bin, name: string, ntypes: number): { args: string[]; subs: HybridSub[] } => {
  const nstyles = b.int();
  if (nstyles < 1 || nstyles > 64) b.fail('Invalid number of sub-styles in restart file');
  b.ints(nstyles); // compute_tally (the flags of the sub-styles' tallies are not kept)
  const subs: HybridSub[] = [];
  for (let m = 0; m < nstyles; m++) {
    const n = b.int();
    if (n < 1 || n > 65536) b.fail('Invalid style name length in restart file');
    const sub = b.chars(n);
    const layout = PAIR_LAYOUTS[sub];
    if (!layout || sub.startsWith('hybrid')) b.fail(`pair style ${name} sub-style ${sub} is not readable by the browser engine yet`);
    const st = layout.settings(b, ntypes);
    for (let k = 0; k < 2; k++) {
      if (b.int()) b.fail(`pair style ${name} sub-style ${sub} with special weights is not readable by the browser engine yet`);
    }
    subs.push({ name: sub, st, scale: null });
  }
  if (name === 'hybrid/scaled') {
    const scaleval = b.dbls(nstyles);
    const scaleidx = b.ints(nstyles);
    const atomvar = b.ints(nstyles);
    const nvars = b.int();
    if (nvars < 0 || nvars > 4096) b.fail('Invalid number of scale variables in restart file');
    for (let k = 0; k < nvars; k++) b.chars(b.int());
    // PairHybridScaled::settings: scaleidx and atomvar are -1 for a constant factor (an index otherwise)
    if (scaleidx.some((v) => v >= 0) || atomvar.some((v) => v >= 0)) {
      b.fail(`pair style ${name} with variable scale factors is not readable by the browser engine yet`);
    }
    subs.forEach((s, m) => { s.scale = scaleval[m]; });
  }
  const args: string[] = [];
  for (const s of subs) {
    if (s.scale !== null) args.push(num(s.scale));
    args.push(s.name, ...s.st.args);
  }
  return { args, subs };
};

/**
 * Pair style data of the file: the settings block, then the per-pair block of the style (none for a hybrid, whose
 * sub-styles' coefficients are not stored, so pair_coeff must be re-issued after read_restart). Any other style stops
 * the read by name.
 */
const readPairStyle = (b: Bin, name: string, ntypes: number): NativePairStyle => {
  if (HYBRID_STYLES.includes(name)) {
    const h = readHybrid(b, name, ntypes);
    return { name, settings: h.args, coeffs: [], modify: {}, subs: h.subs };
  }
  const layout = PAIR_LAYOUTS[name];
  if (!layout) return b.fail(`pair style ${name} is not readable by the browser engine yet (its binary coefficient block is not decoded here)`);
  const st = layout.settings(b, ntypes);
  const coeffs = layout.pairs ? layout.pairs(b, ntypes, st) : [];
  return { name, settings: st.args, coeffs, modify: st.modify };
};

/** Bond and angle style data: the per-type arrays in the order of each style's write_restart; the rows are coeff lines (type first). */
const deg = (rad: number): number => (rad * 180) / Math.PI;
const rows = (n: number, cols: number[][], fmt: (t: number) => string[] = () => []): string[][] =>
  Array.from({ length: n }, (_, t) => [String(t + 1), ...cols.map((c) => num(c[t])), ...fmt(t)]);

/** bond_harmonic.cpp (k, r0), bond_fene.cpp (k, r0, epsilon, sigma), bond_morse.cpp (d0, alpha, r0), bond_class2.cpp (r0, k2, k3, k4), bond_gromos.cpp (k, r0). */
const BOND_READERS: Record<string, (b: Bin, n: number) => string[][]> = {
  harmonic: (b, n) => { const k = b.dbls(n), r0 = b.dbls(n); return rows(n, [k, r0]); },
  fene: (b, n) => { const k = b.dbls(n), r0 = b.dbls(n), eps = b.dbls(n), sig = b.dbls(n); return rows(n, [k, r0, eps, sig]); },
  morse: (b, n) => { const d0 = b.dbls(n), alpha = b.dbls(n), r0 = b.dbls(n); return rows(n, [d0, alpha, r0]); },
  class2: (b, n) => { const r0 = b.dbls(n), k2 = b.dbls(n), k3 = b.dbls(n), k4 = b.dbls(n); return rows(n, [r0, k2, k3, k4]); },
  gromos: (b, n) => { const k = b.dbls(n), r0 = b.dbls(n); return rows(n, [k, r0]); },
};

/**
 * angle_harmonic.cpp (k, theta0), angle_charmm.cpp (k, theta0, k_ub, r_ub), angle_cosine.cpp (k),
 * angle_cosine_squared.cpp (k, theta0), angle_class2.cpp (theta0, k2, k3, k4, then the bb and ba terms). theta0 is
 * stored in radians; the angle_coeff command takes degrees.
 */
const ANGLE_READERS: Record<string, (b: Bin, n: number) => string[][]> = {
  harmonic: (b, n) => { const k = b.dbls(n), th = b.dbls(n); return rows(n, [k, th.map(deg)]); },
  charmm: (b, n) => { const k = b.dbls(n), th = b.dbls(n), kub = b.dbls(n), rub = b.dbls(n); return rows(n, [k, th.map(deg), kub, rub]); },
  cosine: (b, n) => { const k = b.dbls(n); return rows(n, [k]); },
  'cosine/squared': (b, n) => { const k = b.dbls(n), th = b.dbls(n); return rows(n, [k, th.map(deg)]); },
  class2: (b, n) => {
    const th = b.dbls(n), k2 = b.dbls(n), k3 = b.dbls(n), k4 = b.dbls(n);
    const bbK = b.dbls(n), bbR1 = b.dbls(n), bbR2 = b.dbls(n);
    const baK1 = b.dbls(n), baK2 = b.dbls(n), baR1 = b.dbls(n), baR2 = b.dbls(n);
    const out = rows(n, [th.map(deg), k2, k3, k4]);
    for (let t = 0; t < n; t++) {
      out.push([String(t + 1), 'bb', num(bbK[t]), num(bbR1[t]), num(bbR2[t])]);
    }
    for (let t = 0; t < n; t++) {
      out.push([String(t + 1), 'ba', num(baK1[t]), num(baK2[t]), num(baR1[t]), num(baR2[t])]);
    }
    return out;
  },
};

/**
 * dihedral_harmonic.cpp (k, sign, multiplicity), dihedral_charmm.cpp (k, multiplicity, shift, weight per type, then the
 * weightflag), dihedral_opls.cpp (k1..k4), dihedral_multi_harmonic.cpp (a1..a5). A charmm style with weights (weightflag
 * set) is read with its per-type weights, which dihedral_coeff takes.
 */
const DIHEDRAL_READERS: Record<string, (b: Bin, n: number) => string[][]> = {
  harmonic: (b, n) => { const k = b.dbls(n), sign = b.ints(n), mult = b.ints(n); return rows(n, [k, sign, mult]); },
  charmm: (b, n) => {
    const k = b.dbls(n), mult = b.ints(n), shift = b.ints(n), weight = b.dbls(n);
    b.int(); // weightflag: set when any weight is positive (the weights are read per type)
    return rows(n, [k, mult, shift, weight]);
  },
  // dihedral_opls.cpp coeff stores 0.5 * K (k1[i] = 0.5 * k1_one); the command takes K, so the file value is doubled (exact)
  opls: (b, n) => {
    const k1 = b.dbls(n), k2 = b.dbls(n), k3 = b.dbls(n), k4 = b.dbls(n);
    const twice = (v: number[]): number[] => v.map((x) => 2 * x);
    return rows(n, [twice(k1), twice(k2), twice(k3), twice(k4)]);
  },
  'multi/harmonic': (b, n) => {
    const a1 = b.dbls(n), a2 = b.dbls(n), a3 = b.dbls(n), a4 = b.dbls(n), a5 = b.dbls(n);
    return rows(n, [a1, a2, a3, a4, a5]);
  },
};

/**
 * improper_harmonic.cpp (k, chi in radians), improper_cvff.cpp (k, sign, multiplicity), improper_umbrella.cpp (kw, w0 in
 * radians; the C coefficient is derived from kw and w0 and not read as an input).
 */
const IMPROPER_READERS: Record<string, (b: Bin, n: number) => string[][]> = {
  harmonic: (b, n) => { const k = b.dbls(n), chi = b.dbls(n); return rows(n, [k, chi.map(deg)]); },
  cvff: (b, n) => { const k = b.dbls(n), sign = b.ints(n), mult = b.ints(n); return rows(n, [k, sign, mult]); },
  umbrella: (b, n) => { const kw = b.dbls(n), w0 = b.dbls(n); b.dbls(n); return rows(n, [kw, w0.map(deg)]); },
};

/** Reads one bonded style (bond, angle, dihedral or improper) of the file; a style without a reader stops the read by name. */
const readBondedStyle = (b: Bin, kind: 'bond' | 'angle' | 'dihedral' | 'improper', name: string, ntypes: number): NativeBondedStyle => {
  const readers = kind === 'bond' ? BOND_READERS : kind === 'angle' ? ANGLE_READERS : kind === 'dihedral' ? DIHEDRAL_READERS : IMPROPER_READERS;
  const read = readers[name];
  if (!read) b.fail(`${kind} style ${name} is not readable by the browser engine yet`);
  return { name, coeffs: read(b, ntypes) };
};

/** The pair_modify flags of a style read from the file (only the keys the file stores). */
const applyPairFlags = (p: Pair, m: PairFlags): void => {
  if (m.mix !== undefined) p.mix = m.mix;
  if (m.shift !== undefined) p.shift = m.shift;
  if (m.tail !== undefined) p.tail = m.tail;
  if (m.table !== undefined) p.table = m.table;
  if (m.tabinnerRsq !== undefined) p.tabinnerRsq = m.tabinnerRsq;
};

/** ReadRestart::force_fields: the pair style (or the NO_PAIR marker), then the bond and angle styles; dihedrals and impropers are refused. */
const readForceFields = (b: Bin, h: Header): { pair: NativePairStyle | null; noPair: string | null; bond: NativeBondedStyle | null; angle: NativeBondedStyle | null; dihedral: NativeBondedStyle | null; improper: NativeBondedStyle | null } => {
  const out = {
    pair: null as NativePairStyle | null, noPair: null as string | null, bond: null as NativeBondedStyle | null, angle: null as NativeBondedStyle | null,
    dihedral: null as NativeBondedStyle | null, improper: null as NativeBondedStyle | null,
  };
  let flag = b.int();
  while (flag >= 0) {
    if (flag === F.PAIR) out.pair = readPairStyle(b, b.str(), h.ntypes);
    else if (flag === F.NO_PAIR) out.noPair = b.str();
    else if (flag === F.BOND) out.bond = readBondedStyle(b, 'bond', b.str(), h.nbondtypes);
    else if (flag === F.ANGLE) out.angle = readBondedStyle(b, 'angle', b.str(), h.nangletypes);
    else if (flag === F.DIHEDRAL) out.dihedral = readBondedStyle(b, 'dihedral', b.str(), h.ndihedraltypes);
    else if (flag === F.IMPROPER) out.improper = readBondedStyle(b, 'improper', b.str(), h.nimpropertypes);
    else b.fail('Invalid flag in force field section of restart file');
    flag = b.int();
  }
  return out;
};

/**
 * Modify::read_restart: the fix entries with global state (id, style, the state block of its size), then those with
 * per-atom state (id, style, the largest per-atom size). A global block the engine can hold is read into the engine's
 * record for that fix (fixRecordOf); every other entry is a marker (stateNotStored): the engine refuses a definition of
 * that fix after read_restart, as it does for its own records of fixes it does not store (fix/restartRecord.ts
 * checkFixStateNotStored).
 */
const readFixState = (b: Bin, h: Header): { records: FixRecord[]; hasPerAtom: boolean } => {
  const records: FixRecord[] = [];
  const nglobal = b.int();
  if (nglobal < 0 || nglobal > 4096) b.fail('Invalid number of fix entries in restart file');
  for (let i = 0; i < nglobal; i++) {
    const id = b.str(), style = b.str();
    const size = b.int();
    if (size < 0 || size > 1 << 30) b.fail('Invalid fix data size in restart file');
    const dv = new DataView(b.b.buffer, b.take(size), size);
    const rec = fixRecordOf(style, dv, h);
    records.push(rec ? { id, style, props: [], data: {}, ...rec } : { id, style, props: [], data: {}, stateNotStored: true });
  }
  const nperatom = b.int();
  if (nperatom < 0 || nperatom > 4096) b.fail('Invalid number of fix entries in restart file');
  for (let i = 0; i < nperatom; i++) {
    const id = b.str(), style = b.str();
    b.int(); // the largest per-atom size of the entry (its values follow each atom record)
    records.push({ id, style, props: [], data: {}, stateNotStored: true });
  }
  return { records, hasPerAtom: nperatom > 0 };
};

/** The doubles of a state block (the block size is a multiple of 8 for each block read here). */
const blockDoubles = (dv: DataView): number[] | null => {
  if (dv.byteLength % 8 !== 0) return null;
  return Array.from({ length: dv.byteLength / 8 }, (_, k) => dv.getFloat64(8 * k, true));
};

/**
 * The engine's record of one global fix block, or null when the engine does not hold that state (a marker). Each block
 * is the layout its write_restart writes, serial (one processor), with the values in the order its restart() reads them:
 *  - fix gcmc (fix_gcmc.cpp): random_equal state, next_reneighbor, the eight counters (translation, rotation, deletion,
 *    insertion; attempts then successes), ntimestep, nprocs, random_unequal state;
 *  - fix mol/swap (fix_mol_swap.cpp), fix deposit (fix_deposit.cpp): the state, the next step, the counts, ntimestep;
 *  - fix charge/regulation (fix_charge_regulation.cpp): two states, six counters, next_reneighbor, ntimestep;
 *  - fix temp/csvr, temp/csld (energy, nprocs, RanMars state), fix gjf and the brownian fixes (nprocs, RanMars state),
 *    fix baoab (energy): RanMars::get_state, with its -1 marker (random_mars.cpp; a legacy layout is a marker);
 *  - fix nvt, npt, nph (FixNH::pack_restart_data: see nhRecord).
 */
const fixRecordOf = (style: string, dv: DataView, h: Header): Partial<FixRecord> | null => {
  if (style === 'nvt' || style === 'npt' || style === 'nph') return nhRecord(style, dv, h);
  const d = blockDoubles(dv);
  if (!d) return null;
  const n = d.length;
  // a step or a step count is written as ubuf(bigint).d, the 8 bytes of the integer read as a double (native restart)
  const u = (k: number): number => Number(dv.getBigInt64(8 * k, true));
  switch (style) {
    case 'gcmc':
      return n === 13 && d[11] === 1 ? { state: { seed: d[0], seedUnequal: d[12], nextStep: u(1), step: u(10), counters: d.slice(2, 10) } } : null;
    case 'mol/swap':
      return n === 5 ? { state: { seed: d[0], nextStep: u(1), attempts: d[2], accepts: d[3], step: u(4) } } : null;
    case 'deposit':
      return n === 5 ? { state: { seed: d[0], inserted: d[1], nfirst: u(2), nextStep: u(3), step: u(4) } } : null;
    case 'charge/regulation':
      return n === 10 ? { state: { seed: d[0], seedUnequal: d[1], nextEvent: u(8), step: u(9), counters: d.slice(2, 8) } } : null;
    case 'temp/csvr':
    case 'temp/csld':
      return n === 107 && d[1] === 1 && d[2] === -1 ? { state: { energy: d[0], rng: d.slice(2) } } : null;
    case 'gjf':
    case 'brownian':
    case 'brownian/sphere':
    case 'brownian/asphere':
      return n === 106 && d[0] === 1 && d[1] === -1 ? { state: { rng: d.slice(1) } } : null;
    case 'baoab':
      return n === 1 ? { state: { energy: d[0] } } : null;
    default:
      return null;
  }
};

/**
 * A fix nvt, npt or nph block (fix_nh.cpp FixNH::pack_restart_data): tstat_flag, then mtchain, eta[mtchain], eta_dot[mtchain]
 * when set; pstat_flag, then omega[6], omega_dot[6], vol0, t0, mpchain, etap[mpchain], etap_dot[mpchain] (when mpchain
 * is set), deviatoric_flag (then h0_inv[6]) and isochoric (then p_isoch[3] and vol_start). The engine holds the chains
 * and barostat variables (NHRestart), with vol0 and the reference cell h0_inv when the file has it. An isochoric
 * barostat is a marker, since the engine has no isochoric barostat.
 * The style must match the flags (nvt a thermostat only, nph a barostat only, npt both).
 */
const nhRecord = (style: string, dv: DataView, h: Header): Partial<FixRecord> | null => {
  const d = blockDoubles(dv);
  if (!d) return null;
  let k = 0;
  const next = (): number => d[k++];
  const tflag = next();
  let eta: number[] = [], etaDot: number[] = [];
  if (tflag === 1) {
    const m = next();
    if (!Number.isInteger(m) || m < 1 || k + 2 * m > d.length) return null;
    eta = d.slice(k, k + m); k += m;
    etaDot = d.slice(k, k + m); k += m;
  } else if (tflag !== 0) return null;
  const pflag = next();
  let omega = [0, 0, 0, 0, 0, 0], omegaDot = [0, 0, 0, 0, 0, 0], vol0 = 0, t0 = 0;
  let etap: number[] = [], etapDot: number[] = [], h0inv: number[] | null = null;
  if (pflag === 1) {
    if (k + 14 > d.length) return null;
    omega = d.slice(k, k + 6); k += 6;
    omegaDot = d.slice(k, k + 6); k += 6;
    vol0 = next();
    t0 = next();
    const mp = next();
    if (!Number.isInteger(mp) || mp < 0 || k + 2 * mp > d.length) return null;
    if (mp > 0) {
      etap = d.slice(k, k + mp); k += mp;
      etapDot = d.slice(k, k + mp); k += mp;
    }
    const dev = next();
    if (dev === 1) {
      if (k + 6 > d.length) return null;
      h0inv = d.slice(k, k + 6); k += 6;
    } else if (dev !== 0) return null;
    if (next() !== 0) return null; // isochoric: not held by the engine
  } else if (pflag !== 0) return null;
  if (k !== d.length) return null;
  if ((tflag === 1) !== (style !== 'nph') || (pflag === 1) !== (style !== 'nvt')) return null;
  const nh: NHRestart = {
    eta, etaDot: [...etaDot, 0], etap, etapDot: pflag === 1 ? [...etapDot, 0] : [],
    omegaDot, omegaPos: omega, ecoupleCum: 0, vol0,
  };
  if (h0inv) nh.h0Inv = h0inv;
  if (pflag === 1) nh.t0 = t0;
  return { nh };
};

/** One atom of the file, with the fields of its atom_style (AtomVec::unpack_restart order). */
interface NativeAtom {
  x: number[];
  v: number[];
  tag: number;
  type: number;
  mask: number;
  image: number;
  q: number;
  molecule: number;
  radius: number;
  rmass: number;
  omega: number[];
  sp: number[];
  /** [type, atoms...] per entry, as stored on this atom (a bond is stored with its first atom; the reader dedupes). */
  bonds: number[][];
  angles: number[][];
  dihedrals: number[][];
  impropers: number[][];
}

/** Box geometry of the header (Domain: boxlo, prd, h and h_inv of set_global_box). */
interface Geometry {
  lo: number[];
  hi: number[];
  prd: number[];
  triclinic: boolean;
  h: number[];
  hinv: number[];
  periodic: boolean[];
}

const geometryOf = (h: Header): Geometry => {
  const lo = h.boxlo, hi = h.boxhi;
  const prd = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const hh = [prd[0], prd[1], prd[2], h.yz, h.xz, h.xy];
  const hinv = [
    1 / hh[0], 1 / hh[1], 1 / hh[2],
    -hh[3] / (hh[1] * hh[2]),
    (hh[3] * hh[5] - hh[1] * hh[4]) / (hh[0] * hh[1] * hh[2]),
    -hh[5] / (hh[0] * hh[1]),
  ];
  return { lo, hi, prd, triclinic: h.triclinic !== 0, h: hh, hinv, periodic: h.periodic.map((p) => p !== 0) };
};

/** Domain::x2lamda and Domain::lamda2x (triclinic fractional coordinates). */
const x2lamda = (g: Geometry, x: number[]): number[] => {
  const d = [x[0] - g.lo[0], x[1] - g.lo[1], x[2] - g.lo[2]];
  const hi = g.hinv;
  return [hi[0] * d[0] + hi[5] * d[1] + hi[4] * d[2], hi[1] * d[1] + hi[3] * d[2], hi[2] * d[2]];
};
const lamda2x = (g: Geometry, l: number[]): number[] => {
  const hh = g.h;
  return [hh[0] * l[0] + hh[5] * l[1] + hh[4] * l[2] + g.lo[0], hh[1] * l[1] + hh[3] * l[2] + g.lo[1], hh[2] * l[2] + g.lo[2]];
};

/** The packed image flags (lmptype.h IMGBITS 10 with IMGMAX offsets) as box counts per dimension. */
const unpackImage = (img: number): number[] => [
  (img & IMGMASK) - IMGMAX,
  ((img >> IMGBITS) & IMGMASK) - IMGMAX,
  (img >> (2 * IMGBITS)) - IMGMAX,
];
const packImage = (ix: number[]): number =>
  (ix[0] + IMGMAX) | ((ix[1] + IMGMAX) << IMGBITS) | ((ix[2] + IMGMAX) << (2 * IMGBITS));

/** Domain::remap(x, image): wraps a position into the periodic box and moves its image count (triclinic in lamda). */
const remapAtom = (g: Geometry, x0: number[], image: number): { x: number[]; image: number } => {
  const lam = g.triclinic;
  const coord = lam ? x2lamda(g, x0) : x0.slice();
  const lo = lam ? [0, 0, 0] : g.lo;
  const hi = lam ? [1, 1, 1] : g.hi;
  const period = lam ? [1, 1, 1] : g.prd;
  const img = unpackImage(image);
  for (let d = 0; d < 3; d++) {
    if (!g.periodic[d]) continue;
    while (coord[d] < lo[d]) { coord[d] += period[d]; img[d]--; }
    while (coord[d] >= hi[d]) { coord[d] -= period[d]; img[d]++; }
    coord[d] = Math.max(coord[d], lo[d]);
  }
  return { x: lam ? lamda2x(g, coord) : coord, image: packImage(img) };
};

/**
 * ReadRestart (serial file, the chunks of PERPROC): every atom record of every chunk, in file order. A record's first
 * double is its size; the following values follow AtomVec::pack_restart: x, tag, type, mask, image, v, then the fields
 * of the atom_style. Positions are remapped into the box as read_restart does (the default remap).
 */
const readAtoms = (b: Bin, h: Header, fields: readonly string[], g: Geometry, extras: boolean): NativeAtom[] => {
  const atoms: NativeAtom[] = [];
  for (let ip = 0; ip < h.nprocs; ip++) {
    const flag = b.int();
    if (flag !== F.PERPROC) b.fail('Invalid flag in peratom section of restart file');
    const n = b.int();
    if (n < 0) b.fail('Invalid data size in peratom section of restart file');
    const dv = new DataView(b.b.buffer, b.take(8 * n), 8 * n);
    const d = (k: number): number => dv.getFloat64(8 * k, true);
    const L = (k: number): number => Number(dv.getBigInt64(8 * k, true));
    let m = 0;
    while (m < n) {
      const size = d(m);
      if (!Number.isInteger(size) || size < 1 || m + size > n) b.fail('Invalid data in peratom section of restart file');
      let k = m + 1;
      const x = [d(k), d(k + 1), d(k + 2)];
      k += 3;
      const tag = L(k++), type = L(k++), mask = L(k++), image0 = L(k++);
      const v = [d(k), d(k + 1), d(k + 2)];
      k += 3;
      const at: NativeAtom = {
        x, v, tag, type, mask, image: image0, q: 0, molecule: 0, radius: 0, rmass: 0, omega: [0, 0, 0], sp: [0, 0, 0, 0],
        bonds: [], angles: [], dihedrals: [], impropers: [],
      };
      for (const f of fields) {
        if (f === 'q') at.q = d(k++);
        else if (f === 'molecule') at.molecule = L(k++);
        else if (f === 'radius') at.radius = d(k++);
        else if (f === 'rmass') at.rmass = d(k++);
        else if (f === 'omega') { at.omega = [d(k), d(k + 1), d(k + 2)]; k += 3; }
        else if (f === 'sp') { at.sp = [d(k), d(k + 1), d(k + 2), d(k + 3)]; k += 4; }
        else if (f === 'topo') {
          // bonds, angles, dihedrals, impropers in this order: a count, the types, then one column per atom of an entry
          const topo = { k };
          const column = (width: number): number[][] => {
            const count = L(topo.k++);
            if (count < 0) b.fail('Invalid topology count in peratom section of restart file');
            const types = Array.from({ length: count }, () => L(topo.k++));
            const cols = Array.from({ length: width }, () => Array.from({ length: count }, () => L(topo.k++)));
            return types.map((t, e) => [t, ...cols.map((c) => c[e])]);
          };
          at.bonds = column(1);
          at.angles = column(3);
          at.dihedrals = column(4);
          at.impropers = column(4);
          k = topo.k;
        }
      }
      // the values of a fix with per-atom state follow the atom_style fields (AtomVec::pack_restart, extra fixes)
      if (k > m + size || (!extras && k !== m + size)) b.fail('the atom record size does not match its atom_style (corrupt file or another atom_style)');
      const rem = remapAtom(g, at.x, at.image);
      at.x = rem.x;
      at.image = rem.image;
      atoms.push(at);
      m += size;
    }
  }
  return atoms;
};

const BOUNDARY_OF = (b: Bin, code: number): BoundaryStyle => {
  const f = BOUNDARY_CODE[code];
  if (f === undefined) b.fail(`unknown boundary code ${code}`);
  return f;
};

/**
 * Bonds and angles without repeats: a file stores a bond with each of its atoms when newton_bond is off, and a
 * topology entry may repeat with the same atoms; the count must then match the header (NBONDS, NANGLES).
 */
const buildTopology = (b: Bin, h: Header, atoms: NativeAtom[]): Topology => {
  const topo = emptyTopology();
  topo.nbondtypes = h.nbondtypes;
  topo.nangletypes = h.nangletypes;
  const tags = new Set(atoms.map((a) => a.tag));
  const bondSeen = new Set<string>();
  const angleSeen = new Set<string>();
  const dihSeen = new Set<string>();
  const impSeen = new Set<string>();
  for (const at of atoms) {
    for (const [t, partner] of at.bonds.map((e) => [e[0], e[1]] as const)) {
      if (!tags.has(partner)) b.fail(`bond references atom ${partner}, which is not in the file`);
      if (t < 1 || t > h.nbondtypes) b.fail(`bond type ${t} is outside 1..${h.nbondtypes}`);
      const key = at.tag < partner ? `${at.tag}-${partner}` : `${partner}-${at.tag}`;
      if (bondSeen.has(key)) continue;
      bondSeen.add(key);
      pushTopo(topo.bonds, t, [at.tag, partner]);
    }
    for (const [t, a1, a2, a3] of at.angles as [number, number, number, number][]) {
      if (!tags.has(a1) || !tags.has(a2) || !tags.has(a3)) b.fail('angle references an atom that is not in the file');
      if (t < 1 || t > h.nangletypes) b.fail(`angle type ${t} is outside 1..${h.nangletypes}`);
      const key = `${Math.min(a1, a3)},${a2},${Math.max(a1, a3)}`;
      if (angleSeen.has(key)) continue;
      angleSeen.add(key);
      pushTopo(topo.angles, t, [a1, a2, a3]);
    }
    // dihedral and improper: stored once (newton_bond on); a quadruple and its reverse are the same dihedral
    for (const [t, a1, a2, a3, a4] of at.dihedrals as [number, number, number, number, number][]) {
      for (const a of [a1, a2, a3, a4]) if (!tags.has(a)) b.fail('dihedral references an atom that is not in the file');
      if (t < 1 || t > h.ndihedraltypes) b.fail(`dihedral type ${t} is outside 1..${h.ndihedraltypes}`);
      const fwd = `${a1},${a2},${a3},${a4}`, rev = `${a4},${a3},${a2},${a1}`;
      const key = fwd < rev ? fwd : rev;
      if (dihSeen.has(key)) continue;
      dihSeen.add(key);
      pushTopo(topo.dihedrals, t, [a1, a2, a3, a4]);
    }
    for (const [t, a1, a2, a3, a4] of at.impropers as [number, number, number, number, number][]) {
      for (const a of [a1, a2, a3, a4]) if (!tags.has(a)) b.fail('improper references an atom that is not in the file');
      if (t < 1 || t > h.nimpropertypes) b.fail(`improper type ${t} is outside 1..${h.nimpropertypes}`);
      const key = `${a1},${a2},${a3},${a4}`;
      if (impSeen.has(key)) continue;
      impSeen.add(key);
      pushTopo(topo.impropers, t, [a1, a2, a3, a4]);
    }
  }
  topo.ndihedraltypes = h.ndihedraltypes;
  topo.nimpropertypes = h.nimpropertypes;
  if (topo.bonds.n !== h.nbonds) b.fail(`the file holds ${topo.bonds.n} bonds, its header says ${h.nbonds}`);
  if (topo.angles.n !== h.nangles) b.fail(`the file holds ${topo.angles.n} angles, its header says ${h.nangles}`);
  if (topo.dihedrals.n !== h.ndihedrals) b.fail(`the file holds ${topo.dihedrals.n} dihedrals, its header says ${h.ndihedrals}`);
  if (topo.impropers.n !== h.nimpropers) b.fail(`the file holds ${topo.impropers.n} impropers, its header says ${h.nimpropers}`);
  return topo;
};

/** The atom_style fields this reader stores, or a StyleError naming the style. */
const fieldsOf = (b: Bin, h: Header): readonly string[] => {
  const fields = ATOM_FIELDS[h.atomStyle];
  if (!fields || h.atomStyleArgs.length) {
    b.fail(`atom_style ${h.atomStyle}${h.atomStyleArgs.length ? ' with arguments' : ''} is not readable by the browser engine yet (the readable styles: ${Object.keys(ATOM_FIELDS).join(', ')})`);
  }
  return fields;
};

/**
 * Reads a serial native binary restart file (the magic string checked by the caller) and installs it into the System
 * as read_restart does: the box, the atoms (in file order), the masses, the groups, the special bond weights, the
 * pair and bonded styles with their coefficients, and the comm settings. Fix state is refused by name.
 */
export const readNativeRestart = (sys: System, text: string, name: string): void => {
  const b = new Bin(text, name);
  b.take(NATIVE_RESTART_MAGIC.length);
  const endian = b.int();
  if (endian !== ENDIAN) b.fail(endian === 0x1000 ? 'byte ordering is swapped (the file was written on a big-endian machine)' : 'byte ordering is not recognized');
  const revision = b.int();
  if (revision !== FORMAT_REVISION) b.fail(`format revision ${revision} is not readable (revision ${FORMAT_REVISION} is)`);
  if (!text.endsWith(NATIVE_RESTART_MAGIC)) b.fail('incomplete or corrupted restart file (the end marker is missing)');
  const h = readHeader(b);
  const names = readGroups(b);
  const masses = readTypeArrays(b, h);
  const ff = readForceFields(b, h);
  const fixState = readFixState(b, h);
  let multiproc = 0;
  let flag = b.int();
  while (flag >= 0) {
    if (flag !== F.MULTIPROC) b.fail('Invalid flag in file layout section of restart file');
    multiproc = b.int();
    flag = b.int();
  }
  if (multiproc) b.fail('a multi-file restart (a file name with %) is not readable by the browser engine yet; read the single file');
  if (h.atomId !== 1) b.fail('atom IDs that are turned off (atom_modify id no) are not readable by the browser engine yet');
  if (h.nprocs < 1) b.fail('the file has no processor chunks');
  const unitName = h.units;
  if (!isUnitStyle(unitName)) throw new StyleError(`native restart ${name}: units ${unitName} are not known to the browser engine`);
  if (h.dimension !== 2 && h.dimension !== 3) b.fail(`dimension ${h.dimension} is not valid`);

  const fields = fieldsOf(b, h);
  const g = geometryOf(h);
  const atoms = readAtoms(b, h, fields, g, fixState.hasPerAtom);
  if (atoms.length !== h.natoms) b.fail(`the file holds ${atoms.length} atoms, its header says ${h.natoms}`);

  // the box: its boundary codes (0 p, 1 f, 2 s, 3 m) and the minimum bounds of the 'm' faces
  const boundary = [0, 1, 2].map((d) => [BOUNDARY_OF(b, h.boundary[2 * d]), BOUNDARY_OF(b, h.boundary[2 * d + 1])]) as SimBox['boundary'];
  const box = makeBox({
    lo: [h.boxlo[0], h.boxlo[1], h.boxlo[2]], hi: [h.boxhi[0], h.boxhi[1], h.boxhi[2]], boundary,
    tilt: g.triclinic ? [h.xy, h.xz, h.yz] : undefined,
  });
  if (h.boundmin) {
    for (let d = 0; d < 3; d++) {
      if (boundary[d][0] === 'm') box.minLo[d] = h.boundmin[2 * d];
      if (boundary[d][1] === 'm') box.minHi[d] = h.boundmin[2 * d + 1];
    }
  }

  const units = UNIT_SYSTEMS[unitName];
  const style = h.atomStyle as AtomStyle;
  const s: SimState = emptyState(units, h.dimension as 2 | 3, box, h.ntypes, style);
  const n = atoms.length;
  s.n = n;
  s.x = new Float64Array(3 * n);
  s.v = new Float64Array(3 * n);
  s.f = new Float64Array(3 * n);
  s.image = new Int32Array(3 * n);
  s.id = new Int32Array(n);
  s.type = new Int32Array(n);
  s.mask = new Int32Array(n);
  s.molecule = new Int32Array(n);
  s.q = new Float64Array(n);
  s.order = new Int32Array(n);
  atoms.forEach((at, i) => {
    s.x.set(at.x, 3 * i);
    s.v.set(at.v, 3 * i);
    const ix = unpackImage(at.image);
    s.image[3 * i] = ix[0];
    s.image[3 * i + 1] = ix[1];
    s.image[3 * i + 2] = ix[2];
    s.id[i] = at.tag;
    s.type[i] = at.type;
    s.mask[i] = at.mask;
    s.molecule[i] = at.molecule;
    s.q[i] = at.q;
    s.order[i] = i;
  });
  if (fields.includes('radius')) {
    s.radius = Float64Array.from(atoms, (a) => a.radius);
    s.omega = Float64Array.from(atoms.flatMap((a) => a.omega));
    s.torque = new Float64Array(3 * n);
  }
  if (fields.includes('rmass')) s.rmass = Float64Array.from(atoms, (a) => a.rmass);
  if (fields.includes('sp')) {
    s.sp = Float64Array.from(atoms.flatMap((a) => a.sp));
    s.fm = new Float64Array(3 * n);
  }
  s.massByType = new Float64Array(h.ntypes + 1).fill(Number.NaN);
  if (masses) for (let t = 0; t < h.ntypes; t++) s.massByType[t + 1] = masses[t];
  s.topo = buildTopology(b, h, atoms);
  s.step = h.ntimestep;
  s.dt = h.dt;
  s.time = h.atime;
  s.timeStep = h.atimestep;

  sys.units = units;
  sys.dimension = h.dimension as 2 | 3;
  sys.boundary = box.boundary.map((f) => [f[0], f[1]]) as SimBox['boundary'];
  sys.atomStyle = style;
  sys.commStyle = h.commMode === 1 ? 'tiled' : 'brick';
  sys.ghostVelocity = h.commVel !== 0;
  sys.nb.commCutoff = h.commCutoff;
  sys.pendingDt = null;
  sys.pendingFixData = new Map(fixState.records.map((r) => [r.id, r] as const));
  // the fixes whose state the engine does not hold: one warning naming them (their definition is refused by name later)
  const unstored = fixState.records.filter((r) => r.stateNotStored).map((r) => `${r.id} (${r.style})`);
  if (unstored.length) sys.log(`WARNING: restart file state of ${unstored.join(', ')} is not read by the browser engine; defining one of these fixes again is refused`);
  sys.setState(s);

  sys.groups.names = names;
  sys.groups.dynamic.clear();
  sys.ff.special = {
    ...sys.ff.special,
    lj: [h.specialLJ[0], h.specialLJ[1], h.specialLJ[2]],
    coul: [h.specialCoul[0], h.specialCoul[1], h.specialCoul[2]],
  };

  const ctx: StyleContext = sys.styleContext();
  sys.ff.pair = null;
  sys.ff.pairNotRestarted = null;
  if (ff.pair) {
    const make = PAIR_STYLES[ff.pair.name];
    if (!make) b.fail(`pair style ${ff.pair.name} is not available in this engine`);
    const p: Pair = make();
    p.settings(ff.pair.settings, ctx);
    p.allocate(h.ntypes);
    for (const c of ff.pair.coeffs) p.coeff(c, ctx);
    applyPairFlags(p, ff.pair.modify);
    if (p instanceof PairHybrid && ff.pair.subs) {
      ff.pair.subs.forEach((sub, m) => applyPairFlags(p.subs[m].style, sub.st.modify));
    }
    sys.ff.pair = p;
  } else if (ff.noPair) {
    // measured with native LAMMPS (black box): read_restart logs that the pair style stores no restart info
    sys.ff.pairNotRestarted = ff.noPair;
    sys.log(`pair style ${ff.noPair} stores no restart info`);
  }
  const bonded = (kind: 'bond' | 'angle' | 'dihedral' | 'improper', st: NativeBondedStyle | null, reg: Record<string, () => Bonded>, ntypes: number): Bonded | null => {
    if (!st) return null;
    const make = reg[st.name];
    if (!make) b.fail(`${kind} style ${st.name} is not available in this engine`);
    const o = make();
    o.settings([], ctx);
    o.allocate(ntypes);
    for (const c of st.coeffs) o.coeff(c, ctx);
    return o;
  };
  sys.ff.bond = bonded('bond', ff.bond, BOND_STYLES, h.nbondtypes);
  sys.ff.angle = bonded('angle', ff.angle, ANGLE_STYLES, h.nangletypes);
  sys.ff.dihedral = bonded('dihedral', ff.dihedral, DIHEDRAL_STYLES, h.ndihedraltypes);
  sys.ff.improper = bonded('improper', ff.improper, IMPROPER_STYLES, h.nimpropertypes);
  sys.bump();
  sys.log(`Restoring from native restart file ${name}: ${n} atoms at step ${h.ntimestep} (LAMMPS ${h.version})`);
};

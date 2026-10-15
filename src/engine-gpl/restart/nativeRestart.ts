/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the binary restart reader of src/read_restart.cpp
 * (ReadRestart::command, header, type_arrays, force_fields, file_layout, magic_string, endian, format_revision,
 * check_eof_magic, read_int, read_string, read_double_vec), src/write_restart.cpp and src/lmprestart.h (the flags,
 * ENDIAN and FORMAT_REVISION), src/group.cpp (Group::read_restart), src/modify.cpp (Modify::read_restart: the fix
 * entries), src/domain.cpp (Domain::remap), src/atom_vec.cpp (AtomVec::unpack_restart) with the fields of
 * src/atom_vec_*.cpp, MOLECULE/atom_vec_molecular.cpp and MOLECULE/atom_vec_full.cpp, and the per-style read_restart of
 * src/KSPACE/pair_lj_cut_coul_long.cpp, src/pair_lj_cut.cpp, src/SPIN/pair_spin_exchange.cpp, src/MOLECULE/bond_harmonic.cpp
 * and src/MOLECULE/angle_harmonic.cpp. LAMMPS is Copyright (2003) Sandia Corporation and the LAMMPS development team,
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * What it reads: a serial (one-file, no "%") binary restart file of revision 3, written by native LAMMPS on this
 * platform (little-endian, 32-bit imageint and tagint, 64-bit bigint), from one or more processes. The box (orthogonal
 * and triclinic) and the header settings, the masses, the groups, the per-atom attributes of atom_style atomic, charge,
 * molecular, full, sphere and spin, the bonds and angles of the molecular styles, the pair styles lj/cut,
 * lj/cut/coul/long and spin/exchange (or the NO_PAIR marker of a style that stores nothing), the bond and angle styles
 * harmonic, and the fix state of fix nvt (its thermostat chain). Any other fix state is a marker that refuses the
 * definition of that fix after read_restart. Anything else is refused by name (StyleError).
 *
 * The file's values go into the engine's own state (SimState) and style objects, through the public engine API.
 */
import { emptyState, emptyTopology, pushTopo } from '../../engine/atoms';
import { makeBox } from '../../engine/domain';
import { UNIT_SYSTEMS, isUnitStyle } from '../../engine/units';
import { PAIR_STYLES, BOND_STYLES, ANGLE_STYLES } from '../../engine/styles';
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
  nimpropers: number;
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
  modify: { mix?: (typeof MIX_RULE)[number]; shift?: boolean; tail?: boolean; table?: number; tabinnerRsq?: number };
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
    nbonds: 0, nbondtypes: 0, nangles: 0, nangletypes: 0, ndihedrals: 0, nimpropers: 0, triclinic: 0,
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
      case F.NDIHEDRALTYPES: b.int(); break;
      case F.DIHEDRAL_PER_ATOM: b.int(); break;
      case F.NIMPROPERS: h.nimpropers = b.i64(); break;
      case F.NIMPROPERTYPES: b.int(); break;
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

/** Pair style data of the file (the write_restart layouts of the styles listed below); any other style stops the read by name. */
const readPairStyle = (b: Bin, name: string, ntypes: number): NativePairStyle => {
  const mixOf = (k: number): (typeof MIX_RULE)[number] => {
    if (k < 0 || k > 2) b.fail(`pair style ${name}: unknown mixing rule ${k}`);
    return MIX_RULE[k];
  };
  // the (i <= j) loop of each write_restart: a set flag per pair, then the values of a set pair
  const lj3 = (i: number, j: number): string[] => {
    const eps = b.dbl(), sig = b.dbl(), cutIJ = b.dbl();
    return [String(i), String(j), String(eps), String(sig), String(cutIJ)];
  };
  // pair_lj_cut.cpp write_restart_settings: cut_global, offset_flag, mix_flag, tail_flag
  if (name === 'lj/cut') {
    const cut = b.dbl();
    const offset = b.int(), mix = b.int(), tail = b.int();
    const coeffs: string[][] = [];
    for (let i = 1; i <= ntypes; i++) for (let j = i; j <= ntypes; j++) if (b.int()) coeffs.push(lj3(i, j));
    return { name, settings: [String(cut)], coeffs, modify: { shift: offset !== 0, mix: mixOf(mix), tail: tail !== 0 } };
  }
  // KSPACE/pair_lj_cut_coul_long.cpp write_restart_settings: cut_lj_global, cut_coul, offset_flag, mix_flag, tail_flag,
  // ncoultablebits, tabinner (the squared inner cutoff is tabinner^2, as pair_modify tabinner sets it)
  if (name === 'lj/cut/coul/long') {
    const cutLJ = b.dbl(), cutCoul = b.dbl();
    const offset = b.int(), mix = b.int(), tail = b.int(), tableBits = b.int();
    const tabinner = b.dbl();
    const coeffs: string[][] = [];
    for (let i = 1; i <= ntypes; i++) for (let j = i; j <= ntypes; j++) if (b.int()) coeffs.push(lj3(i, j));
    return {
      name, settings: [String(cutLJ), String(cutCoul)], coeffs,
      modify: { shift: offset !== 0, mix: mixOf(mix), tail: tail !== 0, table: tableBits, tabinnerRsq: tabinner * tabinner },
    };
  }
  // SPIN/pair_spin_exchange.cpp write_restart_settings: cut_spin_exchange_global, e_offset, offset_flag, mix_flag;
  // per pair: J1_mag, J1_mech, J2, J3, cut. The pair_coeff command takes J1_mech (J1_mag = J1_mech / hbar is derived
  // from it, as coeff() does), then J2 and J3, with the pair's cutoff.
  if (name === 'spin/exchange') {
    const cutGlobal = b.dbl();
    const eOffset = b.int(), offset = b.int();
    b.int();
    if (offset !== 0) b.fail(`pair style ${name} with offset_flag set is not readable by the browser engine yet`);
    const coeffs: string[][] = [];
    for (let i = 1; i <= ntypes; i++) {
      for (let j = i; j <= ntypes; j++) {
        if (!b.int()) continue;
        b.dbl(); // J1_mag (derived from J1_mech)
        const j1mech = b.dbl(), j2 = b.dbl(), j3 = b.dbl(), rc = b.dbl();
        coeffs.push([String(i), String(j), 'exchange', String(rc), String(j1mech), String(j2), String(j3), ...(eOffset ? ['offset', 'yes'] : [])]);
      }
    }
    return { name, settings: [String(cutGlobal)], coeffs, modify: {} };
  }
  return b.fail(`pair style ${name} is not readable by the browser engine yet (its binary coefficient block is not decoded here)`);
};

/** Bond and angle style data (bond_harmonic.cpp, angle_harmonic.cpp): k per type, then r0 or theta0 per type. */
const readBondedStyle = (b: Bin, kind: 'bond' | 'angle', name: string, ntypes: number): NativeBondedStyle => {
  if (name !== 'harmonic') b.fail(`${kind} style ${name} is not readable by the browser engine yet`);
  const k = b.dbls(ntypes);
  const second = b.dbls(ntypes);
  const coeffs = k.map((kv, t) => {
    // angle_harmonic.cpp stores theta0 in radians; the angle_coeff command takes degrees
    const v2 = kind === 'angle' ? (second[t] * 180) / Math.PI : second[t];
    return [String(t + 1), String(kv), String(v2)];
  });
  return { name, coeffs };
};

/** ReadRestart::force_fields: the pair style (or the NO_PAIR marker), then the bond and angle styles; dihedrals and impropers are refused. */
const readForceFields = (b: Bin, h: Header): { pair: NativePairStyle | null; noPair: string | null; bond: NativeBondedStyle | null; angle: NativeBondedStyle | null } => {
  const out = { pair: null as NativePairStyle | null, noPair: null as string | null, bond: null as NativeBondedStyle | null, angle: null as NativeBondedStyle | null };
  let flag = b.int();
  while (flag >= 0) {
    if (flag === F.PAIR) out.pair = readPairStyle(b, b.str(), h.ntypes);
    else if (flag === F.NO_PAIR) out.noPair = b.str();
    else if (flag === F.BOND) out.bond = readBondedStyle(b, 'bond', b.str(), h.nbondtypes);
    else if (flag === F.ANGLE) out.angle = readBondedStyle(b, 'angle', b.str(), h.nangletypes);
    else if (flag === F.DIHEDRAL) b.fail(`dihedral style ${b.str()} is not readable by the browser engine yet`);
    else if (flag === F.IMPROPER) b.fail(`improper style ${b.str()} is not readable by the browser engine yet`);
    else b.fail('Invalid flag in force field section of restart file');
    flag = b.int();
  }
  return out;
};

/**
 * Modify::read_restart: the fix entries with global state (id, style, the state block of its size), then those with
 * per-atom state (id, style, the largest per-atom size). A fix nvt entry with a thermostat and no barostat is read
 * into the engine's NH state (fix_nh.cpp FixNH::pack_restart_data and FixNH::restart: the chain variables only). Every
 * other entry is a marker (stateNotStored): the engine refuses a definition of that fix after read_restart, as it does
 * for its own records of fixes it does not store (fix/restartRecord.ts checkFixStateNotStored).
 */
const readFixState = (b: Bin): { records: FixRecord[]; hasPerAtom: boolean } => {
  const records: FixRecord[] = [];
  const nglobal = b.int();
  if (nglobal < 0 || nglobal > 4096) b.fail('Invalid number of fix entries in restart file');
  for (let i = 0; i < nglobal; i++) {
    const id = b.str(), style = b.str();
    const size = b.int();
    if (size < 0 || size > 1 << 30) b.fail('Invalid fix data size in restart file');
    const dv = new DataView(b.b.buffer, b.take(size), size);
    const nh = style === 'nvt' ? nvtChainState(dv) : null;
    records.push(nh ? { id, style, props: [], data: {}, nh } : { id, style, props: [], data: {}, stateNotStored: true });
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

/**
 * The thermostat chain of a fix nvt state block (fix_nh.cpp pack_restart_data): tstat_flag, mtchain, eta[mtchain],
 * eta_dot[mtchain], then pstat_flag, which must be 0 (no barostat state). The engine's etaDot holds one more entry
 * than the chain, which the file does not store (zero, as at the start of a run). Null for any other block.
 */
const nvtChainState = (dv: DataView): NHRestart | null => {
  const size = dv.byteLength;
  if (size % 8 !== 0) return null;
  const n = size / 8;
  const d = (k: number): number => dv.getFloat64(8 * k, true);
  if (n < 4 || d(0) !== 1) return null;
  const m = d(1);
  if (!Number.isInteger(m) || m < 1 || 2 + 2 * m + 1 !== n || d(2 + 2 * m) !== 0) return null;
  const eta = Array.from({ length: m }, (_, k) => d(2 + k));
  const etaDot = [...Array.from({ length: m }, (_, k) => d(2 + m + k)), 0];
  return { eta, etaDot, etap: [], etapDot: [], omegaDot: [], omegaPos: [], ecoupleCum: 0, vol0: 0, len0: [0, 0, 0] };
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
  for (const at of atoms) {
    if (at.dihedrals.length || at.impropers.length) b.fail('dihedrals and impropers are not readable by the browser engine yet');
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
  }
  if (topo.bonds.n !== h.nbonds) b.fail(`the file holds ${topo.bonds.n} bonds, its header says ${h.nbonds}`);
  if (topo.angles.n !== h.nangles) b.fail(`the file holds ${topo.angles.n} angles, its header says ${h.nangles}`);
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
  const fixState = readFixState(b);
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
    const m = ff.pair.modify;
    if (m.mix !== undefined) p.mix = m.mix;
    if (m.shift !== undefined) p.shift = m.shift;
    if (m.tail !== undefined) p.tail = m.tail;
    if (m.table !== undefined) p.table = m.table;
    if (m.tabinnerRsq !== undefined) p.tabinnerRsq = m.tabinnerRsq;
    sys.ff.pair = p;
  } else if (ff.noPair) {
    // measured with native LAMMPS (black box): read_restart logs that the pair style stores no restart info
    sys.ff.pairNotRestarted = ff.noPair;
    sys.log(`pair style ${ff.noPair} stores no restart info`);
  }
  const bonded = (kind: 'bond' | 'angle', st: NativeBondedStyle | null, reg: Record<string, () => Bonded>, ntypes: number): Bonded | null => {
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
  sys.bump();
  sys.log(`Restoring from native restart file ${name}: ${n} atoms at step ${h.ntimestep} (LAMMPS ${h.version})`);
};

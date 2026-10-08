import type { SimState } from '../types';
import type { System } from '../system';
import type { SpecialBonds } from '../force/forcefield';
import type { DynamicGroup } from '../group';
import { StyleError, type Bonded, type Pair } from '../force/types';
import { PAIR_STYLES, BOND_STYLES, ANGLE_STYLES, DIHEDRAL_STYLES, IMPROPER_STYLES } from '../styles';

/*
 * The browser engine's restart file (write_restart / read_restart).
 *
 * A native LAMMPS binary restart file cannot be read here. read_restart.rst
 * says: "The binary restart file format was not designed with backward,
 * forward, or cross-platform compatibility in mind". This module therefore
 * writes its own text format: a magic first line, then one JSON document
 * holding what the docs list as stored. Everything else is re-specified by
 * the input script, as the docs require.
 *
 * Stored (read_restart.rst: "Here is the list of information included in a
 * restart file"): units, atom style, box size and shape with boundary
 * settings, timestep size and step number, per-atom attributes including
 * group assignments, image flags and molecular topology, group definitions,
 * per-type masses, force field styles and (for the styles below) their
 * coefficients, pair_modify settings (mix, shift, tail, table), special_bonds
 * settings, and comm_style / comm_modify settings (vel, cutoff).
 *
 * Not stored (read_restart.rst: "Here is a list of information not stored in a
 * restart file"): fixes, computes, variables, regions, neighbor settings,
 * kspace settings, thermo / dump / restart output settings. Molecule templates
 * are not stored either. Forces are recomputed by the next run, so the file
 * does not hold them.
 *
 * Coefficient storage (quoted from the style doc pages):
 *  - pair: pair_lj.rst "All of the *lj/cut* pair styles write their information
 *    to binary restart files"; pair_coul.rst "These pair styles write their
 *    information to binary restart files". Other pair styles are refused on
 *    write, so nothing is silently dropped.
 *  - bond, angle, dihedral, improper: bond_style.rst "All bond potentials store
 *    their coefficient data in binary restart files" (the same sentence is in
 *    angle_style.rst, dihedral_style.rst and improper_style.rst). Table styles
 *    are refused: angle_table.rst "the coefficient information is not stored in
 *    the restart file, since it is tabulated in the potential files".
 *  - hybrid styles: bond_style.rst "only stores the list of sub-styles in the
 *    restart file", so they are refused too.
 */

export const RESTART_MAGIC = 'LAMMPS-WEB-RESTART 1';
const FORMAT = 'lammps-web-restart';
const VERSION = 1;

/** Pair styles whose coefficients are stored (see the doc quotes above). */
const PAIR_RESTART_STYLES = new Set(['lj/cut', 'lj/cut/coul/cut', 'coul/cut']);

const TYPED = {
  Float64Array, Float32Array, Int32Array, Uint32Array, Int16Array, Uint16Array, Int8Array, Uint8Array,
} as const;
type TypedName = keyof typeof TYPED;

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
type Obj = Record<string, unknown>;

const isObject = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v);

/** Encodes one value as JSON; non-finite numbers, undefined, typed arrays, Maps and class instances get tags. */
const encodeValue = (v: unknown, path: string, stack: object[]): Json => {
  if (v === undefined) return { $u: 1 };
  if (v === null || typeof v === 'boolean' || typeof v === 'string') return v;
  if (typeof v === 'number') return Number.isFinite(v) ? v : { $n: String(v) };
  if (typeof v === 'function') throw new StyleError(`write_restart: ${path} holds a function, which a restart file cannot store`);
  if (typeof v !== 'object') throw new StyleError(`write_restart: ${path} has type ${typeof v}, which a restart file cannot store`);
  if (stack.includes(v)) throw new StyleError(`write_restart: ${path} is a circular reference`);
  stack.push(v);
  try {
    if (ArrayBuffer.isView(v)) {
      const name = v.constructor.name as TypedName;
      if (!(name in TYPED)) throw new StyleError(`write_restart: ${path} is a ${name}, which a restart file cannot store`);
      const out: Json[] = [];
      const nums = v as unknown as ArrayLike<number>;
      for (let i = 0; i < nums.length; i++) out.push(Number.isFinite(nums[i]) ? nums[i] : { $n: String(nums[i]) });
      return { $ta: name, v: out };
    }
    if (Array.isArray(v)) return v.map((x, i) => encodeValue(x, `${path}[${i}]`, stack));
    if (v instanceof Map) {
      return { $map: [...v].map(([k, x]) => [encodeValue(k, path, stack), encodeValue(x, `${path}{key}`, stack)]) };
    }
    const entries = Object.entries(v).map(([k, x]) => [k, encodeValue(x, `${path}.${k}`, stack)] as const);
    const proto = Object.getPrototypeOf(v);
    const fields = Object.fromEntries(entries) as { [k: string]: Json };
    if (proto === Object.prototype || proto === null) return fields;
    return { $inst: (v as object).constructor?.name ?? 'Object', f: fields };
  } finally {
    stack.pop();
  }
};

/** Decodes plain data (no class instances; those need a restore target, see restoreInto). */
const decodeValue = (j: Json | undefined, path: string): unknown => {
  if (j === null || typeof j !== 'object') return j;
  if (Array.isArray(j)) return j.map((x, i) => decodeValue(x, `${path}[${i}]`));
  if ('$u' in j) return undefined;
  if ('$n' in j) return Number(j.$n);
  if ('$ta' in j) {
    const C = TYPED[j.$ta as TypedName];
    if (!C) throw new StyleError(`restart file: ${path} has unknown array type ${String(j.$ta)}`);
    return C.from(((j.v as Json[]) ?? []).map((x) => decodeValue(x, path) as number));
  }
  if ('$map' in j) {
    return new Map(((j.$map as Json[][]) ?? []).map(([k, x]) => [decodeValue(k, path), decodeValue(x, path)] as const));
  }
  if ('$inst' in j) throw new StyleError(`restart file: ${path} holds a ${String(j.$inst)} object without a restore target`);
  return Object.fromEntries(Object.entries(j).map(([k, x]) => [k, decodeValue(x, `${path}.${k}`)]));
};

const isInst = (j: Json | undefined): j is { $inst: string; f: { [k: string]: Json } } =>
  typeof j === 'object' && j !== null && !Array.isArray(j) && '$inst' in j;

/** Overwrites the fields of an existing object (a freshly made style) with the stored ones. */
const restoreInto = (target: Obj, j: { f: { [k: string]: Json } }, path: string): void => {
  for (const [k, x] of Object.entries(j.f)) {
    if (!(k in target)) throw new StyleError(`restart file: ${path}.${k} is not a field of this style (the file was written by another version?)`);
    const cur = target[k];
    if (isInst(x) && isObject(cur)) restoreInto(cur, x, `${path}.${k}`);
    else target[k] = decodeValue(x, `${path}.${k}`);
  }
};

const registryOf = (kind: 'bond' | 'angle' | 'dihedral' | 'improper') => ({
  bond: BOND_STYLES, angle: ANGLE_STYLES, dihedral: DIHEDRAL_STYLES, improper: IMPROPER_STYLES,
})[kind];

const BONDED_KINDS = ['bond', 'angle', 'dihedral', 'improper'] as const;

/** Bonded styles with coefficients in the file: all except table styles (and hybrid, which stores only its sub-style list). */
const bondedStorable = (name: string): boolean => !name.includes('table') && !name.includes('hybrid');

const bondedTypeCount = (s: SimState, kind: typeof BONDED_KINDS[number]): number => ({
  bond: s.topo.nbondtypes, angle: s.topo.nangletypes, dihedral: s.topo.ndihedraltypes, improper: s.topo.nimpropertypes,
})[kind];

/** The restart text (magic line, then one JSON line). Throws StyleError for anything it cannot store. */
export const writeRestartText = (sys: System): string => {
  const s = sys.state;
  if (s.custom.size || s.propMol || s.propQ || (s.rmass && s.atomStyle !== 'sphere')) {
    throw new StyleError('write_restart: per-atom properties from fix property/atom are not stored in the browser restart file; use write_data (its fix sections keep them)');
  }
  const pair = sys.ff.pair;
  if (pair && !PAIR_RESTART_STYLES.has(pair.name)) {
    throw new StyleError(`write_restart: pair_style ${pair.name} is not stored in the browser restart file (its coefficients come from files or it is a hybrid style); re-specify it after read_restart, or use write_data`);
  }
  const styles: Record<string, Json> = {};
  for (const kind of BONDED_KINDS) {
    const st = sys.ff[kind];
    if (st && !bondedStorable(st.name)) {
      throw new StyleError(`write_restart: ${kind}_style ${st.name} is not stored in the browser restart file (its coefficients are tabulated or it is hybrid); re-specify it after read_restart`);
    }
  }
  const { f: _f, custom: _c, ...rest } = s;
  const styleEntry = (name: string, obj: unknown) => ({ name, data: encodeValue(obj, name, []) });
  styles.pair = pair ? styleEntry(pair.name, pair) : null;
  for (const kind of BONDED_KINDS) {
    const st = sys.ff[kind];
    styles[kind] = st ? styleEntry(st.name, st) : null;
  }
  const doc = {
    format: FORMAT,
    version: VERSION,
    step: s.step,
    state: encodeValue(rest, 'state', []),
    comm: { style: sys.commStyle, vel: sys.ghostVelocity, cutoff: sys.nb.commCutoff },
    groups: encodeValue({ names: sys.groups.names, dynamic: sys.groups.dynamic }, 'groups', []),
    special: encodeValue(sys.ff.special, 'special', []),
    styles,
  };
  return `${RESTART_MAGIC}\n${JSON.stringify(doc)}\n`;
};

const need = (cond: boolean, msg: string): void => {
  if (!cond) throw new StyleError(`restart file: ${msg}`);
};

/**
 * Reads restart text into an empty System (the read_restart command checks
 * that no box exists). Replaces the box, atoms, groups and force field
 * styles, as read_restart does; everything the file does not hold is left
 * as the System had it.
 */
export const readRestartText = (sys: System, text: string, name: string): void => {
  const nl = text.indexOf('\n');
  const header = (nl < 0 ? text : text.slice(0, nl)).trim();
  if (header !== RESTART_MAGIC) {
    if (header.startsWith('LAMMPS-WEB-RESTART')) throw new StyleError(`${name}: restart format '${header}' is not supported (this engine reads '${RESTART_MAGIC}')`);
    throw new StyleError(`${name} is not a restart file written by this browser engine. Native LAMMPS binary restart files cannot be read in the browser: their format is platform-specific. Convert the native file with the lmp -restart2data command-line flag and use read_data, or write the state here with write_restart (or write_data)`);
  }
  let doc: { format: string; version: number; step: number; state: Json; comm: { style: 'brick' | 'tiled'; vel: boolean; cutoff: number }; groups: Json; special: Json; styles: Record<string, { name: string; data: Json } | null> };
  try {
    doc = JSON.parse(text.slice(nl + 1));
  } catch {
    throw new StyleError(`${name}: the restart file is damaged (its JSON does not parse)`);
  }
  need(doc.format === FORMAT && doc.version === VERSION, `${name} is not a version ${VERSION} restart file`);
  const state = decodeValue(doc.state, 'state') as SimState;
  const n = state.n;
  for (const [k, len] of [['x', 3 * n], ['v', 3 * n], ['image', 3 * n], ['id', n], ['type', n], ['mask', n], ['molecule', n], ['q', n]] as const) {
    need((state[k] as ArrayLike<number>).length === len, `${k} has the wrong length for ${n} atoms`);
  }
  // forces are not stored: the next run computes them
  state.f = new Float64Array(3 * n);
  state.custom = new Map();
  state.propMol = false;
  state.propQ = false;

  sys.units = state.units;
  sys.dimension = state.dimension;
  sys.boundary = state.box.boundary.map((f) => [f[0], f[1]]) as SimState['box']['boundary'];
  sys.atomStyle = state.atomStyle;
  sys.commStyle = doc.comm.style;
  sys.ghostVelocity = doc.comm.vel;
  sys.nb.commCutoff = doc.comm.cutoff;
  // "timestep" given before read_restart is replaced by the file's timestep size
  sys.pendingDt = null;
  sys.setState(state);

  const g = decodeValue(doc.groups, 'groups') as { names: (string | null)[]; dynamic: Map<number, DynamicGroup> };
  sys.groups.names = g.names;
  sys.groups.dynamic.clear();
  for (const [bit, dg] of g.dynamic) sys.groups.dynamic.set(bit, dg);

  sys.ff.special = decodeValue(doc.special, 'special') as SpecialBonds;

  const restoreStyle = <T extends Pair | Bonded>(
    kind: string, registry: Record<string, () => T>, entry: { name: string; data: Json } | null | undefined, ntypes: number,
  ): T | null => {
    if (!entry) return null;
    const make = registry[entry.name];
    if (!make) throw new StyleError(`restart file: ${kind} style ${entry.name} is not available in this engine`);
    const st = make();
    st.allocate(ntypes);
    need(isInst(entry.data), `${kind} data is not an object`);
    restoreInto(st as unknown as Obj, entry.data as { f: { [k: string]: Json } }, kind);
    return st;
  };
  const pair = restoreStyle('pair', PAIR_STYLES as Record<string, () => Pair>, doc.styles.pair, state.ntypes);
  sys.ff.pair = pair;
  for (const kind of BONDED_KINDS) {
    const st = restoreStyle(kind, registryOf(kind) as Record<string, () => Bonded>, doc.styles[kind], bondedTypeCount(state, kind));
    sys.ff[kind] = st;
  }
  sys.bump();
  sys.log(`Restoring from restart file ${name}: ${n} atoms at step ${state.step}`);
};

import type { AtomStyle, CustomProp, SimState } from '../types';
import type { System } from '../system';
import { LABEL_KINDS, LabelMap } from '../labelmap';
import type { SpecialBonds } from '../force/forcefield';
import type { DynamicGroup } from '../group';
import { StyleError, type Bonded, type Pair } from '../force/types';
import { PairHybrid, type HybridMode } from '../force/pair/hybrid';
import { PAIR_STYLES, BOND_STYLES, ANGLE_STYLES, DIHEDRAL_STYLES, IMPROPER_STYLES } from '../styles';
import { FixPropertyAtom } from '../fix/property_atom';
import { FixCmap } from '../fix/cmap';
import { FixNH } from '../fix/nh';
import { type FixRecord, isRestartableFix } from '../fix/restartRecord';
import { BondedHybrid, type HybridKind } from '../force/bonded_hybrid';
import { gplHook } from '../gplHooks';

/** A fix's record in the restart file (fix/restartRecord.ts). */
type FixRestart = FixRecord;

/*
 * Fix styles whose native class keeps state in the binary restart file (restart_global or restart_peratom in
 * the LAMMPS source at b332f28, checked against the class's write_restart / pack_restart) and that this engine
 * does not store yet. write_restart refuses them by name: dropping their state silently would make a restarted
 * run differ from the uninterrupted one without a word. The fixes that are stored are fix nvt/npt/nph and their
 * variants (FixNH), fix nphug, npt/cauchy, msst and qbmsst (restartState), fix property/atom and fix cmap.
 */
const FIX_STATE_NOT_STORED = new Set([
  // widom: fix_widom.cpp restart() reads a third value (list[2]) that write_restart never wrote, so the next event step
  // native restores is not defined; its state cannot be stored exactly
  'widom',
  'adapt', 'amoeba/bitorsion', 'amoeba/pitorsion', 'bocs', 'bond/react', 'deform',
  'graphics/lines', 'rigid/nph', 'rigid/nph/small', 'rigid/npt', 'rigid/npt/small', 'rigid/nvt', 'rigid/nvt/small',
  'smd/adjust_dt', 'spring/self', 'store/state', 'tgnpt/drude', 'tgnvt/drude', 'ti/spring', 'ttm', 'ttm/cascade',
  'ttm/grid', 'ttm/mod', 'ttm/thermal', 'wall/body/polygon', 'wall/body/polyhedron', 'wall/gran', 'wall/gran/region',
  'gle', 'move',
]);
import { SMD_COLS, isSmdStyle } from '../smd';
import { HEAT_FLOW, HEAT_TEMPERATURE, hasChargeStyle, isMolecularStyle, hasRmassStyle, hasDipoleStyle, isEllipsoidStyle, isBpmSphereStyle, isTemplateStyle, templateStyleId, isOxdnaStyle, OXDNA_5P, OXDNA_3P, setOxdnaPartners, isTriStyle, LINE_FLAG, LINE_BONUS, TRI_FLAG, TRI_BONUS, isElectronStyle, ELECTRON_DEFAULTS, ELECTRON_SPIN, isDielectricStyle, DIELECTRIC_DEFAULTS, isSphStyle, SPH_DEFAULTS, SPH_WIDTH, isDpdMesoKey, isDpdMesoStyle, dpdMesoProps, RHEO_DEFAULTS, RHEO_STATUS, isRheoStyle, isRheoThermalStyle, RHEO_THERMAL_DEFAULTS, DPD_DEFAULTS, isDpdStyle } from '../atoms';
import { ErfcTableCache } from '../force/erfc';

/*
 * The browser engine's restart file (write_restart / read_restart).
 *
 * A native LAMMPS binary restart file is read by the lazily loaded reader (engine-gpl/restart/nativeRestart.ts,
 * dispatched from readRestartText by its magic string). read_restart.rst says: "The binary restart file format was not
 * designed with backward, forward, or cross-platform compatibility in mind", so this module writes its own text format
 * and reads only that format and the native binary one: a magic first line, then one JSON document
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
 * Fixes: the state a fix keeps in the native restart file (fix/restartRecord.ts) is stored per fix ID and style and
 * taken back when a fix with the same ID and style is re-specified (fix_nh.html, fix_msst.html, fix_qbmsst.html,
 * fix_property_atom.html, fix_cmap.html and the other pages that say "writes ... to binary restart files"). A fix
 * whose native class keeps state this engine does not store is refused by name (FIX_STATE_NOT_STORED), never dropped.
 *
 * Not stored (read_restart.rst: "Here is a list of information not stored in a
 * restart file"): computes, variables, regions, neighbor settings,
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
 *  - hybrid styles: the sub-style list is stored and restored; the coefficients are
 *    not (bond_hybrid.rst: "Unlike other bond styles, the hybrid bond style does not store bond").
 */

export const RESTART_MAGIC = 'LAMMPS-WEB-RESTART 1';
/** The first 16 bytes of a native binary restart file (lmprestart.h MAGIC_STRING "LammpS RestartT" and its NUL). */
const NATIVE_RESTART_MAGIC_TEXT = 'LammpS RestartT\u0000';
/** The lazily loaded native reader (gplHook 'native-restart'). */
type NativeRestartReader = (sys: System, text: string, name: string) => void;
/** The internal RHEO props stored in a restart file for a style (atom_style rheo, or rheo/thermal with its thermal props). */
const rheoKeys = (st: AtomStyle): string[] => [...Object.keys(RHEO_DEFAULTS), ...(isRheoThermalStyle(st) ? Object.keys(RHEO_THERMAL_DEFAULTS) : [])];

const FORMAT = 'lammps-web-restart';
const VERSION = 1;

/*
 * Pair styles whose doc page says they write their information to binary restart files, so their
 * coefficients are stored here and restored by read_restart. Each name is the registry key; a
 * style family covered by one page lists every member the page names (pair_lj.rst "All of the
 * *lj/cut* pair styles write their information to binary restart files"). The CS variants follow
 * their base style (pair_cs.rst: "See the corresponding doc pages for pair styles without the "cs"
 * suffix").
 */
const PAIR_RESTART_STYLES = new Set([
  // pair_lj.rst, pair_lj_cut_coul.rst, pair_lj_cut_tip4p.rst, pair_lj_long.rst
  'lj/cut', 'lj/cut/coul/cut', 'lj/cut/coul/debye', 'lj/cut/coul/dsf', 'lj/cut/coul/wolf',
  'lj/cut/coul/long', 'lj/cut/tip4p/cut', 'lj/cut/tip4p/long', 'lj/long/coul/long',
  // pair_coul.rst, pair_coul_slater.rst
  'coul/cut', 'coul/debye', 'coul/dsf', 'coul/wolf', 'coul/long', 'coul/long/cs',
  // pair_born.rst, pair_buck.rst
  'born', 'born/coul/long', 'born/coul/long/cs', 'buck', 'buck/coul/cut', 'buck/coul/long', 'buck/coul/long/cs',
  // pair_charmm.rst, pair_class2.rst, pair_gromacs.rst, pair_gauss.rst
  'lj/charmm/coul/charmm', 'lj/charmm/coul/charmm/implicit', 'lj/charmm/coul/long',
  'lj/charmmfsw/coul/charmmfsh', 'lj/charmmfsw/coul/long', 'lj/class2', 'lj/gromacs', 'gauss',
  // pair_lj_cubic.rst, pair_lj_expand.rst, pair_lj_relres.rst, pair_lj_smooth.rst, pair_lj_smooth_linear.rst
  'lj/cubic', 'lj/expand', 'lj/relres', 'lj/smooth', 'lj/smooth/linear',
  // pair_lj96.rst, pair_mie.rst, pair_morse.rst, pair_soft.rst, pair_yukawa.rst, pair_yukawa_colloid.rst
  'lj96/cut', 'mie/cut', 'morse', 'soft', 'yukawa', 'yukawa/colloid',
  // the spin pair styles (pair_spin_exchange.html and the spin pair pages): measured with native LAMMPS (black box):
  // write_restart keeps the pair style and its coefficients (examples/SPIN/read_restart/in.spin.write_restart)
  'spin/exchange', 'spin/exchange/biquadratic', 'spin/dmi', 'spin/magelec', 'spin/dipole/cut', 'spin/neel',
  // pair_colloid.rst, pair_zero.rst, pair_atm.rst, pair_gran.rst, pair_granular.rst
  'colloid', 'zero', 'atm',
  'gran/hooke', 'gran/hooke/history', 'gran/hertz/history', 'granular',
  // pair_eff.rst: "These pair styles write their information to binary restart files" (eff/cut: the settings,
  // the ECP parameters and the per-type cutoffs, as src/EFF/pair_eff_cut.cpp write_restart at b332f28)
  'eff/cut',
]);

/*
 * Pair styles whose doc pages say they keep nothing in restart files, e.g. pair_eam.html: "The eam
 * pair styles do not write their information to :doc:`binary restart files <restart>`, since it is
 * stored in tabulated potential files.  Thus, you need to re-specify the pair_style and pair_coeff
 * commands in an input script that reads a restart file." (pair_sw.html and the others say the same;
 * the list below is every style whose page says so). write_restart stores no pair style for them.
 */
export const PAIR_NOT_IN_RESTART = new Set([
  'adp', 'agni', 'airebo', 'airebo/bc', 'airebo/morse', 'amoeba', 'body/nparticle', 'body/rounded/polygon',
  'body/rounded/polyhedron', 'bop', 'comb', 'comb3', 'dispersion/d3', 'e3b', 'eam', 'eam/alloy',
  'eam/apip', 'eam/cd', 'eam/cd/old', 'eam/fs', 'eam/fs/apip', 'eam/he', 'edip', 'edip/multi',
  'gw', 'gw/zbl', 'hbond/dreiding/lj', 'hbond/dreiding/lj/angleoffset', 'hbond/dreiding/morse',
  'hbond/dreiding/morse/angleoffset', 'hippo', 'kim', 'lcbop', 'line/lj', 'list', 'ldd', 'local/density',
  'meam', 'meam/ms', 'meam/spline', 'meam/sw/spline', 'mesocnt', 'mesocnt/viscous', 'mgpt',
  'mliap', 'pace', 'pace/apip', 'pace/extrapolation', 'pace/fast/apip', 'pace/precise/apip', 'pod',
  'polymorphic', 'python', 'quip', 'reaxff', 'rebo', 'rebomos', 'smtbq', 'snap', 'sw',
  'sw/angle/table', 'sw/mod', 'tersoff', 'tersoff/mod', 'tersoff/mod/c', 'tersoff/table',
  'tersoff/zbl', 'threebody/table', 'tri/lj', 'uf3', 'vashishta', 'vashishta/table',
  'zbl', 'sph/heatconduction', 'sph/idealgas', 'sph/lj', 'sph/rhosum', 'sph/taitwater',
  'sph/taitwater/morris', 'sdpd/taitwater/isothermal', 'hdnnp', 'rheo',
]);

/*
 * Pair styles that store their settings but not their coefficients (pair_table.rst: "This pair style
 * writes the settings for the \"pair_style table\" command to :doc:`binary restart files <restart>`, so a pair_style
 * command does not need to specified in an input script that reads a restart file. However, the
 * coefficient information is not stored in the restart file, since it is tabulated in the potential
 * files. Thus, pair_coeff commands do need to be specified in the restart input script."). The value
 * lists the fields that hold coefficients; they are left out of the file.
 */
const PAIR_SETTINGS_ONLY: Record<string, readonly string[]> = {
  table: ['p', 'inputs', 'tRsq1', 'tInvDelta', 'tE', 'tG', 'tE2', 'tG2'],
};

/*
 * lepton, lepton/coul and lepton/sphere (pair_lepton.rst: "These pair styles write their information to
 * :doc:`binary restart files <restart>`") are not in PAIR_RESTART_STYLES yet: their compiled expressions are functions,
 * and a restore needs a hook that recompiles them from the stored expression text (HOOKS NEEDED).
 */

/** A copy of a pair style object without the listed fields, keeping its class (for encodeValue). */
const withoutFields = <T extends object>(obj: T, drop: readonly string[]): T => {
  const copy = Object.create(Object.getPrototypeOf(obj)) as T;
  for (const [k, v] of Object.entries(obj)) if (!drop.includes(k)) (copy as Record<string, unknown>)[k] = v;
  return copy;
};

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
    // derived caches (the erfc tables of the coul/long family) are rebuilt by the restored style
    const entries = Object.entries(v)
      .filter(([, x]) => !(x instanceof ErfcTableCache))
      .map(([k, x]) => [k, encodeValue(x, `${path}.${k}`, stack)] as const);
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

/** Bonded styles with coefficients in the file: all except table styles (a hybrid style stores only its sub-style list, see below). */
const bondedStorable = (name: string): boolean => !name.includes('table');

/**
 * The sub-style list of a hybrid bonded style. Measured with native LAMMPS (black box): read_restart
 * restores the hybrid style, the listed sub-styles are usable without a new bond_style, and no
 * coefficients come back (bond_coeff must be given again, otherwise the run stops with All bond coeffs
 * are not set). bond_hybrid.rst: "Unlike other bond styles, the hybrid bond style does not store bond".
 */
const restoreHybrid = (kind: HybridKind, data: Json, ntypes: number, reg: Record<string, () => Bonded>): Bonded => {
  const subs = typeof data === 'object' && data !== null && !Array.isArray(data) ? data.subs : undefined;
  need(Array.isArray(subs) && subs.every((x) => typeof x === 'string'), `restart file: ${kind} hybrid data has no sub-style list`);
  const st = new BondedHybrid(kind, reg);
  st.settings(subs as string[]);
  st.allocate(ntypes);
  return st;
};

/**
 * Rebuilds a pair_style hybrid from hybridPairData: the stored arguments go through settings() (the sub-styles are made
 * as the pair_style command makes them), then the stored special weights and scalar settings are put back. No
 * coefficients are set, so pair_coeff is needed again, as after native read_restart.
 */
const restoreHybridPair = (mode: HybridMode, data: Json, ntypes: number, sys: System): PairHybrid => {
  const d = decodeValue(data, 'pair') as { args?: unknown; subs?: unknown };
  need(Array.isArray(d.args) && d.args.every((x) => typeof x === 'string'), `${mode} data has no argument list`);
  need(Array.isArray(d.subs), `${mode} data has no sub-style list`);
  const hy = new PairHybrid(mode, PAIR_STYLES as Record<string, () => Pair>);
  hy.settings(d.args as string[], sys.styleContext());
  hy.allocate(ntypes);
  const subs = d.subs as { name: string; specialLJ: number[] | null; specialCoul: number[] | null; fields: Record<string, unknown> }[];
  need(subs.length === hy.subs.length, `${mode} sub-style count does not match its arguments`);
  hy.subs.forEach((sub, k) => {
    const stored = subs[k];
    need(stored.name === sub.name, `${mode} sub-style ${k + 1} is ${sub.name}, the file has ${stored.name}`);
    for (const [key, val] of Object.entries(stored.fields)) {
      need(key in sub.style, `restart file: ${mode} sub-style ${sub.name} has no field ${key} (the file was written by another version?)`);
      (sub.style as unknown as Record<string, unknown>)[key] = val;
    }
    sub.specialLJ = stored.specialLJ ? Float64Array.from(stored.specialLJ) : null;
    sub.specialCoul = stored.specialCoul ? Float64Array.from(stored.specialCoul) : null;
  });
  return hy;
};

const bondedTypeCount = (s: SimState, kind: typeof BONDED_KINDS[number]): number => ({
  bond: s.topo.nbondtypes, angle: s.topo.nangletypes, dihedral: s.topo.ndihedraltypes, improper: s.topo.nimpropertypes,
})[kind];

/** The restart text (magic line, then one JSON line). Throws StyleError for anything it cannot store. */
export const writeRestartText = (sys: System): string => {
  const s = sys.state;
  // fixes whose native state is not stored: a marker record, and one warning naming them (checkFixStateNotStored
  // refuses a fix with the same ID and style after read_restart)
  const unstored: string[] = [];
  // fix_property_atom.html: "This fix writes the per-atom values it stores to :doc:`binary restart
  // files <restart>`, so that the values can be restored when a simulation is restarted."
  const fixes: FixRestart[] = [];
  for (const f of sys.fixes) {
    if (FIX_STATE_NOT_STORED.has(f.style)) {
      unstored.push(`${f.id} (${f.style})`);
      fixes.push({ id: f.id, style: f.style, props: [], data: {}, stateNotStored: true });
      continue;
    }
    // fix_cmap.html: "This fix writes the list of CMAP cross-terms to binary restart files"
    if (f instanceof FixCmap) { fixes.push({ id: f.id, style: f.style, props: [], data: {}, cmap: f.restartState() }); continue; }
    // fix_nh.html: the thermostat and barostat variables go to the binary restart file
    if (f instanceof FixNH) { fixes.push({ id: f.id, style: f.style, props: [], data: {}, nh: f.restartState() }); continue; }
    // fix_msst.html, fix_qbmsst.html, fix_nphug.html: the omega, e0, v0 and p0 (and T_qm) the fix keeps (restartState)
    if (isRestartableFix(f)) { fixes.push({ id: f.id, style: f.style, props: [], data: {}, state: f.restartState() }); continue; }
    if (!(f instanceof FixPropertyAtom)) continue;
    const data: Record<string, number[]> = {};
    for (const p of f.props) {
      if (p.kind === 'mol') data.mol = Array.from(s.molecule.subarray(0, s.n));
      else if (p.kind === 'q') data.q = Array.from(s.q.subarray(0, s.n));
      else if (p.kind === 'rmass') data.rmass = Array.from(s.rmass!.subarray(0, s.n));
      else if (p.kind === 'temperature' || p.kind === 'heatflow') data[p.kind] = Array.from(s.custom.get(p.kind === 'temperature' ? HEAT_TEMPERATURE : HEAT_FLOW)!.data.subarray(0, s.n));
      else data[`${p.kind}_${p.name}`] = Array.from(s.custom.get(p.name)!.data);
    }
    fixes.push({ id: f.id, style: f.style, props: f.props.map((p) => ({ ...p })), data });
  }
  if (unstored.length) sys.warn(`state of fix ${unstored.join(', ')} is not stored in the browser restart file`);
  const pair = sys.ff.pair;
  const pairStored = pair && !PAIR_NOT_IN_RESTART.has(pair.name) ? pair : null;
  // pair_hybrid.cpp write_restart (b332f28): the sub-style names and settings, no coefficients; each sub-style must be one
  // of the styles stored here (a hybrid cannot store a sub-style whose settings are not in the file)
  if (pair instanceof PairHybrid) {
    for (const sub of pair.subs) {
      if (!PAIR_RESTART_STYLES.has(sub.name) && !(sub.name in PAIR_SETTINGS_ONLY)) {
        throw new StyleError(`write_restart: pair_style ${pair.name} sub-style ${sub.name} is not stored in the browser restart file yet; re-specify it after read_restart, or use write_data`);
      }
    }
  } else if (pairStored && !PAIR_RESTART_STYLES.has(pairStored.name) && !(pairStored.name in PAIR_SETTINGS_ONLY)) {
    throw new StyleError(`write_restart: pair_style ${pairStored.name} is not stored in the browser restart file yet; re-specify it after read_restart, or use write_data`);
  }
  if (pair && !pairStored) sys.log(`write_restart: pair_style ${pair.name} keeps its coefficients in potential files and is not stored; re-specify pair_style and pair_coeff after read_restart`);
  const styles: Record<string, Json> = {};
  for (const kind of BONDED_KINDS) {
    const st = sys.ff[kind];
    if (st && !bondedStorable(st.name)) {
      throw new StyleError(`write_restart: ${kind}_style ${st.name} is not stored in the browser restart file (its coefficients are tabulated); re-specify it after read_restart`);
    }
  }
  const { f: _f, custom: _c, ...rest } = s;
  // atom_style line and tri: the per-atom line/triangle flags and bonus data (atoms.ts LINE_FLAG ... TRI_BONUS) are stored with the atoms
  const lineTri = new Map([...s.custom].filter(([k]) => k === LINE_FLAG || k === LINE_BONUS || k === TRI_FLAG || k === TRI_BONUS));
  const styleEntry = (name: string, obj: unknown) => ({ name, data: encodeValue(obj, name, []) });
  styles.pair = pair instanceof PairHybrid
    ? { name: pair.name, data: hybridPairData(pair) }
    : pairStored
      ? styleEntry(pairStored.name, pairStored.name in PAIR_SETTINGS_ONLY ? withoutFields(pairStored, PAIR_SETTINGS_ONLY[pairStored.name]) : pairStored)
      : null;
  if (pair && !pairStored) styles.pairNotStored = pair.name;
  for (const kind of BONDED_KINDS) {
    const st = sys.ff[kind];
    styles[kind] = st instanceof BondedHybrid ? { name: 'hybrid', data: { subs: st.subNames } } : st ? styleEntry(st.name, st) : null;
  }
  // atom_style electron: the electron properties are internal custom props, not part of the state, so they are stored apart
  const electron = isElectronStyle(s.atomStyle)
    ? encodeValue(Object.fromEntries(Object.keys(ELECTRON_DEFAULTS).map((k) => [k, s.custom.get(k)!.data])), 'electron', [])
    : null;
  // atom_style dielectric: the surface and dielectric properties (area, ed, em, epsilon, curvature, q_scaled) are internal custom props too
  const dielectric = isDielectricStyle(s.atomStyle)
    ? encodeValue(Object.fromEntries(Object.keys(DIELECTRIC_DEFAULTS).map((k) => [k, s.custom.get(k)!.data])), 'dielectric', [])
    : null;
  // atom_style sph: the density, energy, heat capacity and velocity estimate (atom_style.html: "rho, esph, cv") are internal props too
  const sph = isSphStyle(s.atomStyle)
    ? encodeValue(Object.fromEntries(['#rho', '#esph', '#cv', '#vest'].map((k) => [k, s.custom.get(k)!.data])), 'sph', [])
    : null;
  // atom_style edpd / mdpd / tdpd: the DPD-MESO per-atom props (temperature, flux, rho, cc, vest) are internal custom props
  const dpdMeso = isDpdMesoStyle(s.atomStyle)
    ? encodeValue(new Map([...s.custom].filter(([k]) => isDpdMesoKey(k))), 'dpdMeso', [])
    : null;
  // atom_style rheo: the density, status and the RHEO fix's per-atom fields are internal custom props (atoms.ts RHEO_DEFAULTS)
  const rheo = isRheoStyle(s.atomStyle)
    ? encodeValue(Object.fromEntries(rheoKeys(s.atomStyle).map((k) => [k, s.custom.get(k)!.data])), 'rheo', [])
    : null;
  // atom_style dpd: the internal temperature and energies are internal custom props, stored apart like the dielectric ones
  const dpd = isDpdStyle(s.atomStyle)
    ? encodeValue(Object.fromEntries(Object.keys(DPD_DEFAULTS).map((k) => [k, s.custom.get(k)!.data])), 'dpd', [])
    : null;
  // atom_style smd: the SMD data (deformation gradient, stress, estimated velocity, energy, plastic strain, damage, contact radius)
  const smd = isSmdStyle(s.atomStyle)
    ? encodeValue(Object.fromEntries(Object.keys(SMD_COLS).map((k) => [k, s.custom.get(k)!.data])), 'smd', [])
    : null;
  const doc = {
    format: FORMAT,
    version: VERSION,
    step: s.step,
    state: encodeValue(rest, 'state', []),
    lineTri: encodeValue(lineTri, 'lineTri', []),
    electron,
    dielectric,
    sph,
    dpdMeso,
    rheo,
    dpd,
    smd,
    comm: { style: sys.commStyle, vel: sys.ghostVelocity, cutoff: sys.nb.commCutoff },
    groups: encodeValue({ names: sys.groups.names, dynamic: sys.groups.dynamic }, 'groups', []),
    special: encodeValue(sys.ff.special, 'special', []),
    // Howto_type_labels.html: "This map is also written to and read from restart files, by the write_restart and read_restart commands."
    labels: { defined: sys.labels.defined, maps: Object.fromEntries(LABEL_KINDS.map((k) => [k, sys.labels.entries(k)])) },
    styles,
    fixes,
  };
  return `${RESTART_MAGIC}\n${JSON.stringify(doc)}\n`;
};

/**
 * The stored form of a pair_style hybrid (pair_hybrid.cpp write_restart at b332f28): the pair_style arguments, which
 * rebuild the sub-styles and their settings, and per sub-style its special weights and its scalar fields (the settings
 * that pair_modify changed, such as shift, tail or tabinner). Coefficient arrays are left out, as native writes no
 * coefficient info; pair_coeff is specified again after read_restart.
 */
const hybridPairData = (pair: PairHybrid): Json => encodeValue({
  args: pair.settingsArgs,
  subs: pair.subs.map((sub) => ({
    name: sub.name,
    specialLJ: sub.specialLJ ? Array.from(sub.specialLJ) : null,
    specialCoul: sub.specialCoul ? Array.from(sub.specialCoul) : null,
    fields: Object.fromEntries(Object.entries(sub.style).filter(([, v]) => v === null || typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string')),
  })),
}, 'pair', []);

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
  // a native LAMMPS binary restart file (its magic string, lmprestart.h MAGIC_STRING): the lazily loaded reader reads it
  // into this System (src/engine-gpl/restart/nativeRestart.ts); the session loads it before read_restart
  if (text.startsWith(NATIVE_RESTART_MAGIC_TEXT)) {
    gplHook<NativeRestartReader>('native-restart')(sys, text, name);
    return;
  }
  const nl = text.indexOf('\n');
  const header = (nl < 0 ? text : text.slice(0, nl)).trim();
  if (header !== RESTART_MAGIC) {
    if (header.startsWith('LAMMPS-WEB-RESTART')) throw new StyleError(`${name}: restart format '${header}' is not supported (this engine reads '${RESTART_MAGIC}')`);
    throw new StyleError(`${name} is not a restart file written by this browser engine or by native LAMMPS (a native binary restart file starts with 'LammpS RestartT'). Use write_restart or write_data here; a native file of another kind can be converted with the lmp -restart2data command-line flag and read with read_data`);
  }
  let doc: { format: string; version: number; step: number; state: Json; electron?: Json | null; dielectric?: Json | null; sph?: Json | null; dpdMeso?: Json | null; rheo?: Json | null; dpd?: Json | null; smd?: Json | null; comm: { style: 'brick' | 'tiled'; vel: boolean; cutoff: number }; groups: Json; special: Json; styles: Record<string, { name: string; data: Json } | null> & { pairNotStored?: string }; fixes?: FixRestart[]; lineTri?: Json };
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
  // atom_style line and tri: their flags and bonus data are read back from the file (restart.ts writer)
  if (doc.lineTri !== undefined) {
    const lt = decodeValue(doc.lineTri, 'lineTri') as Map<string, CustomProp>;
    for (const [k, c] of lt) {
      need(c.data.length === (c.cols > 0 ? c.cols : 1) * n, `${k} has the wrong length for ${n} atoms`);
      state.custom.set(k, c);
    }
  }
  // atom_style electron: the stored electron properties (espin, eradius, ervel, erforce) come back as they were written
  if (isElectronStyle(state.atomStyle)) {
    const ep = doc.electron ? decodeValue(doc.electron, 'electron') as Record<string, Float64Array> : {};
    for (const [name, def] of Object.entries(ELECTRON_DEFAULTS)) {
      const data = ep[name] ?? new Float64Array(n).fill(def);
      need(data.length === n, `electron ${name} has the wrong length for ${n} atoms`);
      state.custom.set(name, { int: name === ELECTRON_SPIN, cols: 0, data: Float64Array.from(data) });
    }
  }
  // atom_style dielectric: the stored per-atom dielectric properties come back as they were written
  if (isDielectricStyle(state.atomStyle)) {
    const dp = doc.dielectric ? decodeValue(doc.dielectric, 'dielectric') as Record<string, Float64Array> : {};
    for (const [name, def] of Object.entries(DIELECTRIC_DEFAULTS)) {
      const data = dp[name] ?? new Float64Array(n).fill(def);
      need(data.length === n, `dielectric ${name} has the wrong length for ${n} atoms`);
      state.custom.set(name, { int: false, cols: 0, data: Float64Array.from(data) });
    }
  }
  // atom_style sph: the stored density, energy, heat capacity and velocity estimate come back as they were written
  if (isSphStyle(state.atomStyle)) {
    const sp = doc.sph ? decodeValue(doc.sph, 'sph') as Record<string, Float64Array> : {};
    for (const [name, def] of Object.entries(SPH_DEFAULTS)) {
      const w = SPH_WIDTH[name] ?? 0;
      const data = sp[name] ?? new Float64Array(n * Math.max(w, 1)).fill(def);
      need(data.length === n * Math.max(w, 1), `sph ${name} has the wrong length for ${n} atoms`);
      state.custom.set(name, { int: false, cols: w, data: Float64Array.from(data) });
    }
  }
  // atom_style edpd / mdpd / tdpd: the stored DPD-MESO props come back with their columns (dpdMesoProps)
  if (isDpdMesoStyle(state.atomStyle)) {
    const dm = doc.dpdMeso ? decodeValue(doc.dpdMeso, 'dpdMeso') as Map<string, CustomProp> : new Map<string, CustomProp>();
    for (const [name, cols] of dpdMesoProps(state.atomStyle)) {
      const c = dm.get(name) ?? { int: false, cols, data: new Float64Array((cols > 0 ? cols : 1) * n) };
      need(c.data.length === (c.cols > 0 ? c.cols : 1) * n, `${name} has the wrong length for ${n} atoms`);
      state.custom.set(name, c);
    }
  }
  // atom_style rheo: the stored RHEO per-atom fields come back as they were written
  if (isRheoStyle(state.atomStyle)) {
    const rp = doc.rheo ? decodeValue(doc.rheo, 'rheo') as Record<string, Float64Array> : {};
    for (const [name, def] of Object.entries({ ...RHEO_DEFAULTS, ...(isRheoThermalStyle(state.atomStyle) ? RHEO_THERMAL_DEFAULTS : {}) })) {
      const data = rp[name] ?? new Float64Array(n).fill(def);
      need(data.length === n, `rheo ${name} has the wrong length for ${n} atoms`);
      state.custom.set(name, { int: name === RHEO_STATUS, cols: 0, data: Float64Array.from(data) });
    }
  }
  // atom_style dpd: the stored per-atom DPD properties come back as they were written
  if (isDpdStyle(state.atomStyle)) {
    const dp = doc.dpd ? decodeValue(doc.dpd, 'dpd') as Record<string, Float64Array> : {};
    for (const [name, def] of Object.entries(DPD_DEFAULTS)) {
      const data = dp[name] ?? new Float64Array(n).fill(def);
      need(data.length === n, `dpd ${name} has the wrong length for ${n} atoms`);
      state.custom.set(name, { int: false, cols: 0, data: Float64Array.from(data) });
    }
  }
  // atom_style smd: the stored SMD data come back as they were written
  if (isSmdStyle(state.atomStyle)) {
    const sp = doc.smd ? decodeValue(doc.smd, 'smd') as Record<string, Float64Array> : {};
    for (const [name, cols] of Object.entries(SMD_COLS)) {
      const data = sp[name] ?? new Float64Array(n * Math.max(cols, 1));
      need(data.length === n * Math.max(cols, 1), `smd ${name} has the wrong length for ${n} atoms`);
      state.custom.set(name, { int: false, cols, data: Float64Array.from(data) });
    }
  }
  // atom_style oxdna: the strand partners are not stored; they follow from the stored bonds, as after read_data
  if (isOxdnaStyle(state.atomStyle)) {
    state.custom.set(OXDNA_5P, { int: true, cols: 0, data: new Float64Array(n) });
    state.custom.set(OXDNA_3P, { int: true, cols: 0, data: new Float64Array(n) });
    const b = state.topo.bonds;
    setOxdnaPartners(state, Array.from({ length: b.n }, (_, e) => [b.atoms[2 * e], b.atoms[2 * e + 1]] as const));
  }
  state.propMol = false;
  state.propQ = false;
  // per-atom values of fix property/atom wait for the fix to be re-specified (FixPropertyAtom);
  // until then the attributes the atom style lacks are absent
  if (!isMolecularStyle(state.atomStyle)) state.molecule.fill(0);
  // atom_style template: the template index and atom per atom come with the file; the bonded topology
  // they imply is already in the stored topo
  if (isTemplateStyle(state.atomStyle)) {
    need(!!state.tmplIndex && !!state.tmplAtom && state.tmplIndex.length === n && state.tmplAtom.length === n, 'template index and atom have the wrong length');
    // Measured with native LAMMPS (black box): read_restart of an atom_style template file stops with an
    // error unless the molecule command for that template ID was given first
    const tid = templateStyleId(state.atomStyle)!;
    if (!sys.molecules.has(tid)) throw new StyleError(`${name}: atom_style template needs molecule template ${tid}; define it with the molecule command before read_restart`);
  } else { state.tmplIndex = null; state.tmplAtom = null; }
  if (!hasChargeStyle(state.atomStyle)) state.q.fill(0);
  if (!hasRmassStyle(state.atomStyle)) state.rmass = null;
  if (!hasDipoleStyle(state.atomStyle)) state.mu = null;
  if (!isEllipsoidStyle(state.atomStyle)) {
    state.shape = null;
    // atom_style bpm/sphere keeps its quaternion (atoms.ts isBpmSphereStyle)
    if (!isBpmSphereStyle(state.atomStyle) && !isTriStyle(state.atomStyle)) state.quat = null;
    if (!isTriStyle(state.atomStyle)) state.angmom = null;
  }
  sys.pendingFixData = new Map((doc.fixes ?? []).map((f) => [f.id, f]));

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
  // the label map (files written before type labels existed have none: an empty map)
  const lab = (doc as { labels?: { defined: boolean; maps: Record<string, [number, string][]> } }).labels;
  sys.labels = new LabelMap();
  if (lab) {
    for (const k of LABEL_KINDS) for (const [t, l] of lab.maps[k] ?? []) sys.labels.set(k, t, l);
    sys.labels.defined = lab.defined;
  }

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
  const HYBRID_PAIRS = ['hybrid', 'hybrid/overlay', 'hybrid/scaled', 'hybrid/molecular'];
  const pair = doc.styles.pair && HYBRID_PAIRS.includes(doc.styles.pair.name)
    ? restoreHybridPair(doc.styles.pair.name as HybridMode, doc.styles.pair.data, state.ntypes, sys)
    : restoreStyle('pair', PAIR_STYLES as Record<string, () => Pair>, doc.styles.pair, state.ntypes);
  sys.ff.pair = pair;
  // measured with native LAMMPS (black box): read_restart logs pair style sw stores no restart info,
  // and the next run needs a new pair_style (ForceField.pairNotRestarted)
  const notStored = typeof doc.styles.pairNotStored === 'string' ? doc.styles.pairNotStored : null;
  sys.ff.pairNotRestarted = pair ? null : notStored;
  if (!pair && notStored) sys.log(`pair style ${notStored} stores no restart info`);
  for (const kind of BONDED_KINDS) {
    const entry = doc.styles[kind];
    const reg = registryOf(kind) as Record<string, () => Bonded>;
    const st = entry?.name === 'hybrid'
      ? restoreHybrid(kind, entry.data, bondedTypeCount(state, kind), reg)
      : restoreStyle(kind, reg, entry, bondedTypeCount(state, kind));
    sys.ff[kind] = st;
  }
  sys.bump();
  sys.log(`Restoring from restart file ${name}: ${n} atoms at step ${state.step}`);
};

import type { System } from './system';
import { StyleError } from './force/types';
import { FormulaError, errorUrl, resolveIndex, scanBracket, type Index, type Mode, type Value } from './formula';
import { hasCharge, hasMolecule, massOf } from './atoms';
import { finiteSpin } from './compute/finiteSpin';
import type { SimState } from './types';
import { isLabelKind } from './labelmap';
import { groupMotionOf } from './groupMotion';

/*
 * Group, region, special and feature functions of variable formulas —
 * docs.lammps.org/variable.html:
 * "The group function count() is the number of atoms in the group. The group
 * functions mass() and charge() are the total mass and charge of the group.
 * Xcm() and vcm() return components of the position and velocity of the
 * center of mass of the group. Fcm() returns a component of the total force
 * on the group of atoms. Bound() returns the min/max of a particular
 * coordinate for all atoms in the group. Gyration() computes the
 * radius-of-gyration of the group of atoms. ... Angmom() returns components
 * of the angular momentum of the group of atoms around its center of mass.
 * Torque() returns components of the torque on the group of atoms around its
 * center of mass, based on current forces on the atoms. Inertia() returns one
 * of 6 components of the symmetric inertia tensor of the group of atoms
 * around its center of mass, ordered as Ixx,Iyy,Izz,Ixy,Iyz,Ixz. Omega()
 * returns components of the angular velocity of the group of atoms around
 * its center of mass." Region forms add a final region ID: "The function is
 * computed for all atoms that are in both the group and the region."
 * Positions for xcm, gyration, angmom, torque, inertia and omega are
 * unwrapped with the image flags. Special functions sum(x), min(x), max(x),
 * ave(x), trap(x), slope(x), sort(x), rsort(x) work on a global vector; "gmask(x)
 * ... returns a 1 for atoms that are in the group"; rmask, grmask; next().
 * "slope() ... If the line has a single point or is vertical, it returns
 * 1.0e20."
 */

const DIMS: Record<string, number> = { x: 0, y: 1, z: 2 };

/** The operating-system text is_os() matches in the browser engine (native: platform::os_info()). */
const OS_INFO = 'Web browser';

const special = (fn: string) => new FormulaError(fn ? `Invalid special function ${fn}() in variable formula` : 'Invalid special function in variable formula');

export const groupFunction = (sys: System, fn: string, args: string[], mode: Mode, index: Index | null = null): Value => {
  switch (fn) {
    case 'count': case 'mass': case 'charge': case 'xcm': case 'vcm': case 'fcm': case 'bound':
    case 'gyration': case 'ke': case 'angmom': case 'torque': case 'inertia': case 'omega':
      return groupFn(sys, fn, args);
    case 'sum': case 'min': case 'max': case 'ave': case 'trap': case 'slope': case 'sort': case 'rsort':
      return reduceFn(sys, fn, args, mode, index);
    case 'gmask': case 'rmask': case 'grmask':
      return maskFn(sys, fn, args, mode);
    case 'next': {
      // As native LAMMPS does (Variable::special_function, variable.cpp at b332f28)
      if (args.length !== 1) throw special('');
      const v = sys.vars.get(args[0]);
      if (!v) throw new StyleError(`Variable ID '${args[0]}' in variable formula does not exist`);
      if (v.style === 'file') {
        const cur = Number(v.values[0]);
        sys.vars.next([args[0]]);
        return Number.isFinite(cur) ? cur : 0;
      }
      if (v.style !== 'atomfile') throw new FormulaError(`Variable ${args[0]}: Invalid variable style in special function next`, true);
      if (mode === 'equal') throw new FormulaError(`Variable ${args[0]}: Atomfile variable in equal-style variable formula`, true);
      const s = sys.state;
      const cur = sys.vars.evalAtom(args[0], sys.formulaEnv, s.id, s.n);
      sys.vars.next([args[0]]);
      return cur;
    }
    case 'is_file':
      if (args.length !== 1) throw new FormulaError('Invalid is_file() function in variable formula');
      return sys.files.has(sys.resolveFile(args[0])) ? 1 : 0;
    case 'is_os': {
      // variable.html: is_os(name) "checks, if the name (a regular expression) matches the name of the operating
      // system"; the browser engine runs in a web page, whose system name is OS_INFO
      if (args.length !== 1) throw new FormulaError('Invalid is_os() function in variable formula');
      try { return new RegExp(args[0]).test(OS_INFO) ? 1 : 0; } catch { return 0; }
    }
    case 'is_timeout':
      if (args.length !== 1 || args[0] !== '') throw new FormulaError('Invalid is_timeout() function in variable formula');
      return 0;
    case 'extract_setting':
      if (args.length !== 1) throw new FormulaError('Invalid extract_setting() function in variable formula');
      return extractSetting(sys, args[0]);
    case 'is_defined': case 'is_active': case 'is_available':
      if (args.length !== 2) throw new FormulaError(`Invalid ${fn}() function in variable formula`);
      if (fn === 'is_defined') return isDefined(sys, args[0], args[1]);
      return fn === 'is_active' ? isActive(sys, args[0], args[1]) : isAvailable(sys, args[0], args[1]);
    case 'label2type': case 'is_typelabel': return labelFunction(sys, fn, args);
  }
  throw new StyleError(`unknown function ${fn}()`);
};

/**
 * label2type(kind,label) and is_typelabel(kind,label) — variable.html: "The label2type(kind,label) function
 * converts type labels into numeric types, using label maps created by the labelmap or read_data commands ...
 * or triggers an error if the queried label does not exist." "is_typelabel ... returns 1 if the type label has
 * been assigned, otherwise it returns 0." Native messages: "Invalid kind foo in label2type() in variable",
 * "Invalid atom type label ZZ in label2type() in variable", "Cannot use label2type() function without a labelmap".
 */
const labelFunction = (sys: System, fn: string, args: string[]): number => {
  // native's order: the labelmap, the comma, the kind, then the label (Variable::special_function at b332f28)
  if (!sys.labels.defined) throw new FormulaError(`Cannot use ${fn}() function without a labelmap`);
  if (args.length < 2) throw new FormulaError(`Invalid ${fn}(${args[0]}) function in variable formula`);
  const kind = args[0];
  if (!isLabelKind(kind)) throw new FormulaError(`Invalid kind ${kind} in ${fn}() in variable`);
  const label = args.slice(1).join(',');
  const t = sys.labels.typeOf(kind, label);
  if (fn === 'is_typelabel') return t === undefined ? 0 : 1;
  if (t === undefined) throw new FormulaError(`Invalid ${kind} type label ${label} in ${fn}() in variable`);
  return t;
};

/** A region of a formula's function (native Variable::region_function at b332f28). */
const formulaRegion = (sys: System, id: string) => {
  if (!sys.regions.has(id)) throw new FormulaError(`Region ${id} in variable formula does not exist`);
  return sys.region(id);
};

/** Atoms in group (and region, if given), unwrapped positions available. */
const members = (sys: System, group: string, region: string | undefined): number[] => {
  const s = sys.state;
  const bit = sys.groupBit(group);
  const reg = region !== undefined ? formulaRegion(sys, region) : null;
  const out: number[] = [];
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & bit)) continue;
    if (reg && !reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
    out.push(i);
  }
  return out;
};

/** The components of inertia(), in native's order. */
const INERTIA = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz'];

/** The group functions that weigh atoms by mass (native calls atom->check_mass first). */
const MASS_FNS = new Set(['xcm', 'vcm', 'gyration', 'angmom', 'torque', 'inertia', 'omega']);

const groupFn = (sys: System, fn: string, args: string[]): number => {
  const nreq: Record<string, number> = {
    count: 1, mass: 1, charge: 1, xcm: 2, vcm: 2, fcm: 2, bound: 2, gyration: 1, ke: 1, angmom: 2, torque: 2, inertia: 2, omega: 2,
  };
  // As native LAMMPS does (Variable::group_function, variable.cpp at b332f28): the group, the per-type masses of the
  // functions that weigh atoms, the argument count and the region, then the component
  if (sys.groups.find(args[0]) < 0) throw new FormulaError(`Group ${args[0]} in variable formula does not exist`);
  const bad = () => new FormulaError(`Invalid ${fn}() function in variable formula`);
  if (MASS_FNS.has(fn) && sys.hasBox && !sys.state.rmass) {
    const st = sys.state;
    for (let t = 1; t <= st.ntypes; t++) if (!(st.massByType[t] > 0)) throw new StyleError(`Not all per-type masses are set. Type ${t} is missing.`);
  }
  const n0 = nreq[fn];
  if (args.length !== n0 && args.length !== n0 + 1) throw bad();
  const region = args.length === n0 + 1 ? args[n0] : undefined;
  if (region !== undefined) formulaRegion(sys, region);
  if (!sys.hasBox) {
    // before the box there are no atoms: native's sums over none are 0, bound() gives the empty extent
    if (fn === 'bound') {
      if (!/^[xyz](min|max)$/.test(args[1])) throw bad();
      return args[1].endsWith('min') ? 1e20 : -1e20;
    }
    if (fn === 'inertia' ? !INERTIA.includes(args[1]) : n0 === 2 && !(args[1] in DIMS)) throw bad();
    return 0;
  }
  const s = sys.state;
  if (fn === 'fcm' || fn === 'torque') sys.forces();
  const idx = members(sys, args[0], region);
  const m = (i: number) => massOf(s, i);
  const dim = (w: string) => {
    if (!(w in DIMS)) throw bad();
    return DIMS[w];
  };
  const pos = (i: number) => {
    const out = [0, 0, 0];
    sys.geom.unwrap(s.x, s.image, i, out);
    return out;
  };
  const com = () => {
    let mt = 0;
    const c = [0, 0, 0];
    for (const i of idx) { const p = pos(i); const mi = m(i); mt += mi; for (let d = 0; d < 3; d++) c[d] += mi * p[d]; }
    return mt > 0 ? c.map((x) => x / mt) : c;
  };
  switch (fn) {
    case 'count': return idx.length;
    case 'mass': return idx.reduce((a, i) => a + m(i), 0);
    case 'charge':
      // native reads the per-atom charges, which an atom style without them lacks (it crashes there)
      if (idx.length && !hasCharge(s)) throw new StyleError('charge() in a variable formula needs an atom style with charges');
      return idx.reduce((a, i) => a + s.q[i], 0);
    case 'xcm': return com()[dim(args[1])];
    case 'vcm': {
      let mt = 0, p = 0;
      const d = dim(args[1]);
      for (const i of idx) { mt += m(i); p += m(i) * s.v[3 * i + d]; }
      return mt > 0 ? p / mt : 0;
    }
    case 'fcm': {
      const d = dim(args[1]);
      return idx.reduce((a, i) => a + s.f[3 * i + d], 0);
    }
    case 'bound': {
      const w = args[1];
      const d = DIMS[w[0]];
      if (d === undefined || (w.slice(1) !== 'min' && w.slice(1) !== 'max')) throw bad();
      let v = w.slice(1) === 'min' ? Infinity : -Infinity;
      for (const i of idx) v = w.slice(1) === 'min' ? Math.min(v, s.x[3 * i + d]) : Math.max(v, s.x[3 * i + d]);
      return Number.isFinite(v) ? v : (w.slice(1) === 'min' ? 1e20 : -1e20);
    }
    case 'gyration': {
      const c = com();
      let mt = 0, r2 = 0;
      for (const i of idx) {
        const p = pos(i);
        mt += m(i);
        r2 += m(i) * ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2);
      }
      return mt > 0 ? Math.sqrt(r2 / mt) : 0;
    }
    case 'ke': {
      let k = 0;
      for (const i of idx) k += m(i) * (s.v[3 * i] ** 2 + s.v[3 * i + 1] ** 2 + s.v[3 * i + 2] ** 2);
      return 0.5 * s.units.mvv2e * k;
    }
    case 'angmom': return angmom(s, idx, pos, com(), m, true)[dim(args[1])];
    case 'torque': {
      const c = com();
      const t = [0, 0, 0];
      for (const i of idx) {
        const p = pos(i);
        const dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
        const fx = s.f[3 * i], fy = s.f[3 * i + 1], fz = s.f[3 * i + 2];
        t[0] += dy * fz - dz * fy; t[1] += dz * fx - dx * fz; t[2] += dx * fy - dy * fx;
      }
      return t[dim(args[1])];
    }
    case 'inertia': {
      const I = inertia(s, idx, pos, com(), m, true);
      const k = INERTIA.indexOf(args[1]);
      if (k < 0) throw bad();
      return [I[0][0], I[1][1], I[2][2], I[0][1], I[1][2], I[0][2]][k];
    }
    case 'omega': {
      // variable.cpp (omega group function at b332f28): group->xcm, angmom and inertia (point masses), then group->omega
      return groupMotionOf(sys, idx, 'omega').omega[dim(args[1])];
    }
  }
  throw new StyleError(`unknown group function ${fn}()`);
};

/*
 * docs.lammps.org/variable.html (versionchanged 2Sep2026): "The inertia() and angmom() functions now include the
 * contributions of finite-size particles; previously all atoms were treated as point masses." omega() is not
 * listed there and keeps the point-mass values (measured with native LAMMPS, black box).
 */
/** Angular momentum about c; with finite = true the spin of finite-size particles is added (compute/finiteSpin.ts). */
const angmom = (s: SimState, idx: number[], pos: (i: number) => number[], c: number[], m: (i: number) => number, finite: boolean): number[] => {
  const L = [0, 0, 0];
  const v = s.v;
  for (const i of idx) {
    const p = pos(i);
    const dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    const mi = m(i);
    const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
    L[0] += mi * (dy * vz - dz * vy); L[1] += mi * (dz * vx - dx * vz); L[2] += mi * (dx * vy - dy * vx);
    if (finite) {
      const f = finiteSpin(s, i);
      if (f) { L[0] += f.L[0]; L[1] += f.L[1]; L[2] += f.L[2]; }
    }
  }
  return L;
};

/** Inertia tensor about c; with finite = true each finite-size particle's own tensor is added (parallel-axis theorem, compute/finiteSpin.ts). */
const inertia = (s: SimState, idx: number[], pos: (i: number) => number[], c: number[], m: (i: number) => number, finite: boolean): number[][] => {
  const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const i of idx) {
    const p = pos(i);
    const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    const mi = m(i);
    I[0][0] += mi * (d[1] * d[1] + d[2] * d[2]);
    I[1][1] += mi * (d[0] * d[0] + d[2] * d[2]);
    I[2][2] += mi * (d[0] * d[0] + d[1] * d[1]);
    I[0][1] -= mi * d[0] * d[1];
    I[1][2] -= mi * d[1] * d[2];
    I[0][2] -= mi * d[0] * d[2];
    if (finite) {
      const f = finiteSpin(s, i);
      if (f) {
        I[0][0] += f.T[0]; I[1][1] += f.T[1]; I[2][2] += f.T[2];
        I[0][1] += f.T[3]; I[1][2] += f.T[4]; I[0][2] += f.T[5];
      }
    }
  }
  I[1][0] = I[0][1]; I[2][1] = I[1][2]; I[2][0] = I[0][2];
  return I;
};

/**
 * The global vector a special function reduces: c_ID, c_ID[N], f_ID, f_ID[N] or v_name. As native LAMMPS does
 * (Variable::special_function, variable.cpp at b332f28): the messages of a compute or fix name no variable, those of a
 * vector-style variable name it.
 */
const specialVector = (sys: System, fn: string, arg: string): Float64Array => {
  const unnamed = 'Invalid special function in variable formula';
  const br = arg.indexOf('[');
  const index = br >= 0 ? resolveIndex(scanBracket(arg, br + 1, false).index, sys.formulaEnv) : 0;
  const id = (br >= 0 ? arg.slice(0, br) : arg).slice(2);
  const column = (rows: number, at: (r: number) => number) => {
    const out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) out[r] = at(r);
    return out;
  };
  if (arg.startsWith('c_')) {
    const c = sys.computes.find((x) => x.id === id);
    if (!c) throw new StyleError(`Invalid compute ID '${id}' in variable formula`);
    const init = () => {
      if (!c.initialized) throw new StyleError('Variable formula compute cannot be invoked before initialization by a run');
    };
    if (index === 0 && c.vectorFlag) {
      init();
      return c.vectorValues();
    }
    if (index && c.arrayFlag) {
      if (index > c.sizeArrayCols) throw new StyleError(`Variable formula compute array is accessed out-of-range${errorUrl(20)}`);
      init();
      const a = c.arrayValues();
      return column(c.sizeArrayRows, (r) => a[r * c.sizeArrayCols + index - 1]);
    }
    throw new StyleError(`Compute '${c.id}' in variable formula does not compute the requested property. ${errorUrl(37)}`);
  }
  if (arg.startsWith('f_')) {
    const f = sys.fixes.find((x) => x.id === id);
    if (!f) throw new StyleError(`Invalid fix ID '${id}' in variable formula`);
    const off = sys.run.inRun && f.globalFreq > 0 && sys.state.step % f.globalFreq !== 0;
    if (index === 0 && f.vectorFlag) {
      if (off) throw new StyleError(`Fix with ID '${id}' in variable formula not computed at compatible time${errorUrl(7)}`);
      return column(f.sizeVector, (k) => f.computeVector(k));
    }
    if (index && f.arrayFlag) {
      if (index > f.sizeArrayCols) throw new StyleError(`Variable formula fix array is accessed out-of-range${errorUrl(20)}`);
      if (off) throw new StyleError(`Fix in variable not computed at compatible time${errorUrl(7)}`);
      return column(f.sizeArrayRows, (r) => f.computeArray(r, index - 1));
    }
    throw new StyleError(`Fix '${f.id}' in variable formula does not compute the requested property. ${errorUrl(37)}`);
  }
  if (arg.startsWith('v_')) {
    if (index) throw new StyleError(unnamed);
    const v = sys.vars.get(id);
    if (!v) throw new StyleError(unnamed);
    if (v.style !== 'vector') throw new FormulaError(`Variable ${id}: Mis-matched special function variable in variable formula`, true);
    const vec = sys.vars.evalVector(id, sys.formulaEnv);
    if (fn === 'ave' && vec.length === 0) throw new FormulaError(`Variable ${id}: Cannot compute average of empty vector`, true);
    return vec;
  }
  throw new StyleError(unnamed);
};

/** native's start of min() and max(), and slope()'s value for a vertical line (BIG) */
const BIG = 1.0e20;

const reduceFn = (sys: System, fn: string, args: string[], mode: Mode, index: Index | null): Value => {
  if (args.length !== 1) throw special(fn);
  const v = specialVector(sys, fn, args[0]);
  const n = v.length;
  switch (fn) {
    case 'sum': { let t = 0; for (const x of v) t += x; return t; }
    case 'min': { let t = BIG; for (const x of v) if (x < t) t = x; return t; }
    case 'max': { let t = -BIG; for (const x of v) if (x > t) t = x; return t; }
    case 'ave': { let t = 0; for (const x of v) t += x; return t / n; }
    case 'trap': {
      let t = 0;
      for (let k = 0; k < n; k++) t += k === 0 || k === n - 1 ? 0.5 * v[k] : v[k];
      return t;
    }
    case 'slope': {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let k = 0; k < n; k++) { sx += k; sy += v[k]; sxx += k * k; sxy += k * v[k]; }
      const den = n * sxx - sx * sx;
      return den === 0 ? BIG : (n * sxy - sx * sy) / den;
    }
    case 'sort': case 'rsort': {
      const out = Float64Array.from(v).sort();
      if (fn === 'rsort') out.reverse();
      // an equal-style formula takes one element, sort(x)[I]; a vector-style one the whole vector (no brackets)
      if (mode === 'vector' && index) throw new FormulaError('Invalid syntax in variable formula');
      if (mode !== 'equal') return out;
      if (!index) throw new FormulaError('Invalid syntax in variable formula');
      const i = resolveIndex(index, sys.formulaEnv);
      if (i > n) {
        const text = `index ${i} exceeds vector size of ${n}\n`;
        throw args[0].startsWith('v_') ? new FormulaError(`Variable ${args[0].slice(2)}: ${text}`, true) : new StyleError(text);
      }
      return out[i - 1];
    }
  }
  throw new StyleError(`unknown special function ${fn}()`);
};

const maskFn = (sys: System, fn: string, args: string[], mode: Mode): Value => {
  // As native LAMMPS does (Variable::special_function, variable.cpp at b332f28)
  if (mode === 'equal') throw new FormulaError(`${fn[0].toUpperCase()}${fn.slice(1)} function in equal-style variable formula`);
  if (mode !== 'atom') throw new StyleError(`${fn}() can only be used in atom-style variables`);
  if (args.length !== (fn === 'grmask' ? 2 : 1)) throw special('');
  if (fn !== 'rmask' && sys.groups.find(args[0]) < 0) throw new FormulaError('Group ID in variable formula does not exist');
  const s = sys.state;
  const out = new Float64Array(s.n);
  const bit = fn === 'rmask' ? 0 : sys.groupBit(args[0]);
  const reg = fn === 'gmask' ? null : formulaRegion(sys, fn === 'rmask' ? args[0] : args[1]);
  for (let i = 0; i < s.n; i++) {
    let ok = true;
    if (fn !== 'rmask') ok = (s.mask[i] & bit) !== 0;
    if (ok && reg) ok = reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
    out[i] = ok ? 1 : 0;
  }
  return out;
};

const extractSetting = (sys: System, name: string): number => {
  const s = sys.hasBox ? sys.state : null;
  switch (name) {
    case 'dimension': return sys.dimension;
    case 'box_exist': return s ? 1 : 0;
    case 'triclinic': return s?.box.triclinic ? 1 : 0;
    case 'nlocal': case 'natoms': return s?.n ?? 0;
    case 'nghost': return sys.nb.nghost;
    case 'nall': return sys.nb.nall;
    case 'ntypes': return s?.ntypes ?? 0;
    case 'nbondtypes': return s?.topo.nbondtypes ?? 0;
    case 'nangletypes': return s?.topo.nangletypes ?? 0;
    case 'ndihedraltypes': return s?.topo.ndihedraltypes ?? 0;
    case 'nimpropertypes': return s?.topo.nimpropertypes ?? 0;
    case 'molecule_flag': return s && hasMolecule(s) ? 1 : 0;
    case 'q_flag': return s && hasCharge(s) ? 1 : 0;
    case 'world_rank': case 'universe_rank': return 0;
    case 'world_size': case 'universe_size': case 'nthreads': return 1;
    case 'bigint': case 'tagint': case 'imageint': return 4;
  }
  throw new FormulaError(`Unknown setting ${name} for extract_setting() function in variable formula`);
};

// the messages of native's Info::is_defined, is_active and is_available (src/info.cpp at b332f28)
const isDefined = (sys: System, cat: string, id: string): number => {
  switch (cat) {
    case 'compute': return sys.computes.some((c) => c.id === id) ? 1 : 0;
    case 'fix': return sys.fixes.some((f) => f.id === id) ? 1 : 0;
    case 'group': return sys.groups.find(id) >= 0 ? 1 : 0;
    case 'region': return sys.regions.has(id) ? 1 : 0;
    case 'variable': return sys.vars.has(id) ? 1 : 0;
    case 'dump': return sys.dumps.some((d) => d.id === id) ? 1 : 0;
  }
  throw new StyleError(`Unknown category for info is_defined(): ${cat}`);
};

const isActive = (sys: System, cat: string, feature: string): number => {
  switch (cat) {
    case 'package':
      if (['gpu', 'intel', 'kokkos', 'omp'].includes(feature)) return 0;
      throw new StyleError(`Unknown name for info package category: ${feature}`);
    case 'newton':
      if (feature === 'pair' || feature === 'bond' || feature === 'any') return 1;
      throw new StyleError(`Unknown name for info newton category: ${feature}`);
    case 'pair': {
      const p = sys.ff.pair;
      if (!p) return 0;
      switch (feature) {
        case 'single': return p.single ? 1 : 0;
        case 'respa': return 0;
        case 'manybody': return p.manybody ? 1 : 0;
        case 'tail': return p.tail ? 1 : 0;
        case 'shift': return p.shift ? 1 : 0;
      }
      throw new StyleError(`Unknown name for info pair category: ${feature}`);
    }
    case 'comm_style': return feature === 'brick' ? 1 : 0;
    case 'min_style': return feature === sys.minStyle ? 1 : 0;
    case 'run_style': return feature === 'verlet' ? 1 : 0;
    case 'atom_style': return feature === sys.atomStyle ? 1 : 0;
    case 'pair_style': return feature === sys.ff.pair?.name ? 1 : 0;
    case 'comm_mode': return feature === 'single' ? 1 : 0;
  }
  throw new StyleError(`Unknown category for info is_active(): ${cat}`);
};

/** native Info::has_style categories, and the engine's style registry of each */
const INFO_CATEGORIES: Record<string, string> = {
  pair: 'pair_style', bond: 'bond_style', angle: 'angle_style', dihedral: 'dihedral_style', improper: 'improper_style',
  kspace: 'kspace_style', fix: 'fix', compute: 'compute', minimize: 'min_style', command: 'command',
};

const isAvailable = (sys: System, cat: string, feature: string): number => {
  const reg = sys.registries[INFO_CATEGORIES[cat] ?? ''];
  if (reg) return reg.includes(feature) ? 1 : 0;
  if (['atom', 'integrate', 'region', 'dump'].includes(cat)) throw new StyleError(`is_available(${cat},${feature}) is not supported by the browser engine (it lists no ${cat} styles)`);
  if (cat === 'feature') return ['gzip', 'ffmpeg', 'png', 'jpeg', 'exceptions'].includes(feature) ? (feature === 'exceptions' ? 1 : 0) : 0;
  throw new StyleError(`Unknown category for info is_available(): ${cat}`);
};

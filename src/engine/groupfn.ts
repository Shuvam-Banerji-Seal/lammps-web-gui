import type { System } from './system';
import { StyleError } from './force/types';
import type { Mode, Value } from './formula';

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
 * unwrapped with the image flags. Special functions: "sum(), min(), max(),
 * ave(), trap(), slope(), sort(), rsort()" over a global vector; "gmask(x)
 * ... returns a 1 for atoms that are in the group"; rmask, grmask; next().
 * "slope() ... If the line has a single point or is vertical, it returns
 * 1.0e20."
 */

const DIMS: Record<string, number> = { x: 0, y: 1, z: 2 };

export const groupFunction = (sys: System, fn: string, args: string[], mode: Mode): Value => {
  switch (fn) {
    case 'count': case 'mass': case 'charge': case 'xcm': case 'vcm': case 'fcm': case 'bound':
    case 'gyration': case 'ke': case 'angmom': case 'torque': case 'inertia': case 'omega':
      return groupFn(sys, fn, args);
    case 'sum': case 'min': case 'max': case 'ave': case 'trap': case 'slope': case 'sort': case 'rsort':
      return reduceFn(sys, fn, args);
    case 'gmask': case 'rmask': case 'grmask':
      return maskFn(sys, fn, args, mode);
    case 'next': {
      if (args.length !== 1) throw new StyleError('next() takes one variable name');
      const v = sys.vars.get(args[0]);
      if (!v || (v.style !== 'file' && v.style !== 'atomfile')) throw new StyleError(`next(${args[0]}) needs a file- or atomfile-style variable`);
      if (v.style === 'file') {
        const cur = Number(v.values[0]);
        sys.vars.next([args[0]]);
        return Number.isFinite(cur) ? cur : 0;
      }
      const s = sys.state;
      const cur = sys.vars.evalAtom(args[0], sys.formulaEnv, s.id, s.n);
      sys.vars.next([args[0]]);
      return cur;
    }
    case 'is_file': return sys.files.has(args[0]) ? 1 : 0;
    case 'is_os': return 0;
    case 'is_timeout': return 0;
    case 'extract_setting': return extractSetting(sys, args[0]);
    case 'is_defined': return isDefined(sys, args[0], args[1]);
    case 'is_active': return isActive(sys, args[0], args[1]);
    case 'is_available': return isAvailable(sys, args[0], args[1]);
    case 'label2type': case 'is_typelabel':
      throw new StyleError(`${fn}() needs type labels (labelmap), which the browser engine does not support`);
  }
  throw new StyleError(`unknown function ${fn}()`);
};

/** Atoms in group (and region, if given), unwrapped positions available. */
const members = (sys: System, group: string, region: string | undefined): number[] => {
  const s = sys.state;
  const bit = sys.groupBit(group);
  const reg = region !== undefined ? sys.region(region) : null;
  const out: number[] = [];
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & bit)) continue;
    if (reg && !reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
    out.push(i);
  }
  return out;
};

const groupFn = (sys: System, fn: string, args: string[]): number => {
  const nreq: Record<string, number> = {
    count: 1, mass: 1, charge: 1, xcm: 2, vcm: 2, fcm: 2, bound: 2, gyration: 1, ke: 1, angmom: 2, torque: 2, inertia: 2, omega: 2,
  };
  const n0 = nreq[fn];
  if (args.length !== n0 && args.length !== n0 + 1) throw new StyleError(`${fn}() takes ${n0} argument(s) (plus an optional region ID)`);
  const region = args.length === n0 + 1 ? args[n0] : undefined;
  const s = sys.state;
  if (fn === 'fcm' || fn === 'torque') sys.forces();
  const idx = members(sys, args[0], region);
  const m = (i: number) => s.massByType[s.type[i]];
  const dim = (w: string) => {
    if (!(w in DIMS)) throw new StyleError(`${fn}(): dimension must be x, y or z, got '${w}'`);
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
    case 'charge': return idx.reduce((a, i) => a + s.q[i], 0);
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
      if (d === undefined || (w.slice(1) !== 'min' && w.slice(1) !== 'max')) throw new StyleError(`bound(): direction must be xmin, xmax, ymin, ymax, zmin or zmax, got '${w}'`);
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
    case 'angmom': return angmom(idx, pos, com(), m, s.v)[dim(args[1])];
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
      const I = inertia(idx, pos, com(), m);
      const k = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz'].indexOf(args[1]);
      if (k < 0) throw new StyleError(`inertia(): component must be xx, yy, zz, xy, yz or xz, got '${args[1]}'`);
      return [I[0][0], I[1][1], I[2][2], I[0][1], I[1][2], I[0][2]][k];
    }
    case 'omega': {
      const c = com();
      const L = angmom(idx, pos, c, m, s.v);
      const I = inertia(idx, pos, c, m);
      return omegaFrom(I, L)[dim(args[1])];
    }
  }
  throw new StyleError(`unknown group function ${fn}()`);
};

const angmom = (idx: number[], pos: (i: number) => number[], c: number[], m: (i: number) => number, v: Float64Array): number[] => {
  const L = [0, 0, 0];
  for (const i of idx) {
    const p = pos(i);
    const dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
    const mi = m(i);
    const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
    L[0] += mi * (dy * vz - dz * vy); L[1] += mi * (dz * vx - dx * vz); L[2] += mi * (dx * vy - dy * vx);
  }
  return L;
};

const inertia = (idx: number[], pos: (i: number) => number[], c: number[], m: (i: number) => number): number[][] => {
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
  }
  I[1][0] = I[0][1]; I[2][1] = I[1][2]; I[2][0] = I[0][2];
  return I;
};

/** omega = I^-1 L; a singular (e.g. linear or planar) inertia tensor uses its pseudo-inverse along the non-degenerate axes. */
const omegaFrom = (I: number[][], L: number[]): number[] => {
  const [[a, b, c], [d, e, f], [g, h, k]] = I;
  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  const scale = Math.max(Math.abs(a), Math.abs(e), Math.abs(k), 1e-300);
  if (Math.abs(det) > 1e-12 * scale * scale * scale) {
    const inv = [
      [(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det],
      [(f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det],
      [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ];
    return [0, 1, 2].map((r) => inv[r][0] * L[0] + inv[r][1] * L[1] + inv[r][2] * L[2]);
  }
  // degenerate: diagonal approximation where the moment is non-zero
  return [0, 1, 2].map((r) => (Math.abs(I[r][r]) > 0 ? L[r] / I[r][r] : 0));
};

/** The global vector a special function reduces: c_ID, c_ID[N], f_ID, f_ID[N], v_name. */
const specialVector = (sys: System, arg: string): Float64Array => {
  const m = /^([cfv])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(arg);
  if (!m) throw new StyleError(`invalid special function argument '${arg}' (use c_ID, c_ID[N], f_ID, f_ID[N] or v_name)`);
  const [, kind, id, col] = m;
  if (kind === 'v') return sys.vars.evalVector(id, sys.formulaEnv);
  if (kind === 'c') {
    const c = sys.compute(id);
    if (col === undefined) return c.vectorValues();
    const a = c.arrayValues();
    const j = Number(col);
    if (j < 1 || j > c.sizeArrayCols) throw new StyleError(`${arg}: column out of range 1..${c.sizeArrayCols}`);
    const out = new Float64Array(c.sizeArrayRows);
    for (let r = 0; r < c.sizeArrayRows; r++) out[r] = a[r * c.sizeArrayCols + j - 1];
    return out;
  }
  const f = sys.fix(id);
  if (col === undefined) {
    const out = new Float64Array(f.sizeVector);
    for (let k = 0; k < f.sizeVector; k++) out[k] = f.computeVector(k);
    return out;
  }
  const j = Number(col);
  const out = new Float64Array(f.sizeArrayRows);
  for (let r = 0; r < f.sizeArrayRows; r++) out[r] = f.computeArray(r, j - 1);
  return out;
};

const reduceFn = (sys: System, fn: string, args: string[]): Value => {
  if (args.length !== 1) throw new StyleError(`${fn}() takes one argument`);
  const v = specialVector(sys, args[0]);
  const n = v.length;
  switch (fn) {
    case 'sum': { let t = 0; for (const x of v) t += x; return t; }
    case 'min': { let t = Infinity; for (const x of v) if (x < t) t = x; return n ? t : 0; }
    case 'max': { let t = -Infinity; for (const x of v) if (x > t) t = x; return n ? t : 0; }
    case 'ave': { let t = 0; for (const x of v) t += x; return n ? t / n : 0; }
    case 'trap': {
      let t = 0;
      for (let k = 0; k < n; k++) t += k === 0 || k === n - 1 ? 0.5 * v[k] : v[k];
      return t;
    }
    case 'slope': {
      if (n < 2) return 1e20;
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      for (let k = 0; k < n; k++) { const x = k + 1; sx += x; sy += v[k]; sxx += x * x; sxy += x * v[k]; }
      const den = n * sxx - sx * sx;
      return den === 0 ? 1e20 : (n * sxy - sx * sy) / den;
    }
    case 'sort': return Float64Array.from(v).sort();
    case 'rsort': return Float64Array.from(v).sort().reverse();
  }
  throw new StyleError(`unknown special function ${fn}()`);
};

const maskFn = (sys: System, fn: string, args: string[], mode: Mode): Value => {
  if (mode !== 'atom') throw new StyleError(`${fn}() can only be used in atom-style variables`);
  const s = sys.state;
  const out = new Float64Array(s.n);
  const bit = fn === 'rmask' ? 0 : sys.groupBit(args[0]);
  const reg = fn === 'gmask' ? null : sys.region(fn === 'rmask' ? args[0] : args[1]);
  for (let i = 0; i < s.n; i++) {
    let ok = true;
    if (fn !== 'rmask') ok = (s.mask[i] & bit) !== 0;
    if (ok && reg) ok = reg.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
    out[i] = ok ? 1 : 0;
  }
  return out;
};

const extractSetting = (sys: System, name: string | undefined): number => {
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
    case 'molecule_flag': return s && s.atomStyle !== 'atomic' && s.atomStyle !== 'charge' ? 1 : 0;
    case 'q_flag': return s && (s.atomStyle === 'charge' || s.atomStyle === 'full') ? 1 : 0;
    case 'world_rank': case 'universe_rank': return 0;
    case 'world_size': case 'universe_size': case 'nthreads': return 1;
    case 'bigint': case 'tagint': case 'imageint': return 4;
  }
  throw new StyleError(`extract_setting(${name ?? ''}) is not a known setting`);
};

const isDefined = (sys: System, cat: string | undefined, id: string | undefined): number => {
  if (!id) throw new StyleError('is_defined(category,id) needs two arguments');
  switch (cat) {
    case 'compute': return sys.computes.some((c) => c.id === id) ? 1 : 0;
    case 'fix': return sys.fixes.some((f) => f.id === id) ? 1 : 0;
    case 'group': return sys.groups.find(id) >= 0 ? 1 : 0;
    case 'region': return sys.regions.has(id) ? 1 : 0;
    case 'variable': return sys.vars.has(id) ? 1 : 0;
    case 'dump': return sys.dumps.some((d) => d.id === id) ? 1 : 0;
  }
  throw new StyleError(`is_defined(): unknown category '${cat ?? ''}'`);
};

const isActive = (sys: System, cat: string | undefined, feature: string | undefined): number => {
  switch (cat) {
    case 'package': return 0;
    case 'newton': return feature === 'pair' || feature === 'bond' || feature === 'any' ? 1 : 0;
    case 'pair': return sys.ff.pair ? (feature === 'single' ? (sys.ff.pair.single ? 1 : 0) : 0) : 0;
    case 'comm_style': return feature === 'brick' ? 1 : 0;
    case 'min_style': return feature === sys.minStyle ? 1 : 0;
    case 'run_style': return feature === 'verlet' ? 1 : 0;
    case 'atom_style': return feature === sys.atomStyle ? 1 : 0;
    case 'pair_style': return feature === sys.ff.pair?.name ? 1 : 0;
    case 'comm_mode': return feature === 'single' ? 1 : 0;
  }
  throw new StyleError(`is_active(): unknown category '${cat ?? ''}'`);
};

const isAvailable = (sys: System, cat: string | undefined, feature: string | undefined): number => {
  if (!feature) throw new StyleError('is_available(category,name) needs two arguments');
  const reg = sys.registries[cat ?? ''];
  if (reg) return reg.includes(feature) ? 1 : 0;
  if (cat === 'feature') return ['gzip', 'ffmpeg', 'png', 'jpeg', 'exceptions'].includes(feature) ? (feature === 'exceptions' ? 1 : 0) : 0;
  throw new StyleError(`is_available(): unknown category '${cat ?? ''}'`);
};

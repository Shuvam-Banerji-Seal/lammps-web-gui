import type { Handler } from './args';
import { int, num, yesno, latticeScale, keywords, numOrVar } from './args';
import { StyleError, typeBounds } from '../force/types';
import { UNIT_SYSTEMS, isUnitStyle } from '../units';
import { makeBox, parseBoundary, Geometry, cloneBox } from '../domain';
import { emptyState, appendAtoms, maxAtomId, pushTopo, ALL_GROUP_BIT, hasChargeStyle, isMolecularStyle, sphereMass, massOf, gatherAtoms, hasCharge, hasMolecule } from '../atoms';
import { isLatticeStyle, makeLattice, latticeSites } from '../lattice';
import {
  BIG, BlockRegion, CompoundRegion, ConeRegion, EllipsoidRegion, PlaneRegion, PrismRegion, SphereRegion,
  type Param, type Region,
} from '../region';
import { readData, writeData, defaultReadOptions, type ReadDataOptions } from '../output/data';
import { RanPark, Rng } from '../rng';
import { ComputeTemp } from '../compute/temp';
import type { AtomStyle, SimState } from '../types';
import type { System } from '../system';
import { bitOfIndex, type DynamicGroup } from '../group';
import { parseMoleculeFile, geometricCenter, rotationMatrix, type MoleculeTemplate, type MoleculeOptions } from '../molecule';

/*
 * System setup commands. Each handler cites its docs.lammps.org page.
 */

const noBox = (sys: System, cmd: string) => {
  if (sys.hasBox) throw new StyleError(`${cmd} cannot be used after the simulation box is defined`);
};

/** units style — units.html: "This command cannot be used after the simulation box is defined". */
const units: Handler = ({ sys }, a) => {
  noBox(sys, 'units');
  if (a.length !== 1) throw new StyleError('usage: units lj|real|metal|si|cgs|electron|micro|nano');
  if (!isUnitStyle(a[0])) throw new StyleError(`unknown units style '${a[0]}'`);
  sys.units = UNIT_SYSTEMS[a[0]];
  // "The units command also sets the timestep size and neighbor skin distance to default values"
  sys.nb.skin = sys.units.skin;
  sys.pendingDt = null;
};

/** dimension N — dimension.html (default 3). */
const dimension: Handler = ({ sys }, a) => {
  noBox(sys, 'dimension');
  if (a[0] !== '2' && a[0] !== '3') throw new StyleError('usage: dimension 2|3');
  sys.dimension = a[0] === '2' ? 2 : 3;
};

/**
 * boundary x y z — boundary.html: "A single letter assigns the same style to
 * both the lower and upper face of the box. Two letters assigns the first
 * style to the lower face and the second style to the upper face." "The p
 * style must be applied to both faces of a dimension. For 2d simulations the
 * z dimension must be periodic".
 */
const boundary: Handler = ({ sys }, a) => {
  noBox(sys, 'boundary');
  if (a.length !== 3) throw new StyleError('usage: boundary x y z (each p, f, s, m, or two of f/s/m)');
  const b = a.map((w) => {
    const r = parseBoundary(w);
    if (!r) throw new StyleError(`invalid boundary '${w}'`);
    return r;
  }) as SimState['box']['boundary'];
  if (sys.dimension === 2 && b[2][0] !== 'p') throw new StyleError('for 2d simulations the z dimension must be periodic');
  sys.boundary = b;
};

const ATOM_STYLES: AtomStyle[] = ['atomic', 'charge', 'bond', 'angle', 'molecular', 'full', 'sphere'];

/** atom_style — atom_style.html: "The default atom style is atomic." */
const atomStyle: Handler = ({ sys }, a) => {
  noBox(sys, 'atom_style');
  if (a.length < 1) throw new StyleError('usage: atom_style style');
  if (!(ATOM_STYLES as string[]).includes(a[0])) {
    throw new StyleError(`atom_style '${a[0]}' is not supported by the browser engine; supported: ${ATOM_STYLES.join(', ')}`);
  }
  // atom_style.html: "*sphere* arg = 0/1 (optional) for static/dynamic particle radii"; the engine
  // never changes radii during a run, so both values behave the same.
  if (a[0] === 'sphere') {
    if (a.length > 2 || (a.length === 2 && a[1] !== '0' && a[1] !== '1')) throw new StyleError('usage: atom_style sphere [0|1]');
  } else if (a.length > 1) throw new StyleError(`atom_style ${a[0]} takes no arguments`);
  sys.atomStyle = a[0] as AtomStyle;
};

/** atom_modify id/map/first/sort — atom_modify.html; bookkeeping only in a single-process engine. */
const atomModify: Handler = ({ sys }, a) => {
  const kw = keywords(a, { id: 1, map: 1, first: 1, sort: 2 }, 'atom_modify');
  if (kw.get('id') && kw.get('id')![0] !== 'yes') throw new StyleError('atom_modify id no is not supported (atoms always have IDs)');
  // atom_modify.html: "*first* value = group-ID = group whose atoms will appear first in internal
  // atom lists" — an ordering for performance; the engine keeps its own storage order (as it
  // does for the spatial *sort*), so say so instead of reordering
  if (kw.has('first')) {
    const g = kw.get('first')![0];
    sys.log(`atom_modify first ${g}: accepted; the browser engine does not reorder its atom storage (results are unchanged, only the internal order differs)`);
  }
  if (kw.get('map') && !['array', 'hash', 'yes'].includes(kw.get('map')![0])) throw new StyleError('atom_modify map must be array, hash or yes');
  void sys;
};

/** Single-process / no-accelerator commands: accepted, with what they mean here. */
const parallelOnly = (name: string): Handler => (({ sys }, a) => {
  if (name === 'suffix' && a[0] && a[0] !== 'off') {
    throw new StyleError(`suffix ${a[0]}: accelerator packages (gpu, kk, omp, opt, intel) are not part of the browser engine; remove the suffix command`);
  }
  if (name === 'package') {
    throw new StyleError(`package ${a[0] ?? ''}: accelerator packages are not part of the browser engine; choose CPU threads or the GPU in the notebook toolbar instead`);
  }
  sys.log(`${name}: accepted (the browser engine runs one process; this setting has no effect)`);
}) as Handler;

/** comm_style brick|tiled — comm_style.html; recorded for fix balance rcb, otherwise no effect in one process. */
const commStyle: Handler = ({ sys }, a) => {
  if (a.length < 1 || (a[0] !== 'brick' && a[0] !== 'tiled')) throw new StyleError('usage: comm_style brick|tiled');
  sys.commStyle = a[0];
  sys.log(`comm_style ${a[0]}: accepted (the browser engine runs one process)`);
};

/** newton on/off — newton.html; forces are always summed once per pair here, so either setting gives the same result. */
const newton: Handler = ({ sys }, a) => {
  if (a.length < 1 || a.length > 2 || a.some((w) => w !== 'on' && w !== 'off')) throw new StyleError('usage: newton on|off [on|off]');
  sys.log('newton: accepted (the engine always uses Newton\'s 3rd law within one process; results are the same)');
};

/** comm_modify cutoff C — sets the ghost cutoff (comm_modify.html "cutoff value = Rcut"). */
const commModify: Handler = ({ sys }, a) => {
  for (let k = 0; k < a.length;) {
    if (a[k] === 'cutoff') { sys.nb.commCutoff = num(a[k + 1], 'cutoff'); k += 2; } else if (a[k] === 'mode') { k += 2; } else if (a[k] === 'vel') { sys.ghostVelocity = yesno(a[k + 1], 'vel'); k += 2; } else if (a[k] === 'group') { k += 2; } else if (a[k] === 'cutoff/multi' || a[k] === 'reduce/multi') { k += a[k] === 'cutoff/multi' ? 3 : 1; } else throw new StyleError(`unknown comm_modify keyword '${a[k]}'`);
  }
};

/** lattice style scale [keywords] — lattice.html. */
const lattice: Handler = ({ sys }, a) => {
  if (a.length < 2) throw new StyleError('usage: lattice style scale [keyword values ...]');
  if (!isLatticeStyle(a[0])) throw new StyleError(`unknown lattice style '${a[0]}'`);
  sys.lattice = makeLattice(a[0], num(a[1], 'lattice scale'), sys.units, sys.dimension, a.slice(2));
  const sp = sys.lattice.spacing.map((v) => Number(v.toPrecision(8))).join(' ');
  sys.log(`Lattice spacing in x,y,z = ${sp}`);
};

/** A region bound: number, INF, EDGE, or v_name; scaled by the lattice spacing. */
const bound = (sys: System, w: string | undefined, scale: number, d: number, side: 'lo' | 'hi'): Param => {
  if (w === undefined) throw new StyleError('region: missing bound');
  if (w === 'INF' || w === 'EDGE') {
    if (!sys.hasBox) throw new StyleError('region: INF and EDGE need an existing simulation box');
    if (w === 'INF') return side === 'lo' ? -BIG : BIG;
    return side === 'lo' ? sys.state.box.lo[d] : sys.state.box.hi[d];
  }
  if (w.startsWith('v_')) return { variable: w.slice(2), scale };
  return num(w, 'region parameter') * scale;
};

/** region ID style args [side in|out] [units lattice|box] [move ...] [rotate ...] [open N] — region.html. */
const region: Handler = ({ sys }, a) => {
  const [id, style] = a;
  if (!id || !style) throw new StyleError('usage: region ID style args [keywords]');
  if (style === 'delete') {
    if (!sys.regions.delete(id)) throw new StyleError(`region ${id} does not exist`);
    return;
  }
  if (sys.regions.has(id)) throw new StyleError(`region ${id} already exists`);
  const nargs: Record<string, number> = { block: 6, sphere: 4, cylinder: 6, cone: 7, ellipsoid: 6, plane: 6, prism: 9 };
  let n: number;
  if (style === 'union' || style === 'intersect') n = 1 + int(a[2], 'number of regions');
  else if (style in nargs) n = nargs[style];
  else throw new StyleError(`unknown region style '${style}'`);
  const args = a.slice(2, 2 + n);
  if (args.length < n) throw new StyleError(`region ${style} needs ${n} arguments`);
  const rest = a.slice(2 + n);
  // keywords first: units affects scaling
  let side = true;
  let unitsW = 'lattice';
  let move: [string | null, string | null, string | null] | null = null;
  let rotate: Region['rotate'] = null;
  const open: number[] = [];
  for (let k = 0; k < rest.length;) {
    const key = rest[k];
    if (key === 'side') {
      if (rest[k + 1] !== 'in' && rest[k + 1] !== 'out') throw new StyleError('region side must be in or out');
      side = rest[k + 1] === 'in';
      k += 2;
    } else if (key === 'units') { unitsW = rest[k + 1]; k += 2; } else if (key === 'move') {
      move = [0, 1, 2].map((d) => {
        const w = rest[k + 1 + d];
        if (w === 'NULL') return null;
        if (!w?.startsWith('v_')) throw new StyleError('region move needs v_name or NULL for each component');
        return w.slice(2);
      }) as [string | null, string | null, string | null];
      k += 4;
    } else if (key === 'rotate') {
      const th = rest[k + 1];
      if (!th?.startsWith('v_')) throw new StyleError('region rotate needs v_theta');
      const p = [num(rest[k + 2], 'Px'), num(rest[k + 3], 'Py'), num(rest[k + 4], 'Pz')];
      const r = [num(rest[k + 5], 'Rx'), num(rest[k + 6], 'Ry'), num(rest[k + 7], 'Rz')];
      const len = Math.hypot(r[0], r[1], r[2]);
      if (!(len > 0)) throw new StyleError('region rotate axis must be non-zero');
      rotate = { theta: th.slice(2), p: p as [number, number, number], r: r.map((x) => x / len) as [number, number, number] };
      k += 8;
    } else if (key === 'open') { open.push(int(rest[k + 1], 'open face')); k += 2; } else throw new StyleError(`unknown region keyword '${key}'`);
  }
  const sc = latticeScale(sys, unitsW, 'region');
  if (rotate) rotate.p = rotate.p.map((v, d) => v * sc[d]) as [number, number, number];
  const env = sys.regionEnv;
  let reg: Region;
  const axis = (w: string) => {
    const d = 'xyz'.indexOf(w);
    if (d < 0 || w.length !== 1) throw new StyleError(`region ${style}: axis must be x, y or z`);
    return d as 0 | 1 | 2;
  };
  const p = (w: string, s: number) => (w.startsWith('v_') ? { variable: w.slice(2), scale: s } : num(w, 'region parameter') * s);
  switch (style) {
    case 'block':
      reg = new BlockRegion(id, env, [0, 1, 2].flatMap((d) => [bound(sys, args[2 * d], sc[d], d, 'lo'), bound(sys, args[2 * d + 1], sc[d], d, 'hi')]));
      break;
    case 'sphere':
      reg = new SphereRegion(id, env, [0, 1, 2].map((d) => p(args[d], sc[d])), p(args[3], sc[0]));
      break;
    case 'ellipsoid':
      reg = new EllipsoidRegion(id, env, [0, 1, 2].map((d) => p(args[d], sc[d])), [0, 1, 2].map((d) => p(args[3 + d], sc[d])));
      break;
    case 'cylinder': case 'cone': {
      const ax = axis(args[0]);
      const [d1, d2] = ax === 0 ? [1, 2] : ax === 1 ? [0, 2] : [0, 1];
      const r1 = p(args[3], sc[d1]);
      const r2 = style === 'cone' ? p(args[4], sc[d1]) : r1;
      const lo = style === 'cone' ? args[5] : args[4], hi = style === 'cone' ? args[6] : args[5];
      reg = new ConeRegion(id, style, env, ax, p(args[1], sc[d1]), p(args[2], sc[d2]), r1, r2,
        bound(sys, lo, sc[ax], ax, 'lo'), bound(sys, hi, sc[ax], ax, 'hi'));
      break;
    }
    case 'plane':
      reg = new PlaneRegion(id, env, [0, 1, 2].map((d) => p(args[d], sc[d])), [0, 1, 2].map((d) => p(args[3 + d], sc[d])));
      break;
    case 'prism': {
      const b = [0, 1, 2].flatMap((d) => [bound(sys, args[2 * d], sc[d], d, 'lo'), bound(sys, args[2 * d + 1], sc[d], d, 'hi')]);
      // "The lattice spacing in dimension x is applied to xy and xz, and the spacing in dimension y to yz."
      b.push(p(args[6], sc[0]), p(args[7], sc[0]), p(args[8], sc[1]));
      reg = new PrismRegion(id, env, b);
      break;
    }
    default: {
      const ids = args.slice(1);
      if (ids.length < 2) throw new StyleError(`region ${style} needs 2 or more region IDs`);
      for (const r of ids) if (!sys.regions.has(r)) throw new StyleError(`region ${style}: region ${r} does not exist`);
      if (move || rotate) throw new StyleError('move/rotate cannot be used with union or intersect regions');
      reg = new CompoundRegion(id, style as 'union' | 'intersect', env, ids);
    }
  }
  reg.interior = side;
  reg.move = move;
  reg.rotate = rotate;
  reg.openFaces = open;
  sys.regions.set(id, reg);
};

/**
 * create_box N region-ID [bond/types N ...] — create_box.html. A block region
 * gives an orthogonal box, a prism region a restricted triclinic one. "For
 * 2d simulations, the z-axis bounds must bracket zero" (dimension.html: 2d
 * boxes straddle z = 0).
 */
const createBox: Handler = ({ sys }, a) => {
  if (sys.hasBox) throw new StyleError('a simulation box already exists (use clear to start over)');
  const n = int(a[0], 'number of atom types');
  if (n < 1) throw new StyleError('create_box needs at least 1 atom type');
  if (a[1] === 'NULL') throw new StyleError('general triclinic boxes (create_box N NULL ...) are not supported');
  const reg = sys.region(a[1] ?? '');
  const kw = keywords(a.slice(2), {
    'bond/types': 1, 'angle/types': 1, 'dihedral/types': 1, 'improper/types': 1,
    'extra/bond/per/atom': 1, 'extra/angle/per/atom': 1, 'extra/dihedral/per/atom': 1, 'extra/improper/per/atom': 1, 'extra/special/per/atom': 1,
  }, 'create_box');
  let lo: [number, number, number], hi: [number, number, number];
  let tilt: [number, number, number] | undefined;
  if (reg instanceof BlockRegion) {
    const b = reg.bbox();
    if (!b) throw new StyleError('create_box: the region must be a static block with side in');
    lo = b.lo as [number, number, number]; hi = b.hi as [number, number, number];
  } else if (reg instanceof PrismRegion) {
    const v = reg.values();
    lo = [v[0], v[2], v[4]]; hi = [v[1], v[3], v[5]];
    tilt = [v[6], v[7], v[8]];
  } else throw new StyleError('create_box needs a block or prism region');
  if ([...lo, ...hi].some((v) => Math.abs(v) >= BIG)) throw new StyleError('create_box needs a finite region (no INF bounds)');
  if (sys.dimension === 2 && !(lo[2] < 0 && hi[2] > 0)) throw new StyleError('for a 2d simulation the region z bounds must bracket zero (e.g. -0.5 0.5)');
  if (sys.dimension === 2 && tilt && (tilt[1] !== 0 || tilt[2] !== 0)) throw new StyleError('2d triclinic boxes must have xz = yz = 0');
  const box = makeBox({ lo, hi, boundary: sys.boundary, tilt });
  const s = emptyState(sys.units, sys.dimension, box, n, sys.atomStyle);
  const types = (k: string) => (kw.has(k) ? int(kw.get(k)![0], k) : 0);
  s.topo.nbondtypes = types('bond/types');
  s.topo.nangletypes = types('angle/types');
  s.topo.ndihedraltypes = types('dihedral/types');
  s.topo.nimpropertypes = types('improper/types');
  const molecular = isMolecularStyle(s.atomStyle);
  if (!molecular && (s.topo.nbondtypes || s.topo.nangletypes)) throw new StyleError(`atom_style ${s.atomStyle} cannot have bond or angle types`);
  sys.setState(s);
  sys.ff.bond?.allocate(s.topo.nbondtypes);
  sys.ff.angle?.allocate(s.topo.nangletypes);
  sys.ff.dihedral?.allocate(s.topo.ndihedraltypes);
  sys.ff.improper?.allocate(s.topo.nimpropertypes);
  const f = (v: number[]) => v.map((x) => Number(x.toPrecision(8))).join(' ');
  sys.log(`Created ${tilt ? 'triclinic' : 'orthogonal'} box = (${f(lo)}) to (${f(hi)})${tilt ? ` with tilt (${f(tilt)})` : ''}`);
};

/**
 * create_atoms type box | region ID | single x y z | random N seed region-ID —
 * create_atoms.html: "If your box is periodic and a multiple of the lattice
 * spacing in a particular dimension, LAMMPS is careful to put exactly one
 * particle at the boundary (on either side of the box), not zero or two."
 * Native LAMMPS: a lattice point on an upper face is included only when the
 * face is shrink-wrapped (s); p, f and m faces are half-open (2x2x2 sc box:
 * 8 atoms with p/f/m, 27 with s). Keywords basis, ratio, subset, group,
 * remap, var, set, overlap, maxtry, units ("units = lattice" default).
 */
/**
 * Appends copies of a molecule template whose atom positions are given
 * (flat, natoms per copy): types offset by toff, consecutive atom and
 * molecule IDs after the existing maxima, template charges, and the
 * template's bonds / angles / dihedrals / impropers with those IDs.
 * Atoms are wrapped into periodic dimensions with image flags, so each
 * molecule stays whole when unwrapped.
 */
export const insertMolecules = (sys: System, t: MoleculeTemplate, pts: number[], toff: number, gbit: number): void => {
  const s = sys.state;
  const copies = pts.length / (3 * t.natoms);
  if (copies === 0) { sys.log('Created 0 atoms'); return; }
  const types = new Int32Array(copies * t.natoms);
  for (let c = 0; c < copies; c++) for (let i = 0; i < t.natoms; i++) types[c * t.natoms + i] = t.type[i] + toff;
  for (const ty of types) if (ty < 1 || ty > s.ntypes) throw new StyleError(`molecule ${t.id}: atom type ${ty} is outside 1..${s.ntypes}`);
  const molecular = isMolecularStyle(sys.atomStyle);
  const needs: [string, number[][], number, string[]][] = [
    ['bonds', t.bonds, s.topo.nbondtypes, ['bond', 'angle', 'molecular', 'full']],
    ['angles', t.angles, s.topo.nangletypes, ['angle', 'molecular', 'full']],
    ['dihedrals', t.dihedrals, s.topo.ndihedraltypes, ['molecular', 'full']],
    ['impropers', t.impropers, s.topo.nimpropertypes, ['molecular', 'full']],
  ];
  for (const [what, list, ntypes, styles] of needs) {
    if (!list.length) continue;
    if (!styles.includes(sys.atomStyle)) throw new StyleError(`molecule ${t.id} has ${what}, which atom_style ${sys.atomStyle} cannot store`);
    for (const e of list) if (e[0] < 1 || e[0] > ntypes) throw new StyleError(`molecule ${t.id}: ${what.slice(0, -1)} type ${e[0]} is outside 1..${ntypes}`);
  }
  if (t.q && !hasChargeStyle(sys.atomStyle)) throw new StyleError(`molecule ${t.id} has charges, which atom_style ${sys.atomStyle} cannot store`);
  const x = Float64Array.from(pts);
  const image = new Int32Array(x.length);
  const g = sys.geom;
  for (let i = 0; i < x.length / 3; i++) {
    g.remap(x, image, i);
    for (let d = 0; d < 3; d++) {
      if (s.box.periodic[d] || (d === 2 && sys.dimension === 2)) continue;
      if (x[3 * i + d] < s.box.lo[d] || x[3 * i + d] > s.box.hi[d]) throw new StyleError(`molecule ${t.id}: an inserted atom lies outside the non-periodic ${'xyz'[d]} boundary`);
    }
  }
  let maxMol = 0;
  for (let i = 0; i < s.n; i++) if (s.molecule[i] > maxMol) maxMol = s.molecule[i];
  let tmplMaxMol = 1;
  if (t.mol) for (const m of t.mol) tmplMaxMol = Math.max(tmplMaxMol, m);
  const mol = new Int32Array(copies * t.natoms);
  for (let c = 0; c < copies; c++) {
    for (let i = 0; i < t.natoms; i++) mol[c * t.natoms + i] = molecular ? maxMol + c * tmplMaxMol + (t.mol ? t.mol[i] : 1) : 0;
  }
  const q = t.q ? new Float64Array(copies * t.natoms) : undefined;
  if (q) for (let c = 0; c < copies; c++) q.set(t.q!, c * t.natoms);
  const base = maxAtomId(s);
  const added = appendAtoms(s, { x, image, type: types, molecule: mol, q, mask: gbit });
  for (let c = 0; c < copies; c++) {
    const id0 = base + c * t.natoms;
    for (const [what, list] of [['bonds', t.bonds], ['angles', t.angles], ['dihedrals', t.dihedrals], ['impropers', t.impropers]] as const) {
      for (const e of list) pushTopo(s.topo[what], e[0], e.slice(1).map((k) => id0 + k));
    }
  }
  sys.atomsChanged();
  sys.log(`Created ${added} atoms (${copies} molecules of template ${t.id})`);
};

const createAtoms: Handler = ({ sys }, a) => {
  const s = sys.state;
  let type = int(a[0], 'atom type');
  const style = a[1];
  let rest: string[];
  let regionId: string | null = null;
  let single: number[] | null = null;
  let random: { count: number; seed: number } | null = null;
  if (style === 'box') rest = a.slice(2);
  else if (style === 'region') { regionId = a[2]; sys.region(regionId ?? ''); rest = a.slice(3); } else if (style === 'single') {
    single = [num(a[2], 'x'), num(a[3], 'y'), num(a[4], 'z')];
    rest = a.slice(5);
  } else if (style === 'random') {
    random = { count: int(a[2], 'number of atoms'), seed: int(a[3], 'seed') };
    if (random.seed <= 0) throw new StyleError('create_atoms random: seed must be a positive integer');
    regionId = a[4] === 'NULL' ? null : a[4];
    if (a[4] === undefined) throw new StyleError('usage: create_atoms type random N seed region-ID|NULL');
    if (regionId) sys.region(regionId);
    rest = a.slice(5);
  } else if (style === 'mesh') {
    throw new StyleError('create_atoms mesh (STL files) is not supported by the browser engine');
  } else throw new StyleError(`unknown create_atoms style '${style ?? ''}'`);
  const kw = new Map<string, string[]>();
  const basisType = new Map<number, number>();
  for (let k = 0; k < rest.length;) {
    const key = rest[k];
    const nv: Record<string, number> = { basis: 2, ratio: 2, subset: 2, group: 1, remap: 1, var: 1, set: 2, overlap: 1, maxtry: 1, units: 1, rotate: 4, mol: 2, radscale: 1 };
    if (!(key in nv)) throw new StyleError(`unknown create_atoms keyword '${key}'`);
    const vals = rest.slice(k + 1, k + 1 + nv[key]);
    if (vals.length < nv[key]) throw new StyleError(`create_atoms ${key} needs ${nv[key]} value(s)`);
    if (key === 'basis') basisType.set(int(vals[0], 'basis index'), int(vals[1], 'basis type'));
    else if (key === 'set') kw.set(`set_${vals[0]}`, vals);
    else kw.set(key, vals);
    k += 1 + nv[key];
  }
  // create_atoms.html: "type = atom type ... of atoms to create (offset for molecule creation)"
  const molKw = kw.get('mol');
  const tmpl = molKw ? sys.molecules.get(molKw[0])?.[0] : undefined;
  if (molKw && !tmpl) throw new StyleError(`create_atoms mol: molecule template '${molKw[0]}' does not exist`);
  if (!tmpl && (type < 1 || type > s.ntypes)) throw new StyleError(`atom type ${type} is outside 1..${s.ntypes}`);
  if (tmpl && type < 0) throw new StyleError('create_atoms with mol: the type (an offset) must be >= 0');
  const molRng = molKw ? new Rng(int(molKw[1], 'mol seed')) : null;
  if (molKw && int(molKw[1], 'mol seed') <= 0) throw new StyleError('create_atoms mol: seed must be a positive integer');
  const rotKw = kw.get('rotate');
  if (rotKw && !tmpl) throw new StyleError('create_atoms rotate needs the mol keyword');
  if (rotKw && sys.dimension === 2 && (num(rotKw[1], 'Rx') !== 0 || num(rotKw[2], 'Ry') !== 0)) {
    throw new StyleError('create_atoms rotate: "A rotation vector specified for a single molecule must be in the z-direction for a 2d model."');
  }
  const fixedR = rotKw ? rotationMatrix((num(rotKw[0], 'theta') * Math.PI) / 180, num(rotKw[1], 'Rx'), num(rotKw[2], 'Ry'), num(rotKw[3], 'Rz')) : null;
  const molCenter = tmpl ? geometricCenter(tmpl) : [0, 0, 0];
  /** create_atoms.html: "placing the geometric center of the molecule at the lattice point, and (by default) giving the molecule a random orientation about the point" */
  const placeMol = (p: number[]): number[] => {
    let R = fixedR;
    if (!R) {
      if (sys.dimension === 2) R = rotationMatrix(2 * Math.PI * molRng!.uniform(), 0, 0, 1);
      else {
        // uniformly random rotation from a random unit quaternion (Shoemake)
        const u1 = molRng!.uniform(), u2 = molRng!.uniform(), u3 = molRng!.uniform();
        const q0 = Math.sqrt(1 - u1) * Math.sin(2 * Math.PI * u2), q1 = Math.sqrt(1 - u1) * Math.cos(2 * Math.PI * u2);
        const q2 = Math.sqrt(u1) * Math.sin(2 * Math.PI * u3), q3 = Math.sqrt(u1) * Math.cos(2 * Math.PI * u3);
        R = [
          [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
          [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
          [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
        ];
      }
    }
    const out: number[] = [];
    for (let i = 0; i < tmpl!.natoms; i++) {
      const r = [0, 1, 2].map((d) => tmpl!.x[3 * i + d] - molCenter[d]);
      for (let d = 0; d < 3; d++) out.push(p[d] + R[d][0] * r[0] + R[d][1] * r[1] + R[d][2] * r[2]);
    }
    return out;
  };
  const unitsW = kw.get('units')?.[0] ?? 'lattice';
  const g = sys.geom;
  const box = s.box;
  const reg = regionId ? sys.region(regionId) : null;
  // variable test (var + set x/y/z)
  const varName = kw.get('var')?.[0];
  const setVars = ['x', 'y', 'z'].map((d) => kw.get(`set_${d}`)?.[1] ?? null);
  const varOk = (p: number[]) => {
    if (!varName) return true;
    setVars.forEach((v, d) => { if (v) sys.vars.setInternal(v, p[d]); });
    return sys.equalVariable(varName) !== 0;
  };
  const lam = [0, 0, 0];
  const insideBox = (x: number, y: number, z: number) => {
    g.toLamda(x, y, z, lam);
    const eps = 1e-9;
    for (let d = 0; d < (sys.dimension === 2 ? 2 : 3); d++) {
      const upperInclusive = box.boundary[d][1] === 's';
      if (lam[d] < -eps) return false;
      if (upperInclusive ? lam[d] > 1 + eps : lam[d] >= 1 - eps) return false;
    }
    return true;
  };
  let pts: number[] = [];
  let types: number[] = [];
  if (single) {
    const sc = latticeScale(sys, unitsW, 'create_atoms');
    const p = single.map((v, d) => v * sc[d]);
    if (sys.dimension === 2) p[2] = 0;
    const remap = kw.has('remap') ? yesno(kw.get('remap')![0], 'remap') : false;
    if (remap) {
      const xx = Float64Array.from(p), im = new Int32Array(3);
      g.remap(xx, im, 0);
      p[0] = xx[0]; p[1] = xx[1]; p[2] = xx[2];
    }
    g.toLamda(p[0], p[1], p[2], lam);
    const inside = [0, 1, 2].every((d) => (d === 2 && sys.dimension === 2) || (lam[d] >= 0 && (box.boundary[d][1] === 's' || box.boundary[d][1] === 'm' ? lam[d] <= 1 : lam[d] < 1)));
    if (!inside) {
      sys.warn('create_atoms single: the point is outside the box; no atom created');
    } else if (varOk(p)) { pts = tmpl ? placeMol(p) : p; types = [type]; }
  } else if (random) {
    const rng = new Rng(random.seed);
    const overlap = kw.has('overlap') ? num(kw.get('overlap')![0], 'overlap') : 0;
    const maxtry = kw.has('maxtry') ? int(kw.get('maxtry')![0], 'maxtry') : 10;
    // sample in the region's bounding box clipped to the box, accept points inside both
    const bb = reg?.bbox();
    const blo = [0, 1, 2].map((d) => Math.max(box.lo[d] + Math.min(0, d === 0 ? Math.min(box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]) : d === 1 ? box.tilt[2] : 0), bb ? bb.lo[d] : -Infinity));
    const bhi = [0, 1, 2].map((d) => Math.min(box.hi[d] + Math.max(0, d === 0 ? Math.max(box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]) : d === 1 ? box.tilt[2] : 0), bb ? bb.hi[d] : Infinity));
    const existing: number[] = Array.from(s.x.subarray(0, 3 * s.n));
    const tooClose = (p: number[]) => {
      if (overlap <= 0) return false;
      const d = [0, 0, 0];
      for (const arr of [existing, pts]) {
        for (let j = 0; j < arr.length; j += 3) {
          d[0] = p[0] - arr[j]; d[1] = p[1] - arr[j + 1]; d[2] = p[2] - arr[j + 2];
          g.minimumImage(d);
          if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < overlap * overlap) return true;
        }
      }
      return false;
    };
    let failed = 0;
    for (let c = 0; c < random.count; c++) {
      let placed = false;
      for (let t = 0; t < maxtry * 1000 && !placed; t++) {
        const p = [0, 1, 2].map((d) => (d === 2 && sys.dimension === 2 ? 0 : blo[d] + rng.uniform() * (bhi[d] - blo[d])));
        if (!insideBox(p[0], p[1], p[2])) continue;
        if (reg && !reg.match(p[0], p[1], p[2])) continue;
        if (!varOk(p)) continue;
        const atoms = tmpl ? placeMol(p) : p;
        let close = false;
        for (let j = 0; j < atoms.length && !close; j += 3) close = tooClose([atoms[j], atoms[j + 1], atoms[j + 2]]);
        if (close) { if (t >= maxtry - 1) break; continue; }
        pts.push(...atoms);
        types.push(type);
        placed = true;
      }
      if (!placed) failed++;
    }
    if (failed) sys.warn(`create_atoms random: only ${random.count - failed} of ${random.count} atoms could be inserted`);
  } else {
    const lat = sys.lattice;
    if (!lat || lat.style === 'none') throw new StyleError(`create_atoms ${style} needs a lattice (lattice command)`);
    // bounding box of the domain (tilted boxes: extents of the parallelepiped)
    const xlo = box.lo[0] + Math.min(0, box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]);
    const xhi = box.hi[0] + Math.max(0, box.tilt[0], box.tilt[1], box.tilt[0] + box.tilt[1]);
    const ylo = box.lo[1] + Math.min(0, box.tilt[2]), yhi = box.hi[1] + Math.max(0, box.tilt[2]);
    const site = latticeSites(lat, [xlo, ylo, box.lo[2]], [xhi, yhi, box.hi[2]], sys.dimension,
      (x, y, z) => insideBox(x, y, z) && (!reg || reg.match(x, y, z)) && varOk([x, y, z]));
    pts = site.x;
    types = site.basis.map((b) => basisType.get(b + 1) ?? type);
    // ratio / subset: choose a random subset of the sites
    const pick = kw.get('ratio') ?? kw.get('subset');
    if (pick) {
      const nsite = types.length;
      const want = kw.has('ratio') ? Math.floor(num(pick[0], 'ratio') * nsite) : int(pick[0], 'subset');
      if (want > nsite) throw new StyleError(`create_atoms subset: ${want} is more than the ${nsite} lattice sites`);
      const rng = new Rng(int(pick[1], 'seed'));
      const idx = Array.from({ length: nsite }, (_, i) => i);
      for (let i = nsite - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      const keep = idx.slice(0, want).sort((p, q) => p - q);
      pts = keep.flatMap((i) => [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]]);
      types = keep.map((i) => types[i]);
    }
  }
  const groupName = kw.get('group')?.[0];
  const gbit = groupName ? bitOfIndex(sys.groups.create(groupName)) : 0;
  if (tmpl) {
    if (!single && !random) {
      const centers = pts;
      pts = [];
      for (let c = 0; c < centers.length; c += 3) pts.push(...placeMol([centers[c], centers[c + 1], centers[c + 2]]));
    }
    insertMolecules(sys, tmpl, pts, type, gbit);
    return;
  }
  for (const t of types) if (t < 1 || t > s.ntypes) throw new StyleError(`create_atoms basis type ${t} is outside 1..${s.ntypes}`);
  const added = appendAtoms(s, { x: Float64Array.from(pts), type: Int32Array.from(types), mask: gbit });
  type = 0;
  sys.atomsChanged();
  sys.log(`Created ${added} atoms`);
};

/**
 * molecule ID file1 keyword values ... file2 ... — molecule.html: "zero or
 * more keyword/value pairs may be appended after each file"; "keyword =
 * offset or toff or boff or aoff or doff or ioff or scale". The ID "can only
 * contain alphanumeric characters and underscores".
 */
const molecule: Handler = ({ sys }, a) => {
  const id = a[0];
  if (!id || a.length < 2) throw new StyleError('usage: molecule ID file1 keyword values ... file2 ...');
  if (!/^[A-Za-z0-9_]+$/.test(id)) throw new StyleError(`molecule ID '${id}' must be alphanumeric or underscore`);
  if (sys.molecules.has(id)) throw new StyleError(`molecule template ID '${id}' already exists`);
  const sets: MoleculeTemplate[] = [];
  for (let k = 1; k < a.length;) {
    const file = a[k++];
    const o: MoleculeOptions = { toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 1 };
    while (k < a.length && ['offset', 'toff', 'boff', 'aoff', 'doff', 'ioff', 'scale'].includes(a[k])) {
      const key = a[k];
      if (key === 'offset') {
        [o.toff, o.boff, o.aoff, o.doff, o.ioff] = [1, 2, 3, 4, 5].map((j) => int(a[k + j], 'offset'));
        k += 6;
      } else if (key === 'scale') { o.scale = num(a[k + 1], 'scale'); k += 2; } else { o[key as 'toff'] = int(a[k + 1], key); k += 2; }
    }
    const t = parseMoleculeFile(id, file, sys.readFile(file), o);
    sets.push(t);
    sys.log(`Read molecule template ${id}: ${t.natoms} atoms, ${t.bonds.length} bonds, ${t.angles.length} angles, ${t.dihedrals.length} dihedrals, ${t.impropers.length} impropers (${file})`);
  }
  sys.molecules.set(id, sets);
};

/** mass I value — mass.html: "I can be specified ... as a wildcard"; "All masses must be defined before a simulation is run." */
const mass: Handler = ({ sys }, a) => {
  const s = sys.state;
  if (a.length !== 2) throw new StyleError('usage: mass I value');
  // measured with native LAMMPS: "Cannot set per-type atom mass for atom style sphere"
  if (s.atomStyle === 'sphere') throw new StyleError(`Cannot set per-type atom mass for atom style ${s.atomStyle}`);
  const m = num(a[1], 'mass');
  if (!(m > 0)) throw new StyleError('mass must be > 0');
  const [lo, hi] = typeBounds(a[0], s.ntypes);
  for (let t = lo; t <= hi; t++) s.massByType[t] = m;
  sys.bump();
};

/** read_data file [keywords] — read_data.html. */
const readDataCmd: Handler = ({ sys }, a) => {
  if (!a[0]) throw new StyleError('usage: read_data file [keywords]');
  const text = sys.readFile(a[0]);
  const o: ReadDataOptions = defaultReadOptions();
  for (let k = 1; k < a.length;) {
    const key = a[k];
    switch (key) {
      case 'add':
        if (a[k + 1] === 'append') { o.add = 'append'; k += 2; } else if (a[k + 1] === 'merge') { o.add = 'merge'; k += 2; } else {
          const id = int(a[k + 1], 'IDoffset');
          const mol = /^\d+$/.test(a[k + 2] ?? '') ? int(a[k + 2], 'MOLoffset') : 0;
          o.add = { id, mol };
          k += /^\d+$/.test(a[k + 2] ?? '') ? 3 : 2;
        }
        break;
      case 'offset': o.offset = [1, 2, 3, 4, 5].map((j) => int(a[k + j], 'offset')) as ReadDataOptions['offset']; k += 6; break;
      case 'shift': o.shift = [1, 2, 3].map((j) => num(a[k + j], 'shift')) as ReadDataOptions['shift']; k += 4; break;
      case 'extra/atom/types': o.extraTypes[0] = int(a[k + 1], key); k += 2; break;
      case 'extra/bond/types': o.extraTypes[1] = int(a[k + 1], key); k += 2; break;
      case 'extra/angle/types': o.extraTypes[2] = int(a[k + 1], key); k += 2; break;
      case 'extra/dihedral/types': o.extraTypes[3] = int(a[k + 1], key); k += 2; break;
      case 'extra/improper/types': o.extraTypes[4] = int(a[k + 1], key); k += 2; break;
      case 'extra/bond/per/atom': case 'extra/angle/per/atom': case 'extra/dihedral/per/atom':
      case 'extra/improper/per/atom': case 'extra/special/per/atom': int(a[k + 1], key); k += 2; break;
      case 'group': o.group = a[k + 1]; k += 2; break;
      case 'nocoeff': o.nocoeff = true; k += 1; break;
      case 'fix': {
        const [fid, header, section] = [a[k + 1], a[k + 2], a[k + 3]];
        if (!fid || !header || !section) throw new StyleError('usage: read_data file fix fix-ID header-string section-string');
        if (header !== 'NULL') throw new StyleError(`read_data fix ${fid}: header-string must be NULL (fix property/atom reads no header lines)`);
        o.fixSections.set(section, fid);
        k += 4;
        break;
      }
      default: throw new StyleError(`unknown read_data keyword '${key}'`);
    }
  }
  readData(sys, text, o);
  const s = sys.state;
  sys.log(`Read ${s.n} atoms${s.topo.bonds.n ? `, ${s.topo.bonds.n} bonds` : ''}${s.topo.angles.n ? `, ${s.topo.angles.n} angles` : ''}${s.topo.dihedrals.n ? `, ${s.topo.dihedrals.n} dihedrals` : ''}${s.topo.impropers.n ? `, ${s.topo.impropers.n} impropers` : ''} from ${a[0]}`);
};

/** write_data file [nocoeff] [pair ii|ij] [nofix] [nolabelmap] [types numeric] — write_data.html. */
const writeDataCmd: Handler = ({ sys }, a) => {
  if (!a[0]) throw new StyleError('usage: write_data file [keywords]');
  let nocoeff = false, nofix = false;
  let pairStyle: 'ii' | 'ij' | null = null;
  for (let k = 1; k < a.length;) {
    if (a[k] === 'nocoeff') { nocoeff = true; k++; } else if (a[k] === 'nofix') { nofix = true; k++; } else if (a[k] === 'nolabelmap') k++;
    else if (a[k] === 'pair') {
      if (a[k + 1] !== 'ii' && a[k + 1] !== 'ij') throw new StyleError('write_data pair must be ii or ij');
      pairStyle = a[k + 1] as 'ii' | 'ij';
      k += 2;
    } else if (a[k] === 'types' || a[k] === 'triclinic/general') k += 2;
    else throw new StyleError(`unknown write_data keyword '${a[k]}'`);
  }
  // the box must be current (shrink-wrapped faces, remapped atoms): write_data "calls ... pbc" via a setup.
  // Measured with native LAMMPS: atoms that drifted out of a periodic box since the last reneighboring
  // are written wrapped back in, so remap them even when forces are current.
  sys.pbc();
  sys.nb.lastBuild = -1;
  sys.bump();
  sys.forces();
  sys.writeFile(a[0], writeData(sys, { nocoeff, pairStyle, nofix }), false);
  sys.log(`Wrote ${sys.state.n} atoms to ${a[0]}`);
};

/** timestep dt — timestep.html; also keeps the elapsed time ("time") cumulative. */
const timestep: Handler = ({ sys }, a) => {
  const dt = num(a[0], 'timestep');
  if (!(dt > 0)) throw new StyleError('timestep must be > 0');
  if (!sys.hasBox) { sys.pendingDt = dt; return; }
  const s = sys.state;
  s.time += (s.step - s.timeStep) * s.dt;
  s.timeStep = s.step;
  s.dt = dt;
  for (const f of sys.fixes) f.resetDt?.();
};

/** reset_timestep N [time T] — reset_timestep.html. */
const resetTimestep: Handler = ({ sys }, a) => {
  const s = sys.state;
  const n = int(a[0], 'timestep');
  if (n < 0) throw new StyleError('timestep must be >= 0');
  let t: number | null = null;
  if (a[1] === 'time') t = num(a[2], 'time');
  else if (a.length > 1) throw new StyleError('usage: reset_timestep N [time T]');
  s.time = t ?? s.time + (s.step - s.timeStep) * s.dt;
  s.step = n;
  s.timeStep = n;
  sys.nb.lastBuild = -1;
};

// ----------------------------------------------------------------- groups

/** "A:B:C" sequences / single values. */
const parseList = (w: string): [number, number, number] => {
  const p = w.split(':').map((x) => int(x, 'group list entry'));
  if (p.length === 1) return [p[0], p[0], 1];
  if (p.length === 2) return [p[0], p[1], 1];
  if (p.length === 3) return [p[0], p[1], p[2]];
  throw new StyleError(`invalid group list entry '${w}'`);
};

/** group ID style args — group.html. */
const group: Handler = ({ sys }, a) => {
  const [id, style] = a;
  if (!id || !style) throw new StyleError('usage: group ID style args');
  if (style === 'delete') {
    if (id === 'all') throw new StyleError('cannot delete group all');
    const k = sys.groups.delete(id);
    if (sys.hasBox) { const s = sys.state; const m = ~bitOfIndex(k); for (let i = 0; i < s.n; i++) s.mask[i] &= m; }
    return;
  }
  const s = sys.state;
  const args = a.slice(2);
  if (style === 'dynamic') {
    groupDynamic(sys, id, args);
    return;
  }
  if (style === 'static') {
    // Measured with native LAMMPS (black box): group static on a group that is not dynamic only
    // reports its count, and before any run a new dynamic group has no atoms yet
    const k0 = sys.groups.find(id);
    if (k0 < 0) throw new StyleError(`Could not find group static group ID ${id}`);
    if (args.length) throw new StyleError('usage: group ID static');
    sys.groups.dynamic.delete(bitOfIndex(k0));
    let n = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & bitOfIndex(k0)) n++;
    sys.log(`${n} atoms in group ${id}`);
    sys.refreshComputes();
    return;
  }
  // Measured with native LAMMPS (black box): group clear needs an existing group and logs nothing
  if (style === 'clear' && sys.groups.find(id) < 0) throw new StyleError(`Could not find group clear group ID ${id}`);
  const k = sys.groups.create(id);
  const bit = bitOfIndex(k);
  const set = (pred: (i: number) => boolean) => { for (let i = 0; i < s.n; i++) if (pred(i)) s.mask[i] |= bit; };
  switch (style) {
    case 'clear': {
      if (id === 'all') throw new StyleError('cannot clear group all');
      const m = ~bit;
      for (let i = 0; i < s.n; i++) s.mask[i] &= m;
      break;
    }
    case 'empty': break;
    case 'region': {
      const r = sys.region(args[0] ?? '');
      set((i) => r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]));
      break;
    }
    case 'type': case 'id': case 'molecule': {
      const val = (i: number) => (style === 'type' ? s.type[i] : style === 'id' ? s.id[i] : s.molecule[i]);
      if (['<', '<=', '>', '>=', '==', '!=', '<>'].includes(args[0])) {
        const op = args[0];
        const v1 = int(args[1], 'value');
        const v2 = op === '<>' ? int(args[2], 'value2') : 0;
        set((i) => {
          const v = val(i);
          switch (op) {
            case '<': return v < v1; case '<=': return v <= v1; case '>': return v > v1; case '>=': return v >= v1;
            case '==': return v === v1; case '!=': return v !== v1; default: return v >= v1 && v <= v2;
          }
        });
      } else {
        if (!args.length) throw new StyleError(`group ${style} needs a list of values`);
        const lists = args.map(parseList);
        set((i) => {
          const v = val(i);
          return lists.some(([lo, hi, st]) => v >= lo && v <= hi && (v - lo) % st === 0);
        });
      }
      break;
    }
    case 'variable': {
      const v = sys.atomVariable(args[0] ?? '');
      set((i) => v[i] !== 0);
      break;
    }
    case 'include': {
      if (args[0] !== 'molecule') throw new StyleError('usage: group ID include molecule');
      const mols = new Set<number>();
      for (let i = 0; i < s.n; i++) if (s.mask[i] & bit && s.molecule[i] !== 0) mols.add(s.molecule[i]);
      set((i) => mols.has(s.molecule[i]));
      break;
    }
    case 'subtract': case 'union': case 'intersect': {
      if (!args.length) throw new StyleError(`group ${style} needs group IDs`);
      const bits = args.map((g) => sys.groupBit(g));
      // Measured with native LAMMPS (black box): a dynamic group cannot be combined
      if (bits.some((b) => sys.groups.isDynamic(b))) {
        throw new StyleError(style === 'subtract' ? 'Cannot subtract dynamic groups'
          : style === 'union' ? 'Cannot union groups from a dynamic group' : 'Cannot intersect groups using a dynamic group');
      }
      if (style !== 'union' && bits.length < 2) throw new StyleError(`group ${style} needs 2 or more groups`);
      set((i) => {
        const m = s.mask[i];
        if (style === 'union') return bits.some((b) => (m & b) !== 0);
        if (style === 'intersect') return bits.every((b) => (m & b) !== 0);
        return (m & bits[0]) !== 0 && bits.slice(1).every((b) => (m & b) === 0);
      });
      break;
    }
    default: throw new StyleError(`unknown group style '${style}'`);
  }
  sys.refreshComputes();
  if (style === 'clear') return;
  // Measured with native LAMMPS (black box): a dynamic group reports "dynamic group ID defined"
  // after any group command instead of its count
  if (sys.groups.isDynamic(bit)) { sys.log(`dynamic group ${id} defined`); return; }
  let count = 0;
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) count++;
  sys.log(`${count} atoms in group ${id}`);
};

/*
 * group ID dynamic parent-ID keyword value ... — docs.lammps.org/group.html:
 *   *dynamic* args = parent-ID keyword value ...
 *     keyword = *region* or *var* or *property* or *every*
 * "A group with the ID all is predefined. All atoms belong to this group.
 * This group cannot be deleted, or made dynamic." "If the *var* keyword is
 * used, the variable name must be an atom-style or atomfile-style variable."
 * "Note that the name of the custom per-atom vector is specified just as
 * *name*, not as *i_name* or *d_name*".
 * Measured with native LAMMPS (black box): the group keeps its atoms until the
 * next run assigns them; the command logs "dynamic group ID defined"; a
 * missing region, variable or property and a non-positive every stop at the
 * command, the variable style and a dynamic parent at the next run; the
 * error texts below are native's.
 */
const groupDynamic = (sys: System, id: string, args: string[]): void => {
  const parent = args[0];
  if (!parent || args.length < 3) throw new StyleError('Illegal group command: usage group ID dynamic parent-ID keyword value ...');
  if (parent === id) throw new StyleError('Group dynamic cannot reference itself');
  if (sys.groups.find(parent) < 0) throw new StyleError(`Group dynamic parent group ${parent} does not exist`);
  if (id === 'all') throw new StyleError('Group all cannot be made dynamic');
  const g: DynamicGroup = { parent, region: null, variable: null, property: null, every: 1 };
  for (let k = 1; k < args.length; k += 2) {
    const key = args[k], val = args[k + 1];
    if (!['region', 'var', 'property', 'every'].includes(key)) throw new StyleError(`Unknown keyword ${key} in dynamic group command`);
    if (val === undefined) throw new StyleError(`Illegal group dynamic command: missing value for ${key}`);
    if (key === 'region') {
      if (!sys.regions.has(val)) throw new StyleError(`Region ${val} for dynamic group ${id} does not exist`);
      g.region = val;
    } else if (key === 'var') {
      if (!sys.vars.has(val)) throw new StyleError(`Variable '${val}' for dynamic group ${id} does not exist`);
      g.variable = val;
    } else if (key === 'property') {
      const c = sys.state.custom.get(val);
      if (!c || c.cols !== 0) throw new StyleError(`Custom per-atom vector ${val} for dynamic group ${id} does not exist`);
      g.property = val;
    } else {
      const n = Number(val);
      if (!Number.isInteger(n) || n <= 0) throw new StyleError(`Illegal every value ${val} for dynamic group ${id}`);
      g.every = n;
    }
  }
  sys.groups.dynamic.set(bitOfIndex(sys.groups.create(id)), g);
  sys.log(`dynamic group ${id} defined`);
};

// ----------------------------------------------------------------- set

/** Atoms selected by a set style/ID. */
const selected = (sys: System, style: string, id: string): number[] => {
  const s = sys.state;
  const out: number[] = [];
  const range = (w: string, max: number) => typeBounds(w.includes('*') || /^\d+$/.test(w) ? w : (() => { throw new StyleError(`invalid set ID '${w}'`); })(), max);
  if (style === 'atom') {
    const [lo, hi] = range(id, Math.max(1, maxAtomId(s)));
    for (let i = 0; i < s.n; i++) if (s.id[i] >= lo && s.id[i] <= hi) out.push(i);
  } else if (style === 'type') {
    const [lo, hi] = range(id, s.ntypes);
    for (let i = 0; i < s.n; i++) if (s.type[i] >= lo && s.type[i] <= hi) out.push(i);
  } else if (style === 'mol') {
    let mx = 1;
    for (let i = 0; i < s.n; i++) mx = Math.max(mx, s.molecule[i]);
    const [lo, hi] = range(id, mx);
    for (let i = 0; i < s.n; i++) if (s.molecule[i] >= lo && s.molecule[i] <= hi) out.push(i);
  } else if (style === 'group') {
    const bit = sys.groupBit(id);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) out.push(i);
  } else if (style === 'region') {
    const r = sys.region(id);
    for (let i = 0; i < s.n; i++) if (r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) out.push(i);
  } else throw new StyleError(`unknown set style '${style}' (atom, type, mol, group or region)`);
  return out;
};

/**
 * set style ID keyword values — set.html. Values may be atom-style variables
 * (v_name). type/fraction, type/ratio and type/subset choose atoms randomly
 * with our own generator (the chosen atoms differ from LAMMPS for a seed).
 */
const set: Handler = ({ sys }, a) => {
  const s = sys.state;
  if (a.length < 4) throw new StyleError('usage: set style ID keyword values ...');
  const atoms = selected(sys, a[0], a[1]);
  const value = (w: string | undefined, what: string): ((i: number) => number) => {
    if (w?.startsWith('v_')) {
      const v = sys.atomVariable(w.slice(2));
      return (i) => v[i];
    }
    const x = num(w, what);
    return () => x;
  };
  let changed = 0;
  for (let k = 2; k < a.length;) {
    const key = a[k];
    switch (key) {
      case 'type': {
        const v = value(a[k + 1], 'type');
        for (const i of atoms) {
          const t = Math.trunc(v(i));
          if (t < 1 || t > s.ntypes) throw new StyleError(`set type ${t} is outside 1..${s.ntypes}`);
          s.type[i] = t;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'type/fraction': case 'type/ratio': case 'type/subset': {
        const t = int(a[k + 1], 'type');
        const amount = num(a[k + 2], key);
        const rng = new Rng(int(a[k + 3], 'seed'));
        if (t < 1 || t > s.ntypes) throw new StyleError(`set type ${t} is outside 1..${s.ntypes}`);
        if (key === 'type/fraction') {
          for (const i of atoms) if (rng.uniform() < amount) { s.type[i] = t; changed++; }
        } else {
          const want = key === 'type/ratio' ? Math.floor(amount * atoms.length) : Math.trunc(amount);
          if (want > atoms.length) throw new StyleError(`${key}: ${want} is more than the ${atoms.length} selected atoms`);
          const idx = atoms.slice();
          for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
          for (const i of idx.slice(0, want)) s.type[i] = t;
          changed = want;
        }
        k += 4;
        break;
      }
      case 'mol': {
        if (!hasMolecule(s)) throw new StyleError(`Cannot set attribute mol for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], 'mol');
        for (const i of atoms) s.molecule[i] = Math.trunc(v(i));
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'charge': {
        if (!hasCharge(s)) throw new StyleError(`set charge needs atom_style charge or full, or fix property/atom q (current: ${s.atomStyle})`);
        const v = value(a[k + 1], 'charge');
        for (const i of atoms) s.q[i] = v(i);
        changed = atoms.length;
        sys.nb.refreshCharges(s);
        k += 2;
        break;
      }
      case 'x': case 'y': case 'z': case 'vx': case 'vy': case 'vz': {
        const v = value(a[k + 1], key);
        const arr = key[0] === 'v' ? s.v : s.x;
        const d = 'xyz'.indexOf(key[key.length - 1]);
        for (const i of atoms) arr[3 * i + d] = v(i);
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'image': {
        for (let d = 0; d < 3; d++) {
          const w = a[k + 1 + d];
          if (w === 'NULL') continue;
          if (!s.box.periodic[d] && w !== '0') throw new StyleError('set image: non-zero image flags need a periodic dimension');
          const v = value(w, 'image');
          for (const i of atoms) s.image[3 * i + d] = Math.trunc(v(i));
        }
        changed = atoms.length;
        k += 4;
        break;
      }
      // set.html: "Keyword *mass* sets the mass of all selected particles.  The particles must have a
      // per-atom mass attribute"; "Keyword *diameter* sets the size of the selected atoms ... this
      // command does not adjust the particle mass"; "Keyword *density* or *density/disc* also sets the
      // mass" (atoms.ts sphereMass); "Keyword *omega* sets the angular velocity of selected atoms."
      case 'mass': case 'density': case 'density/disc': {
        // errors as measured with native LAMMPS
        if (!s.rmass) throw new StyleError(`Cannot set attribute ${key} for atom style ${s.atomStyle}`);
        if (key === 'density/disc' && sys.dimension !== 2) throw new StyleError('Set density/disc requires 2d simulation');
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const x = v(i);
          if (!(x > 0)) throw new StyleError(key === 'mass' ? `Invalid mass ${x} in set command` : `Invalid density value ${x} in set command`);
          s.rmass[i] = key === 'mass' ? x : sphereMass(s.radius ? s.radius[i] : 0, x, key === 'density/disc');
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'diameter': {
        if (!s.radius) throw new StyleError(`Cannot set attribute diameter for atom style ${s.atomStyle}`);
        const v = value(a[k + 1], key);
        for (const i of atoms) {
          const d = v(i);
          if (!(d >= 0)) throw new StyleError(`Invalid diameter value ${d} in set command`);
          s.radius[i] = d / 2;
        }
        changed = atoms.length;
        k += 2;
        break;
      }
      case 'omega': {
        if (!s.omega) throw new StyleError(`Cannot set attribute omega for atom style ${s.atomStyle}`);
        const vs = [1, 2, 3].map((d) => value(a[k + d], 'omega'));
        for (const i of atoms) for (let d = 0; d < 3; d++) s.omega[3 * i + d] = vs[d](i);
        changed = atoms.length;
        k += 4;
        break;
      }
      case 'bond': case 'angle': case 'dihedral': case 'improper': {
        const t = int(a[k + 1], `${key} type`);
        const list = s.topo[`${key}s` as 'bonds' | 'angles' | 'dihedrals' | 'impropers'];
        const ids = new Set(atoms.map((i) => s.id[i]));
        for (let e = 0; e < list.n; e++) {
          let all = true;
          for (let w = 0; w < list.width; w++) if (!ids.has(list.atoms[e * list.width + w])) { all = false; break; }
          if (all) { list.type[e] = t; changed++; }
        }
        k += 2;
        break;
      }
      default: {
        // set.html: "*i_name* value = custom integer vector with name", "*d_name* value = custom
        // floating-point vector with name", "column specified as i2_name[N] where N is 1 to Ncol"
        const m = /^(i|d)(2?)_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(key);
        const cp = m ? s.custom.get(m[3]) : undefined;
        if (!m || !cp || cp.int !== (m[1] === 'i') || (cp.cols > 0) !== (m[2] === '2')) {
          throw new StyleError(m ? `set ${key}: no fix property/atom defines this property` : `set keyword '${key}' is not supported by the browser engine`);
        }
        const col = m[4] ? Number(m[4]) : 0;
        if (cp.cols > 0 && (col < 1 || col > cp.cols)) throw new StyleError(`set ${key}: column must be 1..${cp.cols}`);
        if (cp.cols === 0 && m[4]) throw new StyleError(`set ${key}: ${m[1]}_${m[3]} is a vector, not an array`);
        const w = Math.max(cp.cols, 1), off = cp.cols ? col - 1 : 0;
        const v = value(a[k + 1], key);
        for (const i of atoms) { const x = v(i); cp.data[w * i + off] = cp.int ? Math.trunc(x) : x; }
        changed = atoms.length;
        k += 2;
        break;
      }
    }
  }
  sys.bump();
  sys.log(`Setting atom values ...\n  ${changed} settings made for ${a[2]}`);
};

// ----------------------------------------------------------------- velocity

/** velocity group style args [keywords] — velocity.html (see the module comment for defaults). */
const velocity: Handler = ({ sys }, a) => {
  const s = sys.state;
  const [gname, style] = a;
  if (!gname || !style) throw new StyleError('usage: velocity group-ID style args [keywords]');
  const bit = sys.groupBit(gname);
  const nargs: Record<string, number> = { create: 2, set: 3, scale: 1, ramp: 6, zero: 1 };
  if (!(style in nargs)) throw new StyleError(`unknown velocity style '${style}'`);
  const args = a.slice(2, 2 + nargs[style]);
  if (args.length < nargs[style]) throw new StyleError(`velocity ${style} needs ${nargs[style]} arguments`);
  const kw = keywords(a.slice(2 + nargs[style]), { dist: 1, sum: 1, mom: 1, rot: 1, temp: 1, bias: 1, loop: 1, rigid: 1, units: 1 }, 'velocity');
  // "The keyword defaults are dist = uniform, sum = no, mom = yes, rot = no, bias = no, loop = all, and units = lattice."
  const dist = kw.get('dist')?.[0] ?? 'uniform';
  if (dist !== 'uniform' && dist !== 'gaussian') throw new StyleError('velocity dist must be uniform or gaussian');
  const sum = kw.has('sum') ? yesno(kw.get('sum')![0], 'sum') : false;
  const mom = kw.has('mom') ? yesno(kw.get('mom')![0], 'mom') : true;
  const rot = kw.has('rot') ? yesno(kw.get('rot')![0], 'rot') : false;
  const bias = kw.has('bias') ? yesno(kw.get('bias')![0], 'bias') : false;
  const loop = kw.get('loop')?.[0] ?? 'all';
  if (!['all', 'local', 'geom'].includes(loop)) throw new StyleError('velocity loop must be all, local or geom');
  if (kw.has('rigid')) throw new StyleError('velocity rigid is not supported by the browser engine');
  const unitsW = kw.get('units')?.[0] ?? 'lattice';
  for (let t = 1; t <= s.ntypes; t++) {
    if (!s.rmass && !(s.massByType[t] > 0)) throw new StyleError(`velocity: the mass of atom type ${t} is not set`);
  }
  const members: number[] = [];
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) members.push(i);
  // "If this keyword is not specified, create and scale calculate temperature using a compute ...
  //  compute velocity_temp group-ID temp"
  const tempCompute = () => {
    if (kw.has('temp')) {
      const c = sys.compute(kw.get('temp')![0]);
      if (!c.tempFlag) throw new StyleError(`velocity temp: compute ${c.id} does not compute a temperature`);
      c.init();
      return c;
    }
    const c = new ComputeTemp(sys, 'velocity_temp', gname, []);
    c.init();
    return c;
  };
  const scaleTo = (target: number) => {
    sys.refreshComputes();
    const c = tempCompute();
    if (bias && c.hasBias()) { c.computeBias(); c.removeBiasAll(); }
    sys.refreshComputes();
    const t = c.scalarValue();
    if (t > 0) {
      const f = Math.sqrt(target / t);
      for (const i of members) { s.v[3 * i] *= f; s.v[3 * i + 1] *= f; s.v[3 * i + 2] *= f; }
    } else if (target > 0) sys.warn('velocity: current temperature is 0, cannot rescale');
    if (bias && c.hasBias()) c.restoreBiasAll();
  };
  const old = sum ? Float64Array.from(s.v) : null;
  switch (style) {
    case 'create': {
      const t = num(args[0], 'temperature');
      const seed = int(args[1], 'seed');
      if (seed <= 0) throw new StyleError('velocity create: seed must be a positive integer');
      if (t < 0) throw new StyleError('velocity create: temperature must be >= 0');
      // Measured with native LAMMPS (black box; rng.ts RanPark): each atom takes
      // three draws, uniform() - 0.5 or gaussian(), scaled by 1/sqrt(mass).
      // velocity.html: "If loop = all, then each processor loops over all
      // atoms in the simulation to create velocities, but only stores
      // velocities for atoms it owns." — draws run over atom IDs 1..N
      // (non-members draw too) and need consecutive IDs (native stops with an
      // error otherwise). "If loop = local, then
      // each processor loops over only its atoms to produce velocities.  The
      // random number seed is adjusted to give a different set of velocities
      // on each processor." — on one processor: group members in storage
      // order, after 100 discarded uniform draws. loop geom seeds every atom
      // from its coordinates ("For each atom a unique random number seed is
      // created, based on the atom's xyz coordinates"); the docs leave the
      // seeding unspecified and say it "will not necessarily assign identical
      // velocities for two simulations run on different machines", so the
      // engine's coordinate hash gives valid but different velocities.
      const draw = (rng: { uniform(): number; gaussian(): number }, i: number) => {
        const c = 1 / Math.sqrt(massOf(s, i));
        for (let d = 0; d < 3; d++) {
          const r = dist === 'gaussian' ? rng.gaussian() : rng.uniform() - 0.5;
          s.v[3 * i + d] = d === 2 && s.dimension === 2 ? 0 : r * c;
        }
      };
      if (loop === 'all') {
        let maxId = 0;
        for (let i = 0; i < s.n; i++) if (s.id[i] > maxId) maxId = s.id[i];
        if (maxId !== s.n) throw new StyleError('Atom IDs must be consecutive for velocity create loop all');
        const slot = new Int32Array(maxId + 1).fill(-1);
        for (const i of members) slot[s.id[i]] = i;
        const rng = new RanPark(seed);
        const sink = new Float64Array(3);
        for (let id = 1; id <= maxId; id++) {
          if (slot[id] >= 0) draw(rng, slot[id]);
          else for (let d = 0; d < 3; d++) sink[d] = dist === 'gaussian' ? rng.gaussian() : rng.uniform();
        }
      } else if (loop === 'local') {
        const rng = new RanPark(seed);
        for (let k = 0; k < 100; k++) rng.uniform();
        for (const i of members) draw(rng, i);
      } else {
        for (const i of members) draw(new Rng(geomSeed(seed, s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])), i);
      }
      if (mom) zeroMomentum(sys, members);
      if (rot) zeroRotation(sys, members);
      scaleTo(t);
      break;
    }
    case 'scale': scaleTo(num(args[0], 'temperature')); break;
    case 'set': {
      const sc = latticeScale(sys, unitsW, 'velocity');
      const comp = args.map((w, d) => {
        if (w === 'NULL') return null;
        if (w.startsWith('v_')) {
          const v = sys.vars.get(w.slice(2));
          if (v?.style === 'atom') { const arr = sys.atomVariable(w.slice(2)); return (i: number) => arr[i]; }
          const x = sys.equalVariable(w.slice(2));
          return () => x;
        }
        const x = num(w, 'velocity') * sc[d];
        return () => x;
      });
      for (const i of members) for (let d = 0; d < 3; d++) { const f = comp[d]; if (f) s.v[3 * i + d] = f(i); }
      break;
    }
    case 'ramp': {
      const vd = ['vx', 'vy', 'vz'].indexOf(args[0]);
      const cd = 'xyz'.indexOf(args[3]);
      if (vd < 0 || cd < 0 || args[3].length !== 1) throw new StyleError('usage: velocity group ramp vx|vy|vz vlo vhi x|y|z clo chi');
      const sc = latticeScale(sys, unitsW, 'velocity');
      const vlo = num(args[1], 'vlo') * sc[vd], vhi = num(args[2], 'vhi') * sc[vd];
      const clo = num(args[4], 'clo') * sc[cd], chi = num(args[5], 'chi') * sc[cd];
      for (const i of members) {
        let frac = (s.x[3 * i + cd] - clo) / (chi - clo);
        frac = Math.max(0, Math.min(1, frac));
        s.v[3 * i + vd] = vlo + frac * (vhi - vlo);
      }
      break;
    }
    case 'zero':
      if (args[0] === 'linear') zeroMomentum(sys, members);
      else if (args[0] === 'angular') zeroRotation(sys, members);
      else throw new StyleError('usage: velocity group zero linear|angular');
      break;
  }
  if (old) for (const i of members) for (let d = 0; d < 3; d++) s.v[3 * i + d] += old[3 * i + d];
  sys.refreshComputes();
};

export const zeroMomentum = (sys: System, members: number[]): void => {
  const s = sys.state;
  const p = [0, 0, 0];
  let mt = 0;
  for (const i of members) {
    const m = massOf(s, i);
    mt += m;
    for (let d = 0; d < 3; d++) p[d] += m * s.v[3 * i + d];
  }
  if (!(mt > 0)) return;
  for (const i of members) for (let d = 0; d < 3; d++) s.v[3 * i + d] -= p[d] / mt;
};

export const zeroRotation = (sys: System, members: number[]): void => {
  const s = sys.state;
  const g = sys.geom;
  const pos = members.map((i) => { const u = [0, 0, 0]; g.unwrap(s.x, s.image, i, u); return u; });
  let mt = 0;
  const c = [0, 0, 0];
  members.forEach((i, k) => { const m = massOf(s, i); mt += m; for (let d = 0; d < 3; d++) c[d] += m * pos[k][d]; });
  if (!(mt > 0)) return;
  for (let d = 0; d < 3; d++) c[d] /= mt;
  const L = [0, 0, 0];
  const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  members.forEach((i, k) => {
    const m = massOf(s, i);
    const r = [pos[k][0] - c[0], pos[k][1] - c[1], pos[k][2] - c[2]];
    const v = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
    L[0] += m * (r[1] * v[2] - r[2] * v[1]); L[1] += m * (r[2] * v[0] - r[0] * v[2]); L[2] += m * (r[0] * v[1] - r[1] * v[0]);
    const r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) I[p][q] += m * ((p === q ? r2 : 0) - r[p] * r[q]);
  });
  let w: number[];
  if (s.dimension === 2) w = [0, 0, I[2][2] > 0 ? L[2] / I[2][2] : 0];
  else {
    const [[a0, b0, c0], [d0, e0, f0], [g0, h0, k0]] = I;
    const det = a0 * (e0 * k0 - f0 * h0) - b0 * (d0 * k0 - f0 * g0) + c0 * (d0 * h0 - e0 * g0);
    if (Math.abs(det) < 1e-300) return;
    const inv = [
      [(e0 * k0 - f0 * h0) / det, (c0 * h0 - b0 * k0) / det, (b0 * f0 - c0 * e0) / det],
      [(f0 * g0 - d0 * k0) / det, (a0 * k0 - c0 * g0) / det, (c0 * d0 - a0 * f0) / det],
      [(d0 * h0 - e0 * g0) / det, (b0 * g0 - a0 * h0) / det, (a0 * e0 - b0 * d0) / det],
    ];
    w = [0, 1, 2].map((p) => inv[p][0] * L[0] + inv[p][1] * L[1] + inv[p][2] * L[2]);
  }
  members.forEach((i, k) => {
    const r = [pos[k][0] - c[0], pos[k][1] - c[1], pos[k][2] - c[2]];
    s.v[3 * i] -= w[1] * r[2] - w[2] * r[1];
    s.v[3 * i + 1] -= w[2] * r[0] - w[0] * r[2];
    s.v[3 * i + 2] -= w[0] * r[1] - w[1] * r[0];
  });
};

// ----------------------------------------------------------------- atom edits

/** delete_atoms style args [compress/bond/mol] — delete_atoms.html. */
const deleteAtomsCmd: Handler = ({ sys }, a) => {
  const s = sys.state;
  const del = new Uint8Array(s.n);
  let rest: string[];
  switch (a[0]) {
    case 'group': {
      const bit = sys.groupBit(a[1] ?? '');
      for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) del[i] = 1;
      rest = a.slice(2);
      break;
    }
    case 'region': {
      const r = sys.region(a[1] ?? '');
      for (let i = 0; i < s.n; i++) if (r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) del[i] = 1;
      rest = a.slice(2);
      break;
    }
    case 'variable': {
      const v = sys.atomVariable(a[1] ?? '');
      for (let i = 0; i < s.n; i++) if (v[i] !== 0) del[i] = 1;
      rest = a.slice(2);
      break;
    }
    case 'overlap': {
      // "delete one atom from pairs of atoms within the cutoff"
      const cut = num(a[1], 'cutoff');
      const b1 = sys.groupBit(a[2] ?? ''), b2 = sys.groupBit(a[3] ?? '');
      const g = sys.geom;
      const d = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (del[i]) continue;
        for (let j = 0; j < s.n; j++) {
          if (j === i || del[j]) continue;
          const pair = ((s.mask[i] & b1) && (s.mask[j] & b2)) || ((s.mask[i] & b2) && (s.mask[j] & b1));
          if (!pair) continue;
          d[0] = s.x[3 * i] - s.x[3 * j]; d[1] = s.x[3 * i + 1] - s.x[3 * j + 1]; d[2] = s.x[3 * i + 2] - s.x[3 * j + 2];
          g.minimumImage(d);
          if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < cut * cut) {
            // delete the atom with the larger ID
            if (s.id[i] > s.id[j]) { del[i] = 1; break; }
            del[j] = 1;
          }
        }
      }
      rest = a.slice(4);
      break;
    }
    case 'random': {
      const [, ranstyle, value, eflag, gid, rid, seed] = a;
      const bit = sys.groupBit(gid ?? '');
      const r = rid && rid !== 'NULL' ? sys.region(rid) : null;
      const elig: number[] = [];
      for (let i = 0; i < s.n; i++) if ((s.mask[i] & bit) && (!r || r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]))) elig.push(i);
      const rng = new Rng(int(seed, 'seed'));
      if (ranstyle === 'fraction') {
        const f = num(value, 'fraction');
        if (yesno(eflag, 'eflag')) {
          const want = Math.round(f * elig.length);
          for (let i = elig.length - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [elig[i], elig[j]] = [elig[j], elig[i]]; }
          for (const i of elig.slice(0, want)) del[i] = 1;
        } else for (const i of elig) if (rng.uniform() < f) del[i] = 1;
      } else if (ranstyle === 'count') {
        let want = int(value, 'count');
        if (want > elig.length) {
          if (yesno(eflag, 'eflag')) throw new StyleError(`delete_atoms random count ${want} is more than the ${elig.length} eligible atoms`);
          sys.warn('delete_atoms random count exceeds eligible atoms; deleting all of them');
          want = elig.length;
        }
        for (let i = elig.length - 1; i > 0; i--) { const j = Math.floor(rng.uniform() * (i + 1)); [elig[i], elig[j]] = [elig[j], elig[i]]; }
        for (const i of elig.slice(0, want)) del[i] = 1;
      } else throw new StyleError('delete_atoms random needs fraction or count');
      rest = a.slice(7);
      break;
    }
    case 'porosity': throw new StyleError('delete_atoms porosity was replaced by delete_atoms random in LAMMPS');
    default: throw new StyleError(`unknown delete_atoms style '${a[0] ?? ''}'`);
  }
  const kw = keywords(rest, { compress: 1, condense: 1, bond: 1, mol: 1 }, 'delete_atoms');
  if (kw.has('mol') && yesno(kw.get('mol')![0], 'mol')) {
    // "mol yes: delete entire molecules if any atom of the molecule is deleted"
    const mols = new Set<number>();
    for (let i = 0; i < s.n; i++) if (del[i] && s.molecule[i]) mols.add(s.molecule[i]);
    for (let i = 0; i < s.n; i++) if (mols.has(s.molecule[i])) del[i] = 1;
  }
  const n = sys.deleteAtoms(del);
  const compress = kw.has('compress') ? yesno(kw.get('compress')![0], 'compress') : true;
  if (compress && n > 0 && s.topo.bonds.n === 0 && s.topo.angles.n === 0) {
    // "compress yes ... atom IDs are re-assigned so that they run from 1 to N"
    const order = Array.from({ length: s.n }, (_, i) => i).sort((p, q) => s.id[p] - s.id[q]);
    order.forEach((i, k) => { s.id[i] = k + 1; });
    sys.atomsChanged();
  }
  sys.log(`Deleted ${n} atoms, new total = ${s.n}`);
};

/** displace_atoms group style args [units] — displace_atoms.html. */
const displaceAtoms: Handler = ({ sys }, a) => {
  const s = sys.state;
  const bit = sys.groupBit(a[0] ?? '');
  const style = a[1];
  const nargs: Record<string, number> = { move: 3, ramp: 6, random: 4, rotate: 7 };
  if (!style || !(style in nargs)) throw new StyleError('usage: displace_atoms group move|ramp|random|rotate args');
  const args = a.slice(2, 2 + nargs[style]);
  const kw = keywords(a.slice(2 + nargs[style]), { units: 1 }, 'displace_atoms');
  const sc = latticeScale(sys, kw.get('units')?.[0] ?? 'lattice', 'displace_atoms');
  const members: number[] = [];
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) members.push(i);
  switch (style) {
    case 'move': {
      // native LAMMPS evaluates the components in turn, each after the previous one moved the
      // atoms (atom-style dx=1, dy=x, dz=y on an atom at (1,1,1) gives (2,3,4))
      for (let d = 0; d < 3; d++) {
        const w = args[d];
        let comp: (i: number) => number;
        if (w.startsWith('v_')) {
          const v = sys.vars.get(w.slice(2));
          if (v?.style === 'atom') { const arr = sys.atomVariable(w.slice(2)); comp = (i) => arr[i] * sc[d]; } else {
            const x = sys.equalVariable(w.slice(2)) * sc[d];
            comp = () => x;
          }
        } else {
          const x = num(w, 'displacement') * sc[d];
          comp = () => x;
        }
        for (const i of members) s.x[3 * i + d] += comp(i);
      }
      break;
    }
    case 'ramp': {
      const dd = 'xyz'.indexOf(args[0]), cd = 'xyz'.indexOf(args[3]);
      if (dd < 0 || cd < 0) throw new StyleError('usage: displace_atoms group ramp x|y|z dlo dhi x|y|z clo chi');
      const dlo = num(args[1], 'dlo') * sc[dd], dhi = num(args[2], 'dhi') * sc[dd];
      const clo = num(args[4], 'clo') * sc[cd], chi = num(args[5], 'chi') * sc[cd];
      for (const i of members) {
        const f = Math.max(0, Math.min(1, (s.x[3 * i + cd] - clo) / (chi - clo)));
        s.x[3 * i + dd] += dlo + f * (dhi - dlo);
      }
      break;
    }
    case 'random': {
      const dx = [0, 1, 2].map((d) => num(args[d], 'magnitude') * sc[d]);
      const rng = new Rng(int(args[3], 'seed'));
      for (const i of members) for (let d = 0; d < 3; d++) s.x[3 * i + d] += dx[d] * 2 * (rng.uniform() - 0.5);
      break;
    }
    case 'rotate': {
      const P = [0, 1, 2].map((d) => num(args[d], 'P') * sc[d]);
      let R = [num(args[3], 'Rx'), num(args[4], 'Ry'), num(args[5], 'Rz')];
      const len = Math.hypot(R[0], R[1], R[2]);
      if (!(len > 0)) throw new StyleError('displace_atoms rotate: zero axis');
      R = R.map((x) => x / len);
      const th = (num(args[6], 'theta') * Math.PI) / 180;
      const c = Math.cos(th), sn = Math.sin(th);
      const g = sys.geom;
      for (const i of members) {
        // rotate the unwrapped position, then let pbc remap it
        const u = [0, 0, 0];
        g.unwrap(s.x, s.image, i, u);
        const v = [u[0] - P[0], u[1] - P[1], u[2] - P[2]];
        const dot = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
        const cr = [R[1] * v[2] - R[2] * v[1], R[2] * v[0] - R[0] * v[2], R[0] * v[1] - R[1] * v[0]];
        for (let d = 0; d < 3; d++) s.x[3 * i + d] = P[d] + v[d] * c + cr[d] * sn + R[d] * dot * (1 - c);
        s.image[3 * i] = s.image[3 * i + 1] = s.image[3 * i + 2] = 0;
      }
      break;
    }
  }
  if (s.dimension === 2) for (const i of members) s.x[3 * i + 2] = 0;
  for (const i of members) sys.geom.remap(s.x, s.image, i);
  sys.bump();
  sys.nb.lastBuild = -1;
};

/**
 * replicate nx ny nz — replicate.html: "Replicate the current simulation
 * one or more times in each dimension." New atoms get IDs offset by the
 * old maximum ID per copy; bonds are replicated with the copies' IDs,
 * using image flags to keep periodic bonds intact.
 */
const replicate: Handler = ({ sys }, a) => {
  const s = sys.state;
  const rep = [int(a[0], 'nx'), int(a[1], 'ny'), int(a[2], 'nz')];
  if (rep.some((r) => r < 1)) throw new StyleError('replicate factors must be >= 1');
  if (s.dimension === 2 && rep[2] !== 1) throw new StyleError('cannot replicate in z for a 2d simulation');
  for (const w of a.slice(3)) if (w !== 'bbox' && w !== 'bond/periodic') throw new StyleError(`unknown replicate keyword '${w}'`);
  const g = sys.geom;
  const old: SimState = { ...s, x: s.x.slice(0, 3 * s.n), v: s.v.slice(0, 3 * s.n), image: s.image.slice(0, 3 * s.n), type: s.type.slice(0, s.n), id: s.id.slice(0, s.n), mask: s.mask.slice(0, s.n), molecule: s.molecule.slice(0, s.n), q: s.q.slice(0, s.n) };
  // measured with native LAMMPS: "Cannot replicate with fixes that store per-atom quantities"
  for (const f of sys.fixes) if (f.style === 'property/atom') throw new StyleError('Cannot replicate with fixes that store per-atom quantities');
  // every other per-atom field (sphere radius/mass/omega) is copied as is
  const base = gatherAtoms(s, Array.from({ length: s.n }, (_, i) => i));
  const maxId = maxAtomId(s);
  let maxMol = 0;
  for (let i = 0; i < s.n; i++) maxMol = Math.max(maxMol, s.molecule[i]);
  const ncopy = rep[0] * rep[1] * rep[2];
  const n0 = s.n;
  const x = new Float64Array(3 * n0 * ncopy), v = new Float64Array(3 * n0 * ncopy);
  const type = new Int32Array(n0 * ncopy), id = new Int32Array(n0 * ncopy), mask = new Int32Array(n0 * ncopy);
  const mol = new Int32Array(n0 * ncopy), q = new Float64Array(n0 * ncopy), image = new Int32Array(3 * n0 * ncopy);
  let c = 0;
  const u = [0, 0, 0];
  for (let kz = 0; kz < rep[2]; kz++) for (let ky = 0; ky < rep[1]; ky++) for (let kx = 0; kx < rep[0]; kx++) {
    const copy = (kz * rep[1] + ky) * rep[0] + kx;
    for (let i = 0; i < n0; i++, c++) {
      g.unwrap(old.x, old.image, i, u);
      x[3 * c] = u[0] + kx * g.lx + ky * g.xy + kz * g.xz;
      x[3 * c + 1] = u[1] + ky * g.ly + kz * g.yz;
      x[3 * c + 2] = u[2] + kz * g.lz;
      for (let d = 0; d < 3; d++) v[3 * c + d] = old.v[3 * i + d];
      type[c] = old.type[i]; mask[c] = old.mask[i]; q[c] = old.q[i];
      id[c] = old.id[i] + copy * maxId;
      mol[c] = old.molecule[i] ? old.molecule[i] + copy * maxMol : 0;
    }
  }
  // new box
  const b = s.box;
  for (let d = 0; d < 3; d++) b.hi[d] = b.lo[d] + (b.hi[d] - b.lo[d]) * rep[d];
  b.tilt = [b.tilt[0] * rep[1], b.tilt[1] * rep[2], b.tilt[2] * rep[2]];
  b.minLo = [...b.lo]; b.minHi = [...b.hi];
  // topology copies: partner atoms belong to the copy whose image keeps the bond short
  const topo = s.topo;
  const lists = [topo.bonds, topo.angles, topo.dihedrals, topo.impropers].map((l) => ({ ...l, atoms: l.atoms.slice(0, l.n * l.width), type: l.type.slice(0, l.n) }));
  for (const l of [topo.bonds, topo.angles, topo.dihedrals, topo.impropers]) l.n = 0;
  s.n = 0;
  s.x = new Float64Array(0); s.v = new Float64Array(0); s.f = new Float64Array(0); s.image = new Int32Array(0);
  s.type = new Int32Array(0); s.id = new Int32Array(0); s.mask = new Int32Array(0); s.molecule = new Int32Array(0); s.q = new Float64Array(0);
  if (s.rmass) s.rmass = new Float64Array(0);
  if (s.radius) s.radius = new Float64Array(0);
  if (s.omega) s.omega = new Float64Array(0);
  if (s.torque) s.torque = new Float64Array(0);
  for (const cp of s.custom.values()) cp.data = new Float64Array(0);
  const tile = (a: Float64Array | undefined): Float64Array | undefined => {
    if (!a) return undefined;
    const out = new Float64Array(a.length * ncopy);
    for (let k = 0; k < ncopy; k++) out.set(a, k * a.length);
    return out;
  };
  const custom = new Map<string, Float64Array>();
  for (const [name, arr] of base.custom!) custom.set(name, tile(arr)!);
  appendAtoms(s, {
    x, v, type, id, mask: 0, molecule: mol, q, image,
    rmass: tile(base.rmass as Float64Array | undefined), radius: tile(base.radius as Float64Array | undefined), omega: tile(base.omega), custom,
  });
  s.mask.set(mask);
  sys.setState(s);
  const gnew = sys.geom;
  // wrap new atoms into the new box
  for (let i = 0; i < s.n; i++) gnew.remap(s.x, s.image, i);
  const oldIdx = new Map<number, number>();
  for (let i = 0; i < n0; i++) oldIdx.set(old.id[i], i);
  const ux = (i: number) => { const p = [0, 0, 0]; g.unwrap(old.x, old.image, i, p); return p; };
  [topo.bonds, topo.angles, topo.dihedrals, topo.impropers].forEach((l, li) => {
    const src = lists[li];
    for (let kz = 0; kz < rep[2]; kz++) for (let ky = 0; ky < rep[1]; ky++) for (let kx = 0; kx < rep[0]; kx++) {
      for (let e = 0; e < src.n; e++) {
        const ids = Array.from(src.atoms.subarray(e * src.width, (e + 1) * src.width));
        const p0 = ux(oldIdx.get(ids[0])!);
        const out = ids.map((aid, w) => {
          // which copy holds the partner nearest to atom 0 of this copy
          const p = ux(oldIdx.get(aid)!);
          const off = [kx, ky, kz];
          if (w > 0) {
            const dd = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]];
            const sh = [Math.round(dd[0] / g.lx), Math.round(dd[1] / g.ly), Math.round(dd[2] / g.lz)];
            for (let d = 0; d < 3; d++) off[d] = (((off[d] + sh[d]) % rep[d]) + rep[d]) % rep[d];
            void sh;
          }
          return aid + ((off[2] * rep[1] + off[1]) * rep[0] + off[0]) * maxId;
        });
        pushTopo(l, src.type[e], out);
      }
    }
  });
  sys.atomsChanged();
  sys.log(`Replicated to ${s.n} atoms`);
};

/** change_box group-ID parameter args ... [units] — change_box.html (a subset: x/y/z final/delta/scale, xy/xz/yz, boundary, ortho, triclinic, remap). */
const changeBox: Handler = ({ sys }, a) => {
  const s = sys.state;
  const bit = sys.groupBit(a[0] ?? '');
  const ops = a.slice(1);
  // units keyword applies to all
  let unitsW = 'lattice';
  const ui = ops.indexOf('units');
  if (ui >= 0) { unitsW = ops[ui + 1]; ops.splice(ui, 2); }
  const sc = latticeScale(sys, unitsW, 'change_box');
  const b = s.box;
  let saved = cloneBox(b);
  const geomOf = (box: SimState['box']) => new Geometry(box);
  for (let k = 0; k < ops.length;) {
    const p = ops[k];
    if (p === 'x' || p === 'y' || p === 'z') {
      const d = 'xyz'.indexOf(p);
      const st = ops[k + 1];
      if (st === 'final') { b.lo[d] = num(ops[k + 2], 'lo') * sc[d]; b.hi[d] = num(ops[k + 3], 'hi') * sc[d]; k += 4; } else if (st === 'delta') { b.lo[d] += num(ops[k + 2], 'dlo') * sc[d]; b.hi[d] += num(ops[k + 3], 'dhi') * sc[d]; k += 4; } else if (st === 'scale') {
        const f = num(ops[k + 2], 'factor');
        const mid = 0.5 * (b.lo[d] + b.hi[d]), half = 0.5 * (b.hi[d] - b.lo[d]) * f;
        b.lo[d] = mid - half; b.hi[d] = mid + half;
        k += 3;
      } else if (st === 'volume') throw new StyleError('change_box volume is not supported');
      else throw new StyleError(`change_box ${p}: style must be final, delta or scale`);
    } else if (p === 'xy' || p === 'xz' || p === 'yz') {
      if (!b.triclinic) throw new StyleError(`change_box ${p} needs a triclinic box (use change_box all triclinic first)`);
      const t = ['xy', 'xz', 'yz'].indexOf(p);
      const scale = t === 2 ? sc[1] : sc[0];
      if (ops[k + 1] === 'final') b.tilt[t] = num(ops[k + 2], 'tilt') * scale;
      else if (ops[k + 1] === 'delta') b.tilt[t] += num(ops[k + 2], 'dtilt') * scale;
      else throw new StyleError(`change_box ${p}: style must be final or delta`);
      k += 3;
    } else if (p === 'boundary') {
      const nb = ops.slice(k + 1, k + 4).map((w) => {
        const r = parseBoundary(w);
        if (!r) throw new StyleError(`invalid boundary '${w}'`);
        return r;
      });
      b.boundary = nb as SimState['box']['boundary'];
      b.periodic = [nb[0][0] === 'p', nb[1][0] === 'p', nb[2][0] === 'p'];
      sys.boundary = b.boundary;
      // "all non-zero image flags for non-periodic dimensions will be reset" (as read_data)
      for (let i = 0; i < s.n; i++) for (let d = 0; d < 3; d++) if (!b.periodic[d]) s.image[3 * i + d] = 0;
      k += 4;
    } else if (p === 'ortho') {
      if (b.tilt.some((t) => t !== 0)) throw new StyleError('change_box ortho needs all tilt factors to be 0');
      b.triclinic = false;
      k++;
    } else if (p === 'triclinic') { b.triclinic = true; k++; } else if (p === 'set') { saved = cloneBox(b); k++; } else if (p === 'remap') {
      // map group atoms from the saved box to the current one, in fractional coordinates
      const g0 = geomOf(saved), g1 = geomOf(b);
      const lam = [0, 0, 0], out = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & bit)) continue;
        g0.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], lam);
        g1.fromLamda(lam[0], lam[1], lam[2], out);
        s.x[3 * i] = out[0]; s.x[3 * i + 1] = out[1]; s.x[3 * i + 2] = out[2];
      }
      saved = cloneBox(b);
      k++;
    } else throw new StyleError(`unknown change_box parameter '${p}'`);
  }
  b.minLo = [...b.lo]; b.minHi = [...b.hi];
  sys.geom.update();
  for (const f of sys.fixes) f.boxChanged?.();
  sys.nb.lastBuild = -1;
  sys.bump();
};

/** create_bonds single/bond|angle|dihedral|improper type atoms... — create_bonds.html (single styles). */
const createBonds: Handler = ({ sys }, a) => {
  const s = sys.state;
  const width: Record<string, number> = { 'single/bond': 2, 'single/angle': 3, 'single/dihedral': 4, 'single/improper': 4 };
  const style = a[0];
  if (!(style in width)) throw new StyleError('create_bonds supports single/bond, single/angle, single/dihedral and single/improper');
  const kind = style.split('/')[1] as 'bond' | 'angle' | 'dihedral' | 'improper';
  const t = int(a[1], `${kind} type`);
  const ntypes = { bond: s.topo.nbondtypes, angle: s.topo.nangletypes, dihedral: s.topo.ndihedraltypes, improper: s.topo.nimpropertypes }[kind];
  if (t < 1 || t > ntypes) throw new StyleError(`${kind} type ${t} is outside 1..${ntypes}`);
  const ids = a.slice(2, 2 + width[style]).map((w) => int(w, 'atom ID'));
  if (ids.length < width[style]) throw new StyleError(`create_bonds ${style} needs ${width[style]} atom IDs`);
  for (const id of ids) if (sys.indexOfId(id) < 0) throw new StyleError(`create_bonds: atom ${id} does not exist`);
  const kw = keywords(a.slice(2 + width[style]), { special: 1 }, 'create_bonds');
  void kw;
  pushTopo(s.topo[`${kind}s` as 'bonds'], t, ids);
  sys.atomsChanged();
};

/** delete_bonds group style ... (a subset: multi / bond|angle|... type with remove). */
const deleteBonds: Handler = ({ sys }, a) => {
  const s = sys.state;
  const bit = sys.groupBit(a[0] ?? '');
  const style = a[1];
  const remove = a.includes('remove');
  if (!remove) throw new StyleError("delete_bonds without 'remove' turns interactions off by negating types; only 'remove' is supported");
  const inGroup = new Set<number>();
  for (let i = 0; i < s.n; i++) if (s.mask[i] & bit) inGroup.add(s.id[i]);
  const lists: Record<string, [keyof SimState['topo'], number | null]> = {
    multi: ['bonds', null], bond: ['bonds', null], angle: ['angles', null], dihedral: ['dihedrals', null], improper: ['impropers', null], atom: ['bonds', null],
  };
  if (!(style in lists)) throw new StyleError(`delete_bonds style '${style}' is not supported`);
  const type = ['bond', 'angle', 'dihedral', 'improper'].includes(style) ? int(a[2], 'type') : null;
  const targets = style === 'multi' ? (['bonds', 'angles', 'dihedrals', 'impropers'] as const) : ([lists[style][0]] as const);
  for (const key of targets) {
    const l = s.topo[key as 'bonds'];
    let k = 0;
    for (let e = 0; e < l.n; e++) {
      let all = true;
      for (let w = 0; w < l.width; w++) if (!inGroup.has(l.atoms[e * l.width + w])) { all = false; break; }
      if (all && (type === null || l.type[e] === type)) continue;
      l.type[k] = l.type[e];
      for (let w = 0; w < l.width; w++) l.atoms[k * l.width + w] = l.atoms[e * l.width + w];
      k++;
    }
    l.n = k;
  }
  sys.atomsChanged();
};

export const SETUP_COMMANDS: Record<string, Handler> = {
  units, dimension, boundary, atom_style: atomStyle, atom_modify: atomModify, newton,
  processors: parallelOnly('processors'), comm_style: commStyle, package: parallelOnly('package'),
  suffix: parallelOnly('suffix'), partition: parallelOnly('partition'), balance: parallelOnly('balance'),
  comm_modify: commModify, lattice, region, create_box: createBox, create_atoms: createAtoms, mass, molecule,
  read_data: readDataCmd, write_data: writeDataCmd, timestep, reset_timestep: resetTimestep,
  group, set, velocity, delete_atoms: deleteAtomsCmd, displace_atoms: displaceAtoms, replicate,
  change_box: changeBox, create_bonds: createBonds, delete_bonds: deleteBonds,
};

void ALL_GROUP_BIT;

/** velocity loop geom: a seed from the user seed and the coordinate bits (engine-defined hash). */
const geomSeed = (seed: number, x: number, y: number, z: number): number => {
  const b = new Uint32Array(new Float64Array([x, y, z]).buffer);
  let h = seed | 0;
  for (let k = 0; k < b.length; k++) h = Math.imul(h ^ b[k], 0x9e3779b1) ^ (h >>> 15);
  return h;
};

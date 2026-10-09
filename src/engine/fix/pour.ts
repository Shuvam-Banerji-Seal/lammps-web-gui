import { Fix } from './fix';
import { StyleError } from '../force/types';
import { RanPark } from '../rng';
import { appendAtoms, maxAtomId, pushTopo, sphereMass, isSphereStyle, hasChargeStyle, SPHERE_DEFAULT_RADIUS } from '../atoms';
import { BlockRegion, ConeRegion, type Param, type Region } from '../region';
import { geometricCenter, rotationMatrix, type MoleculeTemplate } from '../molecule';
import type { System } from '../system';

/*
 * fix ID group-ID pour N type seed keyword values ... — docs.lammps.org/fix_pour.html
 * (plans/lammps-docs/fix_pour.rst). Supported: region (block, or a z-axis
 * cylinder, side in, 3d only), diam one / range / poly, id max / next, vol,
 * rate, dens, vel, ignore, mol (single-molecule templates) with molfrac, rigid
 * and shake. Everything else throws a StyleError.
 *
 * "Insert finite-size particles or molecules into the simulation box
 * every few timesteps within a specified region until N particles or
 * molecules have been inserted." "For the remainder of this doc page, a
 * single inserted atom or molecule is referred to as a "particle"."
 *
 * Random numbers (RanPark, rng.ts). Measured with native LAMMPS (black box,
 * seeds 1, 2, 777, 12345 and 12346 for the first insertion; the oracle cases
 * also use 999 and 4242): the stream is RanPark seeded with the given seed
 * after 30 discarded draws (the first draw used is the 31st of the
 * Park-Miller sequence, index 30). Per
 * insertion event each particle draws, in this order:
 *   1. fall:   the coordinate along the fall axis (z in 3d, y in 2d),
 *              f = flo + (fhi - flo) * (1 - u^2), u one draw (drawn ONCE per
 *              particle, before the attempt loop; overlap retries keep it);
 *   2. diam:   one draw for range (D = Dlo + u (Dhi - Dlo)) or poly (the
 *              diameter whose cumulative percentage first exceeds u); none for one;
 *              atoms only (no draw for molecules);
 *   3. x [, y]: x = xlo + u (xhi - xlo) and, in 3d only, y = ylo + u (yhi - ylo);
 *              repeated on every attempt (a cylinder rejects points outside the
 *              circle, and an overlapping candidate is rejected; both use one attempt);
 *   4. mol:    molecules only, per attempt after the position: one draw samples
 *              the molecule (molfrac), then the random rotation (5 below);
 *   5. vx[, vy]: vxlo + u (vxhi - vxlo) and, in 3d only, vy = vylo + u (vyhi - vylo),
 *              drawn after acceptance;
 *   6. dens:   rholo + u (rhohi - rholo), drawn after acceptance (mass = rho V),
 *              atoms only.
 * Measured: the fall coordinate of an inserted particle is independent of the
 * diam draw, and changing vel or dens ranges does not move the positions.
 *
 * 2d. Measured with native LAMMPS (black box, dimension 2, atom_style sphere,
 * region 2..18 x 10..15, seed 4767548, diam range 0.5 1.0, vel 1 2 -3, dens 2 3):
 * the fall axis is y; the draw order per atom is y (fall), diam, x, vx, dens
 * (no y position draw and no vy range draw); the fall velocity is
 * vy = -sqrt(vy_set^2 + 2 g (yhi - y)) where vy_set is the single vel value
 * ("vel values (2d) = vxlo vxhi vy" and "The vz or vy value for option vel
 * assigns a z-velocity (3d) or y-velocity (2d) to each inserted particle.");
 * the density still converts to a mass through the 3d sphere volume
 * 4/3 pi r^3 (measured mass 0.9362175310843 for r = 0.434926593646,
 * rho = 2.716691159511). "The rate option moves the insertion volume in the z
 * direction (3d) or y direction (2d)." and "For 2d simulations, gravity must be
 * defined in the -y direction." "The cylinder style of region can only be used
 * with 3d simulations."
 *
 * Molecules. Measured with native LAMMPS (black box, seed 12345, a molecule
 * template with a Coords/Types section, region 5..15 x 5..15 x 15..18): per
 * molecule the draw order is fall, x, y (3d), then per attempt the molecule
 * selection draw (one even for a single-molecule template), then the rotation:
 * 3d uses two pi u4 about the axis (u1 - 0.5, u2 - 0.5, u3 - 0.5) from four
 * further draws (same form as fix deposit mol), 2d uses two pi u about the
 * z-axis from one draw (same form as create_atoms mol). After acceptance the
 * velocities are drawn: vx and, in 3d only, vy (the 2d vel keyword has no vy
 * range); there is no dens draw, and every atom of the molecule gets the same
 * velocity vector with the fall component -sqrt(vel^2 + 2 g (fhi - f)) on the
 * fall axis. The molecule's geometric center is placed at the drawn point.
 * "Note that for molecule insertion, the diameters of individual atoms in the
 * molecule can be specified in the file read by the molecule command. If not
 * specified, the diameter of each atom in the molecule has a default diameter
 * of 1.0." The diam keyword is ignored for molecules.
 *
 * Insertion count and timing. "Next, the target number of particles
 * inserted per event (assuming no failed insertions due to overlaps) is
 * calculated as the product of the volume fraction and the volume of the
 * insertion region divided by the volume of a particle (or area in 2D)"
 * (rst lines 166 to 170). Measured: the count is floor(vol V / Vp) (vol 0.9,
 * V = 1, one diam 1: 1 particle; vol 0.3: the error Fix pour insertion count
 * per timestep is 0); for diam range Vp uses Dhi (a range 0.5 to 1.0 gives 30
 * where 'one 1.0' gives 30, not the 73 of the mean); for poly Vp is the
 * percentage-weighted mean volume (poly 3 0.5 0.3 0.8 0.3 1.2 0.4 gives 35).
 * In 2d Vp is the area pi/4 D^2 for one/range and the weighted mean of the
 * atom areas for poly (measured: region 16 x 5, vol 0.2, diam one 2.0 gives 15;
 * the vendored in.pour.2d region 98 x 4.5, vol 0.4, diam range 0.5 1.0 gives
 * 224, and diam poly 2 0.5 0.5 1.0 0.5 gives 359). For molecules Vp is the
 * sphere (3d) / disc (2d) volume of R = max over template atoms of
 * (|x_i - center| + radius_i) (measured: radii 0.5 and 1.0, region 300, vol 0.2
 * gives 4; radii 0.2, 0.2 gives 20; molecule.vshape in 2d gives 26). Measured:
 * the first event is at the first step of the run (step 1), later events are
 * spaced by the rounded fall time: with gravity g, region height H, rate V and
 * fall velocity v, t = (-w + sqrt(w^2 + 2 g H)) / g with w = rate - v; g = 1,
 * H = 1 gives 1414 steps at dt 0.001 (t = 1.41421), 1664 at dt 0.00085, 1115 for
 * g = 2, H = 1, dt = 0.0007, v = -0.5 (round, not floor). Measured with rate
 * 0.5: the event at step 1177 (dt 0.00085) sees the region bottom at
 * zlo + rate (step - 1) dt.
 *
 * Attempts. "LAMMPS will make up to a total of M tries to insert the new
 * particles without overlaps, where M = # of inserted particles \* Nattempt."
 * Measured: M = nnew * Nattempt where nnew = min(count, remaining N); the tries
 * are shared by the whole event (every position pair counts, successful or not,
 * and for molecules the selection and rotation draws too). A particle that
 * exhausts the budget is not inserted and the event ends with the warning Fewer
 * insertions than requested. Overlap: centres closer than the sum of the two
 * radii are overlapping; a molecule is tested atom by atom. Periodic images are
 * included as the doc requires ("including effects due to periodic boundary
 * conditions if applicable"); not measured.
 *
 * The insertion happens in postIntegrate (after the initial half kick and
 * before the neighbour decision), so atomsChanged() forces a neighbour
 * rebuild in the same step. preExchange is defined only so that the
 * accelerated backends (which cannot add atoms) are not chosen.
 */

/** Strict numeric argument. */
export const num = (w: string | undefined, what: string): number => {
  if (w === undefined || w.trim() === '' || !Number.isFinite(Number(w))) throw new StyleError(`fix: expected a number for ${what}, got '${w ?? ''}'`);
  return Number(w);
};

/** Strict positive integer argument. */
export const posInt = (w: string | undefined, what: string): number => {
  const v = num(w, what);
  if (!Number.isInteger(v) || v <= 0) throw new StyleError(`fix: ${what} must be a positive integer, got '${w}'`);
  return v;
};

/**
 * RanPark seeded with the seed and advanced past the 30 draws native LAMMPS
 * consumes before the first insertion (measured, see the header of pour.ts).
 */
export const insertionStream = (seed: number): RanPark => {
  if (!Number.isInteger(seed) || seed <= 0) throw new StyleError('fix: seed must be a positive integer');
  const rng = new RanPark(seed);
  for (let k = 0; k < 30; k++) rng.uniform();
  return rng;
};

/** Parameter value (constant or equal-style variable). */
export const paramValue = (sys: System, p: Param): number =>
  typeof p === 'number' ? p : sys.regionEnv.variable(p.variable) * p.scale;

/** Insertion volume of a block or a z-axis cylinder (side in), as used by pour and deposit. */
export interface InsertionRegion {
  kind: 'block' | 'cylinder';
  xlo: number; xhi: number; ylo: number; yhi: number; zlo: number; zhi: number;
  /** cylinder centre and radius. */
  xc: number; yc: number; radius: number;
}

export const insertionRegion = (sys: System, id: string, what: string): InsertionRegion => {
  const r: Region = sys.region(id);
  if (!r.interior) throw new StyleError(`fix ${what}: the region must be defined with side in`);
  if (r.dynamic) throw new StyleError(`fix ${what}: a dynamic region (move or rotate) is not supported`);
  if (r instanceof BlockRegion) {
    const b = r.b.map((p) => paramValue(sys, p));
    return { kind: 'block', xlo: b[0], xhi: b[1], ylo: b[2], yhi: b[3], zlo: b[4], zhi: b[5], xc: 0, yc: 0, radius: 0 };
  }
  if (r instanceof ConeRegion && r.style === 'cylinder') {
    if (r.axis !== 2) throw new StyleError(`fix ${what}: only a z-axis cylinder region is supported`);
    const xc = paramValue(sys, r.c1), yc = paramValue(sys, r.c2);
    const rad = paramValue(sys, r.radlo);
    if (rad !== paramValue(sys, r.radhi)) throw new StyleError(`fix ${what}: a cylinder region must have equal radii`);
    const zlo = paramValue(sys, r.lo), zhi = paramValue(sys, r.hi);
    return { kind: 'cylinder', xlo: xc - rad, xhi: xc + rad, ylo: yc - rad, yhi: yc + rad, zlo, zhi, xc, yc, radius: rad };
  }
  throw new StyleError(`fix ${what}: the region must be a block or a z-axis cylinder (region style '${r.style}' is not supported)`);
};

/** Minimum-image separation vector components for the periodic dimensions. */
export const separation = (sys: System, dx: number, dy: number, dz: number): [number, number, number] => {
  const s = sys.state;
  const d = [dx, dy, dz];
  for (let k = 0; k < 3; k++) {
    if (!s.box.periodic[k]) continue;
    const L = s.box.hi[k] - s.box.lo[k];
    d[k] -= L * Math.round(d[k] / L);
  }
  return [d[0], d[1], d[2]];
};

/** The fall axis: z in 3d, y in 2d ("gravity must be defined in the -y direction" in 2d). */
export const fallAxis = (sys: System): 1 | 2 => (sys.dimension === 2 ? 1 : 2);

/** Gravitational acceleration magnitude along the fall axis from the defined fix gravity. */
export const gravityMagnitude = (sys: System, what: string): number => {
  const fx = sys.fixes.find((f) => f.style === 'gravity');
  const fb = fallAxis(sys);
  if (!fx) throw new StyleError(`fix ${what} requires a fix gravity in the -${'xyz'[fb]} direction`);
  const out = new Float64Array(3);
  (fx as unknown as { gvec(o: Float64Array): void }).gvec(out);
  const horiz = [0, 1, 2].filter((d) => d !== fb);
  if (horiz.some((d) => Math.abs(out[d]) > 1e-12) || !(out[fb] < 0)) {
    throw new StyleError(`fix ${what}: the gravity fix must point in the -${'xyz'[fb]} direction`);
  }
  return -out[fb];
};

/**
 * Appends one molecule of a template at the given absolute coordinates (flat
 * 3N, already rotated and translated), giving every atom one molecule ID, a
 * common velocity, the template's atom types offset by `toff`, its charges and
 * its bonds / angles / dihedrals / impropers with the new atom IDs. The
 * coordinates are wrapped into the periodic box with image flags, so the
 * molecule stays whole when unwrapped. `ids` (optional) is the explicit atom
 * ID list; otherwise the IDs continue from the current maximum. Returns the
 * first new atom ID. Used by fix deposit mol (fix_deposit.rst) and fix pour mol.
 *
 * Diameters / Masses: when the atom style stores radii (sphere) each atom gets
 * the template diameter / 2 ("If not listed, the default diameter of each atom
 * in the molecule is 1.0."), and when it stores per-atom masses the template
 * mass is used ("If this section is not included, the default mass for each
 * atom is derived from its volume (see Diameters section) and a default density
 * of 1.0"). Measured with native LAMMPS (black box, create_atoms single mol,
 * atom_style sphere): a Diameters 1.0 / 2.0 section gives radii 0.5 / 1.0 and a
 * Masses 0.5 / 1.5 section gives masses 0.5 / 1.5; without the sections the
 * defaults are radius 0.5 and mass 4/3 pi 0.5^3 = 0.5235987755982988.
 */
export const appendMolecule = (
  sys: System, t: MoleculeTemplate, toff: number, pos: Float64Array,
  vel: readonly number[], molId: number, gbit: number, ids?: Int32Array,
): number => {
  const s = sys.state;
  const n = t.natoms;
  const types = new Int32Array(n);
  for (let i = 0; i < n; i++) types[i] = t.type[i] + toff;
  for (const ty of types) if (ty < 1 || ty > s.ntypes) throw new StyleError(`molecule ${t.id}: atom type ${ty} is outside 1..${s.ntypes}`);
  if (t.q && !hasChargeStyle(s.atomStyle)) throw new StyleError(`molecule ${t.id} has charges, which atom_style ${s.atomStyle} cannot store`);
  const image = new Int32Array(3 * n);
  for (let i = 0; i < n; i++) sys.geom.remap(pos, image, i);
  const v = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) { v[3 * i] = vel[0]; v[3 * i + 1] = vel[1]; v[3 * i + 2] = vel[2]; }
  const molecule = new Int32Array(n).fill(molId);
  const q = t.q ? Float64Array.from(t.q) : undefined;
  let radius: Float64Array | undefined;
  let rmass: Float64Array | undefined;
  if (s.radius) {
    radius = new Float64Array(n);
    for (let i = 0; i < n; i++) radius[i] = t.diam ? t.diam[i] / 2 : SPHERE_DEFAULT_RADIUS;
  }
  if (s.rmass && (t.mass || radius)) {
    rmass = new Float64Array(n);
    for (let i = 0; i < n; i++) rmass[i] = t.mass ? t.mass[i] : sphereMass(radius![i], 1);
  }
  const base = ids && ids.length ? ids[0] - 1 : maxAtomId(s);
  appendAtoms(s, { x: pos, type: types, v, image, molecule, q, mask: gbit, id: ids, radius, rmass });
  for (const [what, list] of [['bonds', t.bonds], ['angles', t.angles], ['dihedrals', t.dihedrals], ['impropers', t.impropers]] as const) {
    if (!list.length) continue;
    for (const e of list) pushTopo(s.topo[what], e[0], e.slice(1).map((k) => base + k));
  }
  return base + 1;
};

/** Atom list checks shared by pour and deposit. */
export const insertionBox = (sys: System, reg: InsertionRegion, what: string): void => {
  const s = sys.state;
  const lo = s.box.lo, hi = s.box.hi;
  const bad = reg.xlo < lo[0] || reg.xhi > hi[0] || reg.ylo < lo[1] || reg.yhi > hi[1] || reg.zlo < lo[2] || reg.zhi > hi[2];
  if (bad) throw new StyleError(`fix ${what}: the insertion region extends outside the simulation box`);
};

/** Largest molecule ID currently in the system (0 when none). */
const maxMoleculeId = (s: { n: number; molecule: Int32Array }): number => {
  let m = 0;
  for (let i = 0; i < s.n; i++) if (s.molecule[i] > m) m = s.molecule[i];
  return m;
};

/** Number of molecules a template ID defines (summed over the files loaded under that ID). */
const moleculeCount = (sys: System, id: string): number => {
  const sets = sys.molecules.get(id);
  if (!sets || !sets.length) throw new StyleError(`fix pour mol: molecule template '${id}' does not exist`);
  let n = 0;
  for (const t of sets) n += t.mol ? new Set(t.mol).size : 1;
  return n;
};

/** Interface for the fix rigid/small method pour calls to add a new rigid body. */
interface RigidAdder { addMolecule(atomIds: readonly number[]): void; }
/** Interface for the fix shake method pour calls to rebuild its clusters. */
interface ClusterRebuilder { rebuildClusters(): void; }

type Diam = { style: 'one'; d: number } | { style: 'range'; lo: number; hi: number } | { style: 'poly'; d: number[]; p: number[] };

/** Keyword values of fix pour (docs.lammps.org/fix_pour.html). */
const parseDiam = (args: string[], i: number): [Diam, number] => {
  const st = args[i];
  if (st === 'one') return [{ style: 'one', d: num(args[i + 1], 'diam one D') }, i + 2];
  if (st === 'range') return [{ style: 'range', lo: num(args[i + 1], 'diam range Dlo'), hi: num(args[i + 2], 'diam range Dhi') }, i + 3];
  if (st === 'poly') {
    const n = posInt(args[i + 1], 'diam poly Npoly');
    const d: number[] = [], p: number[] = [];
    for (let k = 0; k < n; k++) {
      d.push(num(args[i + 2 + 2 * k], 'diam poly D'));
      p.push(num(args[i + 3 + 2 * k], 'diam poly P'));
    }
    const sum = p.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 1e-8) throw new StyleError('fix pour: the diam poly percentages must sum to 1');
    return [{ style: 'poly', d, p }, i + 2 + 2 * n];
  }
  throw new StyleError(`fix pour: diam style '${st ?? ''}' is not one, range or poly`);
};

export class FixPour extends Fix {
  readonly style = 'pour';
  readonly N: number;
  readonly type: number;
  private readonly rng: RanPark;
  private readonly regionId: string;
  private reg: InsertionRegion | null = null;
  private diam: Diam = { style: 'one', d: 1 };
  private rateV = 0;
  private vol = 0.25;
  private nattempt = 50;
  private dens: [number, number] = [1, 1];
  private vel: [number, number, number, number, number] = [0, 0, 0, 0, 0];
  private readonly idNext: boolean;
  private nextId: number;
  private nextMolId: number;
  private g = 0;
  private readonly twoD: boolean;
  private readonly molTemplateId: string | null;
  private molTemplate: MoleculeTemplate | null = null;
  private readonly molfrac: number[] | null;
  private readonly rigidId: string | null;
  private readonly shakeId: string | null;
  private rigidFix: Fix | null = null;
  private shakeFix: Fix | null = null;
  private readonly nfirst: number;
  private nextStep: number;
  inserted = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!isSphereStyle(sys.atomStyle)) throw new StyleError('fix pour requires atom_style sphere');
    if (args.length < 3) throw new StyleError('usage: fix ID group-ID pour N type seed keyword values ...');
    const twoD = sys.dimension === 2;
    this.twoD = twoD;
    this.N = posInt(args[0], 'N');
    this.rng = insertionStream(posInt(args[2], 'seed'));
    let region: string | null = null;
    let idNext = false;
    let molTemplateId: string | null = null;
    let molfrac: number[] | null = null;
    let rigidId: string | null = null;
    let shakeId: string | null = null;
    let i = 3;
    while (i < args.length) {
      const key = args[i];
      if (key === 'region') { region = args[i + 1] ?? null; i += 2; }
      else if (key === 'diam') { [this.diam, i] = parseDiam(args, i + 1); }
      else if (key === 'id') {
        const v = args[i + 1];
        if (v !== 'max' && v !== 'next') throw new StyleError(`fix pour: id must be max or next, got '${v ?? ''}'`);
        idNext = v === 'next';
        i += 2;
      } else if (key === 'vol') {
        this.vol = num(args[i + 1], 'vol fraction');
        this.nattempt = posInt(args[i + 2], 'vol Nattempt');
        if (!(this.vol > 0)) throw new StyleError('fix pour: vol fraction must be > 0');
        i += 3;
      } else if (key === 'rate') { this.rateV = num(args[i + 1], 'rate'); i += 2; }
      else if (key === 'dens') {
        this.dens = [num(args[i + 1], 'dens Rholo'), num(args[i + 2], 'dens Rhohi')];
        i += 3;
      } else if (key === 'vel') {
        // "vel values (3d) = vxlo vxhi vylo vyhi vz"; "vel values (2d) = vxlo vxhi vy"
        if (twoD) { this.vel = [num(args[i + 1], 'vel vxlo'), num(args[i + 2], 'vel vxhi'), 0, 0, num(args[i + 3], 'vel vy')]; i += 4; }
        else { this.vel = [num(args[i + 1], 'vel vxlo'), num(args[i + 2], 'vel vxhi'), num(args[i + 3], 'vel vylo'), num(args[i + 4], 'vel vyhi'), num(args[i + 5], 'vel vz')]; i += 6; }
      } else if (key === 'ignore') {
        // this engine has no line or triangle particles, so there is nothing to skip
        i += 1;
      } else if (key === 'mol') { molTemplateId = args[i + 1] ?? null; i += 2; }
      else if (key === 'molfrac') {
        if (!molTemplateId) throw new StyleError('fix pour: molfrac requires the mol keyword before it');
        const n = moleculeCount(sys, molTemplateId);
        const vals: number[] = [];
        for (let k = 1; k <= n; k++) vals.push(num(args[i + k], 'molfrac'));
        if (Math.abs(vals.reduce((a, b) => a + b, 0) - 1) > 1e-8) throw new StyleError('fix pour: molfrac values must sum to 1.0');
        molfrac = vals;
        i += 1 + n;
      } else if (key === 'rigid') { rigidId = args[i + 1] ?? null; i += 2; }
      else if (key === 'shake') { shakeId = args[i + 1] ?? null; i += 2; }
      else throw new StyleError(`fix pour: unknown keyword '${key}'`);
    }
    if (region === null) throw new StyleError('fix pour requires the region keyword');
    this.regionId = region;
    // "type = atom type to assign to inserted particles (offset for molecule insertion)"
    if (molTemplateId) {
      const tv = num(args[1], 'type');
      if (!Number.isInteger(tv) || tv < 0) throw new StyleError(`fix pour: type must be a non-negative integer offset with mol, got '${args[1]}'`);
      this.type = tv;
    } else {
      this.type = posInt(args[1], 'type');
    }
    this.molTemplateId = molTemplateId;
    this.molfrac = molfrac;
    this.rigidId = rigidId;
    this.shakeId = shakeId;
    if ((rigidId || shakeId) && !molTemplateId) throw new StyleError(`fix pour: the ${rigidId ? 'rigid' : 'shake'} keyword requires the mol keyword`);
    if (this.diam.style === 'one' && !(this.diam.d > 0)) throw new StyleError('fix pour: diam must be > 0');
    if (this.diam.style === 'range' && !(this.diam.lo > 0 && this.diam.hi >= this.diam.lo)) throw new StyleError('fix pour: diam range needs 0 < Dlo <= Dhi');
    if (!(this.dens[0] > 0 && this.dens[1] >= this.dens[0])) throw new StyleError('fix pour: dens needs 0 < Rholo <= Rhohi');
    // "id next": the maximum ID is read once, when the fix is defined
    this.idNext = idNext;
    this.nextId = maxAtomId(sys.state);
    this.nextMolId = maxMoleculeId(sys.state);
    this.nfirst = sys.state.step + 1;
    this.nextStep = this.nfirst;
    this.scalarFlag = true;
  }

  init(): void {
    const s = this.sys.state;
    if (this.molTemplateId) {
      const count = moleculeCount(this.sys, this.molTemplateId);
      if (count > 1) {
        throw new StyleError(`fix pour mol: molecule template '${this.molTemplateId}' defines ${count} molecules (molfrac); only a single-molecule template is supported by the browser engine`);
      }
      this.molTemplate = this.sys.molecules.get(this.molTemplateId)![0];
      let maxT = 0;
      for (const ty of this.molTemplate.type) if (ty > maxT) maxT = ty;
      if (this.type + maxT > s.ntypes) {
        throw new StyleError(`fix pour mol: type offset ${this.type} plus template type ${maxT} is larger than ntypes ${s.ntypes}`);
      }
      if (this.rigidId) {
        const rf = this.sys.fix(this.rigidId);
        if (typeof (rf as { addMolecule?: unknown }).addMolecule !== 'function') {
          throw new StyleError(`fix pour rigid: fix ${this.rigidId} (${rf.style}) is not a fix rigid/small`);
        }
        this.rigidFix = rf;
      }
      if (this.shakeId) {
        const sf = this.sys.fix(this.shakeId);
        if (typeof (sf as { rebuildClusters?: unknown }).rebuildClusters !== 'function') {
          throw new StyleError(`fix pour shake: fix ${this.shakeId} (${sf.style}) is not a fix shake`);
        }
        this.shakeFix = sf;
      }
    } else if (this.type > s.ntypes) throw new StyleError(`fix pour: atom type ${this.type} is larger than ntypes ${s.ntypes}`);
    this.reg = insertionRegion(this.sys, this.regionId, 'pour');
    if (this.twoD && this.reg.kind === 'cylinder') {
      throw new StyleError('fix pour: the cylinder style of region can only be used with 3d simulations');
    }
    insertionBox(this.sys, this.reg, 'pour');
    this.g = gravityMagnitude(this.sys, 'pour');
  }

  /** Particle volume used for the count: diam range uses Dhi, poly the percentage-weighted mean. */
  private particleVolume(): number {
    if (this.molTemplate) return this.moleculeVolume(this.molTemplate);
    const a = (d: number) => (this.twoD ? (Math.PI / 4) * d * d : (Math.PI / 6) * d * d * d);
    if (this.diam.style === 'one') return a(this.diam.d);
    if (this.diam.style === 'range') return a(this.diam.hi);
    let m = 0;
    for (let k = 0; k < this.diam.d.length; k++) m += this.diam.p[k] * a(this.diam.d[k]);
    return m;
  }

  /** Sphere (3d) / disc (2d) volume of R = max over atoms of (|x - center| + radius). */
  private moleculeVolume(t: MoleculeTemplate): number {
    const c = geometricCenter(t);
    let R = 0;
    for (let i = 0; i < t.natoms; i++) {
      const dx = t.x[3 * i] - c[0], dy = t.x[3 * i + 1] - c[1], dz = t.x[3 * i + 2] - c[2];
      const r = t.diam ? t.diam[i] / 2 : SPHERE_DEFAULT_RADIUS;
      const d = Math.hypot(dx, dy, dz) + r;
      if (d > R) R = d;
    }
    return this.twoD ? Math.PI * R * R : (4 / 3) * Math.PI * R * R * R;
  }

  private regionVolume(reg: InsertionRegion): number {
    if (reg.kind === 'cylinder') return Math.PI * reg.radius * reg.radius * (reg.zhi - reg.zlo);
    if (this.twoD) return (reg.xhi - reg.xlo) * (reg.yhi - reg.ylo);
    return (reg.xhi - reg.xlo) * (reg.yhi - reg.ylo) * (reg.zhi - reg.zlo);
  }

  /** Insertion happens here: after the initial half kick, before the neighbour decision. */
  postIntegrate(): void {
    if (this.inserted >= this.N || this.sys.state.step !== this.nextStep) return;
    this.insertEvent();
  }

  /** The run loop calls preExchange only on neighbour rebuilds; it is a stub (see the header). */
  preExchange(): void { /* insertion is done in postIntegrate */ }

  /** Flat 3N coordinates of the molecule rotated by `rot` about its geometric center and placed at p. */
  private placeMolecule(t: MoleculeTemplate, center: number[], rot: number[][], p: [number, number, number]): Float64Array {
    const out = new Float64Array(3 * t.natoms);
    for (let i = 0; i < t.natoms; i++) {
      const rx = t.x[3 * i] - center[0], ry = t.x[3 * i + 1] - center[1], rz = t.x[3 * i + 2] - center[2];
      out[3 * i] = p[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
      out[3 * i + 1] = p[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
      out[3 * i + 2] = p[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
    }
    return out;
  }

  private insertEvent(): void {
    const sys = this.sys;
    const s = sys.state;
    const reg = this.reg!;
    const dt = s.dt;
    const step = s.step;
    const g = this.g;
    const twoD = this.twoD;
    const fb = twoD ? 1 : 2;
    const shift = this.rateV * (step - this.nfirst) * dt;
    const fLo = (fb === 2 ? reg.zlo : reg.ylo) + shift;
    const fHi = (fb === 2 ? reg.zhi : reg.yhi) + shift;
    const H = fHi - fLo;
    const nper = Math.floor((this.vol * this.regionVolume(reg)) / this.particleVolume());
    if (nper === 0) throw new StyleError('Fix pour insertion count per timestep is 0');
    const nnew = Math.min(nper, this.N - this.inserted);
    const budget = nnew * this.nattempt;
    let tries = 0;
    const nOld = s.n;
    const t = this.molTemplate;
    const center = t ? geometricCenter(t) : [0, 0, 0];
    // candidate atoms appended during this event (molecules register all their atoms)
    const cx: number[] = [], cy: number[] = [], cz: number[] = [], cr: number[] = [];
    const at = (i: number): [number, number, number, number] => (i < nOld
      ? [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], s.radius![i]]
      : [cx[i - nOld], cy[i - nOld], cz[i - nOld], cr[i - nOld]]);
    const overlaps = (x: number, y: number, z: number, rad: number, n: number): boolean => {
      for (let j = 0; j < n; j++) {
        const [px, py, pz, pr] = at(j);
        const [dx, dy, dz] = separation(sys, x - px, y - py, z - pz);
        const rs = rad + pr;
        if (dx * dx + dy * dy + dz * dz < rs * rs) return true;
      }
      return false;
    };
    // accepted atoms (no mol) and molecules
    const ax: number[] = [], ay: number[] = [], az: number[] = [], ar: number[] = [], am: number[] = [];
    const avx: number[] = [], avy: number[] = [], avz: number[] = [];
    const mols: { pos: Float64Array; v: [number, number, number] }[] = [];
    for (let k = 0; k < nnew; k++) {
      // the fall coordinate is drawn once per particle; overlap retries keep it
      const uz = this.rng.uniform();
      const f = fLo + H * (1 - uz * uz);
      if (t) {
        let placed = false, pos: Float64Array | null = null;
        while (tries < budget) {
          tries++;
          const x = reg.xlo + this.rng.uniform() * (reg.xhi - reg.xlo);
          let y = 0, z = 0;
          if (twoD) y = f;
          else { y = reg.ylo + this.rng.uniform() * (reg.yhi - reg.ylo); z = f; }
          if (reg.kind === 'cylinder' && (x - reg.xc) ** 2 + (y - reg.yc) ** 2 > reg.radius * reg.radius) continue;
          // one draw samples the molecule list; then the random rotation
          const sel = this.rng.uniform();
          if (this.molfrac) { let c = 0; for (const fr of this.molfrac) { c += fr; if (sel < c) break; } }
          let R: number[][];
          if (twoD) R = rotationMatrix(2 * Math.PI * this.rng.uniform(), 0, 0, 1);
          else {
            const u1 = this.rng.uniform(), u2 = this.rng.uniform(), u3 = this.rng.uniform(), u4 = this.rng.uniform();
            R = rotationMatrix(2 * Math.PI * u4, u1 - 0.5, u2 - 0.5, u3 - 0.5);
          }
          const cand = this.placeMolecule(t, center, R, [x, y, z]);
          let hit = false;
          for (let a = 0; a < t.natoms && !hit; a++) {
            const rad = t.diam ? t.diam[a] / 2 : SPHERE_DEFAULT_RADIUS;
            hit = overlaps(cand[3 * a], cand[3 * a + 1], cand[3 * a + 2], rad, nOld + cx.length);
          }
          if (hit) continue;
          placed = true; pos = cand; break;
        }
        if (!placed) break;
        const vx = this.vel[0] + this.rng.uniform() * (this.vel[1] - this.vel[0]);
        const vyDraw = twoD ? 0 : this.vel[2] + this.rng.uniform() * (this.vel[3] - this.vel[2]);
        const vfall = -Math.sqrt(this.vel[4] * this.vel[4] + 2 * g * (fHi - f));
        const v: [number, number, number] = twoD ? [vx, vfall, 0] : [vx, vyDraw, vfall];
        for (let a = 0; a < t.natoms; a++) {
          cx.push(pos![3 * a]); cy.push(pos![3 * a + 1]); cz.push(pos![3 * a + 2]);
          cr.push(t.diam ? t.diam[a] / 2 : SPHERE_DEFAULT_RADIUS);
        }
        mols.push({ pos: pos!, v });
      } else {
        const rad = this.drawRadius() / 2;
        let x = 0, y = 0, z = 0, found = false;
        while (tries < budget) {
          tries++;
          x = reg.xlo + this.rng.uniform() * (reg.xhi - reg.xlo);
          if (twoD) y = f;
          else { y = reg.ylo + this.rng.uniform() * (reg.yhi - reg.ylo); z = f; }
          if (reg.kind === 'cylinder' && (x - reg.xc) ** 2 + (y - reg.yc) ** 2 > reg.radius * reg.radius) continue;
          if (overlaps(x, y, z, rad, nOld + cx.length)) continue;
          found = true;
          break;
        }
        if (!found) break;
        const vx = this.vel[0] + this.rng.uniform() * (this.vel[1] - this.vel[0]);
        const vyDraw = twoD ? 0 : this.vel[2] + this.rng.uniform() * (this.vel[3] - this.vel[2]);
        const rho = this.dens[0] + this.rng.uniform() * (this.dens[1] - this.dens[0]);
        const vfall = -Math.sqrt(this.vel[4] * this.vel[4] + 2 * g * (fHi - f));
        cx.push(x); cy.push(y); cz.push(z); cr.push(rad);
        ax.push(x); ay.push(y); az.push(z); ar.push(rad); am.push(sphereMass(rad, rho));
        avx.push(vx);
        if (twoD) { avy.push(vfall); avz.push(0); }
        else { avy.push(vyDraw); avz.push(vfall); }
      }
    }
    const n = mols.length + ax.length;
    if (n < nnew) sys.warn(`Fewer insertions than requested (${n} vs ${nnew}) on step ${step}`);
    if (n > 0) {
      if (t) {
        for (const m of mols) {
          const ids = this.idNext ? new Int32Array(t.natoms) : undefined;
          if (ids) for (let k = 0; k < t.natoms; k++) ids[k] = ++this.nextId;
          const molId = this.idNext ? ++this.nextMolId : maxMoleculeId(s) + 1;
          const firstId = appendMolecule(sys, t, this.type, m.pos, m.v, molId, this.groupBit, ids);
          if (this.rigidFix) (this.rigidFix as unknown as RigidAdder).addMolecule(Array.from({ length: t.natoms }, (_, k) => firstId + k));
          if (this.shakeFix) (this.shakeFix as unknown as ClusterRebuilder).rebuildClusters();
        }
      } else {
        const na = ax.length;
        const xa = new Float64Array(3 * na), va = new Float64Array(3 * na);
        const ra = new Float64Array(na), ma = new Float64Array(na);
        const ids = this.idNext ? new Int32Array(na) : undefined;
        for (let k = 0; k < na; k++) {
          xa[3 * k] = ax[k]; xa[3 * k + 1] = ay[k]; xa[3 * k + 2] = az[k];
          va[3 * k] = avx[k]; va[3 * k + 1] = avy[k]; va[3 * k + 2] = avz[k];
          ra[k] = ar[k]; ma[k] = am[k];
          if (ids) ids[k] = ++this.nextId;
        }
        appendAtoms(s, { x: xa, type: this.type, v: va, radius: ra, rmass: ma, id: ids, mask: this.groupBit });
      }
      sys.atomsChanged();
      // the pair cutoffs (granular: from the largest radii present) are read when the lists are built
      sys.setupNeighbors();
      this.inserted += n;
    }
    // time until the particles fall out of the region: (-w + sqrt(w^2 + 2 g H)) / g, w = rate - vfall
    const w = this.rateV - this.vel[4];
    const tFall = (-w + Math.sqrt(w * w + 2 * g * H)) / g;
    this.nextStep = step + Math.max(1, Math.round(tFall / dt));
  }

  /** Diameter for this particle (one draw for range and poly, none for one). */
  private drawRadius(): number {
    if (this.diam.style === 'one') return this.diam.d;
    if (this.diam.style === 'range') return this.diam.lo + this.rng.uniform() * (this.diam.hi - this.diam.lo);
    const u = this.rng.uniform();
    let c = 0;
    for (let q = 0; q < this.diam.d.length; q++) {
      c += this.diam.p[q];
      if (u < c) return this.diam.d[q];
    }
    return this.diam.d[this.diam.d.length - 1];
  }

  computeScalar(): number { return this.inserted; }
}

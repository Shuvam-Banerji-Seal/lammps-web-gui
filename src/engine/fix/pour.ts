import { Fix } from './fix';
import { StyleError } from '../force/types';
import { RanPark } from '../rng';
import { appendAtoms, maxAtomId, pushTopo, sphereMass, isSphereStyle, hasChargeStyle } from '../atoms';
import { BlockRegion, ConeRegion, type Param, type Region } from '../region';
import type { MoleculeTemplate } from '../molecule';
import type { System } from '../system';

/*
 * fix ID group-ID pour N type seed keyword values ... — docs.lammps.org/fix_pour.html
 * (plans/lammps-docs/fix_pour.rst). Supported: region (block, or a z-axis
 * cylinder, side in), diam one / range / poly, id max / next, vol, rate,
 * dens, vel (3d), ignore. Everything else throws a StyleError: mol, molfrac,
 * rigid, shake, 2d simulations, and atom styles other than sphere.
 *
 * "Insert finite-size particles or molecules into the simulation box
 * every few timesteps within a specified region until N particles or
 * molecules have been inserted."
 *
 * Random numbers (RanPark, rng.ts). Measured with native LAMMPS (black box,
 * seeds 1, 2, 777, 12345 and 12346 for the first insertion; the oracle cases
 * also use 999 and 4242): the stream is RanPark seeded with the given seed
 * after 30 discarded draws (the first draw used is the 31st of the
 * Park-Miller sequence, index 30). Per
 * insertion event each particle draws, in this order:
 *   1. z:      z = zlo + (zhi - zlo) * (1 - u^2), u one draw (drawn ONCE per
 *              particle, before the attempt loop; overlap retries keep it);
 *   2. diam:   one draw for range (D = Dlo + u (Dhi - Dlo)) or poly (the
 *              diameter whose cumulative percentage first exceeds u); none for one;
 *   3. x, y:   xlo + u (xhi - xlo), ylo + u (yhi - ylo); repeated on every
 *              attempt (a cylinder rejects points outside the circle, and
 *              an overlapping candidate is rejected; both use one attempt);
 *   4. vx, vy: vxlo + u (vxhi - vxlo) and likewise vy, drawn after acceptance;
 *   5. dens:   rholo + u (rhohi - rholo), drawn after acceptance (mass = rho V).
 * Measured: the z of an inserted particle is independent of the diam draw,
 * and changing vel or dens ranges does not move the positions.
 *
 * Insertion count and timing. "Next, the target number of particles
 * inserted per event (assuming no failed insertions due to overlaps) is
 * calculated as the product of the volume fraction and the volume of the
 * insertion region divided by the volume of a particle" (rst lines 166 to 170).
 * Measured: the count is floor(vol V / Vp) (vol 0.9, V = 1, one diam 1:
 * 1 particle; vol 0.3: the error Fix pour insertion count per timestep is 0); for
 * diam range Vp uses Dhi (a range 0.5 to 1.0 gives 30 where 'one 1.0' gives
 * 30, not the 73 of the mean); for poly Vp is the percentage-weighted mean
 * volume (poly 3 0.5 0.3 0.8 0.3 1.2 0.4 gives 35, which is floor(0.2 * 81 /
 * 0.4620)). Measured: the first event is at the first step of the run
 * (step 1), later events are spaced by the rounded fall time: with gravity
 * g, region height H, rate V (the region rises at rate) and vz given,
 * t = (-w + sqrt(w^2 + 2 g H)) / g with w = rate - vz; g = 1, H = 1 gives
 * 1414 steps at dt 0.001 (t = 1.41421), 1664 at dt 0.00085, 1115 for g = 2,
 * H = 1, dt = 0.0007, vz = -0.5 (round, not floor: 1414.21 -> 1414, 1663.78 ->
 * 1664). Measured with rate 0.5: the event at step 1177 (dt 0.00085) sees the
 * region bottom at zlo + rate (step - 1) dt.
 *
 * Velocity. "The *vz* or *vy* value for option *vel* assigns a z-velocity
 * (3d) or y-velocity (2d) to each inserted particle." Measured: the initial
 * vz of a particle is -sqrt(vz^2 + 2 g (zhi - z)) with zhi the top of the
 * region at the event (sign of the vel vz value does not matter), and a step
 * step after insertion the dump shows v0 - g dt/2: the new atom takes no
 * initial half kick (insertion precedes it) but takes the final one.
 *
 * Attempts. "LAMMPS will make up to a total of M tries to insert the new
 * particles without overlaps, where M = # of inserted particles \* Nattempt."
 * Measured: M = nnew * Nattempt where
 * nnew = min(count, remaining N); the tries are shared by the whole event
 * (every x, y pair counts, successful or not). A particle that exhausts the
 * budget is not inserted and the event ends with the warning Fewer insertions
 * than requested. Overlap: centres closer than the sum of the two radii are
 * overlapping (measured: every rejected candidate of a 30 particle event with
 * mixed radii lay within r_i + r_j of an earlier particle and every accepted
 * one outside). Periodic images are included as the doc requires ("including
 * effects due to periodic boundary conditions if applicable"); not measured.
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

/** Gravitational acceleration magnitude along -z from the defined fix gravity. */
export const gravityMagnitude = (sys: System, what: string): number => {
  const fx = sys.fixes.find((f) => f.style === 'gravity');
  if (!fx) throw new StyleError(`fix ${what} requires a fix gravity in the -z direction`);
  const out = new Float64Array(3);
  (fx as unknown as { gvec(o: Float64Array): void }).gvec(out);
  if (Math.abs(out[0]) > 1e-12 || Math.abs(out[1]) > 1e-12 || !(out[2] < 0)) {
    throw new StyleError(`fix ${what}: the gravity fix must point in the -z direction`);
  }
  return -out[2];
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
  const base = ids && ids.length ? ids[0] - 1 : maxAtomId(s);
  appendAtoms(s, { x: pos, type: types, v, image, molecule, q, mask: gbit, id: ids });
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
}

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
  private g = 0;
  private readonly nfirst: number;
  private nextStep: number;
  inserted = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (sys.dimension !== 3) throw new StyleError('fix pour: only 3d simulations are supported');
    if (!isSphereStyle(sys.atomStyle)) throw new StyleError('fix pour requires atom_style sphere');
    if (args.length < 3) throw new StyleError('usage: fix ID group-ID pour N type seed keyword values ...');
    this.N = posInt(args[0], 'N');
    // molecule insertion (fix_pour.html: type is an offset with mol, so 0 is valid there) is not
    // supported; name it before the type check rejects a 0 offset
    for (const key of ['mol', 'molfrac', 'rigid', 'shake']) {
      if (args.slice(3).includes(key)) throw new StyleError(`fix pour keyword '${key}' is not supported by the browser engine (molecule insertion)`);
    }
    this.type = posInt(args[1], 'type');
    this.rng = insertionStream(posInt(args[2], 'seed'));
    let region: string | null = null;
    let idNext = false;
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
        this.vel = [num(args[i + 1], 'vel vxlo'), num(args[i + 2], 'vel vxhi'), num(args[i + 3], 'vel vylo'), num(args[i + 4], 'vel vyhi'), num(args[i + 5], 'vel vz')];
        i += 6;
      } else if (key === 'ignore') {
        // this engine has no line or triangle particles, so there is nothing to skip
        i += 1;
      } else if (key === 'mol' || key === 'molfrac' || key === 'rigid' || key === 'shake') {
        throw new StyleError(`fix pour keyword '${key}' is not supported by the browser engine`);
      } else throw new StyleError(`fix pour: unknown keyword '${key}'`);
    }
    if (region === null) throw new StyleError('fix pour requires the region keyword');
    this.regionId = region;
    if (this.diam.style === 'one' && !(this.diam.d > 0)) throw new StyleError('fix pour: diam must be > 0');
    if (this.diam.style === 'range' && !(this.diam.lo > 0 && this.diam.hi >= this.diam.lo)) throw new StyleError('fix pour: diam range needs 0 < Dlo <= Dhi');
    if (!(this.dens[0] > 0 && this.dens[1] >= this.dens[0])) throw new StyleError('fix pour: dens needs 0 < Rholo <= Rhohi');
    // "id next": the maximum ID is read once, when the fix is defined
    this.idNext = idNext;
    this.nextId = maxAtomId(sys.state);
    this.nfirst = sys.state.step + 1;
    this.nextStep = this.nfirst;
    this.scalarFlag = true;
  }

  init(): void {
    const s = this.sys.state;
    if (this.type > s.ntypes) throw new StyleError(`fix pour: atom type ${this.type} is larger than ntypes ${s.ntypes}`);
    this.reg = insertionRegion(this.sys, this.regionId, 'pour');
    insertionBox(this.sys, this.reg, 'pour');
    this.g = gravityMagnitude(this.sys, 'pour');
  }

  /** Particle volume used for the count: diam range uses Dhi, poly the percentage-weighted mean. */
  private particleVolume(): number {
    const v = (d: number) => (Math.PI / 6) * d * d * d;
    if (this.diam.style === 'one') return v(this.diam.d);
    if (this.diam.style === 'range') return v(this.diam.hi);
    let m = 0;
    for (let k = 0; k < this.diam.d.length; k++) m += this.diam.p[k] * v(this.diam.d[k]);
    return m;
  }

  private regionVolume(reg: InsertionRegion): number {
    if (reg.kind === 'block') return (reg.xhi - reg.xlo) * (reg.yhi - reg.ylo) * (reg.zhi - reg.zlo);
    return Math.PI * reg.radius * reg.radius * (reg.zhi - reg.zlo);
  }

  /** Insertion happens here: after the initial half kick, before the neighbour decision. */
  postIntegrate(): void {
    if (this.inserted >= this.N || this.sys.state.step !== this.nextStep) return;
    this.insertEvent();
  }

  /** The run loop calls preExchange only on neighbour rebuilds; it is a stub (see the header). */
  preExchange(): void { /* insertion is done in postIntegrate */ }

  private insertEvent(): void {
    const sys = this.sys;
    const s = sys.state;
    const reg = this.reg!;
    const dt = s.dt;
    const step = s.step;
    const g = this.g;
    const shift = this.rateV * (step - this.nfirst) * dt;
    const zlo = reg.zlo + shift, zhi = reg.zhi + shift;
    const H = zhi - zlo;
    const nper = Math.floor((this.vol * this.regionVolume(reg)) / this.particleVolume());
    if (nper === 0) throw new StyleError('Fix pour insertion count per timestep is 0');
    const nnew = Math.min(nper, this.N - this.inserted);
    const budget = nnew * this.nattempt;
    let tries = 0;
    const nOld = s.n;
    const acc = { x: [] as number[], y: [] as number[], z: [] as number[], r: [] as number[], vx: [] as number[], vy: [] as number[], vz: [] as number[], m: [] as number[] };
    const vzg = this.vel[4];
    const at = (i: number): [number, number, number, number] => (i < nOld
      ? [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], s.radius![i]]
      : [acc.x[i - nOld], acc.y[i - nOld], acc.z[i - nOld], acc.r[i - nOld]]);
    for (let k = 0; k < nnew; k++) {
      // z is drawn once per particle; overlap retries redraw only x and y
      const uz = this.rng.uniform();
      const z = zlo + H * (1 - uz * uz);
      const rad = this.drawRadius() / 2;
      let x = 0, y = 0, found = false;
      while (tries < budget) {
        tries++;
        x = reg.xlo + this.rng.uniform() * (reg.xhi - reg.xlo);
        y = reg.ylo + this.rng.uniform() * (reg.yhi - reg.ylo);
        if (reg.kind === 'cylinder' && (x - reg.xc) ** 2 + (y - reg.yc) ** 2 > reg.radius * reg.radius) continue;
        if (this.overlaps(sys, x, y, z, rad, nOld + acc.x.length, at)) continue;
        found = true;
        break;
      }
      if (!found) break;
      const vx = this.vel[0] + this.rng.uniform() * (this.vel[1] - this.vel[0]);
      const vy = this.vel[2] + this.rng.uniform() * (this.vel[3] - this.vel[2]);
      const rho = this.dens[0] + this.rng.uniform() * (this.dens[1] - this.dens[0]);
      acc.x.push(x); acc.y.push(y); acc.z.push(z); acc.r.push(rad);
      acc.vx.push(vx); acc.vy.push(vy);
      // initial vz: the fall from the region top to z (the vel vz value adds in quadrature)
      acc.vz.push(-Math.sqrt(vzg * vzg + 2 * g * (zhi - z)));
      acc.m.push(sphereMass(rad, rho));
    }
    const n = acc.x.length;
    if (n < nnew) sys.warn(`Fewer insertions than requested (${n} vs ${nnew}) on step ${step}`);
    if (n > 0) {
      const xa = new Float64Array(3 * n), va = new Float64Array(3 * n);
      const ra = new Float64Array(n), ma = new Float64Array(n);
      const ids = this.idNext ? new Int32Array(n) : undefined;
      for (let k = 0; k < n; k++) {
        xa[3 * k] = acc.x[k]; xa[3 * k + 1] = acc.y[k]; xa[3 * k + 2] = acc.z[k];
        va[3 * k] = acc.vx[k]; va[3 * k + 1] = acc.vy[k]; va[3 * k + 2] = acc.vz[k];
        ra[k] = acc.r[k]; ma[k] = acc.m[k];
        if (ids) ids[k] = ++this.nextId;
      }
      appendAtoms(s, { x: xa, type: this.type, v: va, radius: ra, rmass: ma, id: ids, mask: this.groupBit });
      sys.atomsChanged();
      // the pair cutoffs (granular: from the largest radii present) are read when the lists are built
      sys.setupNeighbors();
      this.inserted += n;
    }
    // time until the particles fall out of the region: (-w + sqrt(w^2 + 2 g H)) / g, w = rate - vz
    const w = this.rateV - vzg;
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

  /** Overlap: centres closer than the sum of the radii, minimum image in periodic dimensions. */
  private overlaps(sys: System, x: number, y: number, z: number, rad: number, n: number,
    at: (i: number) => [number, number, number, number]): boolean {
    for (let j = 0; j < n; j++) {
      const [px, py, pz, pr] = at(j);
      const [dx, dy, dz] = separation(sys, x - px, y - py, z - pz);
      const rs = rad + pr;
      if (dx * dx + dy * dy + dz * dz < rs * rs) return true;
    }
    return false;
  }

  computeScalar(): number { return this.inserted; }
}

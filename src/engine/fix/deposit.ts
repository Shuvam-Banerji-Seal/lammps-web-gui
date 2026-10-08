import { Fix } from './fix';
import { StyleError } from '../force/types';
import { appendAtoms, maxAtomId } from '../atoms';
import type { System } from '../system';
import { insertionRegion, insertionStream, insertionBox, num, posInt, separation, type InsertionRegion } from './pour';
import type { RanPark } from '../rng';

/*
 * fix ID group-ID deposit N type M seed keyword values ... — docs.lammps.org/fix_deposit.html
 * (plans/lammps-docs/fix_deposit.rst). "Insert a single atom or molecule into
 * the simulation domain every M timesteps until N atoms or molecules have
 * been inserted." Supported keywords: region (block), id max|next, global,
 * local, near, attempt, rate, vx, vy, vz, target, units box. Everything else
 * throws a StyleError: var, set, gaussian, mol, molfrac, rigid, shake, orient,
 * a cylinder region, and units lattice (the default; use units box).
 *
 * "The locations of inserted particles are taken from uniform distributed
 * random numbers, unless the *gaussian* keyword is used." Measured with native
 * LAMMPS (black box, seed 12345, units box, block region 0..1, M = 2, 3, 1):
 * the stream is RanPark seeded with the seed after the same 30 discarded draws
 * as fix pour (see pour.ts). Per attempt the draws are x, y, z (each
 * lo + u (hi - lo)); a rejected attempt (near) costs the same three draws; an
 * accepted particle then draws vx, vy, vz (vxlo + u (vxhi - vxlo) and so on,
 * drawn even when the bounds are 0 0). With the global or local keyword the z
 * draw is still made and a further draw gives the height above the reference
 * (the atom stride is 7 draws instead of 6). Insertions happen at steps
 * 1, 1 + M, 1 + 2M, ... of a run started at step 0 (measured with M = 2:
 * steps 1, 3, 5). An insertion step that fails after Q attempts inserts
 * nothing and the next try is M steps later (measured with attempt 1).
 *
 * The doc's id next rule (fix_deposit.rst: "Each time a new particle is added,
 * this value is incremented to assign IDs to the new atom(s) or molecule.")
 * is what the engine does. Measured region motion with rate V: at an insertion
 * step s the region bounds in z are shifted by V (s + M - 1) dt (M = 1, 2, 3
 * with V = 0.5 and dt = 0.001 gave the shifts V s dt for M = 1 and V (s + 1) dt
 * for M = 2, V (s + 2) dt for M = 3, measured at the first insertion).
 *
 * near (measured with R = 0.9): a candidate closer than R to an existing atom
 * is rejected (0.835 rejected; 0.968 accepted). Periodic images are included
 * as the doc requires; not measured.
 *
 * target: the velocity keeps its magnitude and points from the insertion site to
 * the target point (fix_deposit.rst: "the velocity vector of the inserted" /
 * "particle is changed so that it points from the insertion position" /
 * "towards the specified target point"); verified by the w7dep_target oracle case.
 * Not measured: the local reference height when atoms exist but none is within
 * delta (the box lower bound is assumed, as for global with an empty box), the
 * global/local draw inside a rejected near attempt (assumed to be redrawn per
 * attempt), and periodic images in the near test.
 */

type Vec3 = [number, number, number];

export class FixDeposit extends Fix {
  readonly style = 'deposit';
  readonly N: number;
  readonly type: number;
  readonly M: number;
  private readonly rng: RanPark;
  private readonly regionId: string;
  private reg: InsertionRegion | null = null;
  private readonly idNext: boolean;
  private nextId: number;
  private globalRange: [number, number] | null = null;
  private localRange: [number, number, number] | null = null;
  private near = 0;
  private attempt = 10;
  private rateV = 0;
  private vx: [number, number] = [0, 0];
  private vy: [number, number] = [0, 0];
  private vz: [number, number] = [0, 0];
  private target: Vec3 | null = null;
  private readonly nfirst: number;
  private nextStep: number;
  inserted = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (sys.dimension !== 3) throw new StyleError('fix deposit: only 3d simulations are supported');
    if (args.length < 4) throw new StyleError('usage: fix ID group-ID deposit N type M seed keyword values ...');
    // molecule insertion (fix_deposit.html: type is an offset with mol, so 0 is valid there) is not
    // supported; name it before the type check rejects a 0 offset
    for (const key of ['mol', 'molfrac', 'rigid', 'shake']) {
      if (args.slice(4).includes(key)) throw new StyleError(`fix deposit keyword '${key}' is not supported by the browser engine (molecule insertion)`);
    }
    this.N = posInt(args[0], 'N');
    this.type = posInt(args[1], 'type');
    this.M = posInt(args[2], 'M');
    this.rng = insertionStream(posInt(args[3], 'seed'));
    let region: string | null = null;
    let idNext = false;
    let units: string | null = null;
    let i = 4;
    while (i < args.length) {
      const key = args[i];
      const v = (k: number) => {
        if (args[i + k] === undefined) throw new StyleError(`fix deposit: keyword '${key}' needs ${k} value(s)`);
        return args[i + k];
      };
      if (key === 'region') { region = v(1); i += 2; }
      else if (key === 'id') {
        const w = v(1);
        if (w !== 'max' && w !== 'next') throw new StyleError(`fix deposit: id must be max or next, got '${w}'`);
        idNext = w === 'next';
        i += 2;
      } else if (key === 'global') { this.globalRange = [num(v(1), 'global lo'), num(v(2), 'global hi')]; i += 3; }
      else if (key === 'local') {
        this.localRange = [num(v(1), 'local lo'), num(v(2), 'local hi'), num(v(3), 'local delta')];
        i += 4;
      } else if (key === 'near') { this.near = num(v(1), 'near R'); i += 2; }
      else if (key === 'attempt') { this.attempt = posInt(v(1), 'attempt Q'); i += 2; }
      else if (key === 'rate') { this.rateV = num(v(1), 'rate'); i += 2; }
      else if (key === 'vx') { this.vx = [num(v(1), 'vx lo'), num(v(2), 'vx hi')]; i += 3; }
      else if (key === 'vy') { this.vy = [num(v(1), 'vy lo'), num(v(2), 'vy hi')]; i += 3; }
      else if (key === 'vz') { this.vz = [num(v(1), 'vz lo'), num(v(2), 'vz hi')]; i += 3; }
      else if (key === 'target') {
        this.target = [num(v(1), 'target tx'), num(v(2), 'target ty'), num(v(3), 'target tz')];
        i += 4;
      } else if (key === 'units') {
        units = v(1);
        if (units !== 'box' && units !== 'lattice') throw new StyleError(`fix deposit: units must be lattice or box, got '${units}'`);
        i += 2;
      } else if (['var', 'set', 'gaussian', 'mol', 'molfrac', 'rigid', 'shake', 'orient'].includes(key)) {
        throw new StyleError(`fix deposit keyword '${key}' is not supported by the browser engine`);
      } else throw new StyleError(`fix deposit: unknown keyword '${key}'`);
    }
    if (region === null) throw new StyleError('fix deposit requires the region keyword');
    if (units !== 'box') throw new StyleError('fix deposit: units lattice (the default) is not supported by the browser engine; use units box');
    if (this.globalRange && this.localRange) throw new StyleError('fix deposit: global and local cannot both be used');
    if (this.globalRange && !(this.globalRange[1] >= this.globalRange[0])) throw new StyleError('fix deposit: global needs lo <= hi');
    this.regionId = region;
    if (this.vx[0] > this.vx[1] || this.vy[0] > this.vy[1] || this.vz[0] > this.vz[1]) throw new StyleError('fix deposit: velocity ranges need lo <= hi');
    this.idNext = idNext;
    this.nextId = maxAtomId(sys.state);
    this.nfirst = sys.state.step + 1;
    this.nextStep = this.nfirst;
    this.scalarFlag = true;
  }

  init(): void {
    const s = this.sys.state;
    if (this.type > s.ntypes) throw new StyleError(`fix deposit: atom type ${this.type} is larger than ntypes ${s.ntypes}`);
    this.reg = insertionRegion(this.sys, this.regionId, 'deposit');
    if (this.reg.kind !== 'block') throw new StyleError('fix deposit: only a block region is supported');
    insertionBox(this.sys, this.reg, 'deposit');
  }

  /** The run loop calls preExchange only on neighbour rebuilds; it is a stub (see fix pour). */
  preExchange(): void { /* insertion is done in postIntegrate */ }

  postIntegrate(): void {
    if (this.inserted >= this.N || this.sys.state.step !== this.nextStep) return;
    this.depositOne();
    this.nextStep += this.M;
  }

  /** Up to Q attempts at this step; accepted particles get the velocity draws after the position. */
  private depositOne(): void {
    const sys = this.sys;
    const s = sys.state;
    const reg = this.reg!;
    const step = s.step;
    const shift = this.rateV * (step - this.nfirst + this.M) * s.dt;
    const zlo = reg.zlo + shift, zhi = reg.zhi + shift;
    let placed: { x: Vec3; v: Vec3 } | null = null;
    for (let t = 0; t < this.attempt && !placed; t++) {
      const x = reg.xlo + this.rng.uniform() * (reg.xhi - reg.xlo);
      const y = reg.ylo + this.rng.uniform() * (reg.yhi - reg.ylo);
      let z = zlo + this.rng.uniform() * (zhi - zlo);
      if (this.globalRange) {
        const top = this.highestZ(null);
        z = top + this.globalRange[0] + this.rng.uniform() * (this.globalRange[1] - this.globalRange[0]);
      } else if (this.localRange) {
        const top = this.highestZ([x, y]);
        z = top + this.localRange[0] + this.rng.uniform() * (this.localRange[1] - this.localRange[0]);
      }
      if (this.near > 0 && this.tooNear(x, y, z)) continue;
      let v: Vec3 = [
        this.vx[0] + this.rng.uniform() * (this.vx[1] - this.vx[0]),
        this.vy[0] + this.rng.uniform() * (this.vy[1] - this.vy[0]),
        this.vz[0] + this.rng.uniform() * (this.vz[1] - this.vz[0]),
      ];
      if (this.target) v = this.aimAt([x, y, z], v);
      placed = { x: [x, y, z], v };
    }
    if (!placed) {
      sys.warn('Particle deposition was unsuccessful');
      return;
    }
    const xa = Float64Array.from(placed.x);
    const va = Float64Array.from(placed.v);
    const id = this.idNext ? Int32Array.of(++this.nextId) : undefined;
    appendAtoms(s, { x: xa, type: this.type, v: va, id, mask: this.groupBit });
    sys.atomsChanged();
    sys.setupNeighbors();
    this.inserted++;
  }

  /** Highest z of the atoms (within delta laterally of (x, y) when given); the box bottom if none. */
  private highestZ(lateral: [number, number] | null): number {
    const s = this.sys.state;
    const delta = this.localRange ? this.localRange[2] : 0;
    let top = s.box.lo[2];
    let any = false;
    for (let i = 0; i < s.n; i++) {
      if (lateral) {
        const [dx, dy] = separation(this.sys, lateral[0] - s.x[3 * i], lateral[1] - s.x[3 * i + 1], 0);
        if (Math.sqrt(dx * dx + dy * dy) >= delta) continue;
      }
      const z = s.x[3 * i + 2];
      if (!any || z > top) top = z;
      any = true;
    }
    return top;
  }

  /** True if any atom lies closer than the near distance (minimum image in periodic dimensions). */
  private tooNear(x: number, y: number, z: number): boolean {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      const [dx, dy, dz] = separation(this.sys, x - s.x[3 * i], y - s.x[3 * i + 1], z - s.x[3 * i + 2]);
      if (dx * dx + dy * dy + dz * dz < this.near * this.near) return true;
    }
    return false;
  }

  /** Rotates the velocity to point from the insertion site to the target, keeping its magnitude. */
  private aimAt(p: Vec3, v: Vec3): Vec3 {
    const t = this.target!;
    const d: Vec3 = [t[0] - p[0], t[1] - p[1], t[2] - p[2]];
    const dn = Math.hypot(d[0], d[1], d[2]);
    const vn = Math.hypot(v[0], v[1], v[2]);
    if (dn === 0) return v;
    return [vn * d[0] / dn, vn * d[1] / dn, vn * d[2] / dn];
  }

  computeScalar(): number { return this.inserted; }
}

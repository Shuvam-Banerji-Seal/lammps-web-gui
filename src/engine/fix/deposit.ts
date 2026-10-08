import { Fix } from './fix';
import { StyleError } from '../force/types';
import { appendAtoms, maxAtomId, isMolecularStyle } from '../atoms';
import type { System } from '../system';
import { insertionRegion, insertionStream, insertionBox, num, posInt, separation, appendMolecule, type InsertionRegion } from './pour';
import { geometricCenter, rotationMatrix, type MoleculeTemplate } from '../molecule';
import { latticeScale } from '../commands/args';
import type { RanPark } from '../rng';

/*
 * fix ID group-ID deposit N type M seed keyword values ... — docs.lammps.org/fix_deposit.html
 * (plans/lammps-docs/fix_deposit.rst). "Insert a single atom or molecule into
 * the simulation domain every M timesteps until N atoms or molecules have
 * been inserted." Supported keywords: region (block), id max|next, global,
 * local, near, attempt, rate, vx, vy, vz, target, units box, and mol (one
 * molecule template, single-molecule templates only) with molfrac, rigid,
 * shake and orient, and both units box and units lattice (the default).
 * Everything else throws a StyleError: var, set, gaussian, a cylinder region,
 * and global/local combined with mol.
 *
 * Molecule insertion (measured with native LAMMPS, black box, seed 12345,
 * block region 0..10 x 0..10 x 8..9, M = 1, molecule.dimer): the insertion position
 * is the geometric center and each trial draws, from the same RanPark stream as
 * atom insertion, x, y, z, then the molecule selection (one draw), then the
 * random rotation, and only on acceptance vx, vy, vz. The rotation of molecule
 * k at trial m is a rotation of 2 pi u4 about the axis
 * (u1 - 0.5, u2 - 0.5, u3 - 0.5), with u1..u4 the four draws after the
 * selection draw (same form as create_atoms mol, but from the fix's stream, not
 * a RanMars). The near test runs after the rotation and tests every atom of the
 * rotated molecule against the current atoms, so a rejected trial redraws
 * position, selection and rotation (measured: attempts 1 and 2 positioned at
 * stream indices 0..2 and 8..10, the five draws 3..7 being selection+rotation).
 * A molecule's atoms all get the same velocity. Type offset: each template atom
 * type plus the type value. One molecule ID per deposited particle, atoms
 * consecutive after the current maxima (id max) or the counters stored at fix
 * definition (id next; measured: existing atoms 1,2 -> particle atoms 3,4 and
 * molecule 1, next 5,6 and 2). The `rigid` and `shake` values name a fix
 * rigid/small and a fix shake that are told about the new molecule.
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

/** Number of molecules a template ID defines (summed over the files loaded under that ID). */
const moleculeCount = (sys: System, id: string): number => {
  const sets = sys.molecules.get(id);
  if (!sets || !sets.length) throw new StyleError(`fix deposit mol: molecule template '${id}' does not exist`);
  let n = 0;
  for (const t of sets) n += t.mol ? new Set(t.mol).size : 1;
  return n;
};

/** Largest molecule ID currently in the system (0 when none). */
const maxMoleculeId = (s: { n: number; molecule: Int32Array }): number => {
  let m = 0;
  for (let i = 0; i < s.n; i++) if (s.molecule[i] > m) m = s.molecule[i];
  return m;
};

/** Interface for the fix rigid/small method deposit calls to add a new rigid body. */
interface RigidAdder { addMolecule(atomIds: readonly number[]): void; }
/** Interface for the fix shake method deposit calls to rebuild its clusters. */
interface ClusterRebuilder { rebuildClusters(): void; }

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
  private readonly molTemplateId: string | null;
  private molTemplate: MoleculeTemplate | null = null;
  private readonly molfrac: number[] | null;
  private readonly rigidId: string | null;
  private readonly shakeId: string | null;
  private readonly orient: Vec3 | null;
  private nextMolId: number;
  private rigidFix: Fix | null = null;
  private shakeFix: Fix | null = null;
  private readonly nfirst: number;
  private nextStep: number;
  inserted = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (sys.dimension !== 3) throw new StyleError('fix deposit: only 3d simulations are supported');
    if (args.length < 4) throw new StyleError('usage: fix ID group-ID deposit N type M seed keyword values ...');
    this.N = posInt(args[0], 'N');
    this.M = posInt(args[2], 'M');
    this.rng = insertionStream(posInt(args[3], 'seed'));
    let region: string | null = null;
    let idNext = false;
    let units: string | null = null;
    let molTemplateId: string | null = null;
    let molfrac: number[] | null = null;
    let rigidId: string | null = null;
    let shakeId: string | null = null;
    let orient: Vec3 | null = null;
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
      } else if (key === 'mol') { molTemplateId = v(1); i += 2; }
      else if (key === 'molfrac') {
        if (!molTemplateId) throw new StyleError('fix deposit: molfrac requires the mol keyword before it');
        const n = moleculeCount(sys, molTemplateId);
        const vals: number[] = [];
        for (let k = 1; k <= n; k++) vals.push(num(v(k), 'molfrac'));
        const sum = vals.reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1) > 1e-8) throw new StyleError('fix deposit: molfrac values must sum to 1.0');
        molfrac = vals;
        i += 1 + n;
      } else if (key === 'rigid') { rigidId = v(1); i += 2; }
      else if (key === 'shake') { shakeId = v(1); i += 2; }
      else if (key === 'orient') { orient = [num(v(1), 'orient rx'), num(v(2), 'orient ry'), num(v(3), 'orient rz')]; i += 4; }
      else if (key === 'units') {
        units = v(1);
        if (units !== 'box' && units !== 'lattice') throw new StyleError(`fix deposit: units must be lattice or box, got '${units}'`);
        i += 2;
      } else if (['var', 'set', 'gaussian'].includes(key)) {
        throw new StyleError(`fix deposit keyword '${key}' is not supported by the browser engine`);
      } else throw new StyleError(`fix deposit: unknown keyword '${key}'`);
    }
    if (region === null) throw new StyleError('fix deposit requires the region keyword');
    // fix_deposit.rst: "A lattice value means the distance units are in lattice spacings." and "the
    // units choice affects all the keyword values that have units of distance or velocity"; the
    // default is units lattice.
    const sc = latticeScale(sys, units ?? 'lattice', 'deposit');
    this.near *= sc[0];
    if (this.globalRange) this.globalRange = [this.globalRange[0] * sc[2], this.globalRange[1] * sc[2]];
    if (this.localRange) this.localRange = [this.localRange[0] * sc[2], this.localRange[1] * sc[2], this.localRange[2] * sc[0]];
    if (this.target) this.target = [this.target[0] * sc[0], this.target[1] * sc[1], this.target[2] * sc[2]];
    this.vx = [this.vx[0] * sc[0], this.vx[1] * sc[0]];
    this.vy = [this.vy[0] * sc[1], this.vy[1] * sc[1]];
    this.vz = [this.vz[0] * sc[2], this.vz[1] * sc[2]];
    if (this.globalRange && this.localRange) throw new StyleError('fix deposit: global and local cannot both be used');
    if (this.globalRange && !(this.globalRange[1] >= this.globalRange[0])) throw new StyleError('fix deposit: global needs lo <= hi');
    if (molTemplateId && (this.globalRange || this.localRange)) throw new StyleError('fix deposit: global and local with mol are not supported by the browser engine');
    if (orient && !molTemplateId) throw new StyleError('fix deposit: orient requires the mol keyword');
    if (rigidId && !molTemplateId) throw new StyleError('fix deposit: rigid requires the mol keyword');
    if (shakeId && !molTemplateId) throw new StyleError('fix deposit: shake requires the mol keyword');
    // fix_deposit.html: "type = atom type ... to assign to inserted atoms (offset for molecule insertion)"
    const typeInt = num(args[1], 'type');
    if (!Number.isInteger(typeInt) || typeInt < (molTemplateId ? 0 : 1)) {
      throw new StyleError(`fix deposit: type must be ${molTemplateId ? 'a non-negative integer offset' : 'a positive integer'}, got '${args[1]}'`);
    }
    this.type = typeInt;
    this.regionId = region;
    this.molTemplateId = molTemplateId;
    this.molfrac = molfrac;
    this.rigidId = rigidId;
    this.shakeId = shakeId;
    this.orient = orient;
    if (this.vx[0] > this.vx[1] || this.vy[0] > this.vy[1] || this.vz[0] > this.vz[1]) throw new StyleError('fix deposit: velocity ranges need lo <= hi');
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
      if (!isMolecularStyle(s.atomStyle)) throw new StyleError(`fix deposit mol: atom_style ${s.atomStyle} cannot store molecule IDs and bonds`);
      const sets = this.sys.molecules.get(this.molTemplateId);
      if (!sets || !sets.length) throw new StyleError(`fix deposit mol: molecule template '${this.molTemplateId}' does not exist`);
      const count = moleculeCount(this.sys, this.molTemplateId);
      if (count > 1) {
        throw new StyleError(`fix deposit mol: molecule template '${this.molTemplateId}' defines ${count} molecules (molfrac); only a single-molecule template is supported by the browser engine`);
      }
      this.molTemplate = sets[0];
      let maxT = 0;
      for (const ty of this.molTemplate.type) if (ty > maxT) maxT = ty;
      if (this.type + maxT > s.ntypes) {
        throw new StyleError(`fix deposit mol: type offset ${this.type} plus template type ${maxT} is larger than ntypes ${s.ntypes}`);
      }
    } else if (this.type > s.ntypes) throw new StyleError(`fix deposit: atom type ${this.type} is larger than ntypes ${s.ntypes}`);
    if (this.rigidId) {
      const rf = this.sys.fix(this.rigidId);
      if (typeof (rf as { addMolecule?: unknown }).addMolecule !== 'function') {
        throw new StyleError(`fix deposit rigid: fix ${this.rigidId} (${rf.style}) is not a fix rigid/small`);
      }
      this.rigidFix = rf;
    }
    if (this.shakeId) {
      const sf = this.sys.fix(this.shakeId);
      if (typeof (sf as { rebuildClusters?: unknown }).rebuildClusters !== 'function') {
        throw new StyleError(`fix deposit shake: fix ${this.shakeId} (${sf.style}) is not a fix shake`);
      }
      this.shakeFix = sf;
    }
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
    const t = this.molTemplate;
    const center = t ? geometricCenter(t) : [0, 0, 0];
    let placed: { p: Vec3; v: Vec3; atoms: Float64Array | null } | null = null;
    for (let a = 0; a < this.attempt && !placed; a++) {
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
      let atoms: Float64Array | null = null;
      if (t) {
        // one draw samples the molfrac list; a single-molecule template always takes molecule 0
        const sel = this.rng.uniform();
        if (this.molfrac) { let c = 0; for (const f of this.molfrac) { c += f; if (sel < c) break; } }
        const rot = this.orient
          ? rotationMatrix(2 * Math.PI * this.rng.uniform(), this.orient[0], this.orient[1], this.orient[2])
          : (() => {
            const u1 = this.rng.uniform(), u2 = this.rng.uniform(), u3 = this.rng.uniform(), u4 = this.rng.uniform();
            return rotationMatrix(2 * Math.PI * u4, u1 - 0.5, u2 - 0.5, u3 - 0.5);
          })();
        atoms = this.moleculeCoords(t, center, rot, [x, y, z]);
      }
      if (this.near > 0 && this.tooNear(atoms ?? Float64Array.of(x, y, z), t !== null)) continue;
      let v: Vec3 = [
        this.vx[0] + this.rng.uniform() * (this.vx[1] - this.vx[0]),
        this.vy[0] + this.rng.uniform() * (this.vy[1] - this.vy[0]),
        this.vz[0] + this.rng.uniform() * (this.vz[1] - this.vz[0]),
      ];
      if (this.target) v = this.aimAt([x, y, z], v);
      placed = { p: [x, y, z], v, atoms };
    }
    if (!placed) {
      sys.warn('Particle deposition was unsuccessful');
      return;
    }
    if (t && placed.atoms) {
      const n = t.natoms;
      const ids = this.idNext ? new Int32Array(n) : undefined;
      if (ids) for (let k = 0; k < n; k++) ids[k] = ++this.nextId;
      const molId = this.idNext ? ++this.nextMolId : maxMoleculeId(s) + 1;
      const firstId = appendMolecule(sys, t, this.type, placed.atoms, placed.v, molId, this.groupBit, ids);
      sys.atomsChanged();
      sys.setupNeighbors();
      if (this.rigidFix) (this.rigidFix as unknown as RigidAdder).addMolecule(Array.from({ length: n }, (_, k) => firstId + k));
      if (this.shakeFix) (this.shakeFix as unknown as ClusterRebuilder).rebuildClusters();
    } else {
      const xa = Float64Array.from(placed.p);
      const va = Float64Array.from(placed.v);
      const id = this.idNext ? Int32Array.of(++this.nextId) : undefined;
      appendAtoms(s, { x: xa, type: this.type, v: va, id, mask: this.groupBit });
      sys.atomsChanged();
      sys.setupNeighbors();
    }
    this.inserted++;
  }

  /** Flat 3N coordinates of the molecule rotated by `rot` about its geometric center and placed at p. */
  private moleculeCoords(t: MoleculeTemplate, center: number[], rot: number[][], p: Vec3): Float64Array {
    const out = new Float64Array(3 * t.natoms);
    for (let i = 0; i < t.natoms; i++) {
      const rx = t.x[3 * i] - center[0], ry = t.x[3 * i + 1] - center[1], rz = t.x[3 * i + 2] - center[2];
      out[3 * i] = p[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
      out[3 * i + 1] = p[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
      out[3 * i + 2] = p[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
    }
    return out;
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

  /**
   * True if any given point (a single atom, or every atom of a molecule when `isMolecule`) lies
   * closer than the near distance to a current atom (minimum image in periodic dimensions).
   */
  private tooNear(pts: Float64Array, isMolecule: boolean): boolean {
    const s = this.sys.state;
    const n = isMolecule ? pts.length / 3 : 1;
    for (let a = 0; a < n; a++) {
      const x = pts[3 * a], y = pts[3 * a + 1], z = pts[3 * a + 2];
      for (let i = 0; i < s.n; i++) {
        const [dx, dy, dz] = separation(this.sys, x - s.x[3 * i], y - s.x[3 * i + 1], z - s.x[3 * i + 2]);
        if (dx * dx + dy * dy + dz * dz < this.near * this.near) return true;
      }
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

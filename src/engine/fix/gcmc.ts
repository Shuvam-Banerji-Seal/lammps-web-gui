import { Fix } from './fix';
import { StyleError } from '../force/types';
import { RanPark } from '../rng';
import { appendAtoms, deleteAtoms, hasChargeStyle, isMolecularStyle } from '../atoms';
import { appendMolecule, num, posInt } from './pour';
import { geometricCenter, rotationMatrix, type MoleculeTemplate } from '../molecule';
import { regionBox, maxMoleculeId, topoList } from './widom';
import { BlockRegion, type Region } from '../region';
import { newAccum, type Bonded, type BondedCompute } from '../force/types';
import type { System } from '../system';
import type { SimState, Topology, TopoList } from '../types';

/*
 * fix ID group-ID gcmc N X M type seed T mu displace keyword values ...
 *   — docs.lammps.org/fix_gcmc.html (plans/lammps-docs/fix_gcmc.rst).
 *
 * "This fix performs grand canonical Monte Carlo (GCMC) exchanges of atoms or
 * molecules with an imaginary ideal gas reservoir at the specified T and
 * chemical potential (mu) as discussed in (Frenkel). It also attempts Monte
 * Carlo (MC) moves (translations and molecule rotations) within the simulation
 * cell or region." "Every N timesteps the fix attempts both GCMC exchanges
 * (insertions or deletions) and MC moves of gas atoms or molecules. On those
 * timesteps, the average number of attempted GCMC exchanges is X, while the
 * average number of attempted MC moves is M. For GCMC exchanges of either
 * molecular or atomic gasses, these exchanges can be either deletions or
 * insertions, with equal probability."
 *
 * Supported keywords: mol, mcmoves, region, maxangle, full_energy, charge,
 * group, grouptype, intra_energy, tfac_insert, overlap_cutoff, max, min.
 * rigid, shake, pressure and fugacity_coeff are rejected with a StyleError
 * naming them (they need a fix rigid/small, fix shake, or the per-unit
 * pressure/fugacity conversion the browser engine does not implement).
 *
 * Random stream: measured with native LAMMPS (black box) that the number of
 * attempts per event is exactly X exchanges and M moves (not a Poisson draw),
 * that the move type of each exchange is a fixed random draw, and that an
 * inserted atom's velocity comes from a separate generator. The engine does
 * NOT reproduce native's stream byte for byte (the choice/deletion draw order
 * depends on the occupied state in a way this work could not pin down); it
 * draws the documented algorithm from its own RanPark (seed, no discarded
 * draws) and is checked statistically (acceptance ratios, <N> vs mu) by
 * tests/engineGcmc32.test.ts. The one exact oracle, w32gcmc_move, uses
 * displace 0 so every translation has dU = 0 and is accepted regardless of the
 * stream: measured with native LAMMPS (black box) with X = 0, M = 1,
 * displace = 0, the vector counts one translation attempt and success every
 * step and the atoms and pe are unchanged.
 *
 * Acceptance (muVT detailed balance). Insertion with N particles and volume V
 * (the region volume with the region keyword, else the box volume) and thermal
 * de Broglie length Lambda (Lambda = 1 for units lj): the engine accepts with
 * probability min(1, V / ((N+1) Lambda^3) exp(beta (mu - dU))), deletion with
 * min(1, N Lambda^3 / V exp(-beta (mu + dU))), where dU is the potential-energy
 * change of the proposed move. u is the user chemical potential (docs:
 * "mu = chemical potential of the ideal gas reservoir (energy units)").
 *
 * Only units lj are accepted: for the other unit styles Lambda needs Planck's
 * constant, which the browser engine does not carry.
 */

type Vec3 = [number, number, number];

/** A non-negative integer keyword value (X, M and the max/min bounds allow 0). */
const nonNegInt = (w: string | undefined, what: string): number => {
  const v = num(w, what);
  if (!Number.isInteger(v) || v < 0) throw new StyleError(`fix gcmc: ${what} must be a non-negative integer, got '${w ?? ''}'`);
  return v;
};

const copyList = (l: TopoList): TopoList => ({ n: l.n, width: l.width, type: l.type.slice(), atoms: l.atoms.slice() });
const copyTopo = (t: Topology): Topology => ({
  nbondtypes: t.nbondtypes, nangletypes: t.nangletypes, ndihedraltypes: t.ndihedraltypes, nimpropertypes: t.nimpropertypes,
  bonds: copyList(t.bonds), angles: copyList(t.angles), dihedrals: copyList(t.dihedrals), impropers: copyList(t.impropers),
});

/** Full per-atom snapshot used to undo a rejected insertion or deletion. */
interface Snap {
  n: number;
  x: Float64Array; v: Float64Array; f: Float64Array;
  image: Int32Array; id: Int32Array; type: Int32Array; mask: Int32Array; molecule: Int32Array; q: Float64Array; order: Int32Array;
  rmass: Float64Array | null; radius: Float64Array | null; omega: Float64Array | null; torque: Float64Array | null;
  mu: Float64Array | null; shape: Float64Array | null; quat: Float64Array | null; angmom: Float64Array | null;
  vfrac: Float64Array | null; x0: Float64Array | null; tmplIndex: Int32Array | null; tmplAtom: Int32Array | null;
  custom: Map<string, Float64Array>;
  topo: Topology;
}

const capture = (s: SimState): Snap => ({
  n: s.n,
  x: s.x.slice(), v: s.v.slice(), f: s.f.slice(), image: s.image.slice(), id: s.id.slice(), type: s.type.slice(),
  mask: s.mask.slice(), molecule: s.molecule.slice(), q: s.q.slice(), order: s.order.slice(),
  rmass: s.rmass?.slice() ?? null, radius: s.radius?.slice() ?? null, omega: s.omega?.slice() ?? null, torque: s.torque?.slice() ?? null,
  mu: s.mu?.slice() ?? null, shape: s.shape?.slice() ?? null, quat: s.quat?.slice() ?? null, angmom: s.angmom?.slice() ?? null,
  vfrac: s.vfrac?.slice() ?? null, x0: s.x0?.slice() ?? null, tmplIndex: s.tmplIndex?.slice() ?? null, tmplAtom: s.tmplAtom?.slice() ?? null,
  custom: new Map([...s.custom].map(([k, c]) => [k, c.data.slice()])),
  topo: copyTopo(s.topo),
});

const restore = (s: SimState, k: Snap): void => {
  s.n = k.n;
  s.x = k.x; s.v = k.v; s.f = k.f; s.image = k.image; s.id = k.id; s.type = k.type; s.mask = k.mask;
  s.molecule = k.molecule; s.q = k.q; s.order = k.order;
  s.rmass = k.rmass; s.radius = k.radius; s.omega = k.omega; s.torque = k.torque;
  s.mu = k.mu; s.shape = k.shape; s.quat = k.quat; s.angmom = k.angmom; s.vfrac = k.vfrac; s.x0 = k.x0;
  s.tmplIndex = k.tmplIndex; s.tmplAtom = k.tmplAtom;
  for (const [name, data] of k.custom) { const c = s.custom.get(name); if (c) c.data = data; }
  s.topo = k.topo;
};

export class FixGcmc extends Fix {
  readonly style = 'gcmc';
  readonly N: number;
  readonly X: number;
  readonly M: number;
  readonly type: number;
  readonly T: number;
  private readonly mu: number;
  private readonly displace: number;
  private readonly rng: RanPark;
  private readonly molTemplateId: string | null;
  private molTemplate: MoleculeTemplate | null = null;
  private readonly regionId: string | null;
  private reg: { lo: Vec3; hi: Vec3; volume: number } | null = null;
  private region: Region | null = null;
  private fullEnergy = false;
  private readonly charge: number | null;
  private readonly intraEnergy: number;
  private intraMolEnergy = 0;
  private readonly tfacInsert: number;
  private readonly overlapCutoff: number;
  private readonly maxAtoms: number | null;
  private readonly minAtoms: number | null;
  private readonly groupId: string | null;
  private readonly groupTypes: { type: number; group: string }[] = [];
  private patomtrans = 1;
  private pmoltrans = 0;
  private pmolrotate = 0;
  private maxangle = (10 * Math.PI) / 180;
  private nextStep: number;
  private ntransAtt = 0;
  private ntransSucc = 0;
  private ninsAtt = 0;
  private ninsSucc = 0;
  private ndelAtt = 0;
  private ndelSucc = 0;
  private nrotAtt = 0;
  private nrotSucc = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (sys.dimension !== 3) throw new StyleError('fix gcmc: only 3d simulations are supported');
    if (sys.state.units.style !== 'lj') {
      throw new StyleError('fix gcmc: only the lj unit style is supported by the browser engine (the thermal de Broglie length for other units needs Planck\u2019s constant)');
    }
    if (args.length < 8) throw new StyleError('usage: fix ID group-ID gcmc N X M type seed T mu displace keyword values ...');
    this.N = posInt(args[0], 'N');
    this.X = nonNegInt(args[1], 'X');
    this.M = nonNegInt(args[2], 'M');
    this.rng = new RanPark(posInt(args[4], 'seed'));
    this.T = num(args[5], 'T');
    if (!(this.T > 0)) throw new StyleError('fix gcmc: T must be > 0');
    this.mu = num(args[6], 'mu');
    this.displace = num(args[7], 'displace');
    if (this.displace < 0) throw new StyleError('fix gcmc: displace must be >= 0');
    let molTemplateId: string | null = null;
    let regionId: string | null = null;
    let fullEnergy = false;
    let charge: number | null = null;
    let intraEnergy = 0;
    let tfacInsert = 1;
    let overlapCutoff = 0;
    let maxAtoms: number | null = null;
    let minAtoms: number | null = null;
    let groupId: string | null = null;
    let sawMcmoves = false;
    let i = 8;
    while (i < args.length) {
      const key = args[i];
      const v = (k: number) => {
        if (args[i + k] === undefined) throw new StyleError(`fix gcmc: keyword '${key}' needs ${k} value(s)`);
        return args[i + k];
      };
      if (key === 'mol') { molTemplateId = v(1); i += 2; }
      else if (key === 'mcmoves') {
        this.patomtrans = nonNegInt(v(1), 'mcmoves Patomtrans');
        this.pmoltrans = nonNegInt(v(2), 'mcmoves Pmoltrans');
        this.pmolrotate = nonNegInt(v(3), 'mcmoves Pmolrotate');
        if (this.patomtrans + this.pmoltrans + this.pmolrotate === 0) throw new StyleError('fix gcmc mcmoves: at least one proportion must be non-zero');
        sawMcmoves = true;
        i += 4;
      } else if (key === 'region') { regionId = v(1); i += 2; }
      else if (key === 'maxangle') { this.maxangle = (num(v(1), 'maxangle') * Math.PI) / 180; i += 2; }
      else if (key === 'full_energy') { fullEnergy = true; i += 1; }
      else if (key === 'charge') { charge = num(v(1), 'charge'); i += 2; }
      else if (key === 'intra_energy') { intraEnergy = num(v(1), 'intra_energy'); i += 2; }
      else if (key === 'tfac_insert') { tfacInsert = num(v(1), 'tfac_insert'); i += 2; }
      else if (key === 'overlap_cutoff') { overlapCutoff = num(v(1), 'overlap_cutoff'); i += 2; }
      else if (key === 'max') { maxAtoms = nonNegInt(v(1), 'max'); i += 2; }
      else if (key === 'min') { minAtoms = nonNegInt(v(1), 'min'); i += 2; }
      else if (key === 'group') { groupId = v(1); i += 2; }
      else if (key === 'grouptype') { this.groupTypes.push({ type: nonNegInt(v(1), 'grouptype type'), group: v(2) }); i += 3; }
      else if (key === 'rigid' || key === 'shake' || key === 'pressure' || key === 'fugacity_coeff') {
        throw new StyleError(`fix gcmc keyword '${key}' is not supported by the browser engine`);
      } else throw new StyleError(`fix gcmc: unknown keyword '${key}'`);
    }
    const typeInt = num(args[3], 'type');
    if (!Number.isInteger(typeInt)) throw new StyleError(`fix gcmc: type must be an integer, got '${args[3]}'`);
    if (molTemplateId) {
      if (typeInt !== 0) throw new StyleError('fix gcmc: type must be 0 when the mol keyword is used');
    } else if (typeInt < 1) throw new StyleError('fix gcmc: type must be a positive integer without the mol keyword');
    this.type = typeInt;
    this.molTemplateId = molTemplateId;
    this.regionId = regionId;
    this.fullEnergy = fullEnergy;
    this.charge = charge;
    this.intraEnergy = intraEnergy;
    this.tfacInsert = tfacInsert;
    this.overlapCutoff = overlapCutoff;
    this.maxAtoms = maxAtoms;
    this.minAtoms = minAtoms;
    this.groupId = groupId;
    // docs fix_gcmc.html defaults: "(Patomtrans, Pmoltrans, Pmolrotate) = (1, 0, 0) for mol = no and (0, 1, 1) for mol = yes"
    if (!sawMcmoves && molTemplateId) { this.patomtrans = 0; this.pmoltrans = 1; this.pmolrotate = 1; }
    if (!molTemplateId && (this.pmoltrans !== 0 || this.pmolrotate !== 0)) {
      throw new StyleError('fix gcmc: molecule translation/rotation moves need the mol keyword');
    }
    if (molTemplateId && this.patomtrans !== 0) throw new StyleError('fix gcmc: atom translation moves are not allowed with the mol keyword');
    this.nextStep = sys.hasBox ? sys.state.step + 1 : 1;
    this.vectorFlag = true;
    this.sizeVector = 8;
    this.extvector = 0;
    this.extscalar = 0;
  }

  init(): void {
    const s = this.sys.state;
    const pair = this.sys.ff.pair;
    if (!pair) throw new StyleError('fix gcmc requires a pair style');
    const auto = !!this.sys.ff.kspace || pair.coulLong || pair.tail || pair.manybody
      || pair.name === 'eam' || pair.name.startsWith('eam/');
    if (auto && !this.fullEnergy) {
      this.fullEnergy = true;
      this.sys.warn('fix gcmc: full_energy is required for this pair style and was enabled automatically');
    }
    if (this.molTemplateId) {
      if (!isMolecularStyle(s.atomStyle)) throw new StyleError(`fix gcmc mol: atom_style ${s.atomStyle} cannot store molecule IDs and bonds`);
      const sets = this.sys.molecules.get(this.molTemplateId);
      if (!sets || !sets.length) throw new StyleError(`fix gcmc mol: molecule template '${this.molTemplateId}' does not exist`);
      this.molTemplate = sets[0];
      if (sets.length > 1 || new Set(this.molTemplate.mol ?? []).size > 1) {
        throw new StyleError(`fix gcmc mol: molecule template '${this.molTemplateId}' defines more than one molecule, which the browser engine does not support`);
      }
      for (const ty of this.molTemplate.type) if (ty < 1 || ty > s.ntypes) throw new StyleError(`fix gcmc mol: molecule template atom type ${ty} is outside 1..${s.ntypes}`);
      if (this.molTemplate.q && !hasChargeStyle(s.atomStyle)) throw new StyleError(`fix gcmc mol: molecule template '${this.molTemplateId}' has charges, which atom_style ${s.atomStyle} cannot store`);
      if (!this.fullEnergy) this.intraMolEnergy = this.moleculeIntraEnergy(this.molTemplate);
    } else {
      if (this.type > s.ntypes) throw new StyleError(`fix gcmc: atom type ${this.type} is larger than ntypes ${s.ntypes}`);
      if (this.charge !== null && !hasChargeStyle(s.atomStyle)) throw new StyleError(`fix gcmc charge: atom_style ${s.atomStyle} cannot store a charge`);
    }
    if (this.maxAtoms !== null && this.minAtoms !== null && this.minAtoms > this.maxAtoms) throw new StyleError('fix gcmc: min cannot be larger than max');
    if (this.groupId) this.sys.groups.bit(this.groupId);
    for (const g of this.groupTypes) if (g.type < 1 || g.type > s.ntypes) throw new StyleError(`fix gcmc grouptype: type ${g.type} is outside 1..${s.ntypes}`);
    if (this.regionId) {
      const r = this.sys.region(this.regionId);
      if (!r.interior) throw new StyleError('fix gcmc: the region must be defined with side in');
      if (r.dynamic) throw new StyleError('fix gcmc: a dynamic region (move or rotate) is not supported');
      this.region = r;
      this.reg = regionBox(this.sys, r);
    } else {
      this.reg = null;
    }
  }

  /** The run loop calls preExchange only on neighbour rebuilds; it is a stub (see fix pour). */
  preExchange(): void { /* MC changes are done in postIntegrate */ }

  postIntegrate(): void {
    if (this.sys.state.step !== this.nextStep) return;
    this.event();
    this.nextStep += this.N;
  }

  /** One Monte Carlo event: X exchange attempts then M move attempts. */
  private event(): void {
    for (let i = 0; i < this.X; i++) {
      if (this.rng.uniform() < 0.5) this.deletion();
      else this.insertion();
    }
    for (let i = 0; i < this.M; i++) this.move();
    // docs fix_gcmc.html: "Note that neighbor lists are re-built every timestep that this fix is
    // invoked, so you should not set N to be too small." Rebuild for the final atom count so the
    // run loop's force evaluation (which may decide not to reneighbor) sees the exchanged atoms.
    this.sys.setupNeighbors();
  }

  private get volume(): number {
    if (this.reg) return this.reg.volume;
    const b = this.sys.state.box;
    return (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]);
  }

  /** Random insertion point: the box, or the region bounding box with rejection for non-rectangular regions. */
  private drawPosition(): Vec3 | null {
    const s = this.sys.state;
    const lo = this.reg ? this.reg.lo : (s.box.lo as unknown as Vec3);
    const hi = this.reg ? this.reg.hi : (s.box.hi as unknown as Vec3);
    if (!this.region || this.region instanceof BlockRegion) {
      return [lo[0] + this.rng.uniform() * (hi[0] - lo[0]), lo[1] + this.rng.uniform() * (hi[1] - lo[1]), lo[2] + this.rng.uniform() * (hi[2] - lo[2])];
    }
    // docs fix_gcmc.html: "For non-rectangular regions, random trial points are generated within the
    // rectangular bounding box until a point is found that lies inside the region. If no valid point
    // is generated after 1000 trials, no insertion is performed, but it is counted as an attempted insertion."
    for (let t = 0; t < 1000; t++) {
      const p: Vec3 = [lo[0] + this.rng.uniform() * (hi[0] - lo[0]), lo[1] + this.rng.uniform() * (hi[1] - lo[1]), lo[2] + this.rng.uniform() * (hi[2] - lo[2])];
      if (this.region.match(p[0], p[1], p[2])) return p;
    }
    return null;
  }

  private potentialEnergy(): number {
    const a = this.sys.forces();
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp;
  }

  /** Atoms of the fix group inside the region. */
  private eligibleAtoms(): number[] {
    const s = this.sys.state;
    const out: number[] = [];
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) === 0) continue;
      if (this.region && !this.region.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      out.push(i);
    }
    return out;
  }

  /** Distinct molecule IDs of group atoms inside the region. */
  private eligibleMolecules(): number[] {
    const s = this.sys.state;
    const seen = new Set<number>();
    for (const i of this.eligibleAtoms()) if (s.molecule[i] > 0) seen.add(s.molecule[i]);
    return [...seen];
  }

  /** Number of exchanged particles: atoms, or distinct molecules with the mol keyword. */
  private particleCount(): number {
    if (this.molTemplateId) return this.eligibleMolecules().length;
    return this.eligibleAtoms().length;
  }

  private extraMask(): number {
    let m = this.groupBit;
    if (this.groupId) m |= this.sys.groups.bit(this.groupId);
    return m;
  }

  private drawVelocity(mass: number): Vec3 {
    const sd = Math.sqrt((this.sys.state.units.boltz * this.T * this.tfacInsert) / mass);
    return [sd * this.rng.gaussian(), sd * this.rng.gaussian(), sd * this.rng.gaussian()];
  }

  private metropolis(exponent: number): boolean {
    return this.rng.uniform() < Math.exp(Math.min(0, exponent));
  }

  private insertion(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ninsAtt++;
    const p = this.drawPosition();
    if (!p) return;
    const nPart = this.particleCount();
    const add = this.molTemplate ? this.molTemplate.natoms : 1;
    if (this.maxAtoms !== null && this.groupAtomCount() + add > this.maxAtoms) return;
    const eBefore = this.potentialEnergy();
    const snap = capture(s);
    const kT = s.units.boltz * this.T;
    let mass = 0;
    if (this.molTemplate) {
      const t = this.molTemplate;
      const rot = this.randomRotation(Math.PI * 2);
      const center = geometricCenter(t);
      const coords = new Float64Array(3 * t.natoms);
      for (let a = 0; a < t.natoms; a++) {
        const rx = t.x[3 * a] - center[0], ry = t.x[3 * a + 1] - center[1], rz = t.x[3 * a + 2] - center[2];
        coords[3 * a] = p[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
        coords[3 * a + 1] = p[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
        coords[3 * a + 2] = p[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
      }
      for (const ty of t.type) mass += s.massByType[ty];
      const vel = this.drawVelocity(mass);
      appendMolecule(sys, t, 0, coords, vel, maxMoleculeId(s) + 1, this.extraMask());
      sys.atomsChanged();
    } else {
      const vel = this.drawVelocity(s.massByType[this.type]);
      appendAtoms(s, { x: Float64Array.from(p), type: this.type, v: Float64Array.from(vel), q: this.charge ?? undefined, mask: this.extraMask() });
      sys.atomsChanged();
    }
    let dU = this.potentialEnergy() - eBefore;
    if (!this.fullEnergy) dU -= this.molTemplate ? this.intraMolEnergy : 0;
    dU -= this.intraEnergy;
    const overlap = this.overlapCutoff > 0 && this.hasOverlap(snap.n, this.overlapCutoff);
    const exponent = (this.mu - dU) / kT + Math.log(this.volume / (nPart + 1));
    if (!overlap && this.metropolis(exponent)) {
      this.ninsSucc++;
      this.applyGroupTypes(snap.n);
    } else {
      restore(s, snap);
      sys.atomsChanged();
    }
  }

  private deletion(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ndelAtt++;
    let idx: number[];
    if (this.molTemplate) {
      const mols = this.eligibleMolecules();
      if (!mols.length) return;
      const mid = mols[Math.floor(this.rng.uniform() * mols.length)];
      idx = [];
      for (let i = 0; i < s.n; i++) if (s.molecule[i] === mid) idx.push(i);
    } else {
      const cands = this.eligibleAtoms();
      if (!cands.length) return;
      idx = [cands[Math.floor(this.rng.uniform() * cands.length)]];
    }
    const nPart = this.particleCount();
    if (this.minAtoms !== null && this.groupAtomCount() - idx.length < this.minAtoms) return;
    const eBefore = this.potentialEnergy();
    const snap = capture(s);
    const flags = new Uint8Array(s.n);
    for (const i of idx) flags[i] = 1;
    deleteAtoms(s, flags);
    sys.atomsChanged();
    let dU = this.potentialEnergy() - eBefore;
    if (!this.fullEnergy) dU -= this.molTemplate ? this.intraMolEnergy : 0;
    dU -= this.intraEnergy;
    const kT = s.units.boltz * this.T;
    const exponent = -(this.mu + dU) / kT + Math.log(nPart / this.volume);
    if (this.metropolis(exponent)) {
      this.ndelSucc++;
    } else {
      restore(s, snap);
      sys.atomsChanged();
    }
  }

  /** True if any atom appended after n0 lies closer than the cutoff to a pre-existing atom. */
  private hasOverlap(n0: number, cutoff: number): boolean {
    const s = this.sys.state;
    const c2 = cutoff * cutoff;
    for (let i = n0; i < s.n; i++) {
      for (let j = 0; j < n0; j++) {
        const dx = s.x[3 * i] - s.x[3 * j], dy = s.x[3 * i + 1] - s.x[3 * j + 1], dz = s.x[3 * i + 2] - s.x[3 * j + 2];
        if (dx * dx + dy * dy + dz * dz < c2) return true;
      }
    }
    return false;
  }

  /** Adds the grouptype group bits to the atoms appended after n0. */
  private applyGroupTypes(n0: number): void {
    const s = this.sys.state;
    for (let i = n0; i < s.n; i++) {
      for (const g of this.groupTypes) if (g.type === s.type[i]) s.mask[i] |= this.sys.groups.bit(g.group);
    }
  }

  /** Number of atoms (not molecules) in the fix group inside the region, for max/min. */
  private groupAtomCount(): number {
    return this.eligibleAtoms().length;
  }

  /** Random rotation: axis by rejection in the unit sphere, angle 2 pi u (insertion) or [0, maxangle). */
  private randomRotation(angle: number): number[][] {
    let ax: number, ay: number, az: number;
    do {
      ax = 2 * this.rng.uniform() - 1;
      ay = 2 * this.rng.uniform() - 1;
      az = 2 * this.rng.uniform() - 1;
    } while (ax * ax + ay * ay + az * az >= 1 || (ax === 0 && ay === 0 && az === 0));
    const theta = this.rng.uniform() * angle;
    return rotationMatrix(theta, ax, ay, az);
  }

  /** One MC move: atom translation, molecule translation, or molecule rotation. */
  private move(): void {
    const total = this.patomtrans + this.pmoltrans + this.pmolrotate;
    if (total <= 0) return;
    const u = this.rng.uniform() * total;
    if (u < this.patomtrans) this.translateAtom();
    else if (u < this.patomtrans + this.pmoltrans) this.translateMolecule();
    else this.rotateMolecule();
  }

  private translateAtom(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ntransAtt++;
    const cands = this.eligibleAtoms();
    if (!cands.length) return;
    const i = cands[Math.floor(this.rng.uniform() * cands.length)];
    const old: Vec3 = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
    let nw: Vec3;
    let tries = 0;
    do {
      nw = [old[0] + (2 * this.rng.uniform() - 1) * this.displace, old[1] + (2 * this.rng.uniform() - 1) * this.displace, old[2] + (2 * this.rng.uniform() - 1) * this.displace];
    } while (this.region && !this.region.match(nw[0], nw[1], nw[2]) && ++tries < 1000);
    const eBefore = this.potentialEnergy();
    for (let d = 0; d < 3; d++) s.x[3 * i + d] = nw[d];
    sys.bump();
    const dU = this.potentialEnergy() - eBefore;
    if (this.metropolis(-dU / (s.units.boltz * this.T))) {
      this.ntransSucc++;
    } else {
      for (let d = 0; d < 3; d++) s.x[3 * i + d] = old[d];
      sys.bump();
    }
  }

  private translateMolecule(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ntransAtt++;
    const mols = this.eligibleMolecules();
    if (!mols.length) return;
    const mid = mols[Math.floor(this.rng.uniform() * mols.length)];
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.molecule[i] === mid) idx.push(i);
    const com = this.moleculeCom(idx);
    let nw: Vec3;
    let tries = 0;
    do {
      nw = [com[0] + (2 * this.rng.uniform() - 1) * this.displace, com[1] + (2 * this.rng.uniform() - 1) * this.displace, com[2] + (2 * this.rng.uniform() - 1) * this.displace];
    } while (this.region && !this.region.match(nw[0], nw[1], nw[2]) && ++tries < 1000);
    const shift: Vec3 = [nw[0] - com[0], nw[1] - com[1], nw[2] - com[2]];
    const old = this.snapshotCoords(idx);
    const eBefore = this.potentialEnergy();
    for (const i of idx) for (let d = 0; d < 3; d++) s.x[3 * i + d] += shift[d];
    sys.bump();
    const dU = this.potentialEnergy() - eBefore;
    if (this.metropolis(-dU / (s.units.boltz * this.T))) {
      this.ntransSucc++;
    } else {
      for (let k = 0; k < idx.length; k++) for (let d = 0; d < 3; d++) s.x[3 * idx[k] + d] = old[3 * k + d];
      sys.bump();
    }
  }

  private rotateMolecule(): void {
    const sys = this.sys;
    const s = sys.state;
    this.nrotAtt++;
    const mols = this.eligibleMolecules();
    if (!mols.length) return;
    const mid = mols[Math.floor(this.rng.uniform() * mols.length)];
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.molecule[i] === mid) idx.push(i);
    const com = this.moleculeCom(idx);
    const rot = this.randomRotation(this.maxangle);
    const old = this.snapshotCoords(idx);
    const eBefore = this.potentialEnergy();
    for (const i of idx) {
      const rx = s.x[3 * i] - com[0], ry = s.x[3 * i + 1] - com[1], rz = s.x[3 * i + 2] - com[2];
      s.x[3 * i] = com[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
      s.x[3 * i + 1] = com[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
      s.x[3 * i + 2] = com[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
    }
    sys.bump();
    const dU = this.potentialEnergy() - eBefore;
    if (this.metropolis(-dU / (s.units.boltz * this.T))) {
      this.nrotSucc++;
    } else {
      for (let k = 0; k < idx.length; k++) for (let d = 0; d < 3; d++) s.x[3 * idx[k] + d] = old[3 * k + d];
      sys.bump();
    }
  }

  private moleculeCom(idx: number[]): Vec3 {
    const s = this.sys.state;
    const c: Vec3 = [0, 0, 0];
    for (const i of idx) for (let d = 0; d < 3; d++) c[d] += s.x[3 * i + d];
    return [c[0] / idx.length, c[1] / idx.length, c[2] / idx.length];
  }

  private snapshotCoords(idx: number[]): Float64Array {
    const s = this.sys.state;
    const out = new Float64Array(3 * idx.length);
    idx.forEach((i, k) => { for (let d = 0; d < 3; d++) out[3 * k + d] = s.x[3 * i + d]; });
    return out;
  }

  /**
   * The inserted molecule's intramolecular energy (bonds, angles, dihedrals,
   * impropers), evaluated on the rigid template coordinates, subtracted when
   * full_energy is off ("this energy is not included when full_energy is not
   * used", fix_gcmc.rst).
   */
  private moleculeIntraEnergy(t: MoleculeTemplate): number {
    const n = t.natoms;
    const topo = {
      bonds: topoList(t.bonds, 2), angles: topoList(t.angles, 3),
      dihedrals: topoList(t.dihedrals, 4), impropers: topoList(t.impropers, 4),
      nbondtypes: 0, nangletypes: 0, ndihedraltypes: 0, nimpropertypes: 0,
    };
    const fake = { n, x: Float64Array.from(t.x), topo } as unknown as SimState;
    const map = new Int32Array(n + 1).fill(-1);
    for (let i = 0; i < n; i++) map[i + 1] = i;
    const acc = newAccum();
    const f = new Float64Array(3 * n);
    const bc: BondedCompute = { s: fake, geom: this.sys.geom, map, f, acc, virial: acc.vbond, eatom: null, vatom: null, warn: () => {} };
    const run = (style: Bonded | null, virial: Float64Array) => { if (style) { bc.virial = virial; style.compute(bc); } };
    run(this.sys.ff.bond, acc.vbond);
    run(this.sys.ff.angle, acc.vangle);
    run(this.sys.ff.dihedral, acc.vdihed);
    run(this.sys.ff.improper, acc.vimp);
    return acc.ebond + acc.eangle + acc.edihed + acc.eimp;
  }

  computeVector(i: number): number {
    const v = [this.ntransAtt, this.ntransSucc, this.ninsAtt, this.ninsSucc, this.ndelAtt, this.ndelSucc, this.nrotAtt, this.nrotSucc];
    if (i < 0 || i >= 8) throw new StyleError(`fix ${this.id} does not compute vector element ${i + 1}`);
    return v[i];
  }
}

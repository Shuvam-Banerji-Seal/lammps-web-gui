import { Fix } from './fix';
import { StyleError } from '../force/types';
import { RanPark } from '../rng';
import { appendAtoms, deleteAtoms, hasChargeStyle, isMolecularStyle } from '../atoms';
import { appendMolecule, num, posInt } from './pour';
import { rotationMatrix, type MoleculeTemplate } from '../molecule';
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
 * group, grouptype, intra_energy, tfac_insert, overlap_cutoff, max, min, shake.
 * rigid, pressure and fugacity_coeff are rejected with a StyleError naming
 * them (they need a fix rigid/small or the per-unit pressure/fugacity
 * conversion the browser engine does not implement).
 *
 * shake (docs.lammps.org/fix_gcmc.html, plans/lammps-docs/fix_gcmc.rst): "If
 * you wish to insert molecules via the *mol* keyword, that will have their
 * bonds or angles constrained via SHAKE, use the *shake* keyword, specifying
 * as its value the ID of a separate fix shake command which also appears in
 * your input script." "When using fix gcmc in combination with fix shake or
 * fix rigid, only GCMC exchange moves are supported, so the argument *M* must
 * be zero." Measured with native LAMMPS (black box): with shake, M > 0 stops
 * with Cannot use fix gcmc shake with MC moves; without mol, with Cannot use
 * fix gcmc shake and not molecule; a fix shake whose mol template differs (or
 * has none) gives Fix gcmc and fix shake not using same molecule template ID.
 * Registration (measured with native LAMMPS, black box, w37gcmc_shake_ins and
 * w37gcmc_shake_rej): the trial insertion's energy includes the new molecule's
 * bonds (with full_energy and a stretched template at mu 0 every trial is
 * rejected); the accepted molecule keeps its bonds in the thermo energy of
 * the inserting step and is constrained from the next step on, so the engine
 * registers it in postIntegrate of that step; a deletion rebuilds the clusters
 * at once; a rejected trial restores them. Molecule placement (measured,
 * w37gcmc_* water and dimer, which match native to 1e-12): the centre of mass
 * (mass weighted) is placed at the insertion point, and the velocities of an
 * accepted molecule are the Gaussians of the second stream from its draw 11
 * (nine draws follow the acceptance draw). Not matched: the exchange stream of
 * a second molecule event (native's choice sequence differs from the engine's
 * after the first insertion of a mol run), so multi-event mol runs with shake
 * (and without) are not oracle-checked. Native's mid-run SHAKE for a
 * three-atom water cluster inserted by gcmc or fix deposit collapses the H-H
 * distance (it is not constrained as a rigid angle); the engine constrains it
 * as the documented SHAKE does, so water inserted mid-run differs from native.
 *
 * Acceptance (muVT detailed balance). Insertion with N particles and volume V
 * (the region volume with the region keyword, else the box volume) and thermal
 * de Broglie length Lambda: the engine accepts with probability
 * min(1, V / ((N+1) Lambda^3) exp(beta (mu - dU))), deletion with
 * min(1, N Lambda^3 / V exp(-beta (mu + dU))), where dU is the potential-energy
 * change of the proposed move. u is the user chemical potential (docs:
 * "mu = chemical potential of the ideal gas reservoir (energy units)").
 *
 * Lambda is from docs.lammps.org/fix_gcmc.html: "For all unit styles except
 * *lj* it is defined as the thermal de Broglie wavelength" Lambda =
 * sqrt(h^2 / (2 pi m k_B T)) "where *h* is Planck's constant, and *m* is the
 * mass of the exchanged atom or molecule. For unit style *lj*, Lambda is
 * simply set to unity." h is the style's Planck constant (units.ts hplanck),
 * m the exchanged particle's mass (atom mass, or the molecule's total mass).
 *
 * Random streams, measured with native LAMMPS (black box). The number of
 * attempts per event is exactly X exchanges and M moves (not a Poisson draw).
 * There are two Park-Miller streams, both seeded with the fix's seed:
 *  - the exchange stream skips one draw at creation; each exchange then takes
 *    the choice draw (insertion when u >= 0.5, deletion otherwise), then for
 *    an insertion three position draws, for a deletion the candidate-index
 *    draw (only when there is a candidate), and in both cases one more draw
 *    where the decision is made (also for a deletion with no candidate);
 *  - the second stream holds the decisions: one uniform per accept/reject
 *    test (accepted when u < min(1, P)), none for a deletion with no
 *    candidate, and after an accepted insertion the three velocity
 *    components, each insertion starting a new polar pair.
 * Evidence: bisecting mu at the first insertion into an empty box puts the
 * threshold exactly at the first draw of the second stream (units real and
 * metal, see units.ts hplanck); the first four inserted atoms' velocities in
 * w33gcmc_lj_acc are Gaussians starting at its draws 1, 6, 11 and 16,
 * times sqrt(k_B T / (m mvv2e)); and w33gcmc_lj_acc (lj, dU = 0) and
 * w33gcmc_real_acc (real, interacting argon) match native's atom count,
 * energy and kinetic energy at every step, through 120 and 150 exchanges with
 * both acceptances and rejections. Translations and molecule moves use the
 * same decision rule but are checked exactly only by w32gcmc_move, which uses
 * displace 0 so every translation has dU = 0 and is accepted: measured with
 * native LAMMPS (black box) with X = 0, M = 1, displace = 0, the vector counts
 * one translation attempt and success every step and the atoms and pe are
 * unchanged. tests/engineGcmc32.test.ts checks acceptance ratios and <N> vs mu
 * statistically.
 */

type Vec3 = [number, number, number];

/** What fix gcmc needs from a fix shake (its mol template and its cluster rebuild). */
interface ShakeClusters {
  rebuildClusters(): void;
  extendTopology(): void;
  moleculeTemplateId(): string | null;
  style: string;
}

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
  /** Second stream with the same seed: acceptance numbers and inserted velocities (see header). */
  private readonly accRng: RanPark;
  private readonly molTemplateId: string | null;
  private molTemplate: MoleculeTemplate | null = null;
  private readonly shakeId: string | null;
  private shakeFix: ShakeClusters | null = null;
  /** An accepted insertion waits for fix shake to register it at the next step (see postIntegrate). */
  private shakePending = false;
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
    if (args.length < 8) throw new StyleError('usage: fix ID group-ID gcmc N X M type seed T mu displace keyword values ...');
    this.N = posInt(args[0], 'N');
    this.X = nonNegInt(args[1], 'X');
    this.M = nonNegInt(args[2], 'M');
    const seed = posInt(args[4], 'seed');
    this.rng = new RanPark(seed);
    // Measured with native LAMMPS (black box): the fix consumes one draw before its first
    // exchange choice; consuming it here keeps the choice and insertion-position stream aligned
    // with native (see the header).
    this.rng.uniform();
    this.accRng = new RanPark(seed);
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
    let shakeId: string | null = null;
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
      else if (key === 'shake') { shakeId = v(1); i += 2; }
      else if (key === 'rigid' || key === 'pressure' || key === 'fugacity_coeff') {
        throw new StyleError(`fix gcmc keyword '${key}' is not supported by the browser engine`);
      } else throw new StyleError(`fix gcmc: unknown keyword '${key}'`);
    }
    if (shakeId) {
      if (!molTemplateId) throw new StyleError('fix gcmc: Cannot use fix gcmc shake and not molecule (shake needs the mol keyword)');
      // docs fix_gcmc.html: "When using fix gcmc in combination with fix shake or fix rigid, only GCMC exchange moves are supported, so the argument *M* must be zero."
      if (this.M !== 0) throw new StyleError('fix gcmc: Cannot use fix gcmc shake with MC moves (M must be zero with shake)');
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
    this.shakeId = shakeId;
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
    if (this.shakeId) {
      const sf = this.sys.fix(this.shakeId) as unknown as Partial<ShakeClusters> & { style: string };
      if (typeof sf.rebuildClusters !== 'function' || typeof sf.extendTopology !== 'function' || typeof sf.moleculeTemplateId !== 'function') {
        throw new StyleError(`fix gcmc shake: fix ${this.shakeId} (${sf.style}) is not a fix shake`);
      }
      if (sf.moleculeTemplateId() !== this.molTemplateId) {
        throw new StyleError('fix gcmc: Fix gcmc and fix shake not using same molecule template ID');
      }
      this.shakeFix = sf as ShakeClusters;
    }
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
    // an accepted insertion is registered with fix shake at the next step, before its forces
    // (measured with native LAMMPS, black box: the bonds of the step that inserted the molecule are
    // still counted in its thermo energy, and they are constrained from the next step on)
    if (this.shakePending) {
      this.shakeFix!.rebuildClusters();
      this.shakePending = false;
    }
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
    const u = this.sys.state.units;
    // kT / m is energy per mass; mvv2e turns it into the style's velocity^2
    const sd = Math.sqrt((u.boltz * this.T * this.tfacInsert) / (mass * u.mvv2e));
    const v: Vec3 = [sd * this.accRng.gaussian(), sd * this.accRng.gaussian(), sd * this.accRng.gaussian()];
    // Measured with native LAMMPS (black box): each insertion starts a new polar pair, so the
    // unused second value of the last pair is dropped.
    this.accRng.discardGaussian();
    return v;
  }

  /**
   * Thermal de Broglie length Lambda of an exchanged particle of the given mass
   * (docs.lammps.org/fix_gcmc.html): sqrt(h^2 / (2 pi m k_B T)) for every style
   * but lj, where it is set to unity. hplanck is stored in the style's energy *
   * time units; mvv2e brings the style's energy unit back to mass*length^2/time^2
   * so the result is in the style's length unit (see units.ts).
   */
  private deBroglie(mass: number): number {
    const u = this.sys.state.units;
    if (u.style === 'lj') return 1;
    return Math.sqrt((u.hplanck * u.hplanck) / (2 * Math.PI * mass * u.boltz * this.T * u.mvv2e));
  }

  private metropolis(exponent: number): boolean {
    // the exchange stream still advances by one draw here (see the header)
    this.rng.uniform();
    return this.accRng.uniform() < Math.exp(Math.min(0, exponent));
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
      // docs fix_gcmc.rst: "The center of mass of the molecule is placed at the insertion point. The
      // orientation of the molecule is chosen at random by rotating about this point."
      const center = this.templateCenterOfMass(t);
      const coords = new Float64Array(3 * t.natoms);
      for (let a = 0; a < t.natoms; a++) {
        const rx = t.x[3 * a] - center[0], ry = t.x[3 * a + 1] - center[1], rz = t.x[3 * a + 2] - center[2];
        coords[3 * a] = p[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
        coords[3 * a + 1] = p[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
        coords[3 * a + 2] = p[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
      }
      for (const ty of t.type) mass += s.massByType[ty];
      appendMolecule(sys, t, 0, coords, [0, 0, 0], maxMoleculeId(s) + 1, this.extraMask());
      sys.atomsChanged();
    } else {
      mass = s.massByType[this.type];
      appendAtoms(s, { x: Float64Array.from(p), type: this.type, v: new Float64Array(3), q: this.charge ?? undefined, mask: this.extraMask() });
      sys.atomsChanged();
    }
    // the trial molecule's bonds stay in the energy until it is accepted (see the header)
    this.shakeFix?.extendTopology();
    let dU = this.potentialEnergy() - eBefore;
    if (!this.fullEnergy) dU -= this.molTemplate ? this.intraMolEnergy : 0;
    dU -= this.intraEnergy;
    const overlap = this.overlapCutoff > 0 && this.hasOverlap(snap.n, this.overlapCutoff);
    const l3 = this.deBroglie(mass) ** 3;
    const exponent = (this.mu - dU) / kT + Math.log(this.volume / ((nPart + 1) * l3));
    if (!overlap && this.metropolis(exponent)) {
      this.ninsSucc++;
      // Measured with native LAMMPS (black box): an accepted molecule's velocities are the Gaussians
      // that start at draw 11 of the second stream, for the water and the dimer of the w37 cases (the
      // acceptance number is draw 1), so nine more draws come first; the atom path starts at draw 2.
      if (this.molTemplate) for (let k = 0; k < 9; k++) this.accRng.uniform();
      // the velocity is drawn after the acceptance number, from the same stream (see the header)
      const vel = this.drawVelocity(mass);
      for (let i = snap.n; i < s.n; i++) {
        s.v[3 * i] = vel[0]; s.v[3 * i + 1] = vel[1]; s.v[3 * i + 2] = vel[2];
      }
      this.applyGroupTypes(snap.n);
      if (this.shakeFix) this.shakePending = true;
    } else {
      restore(s, snap);
      sys.atomsChanged();
      this.shakeFix?.extendTopology();
    }
  }

  private deletion(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ndelAtt++;
    let idx: number[] = [];
    if (this.molTemplate) {
      const mols = this.eligibleMolecules();
      if (mols.length) {
        const mid = mols[Math.floor(this.rng.uniform() * mols.length)];
        for (let i = 0; i < s.n; i++) if (s.molecule[i] === mid) idx.push(i);
      }
    } else {
      const cands = this.eligibleAtoms();
      if (cands.length) idx = [cands[Math.floor(this.rng.uniform() * cands.length)]];
    }
    const blocked = this.minAtoms !== null && this.groupAtomCount() - idx.length < this.minAtoms;
    if (!idx.length || blocked) {
      // No candidate: native still advances the exchange stream by its decision draw but takes
      // nothing from the second stream (w33gcmc_lj_empty). The min bound is assumed to act the same.
      this.rng.uniform();
      return;
    }
    const nPart = this.particleCount();
    const eBefore = this.potentialEnergy();
    const snap = capture(s);
    let mass = 0;
    for (const i of idx) mass += s.massByType[s.type[i]];
    const flags = new Uint8Array(s.n);
    for (const i of idx) flags[i] = 1;
    deleteAtoms(s, flags);
    sys.atomsChanged();
    this.shakeFix?.rebuildClusters();
    let dU = this.potentialEnergy() - eBefore;
    if (!this.fullEnergy) dU -= this.molTemplate ? this.intraMolEnergy : 0;
    dU -= this.intraEnergy;
    const kT = s.units.boltz * this.T;
    const l3 = this.deBroglie(mass) ** 3;
    const exponent = -(this.mu + dU) / kT + Math.log((nPart * l3) / this.volume);
    if (this.metropolis(exponent)) {
      this.ndelSucc++;
    } else {
      restore(s, snap);
      sys.atomsChanged();
      this.shakeFix?.rebuildClusters();
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

  /** Mass-weighted centre of an inserted molecule's atoms (docs fix_gcmc.rst: "center-of-mass"). */
  private moleculeCom(idx: number[]): Vec3 {
    const s = this.sys.state;
    const c: Vec3 = [0, 0, 0];
    let mtot = 0;
    for (const i of idx) {
      const m = s.massByType[s.type[i]];
      mtot += m;
      for (let d = 0; d < 3; d++) c[d] += m * s.x[3 * i + d];
    }
    return [c[0] / mtot, c[1] / mtot, c[2] / mtot];
  }

  /** Mass-weighted centre of a molecule template (the point the insertion places). */
  private templateCenterOfMass(t: MoleculeTemplate): Vec3 {
    const s = this.sys.state;
    const c: Vec3 = [0, 0, 0];
    let mtot = 0;
    for (let a = 0; a < t.natoms; a++) {
      const m = s.massByType[t.type[a]];
      mtot += m;
      for (let d = 0; d < 3; d++) c[d] += m * t.x[3 * a + d];
    }
    return [c[0] / mtot, c[1] / mtot, c[2] / mtot];
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

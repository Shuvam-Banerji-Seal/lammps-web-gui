import { Fix } from './fix';
import { StyleError } from '../force/types';
import { RanPark } from '../rng';
import { takeFixRecord } from './restartRecord';
import { appendAtoms, deleteAtoms, hasChargeStyle, isMolecularStyle } from '../atoms';
import { appendMolecule, num, posInt } from './pour';
import { rotationMatrix, type MoleculeTemplate } from '../molecule';
import { estimateRegion, boxInsertionPoint, maxMoleculeId, topoList } from './widom';
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
 * group, grouptype, intra_energy, tfac_insert, overlap_cutoff, max, min, shake,
 * rigid. pressure and fugacity_coeff are rejected with a StyleError naming
 * them (they need the per-unit pressure/fugacity conversion the browser
 * engine does not implement).
 *
 * rigid (docs.lammps.org/fix_gcmc.html, plans/lammps-docs-live/fix_gcmc.rst): "If you wish to insert
 * molecules using the *mol* keyword that will be treated as rigid bodies, use the *rigid* keyword,
 * specifying as its value the ID of a separate fix rigid/small command which also appears in your input
 * script." "When using fix gcmc in combination with fix shake or fix rigid, only GCMC exchange moves are
 * supported, so the argument *M* must be zero." "When using fix gcmc in combination with fix rigid,
 * deletion of the last remaining molecule is not allowed for technical reasons, and so the molecule count
 * will never drop below 1, regardless of the specified chemical potential."
 * Measured with native LAMMPS (black box): rigid without mol stops with Cannot use fix gcmc rigid and not
 * molecule; rigid with shake stops with Cannot use fix gcmc rigid and shake; rigid with M > 0 stops with
 * Cannot use fix gcmc rigid with MC moves; a rigid fix ID that is not defined stops with Fix gcmc rigid
 * fix ID nosuch does not exist; a rigid/small fix whose template differs from the mol keyword (or that has
 * none) stops with Fix gcmc and fix rigid/small not using same molecule template ID. No full_energy is
 * needed for rigid (the lj/cut run with rigid did not switch it on). The rigid fix adds the inserted body
 * from its atoms, so its velocities are the atom velocities and its centre of mass is the mass-weighted
 * centre (w38gcr_ins). Body registration, the removal of a deleted body and its dof are in the code below.
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
 * accepted molecule are drawn from the equal stream after its acceptance draw (before it with full_energy),
 * as the Random streams paragraph above describes. Not oracle-checked: multi-event mol runs with shake (the w37
 * cases are single insertions). Native's mid-run SHAKE for a
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
 * Random streams and event order (fix_gcmc.cpp pre_exchange, b332f28; the first version of this engine was measured
 * against native LAMMPS, black box). Two Park-Miller generators are seeded with the fix's seed: random_equal (here
 * rng) and random_unequal (here accRng). Each event runs ncycles = X + M cycles. A cycle draws
 * ixm = int(u * ncycles) + 1 from random_equal; ixm <= M is an MC move, else a GCMC exchange (both always draw from
 * random_equal, even when X or M is zero). The move choice (one equal draw, normalised proportions) and the
 * exchange choice (one equal draw, below 0.5 is a deletion) follow. Then each attempt draws in native's order:
 *  - atom translation: pick (equal), displacement on the sphere (unequal), acceptance (unequal, or equal with
 *    full_energy);
 *  - molecule translation and rotation: everything on the equal stream;
 *  - atom or molecule deletion: pick (equal, none when the gas is empty or at min), acceptance (unequal for partial
 *    atoms, equal otherwise);
 *  - insertion: three position draws (equal), the orientation draws for a molecule (equal), acceptance (unequal for
 *    partial atoms, equal otherwise), and the velocities: three Gaussians and one discarded (unequal for atoms,
 *    equal for molecules), drawn after an accepted partial insertion but before the acceptance of a full_energy one.
 * Evidence: the w33 cases (X only) and the w37/w38 cases pass with this order; w33gcmc_lj_acc (lj, dU = 0) and
 * w33gcmc_real_acc (real, interacting argon) match native's atom count, energy and kinetic energy at every step,
 * through 120 and 150 exchanges with both acceptances and rejections. w32gcmc_move uses displace 0 so every
 * translation has dU = 0 and is accepted: measured with native LAMMPS (black box) with X = 0, M = 1, displace = 0,
 * the vector counts one translation attempt and success every step and the atoms and pe are unchanged.
 * tests/engineGcmc32.test.ts checks acceptance ratios and <N> vs mu statistically.
 * Partial energy (fix_gcmc.cpp energy(), b332f28; the default when canPartial() holds). Every trial sums pair.single()
 * of the trial particle with the atoms inside the cutoff (minimum image), skipping its own index and, for a molecule,
 * every atom of its molecule (native leaves out every intramolecular pair, bonded or not). Nothing is rebuilt and no
 * forces are evaluated; a rejected trial changes nothing.
 *
 * Triclinic boxes (fix_gcmc.cpp attempt_atomic_insertion and attempt_molecule_insertion, b332f28): without a region
 * the insertion point is drawn in fractional coordinates, three draws lamda = u mapped by lamda2x (widom.ts
 * boxInsertionPoint); with a region it is drawn in the region's box bounding box as in the orthogonal case. Moved atoms
 * are wrapped by the fractional rule of Domain::remap (domain.ts remapTriclinic), which forces() applies before every
 * energy (native energy_full does pbc() in lamda space). MC translation displacements are Cartesian, as native's
 * x += displace. Known gap (fallback only): native's partial energy also runs for a triclinic box (through ghost atoms),
 * but this engine's partial sum uses the orthogonal minimum image, so a triclinic box takes the total-energy difference.
 * For atoms the two agree to round-off with the same random streams; for a mol template the total-energy difference
 * still counts the template's own non-bonded pairs (only its bonded energy is subtracted), so full_energy is exact for
 * such molecules (w188gc_mol_xm). neigh_modify exclude/include, a periodic length below twice the largest cutoff, and a
 * hybrid pair style take the same fallback.
 */

type Vec3 = [number, number, number];

/** What fix gcmc needs from a fix shake (its mol template and its cluster rebuild). */
interface ShakeClusters {
  rebuildClusters(): void;
  extendTopology(): void;
  moleculeTemplateId(): string | null;
  style: string;
}

/** What fix gcmc needs from a fix rigid/small (its molecule bodies and its template ID). */
interface RigidBodies {
  addMolecule(atomIds: readonly number[]): void;
  removeMolecule(atomIds: readonly number[]): void;
  snapshotBodies(): object[];
  restoreBodies(snap: object[]): void;
  moleculeTemplateId(): string | null;
  style: string;
}

/** A non-negative integer keyword value (X, M and the max/min bounds allow 0). */
const nonNegInt = (w: string | undefined, what: string): number => {
  const v = num(w, what);
  if (!Number.isInteger(v) || v < 0) throw new StyleError(`fix gcmc: ${what} must be a non-negative integer, got '${w ?? ''}'`);
  return v;
};

/**
 * A non-negative real keyword value. The mcmoves proportions are real numbers (docs.lammps.org/fix_gcmc.html:
 * "The values must be non-negative integers or real numbers, with at least one non-zero value."), so only the sign
 * is checked here; native fix_gcmc.cpp options() (lines 292-300) refuses a negative one with "Illegal fix gcmc mcmoves parameters".
 */
const nonNegReal = (w: string | undefined, what: string): number => {
  const v = num(w, what);
  if (v < 0) throw new StyleError(`fix gcmc: Illegal fix gcmc mcmoves parameters (${what} must be non-negative, got '${w ?? ''}')`);
  return v;
};

/** native max_region_attempts (fix_gcmc.cpp constructor): redraws of a region-restricted trial point. */
const MAX_REGION_ATTEMPTS = 1000;

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

/** The state fix gcmc keeps in a restart file (fix_gcmc.cpp write_restart); counters in native order (trans, rot, del, ins). */
export interface GcmcRestart { seed: number; seedUnequal: number; nextStep: number; step: number; counters: number[] }

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
  private readonly rigidId: string | null;
  private rigidFix: RigidBodies | null = null;
  /** a region-restricted translation that found no trial inside the region is rejected; the warning is printed once */
  private regionWarned = false;
  /** An accepted insertion waits for fix shake to register it at the next step (see postIntegrate). */
  private shakePending = false;
  private readonly regionId: string | null;
  private reg: { lo: Vec3; hi: Vec3; volume: number } | null = null;
  private region: Region | null = null;
  /** This event uses the partial-energy path (set in event() by canPartial()). */
  private partial = false;
  /** Image shells per periodic side summed by particleEnergy (set by canPartial). */
  private imgN: [number, number, number] = [0, 0, 0];
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
    let rigidId: string | null = null;
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
        this.patomtrans = nonNegReal(v(1), 'mcmoves Patomtrans');
        this.pmoltrans = nonNegReal(v(2), 'mcmoves Pmoltrans');
        this.pmolrotate = nonNegReal(v(3), 'mcmoves Pmolrotate');
        // native: pmctot = patomtrans + pmoltrans + pmolrotate; "if (pmctot <= 0) error->all(...Illegal fix gcmc mcmoves parameters)"
        if (this.patomtrans + this.pmoltrans + this.pmolrotate <= 0) throw new StyleError('fix gcmc mcmoves: at least one proportion must be non-zero');
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
      else if (key === 'rigid') { rigidId = v(1); i += 2; }
      else if (key === 'pressure' || key === 'fugacity_coeff') {
        throw new StyleError(`fix gcmc keyword '${key}' is not supported by the browser engine`);
      } else throw new StyleError(`fix gcmc: unknown keyword '${key}'`);
    }
    if (shakeId) {
      if (!molTemplateId) throw new StyleError('fix gcmc: Cannot use fix gcmc shake and not molecule (shake needs the mol keyword)');
      // docs fix_gcmc.html: "When using fix gcmc in combination with fix shake or fix rigid, only GCMC exchange moves are supported, so the argument *M* must be zero."
      if (this.M !== 0) throw new StyleError('fix gcmc: Cannot use fix gcmc shake with MC moves (M must be zero with shake)');
    }
    if (rigidId) {
      // Measured with native LAMMPS (black box): these three refusals, in this order for rigid with shake.
      if (shakeId) throw new StyleError('fix gcmc: Cannot use fix gcmc rigid and shake');
      if (!molTemplateId) throw new StyleError('fix gcmc: Cannot use fix gcmc rigid and not molecule (rigid needs the mol keyword)');
      if (this.M !== 0) throw new StyleError('fix gcmc: Cannot use fix gcmc rigid with MC moves (M must be zero with rigid)');
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
    this.rigidId = rigidId;
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
    // native estimates the region's volume here, before any other draw of the equal stream (widom.ts estimateRegion);
    // a restart file's generator state replaces the stream afterwards
    if (regionId) this.reg = estimateRegion(sys.region(regionId), this.rng, 'gcmc');
    const saved = takeFixRecord(sys, id, this.style)?.state as GcmcRestart | undefined;
    if (saved) this.restoreState(saved);
  }

  /**
   * fix_gcmc.cpp write_restart: the two generators (random_equal, random_unequal), the next step, the eight attempt and
   * success counters (native order: translation, rotation, deletion, insertion) and the step the file was written at.
   */
  restartState(): GcmcRestart {
    return {
      seed: this.rng.state(), seedUnequal: this.accRng.state(), nextStep: this.nextStep, step: this.sys.state.step,
      counters: [this.ntransAtt, this.ntransSucc, this.nrotAtt, this.nrotSucc, this.ndelAtt, this.ndelSucc, this.ninsAtt, this.ninsSucc],
    };
  }

  /** FixGCMC::restart (fix_gcmc.cpp): the generators, the next step and the counters; the step must be the file's. */
  private restoreState(r: GcmcRestart): void {
    if (r.step !== this.sys.state.step) throw new StyleError(`fix ${this.id}: must not reset timestep when restarting fix gcmc`);
    this.rng.reset(r.seed);
    this.accRng.reset(r.seedUnequal);
    this.nextStep = r.nextStep;
    [this.ntransAtt, this.ntransSucc, this.nrotAtt, this.nrotSucc, this.ndelAtt, this.ndelSucc, this.ninsAtt, this.ninsSucc] = r.counters;
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
    if (this.rigidId) {
      const rf = this.sys.fixes.find((x) => x.id === this.rigidId) as unknown as Partial<RigidBodies> | undefined;
      if (!rf) throw new StyleError(`fix gcmc rigid: fix ID ${this.rigidId} does not exist`);
      if (typeof rf.addMolecule !== 'function' || typeof rf.removeMolecule !== 'function' || typeof rf.snapshotBodies !== 'function'
        || typeof rf.restoreBodies !== 'function' || typeof rf.moleculeTemplateId !== 'function' || !rf.style?.includes('small')) {
        throw new StyleError(`fix gcmc rigid: fix ${this.rigidId} (${rf.style ?? ''}) is not a fix rigid/small`);
      }
      if (rf.moleculeTemplateId!() !== this.molTemplateId) {
        throw new StyleError('fix gcmc: Fix gcmc and fix rigid/small not using same molecule template ID');
      }
      this.rigidFix = rf as RigidBodies;
    }
    for (const g of this.groupTypes) if (g.type < 1 || g.type > s.ntypes) throw new StyleError(`fix gcmc grouptype: type ${g.type} is outside 1..${s.ntypes}`);
    // the region's bounding box and volume come from the definition of the fix (constructor)
    this.region = this.regionId ? this.sys.region(this.regionId) : null;
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

  /**
   * One GCMC event, as native FixGCMC::pre_exchange does it (fix_gcmc.cpp at b332f28): ncycles = X + M cycles.
   * Each cycle draws ixm = int(u * ncycles) + 1 from the equal stream first; ixm <= M is an MC move, otherwise a
   * GCMC exchange. Moves and exchanges interleave in cycle order (they are not all exchanges and then all moves).
   */
  private event(): void {
    this.setImageShells();
    this.partial = this.canPartial();
    const before = this.successCount();
    const ncycles = this.X + this.M;
    for (let c = 0; c < ncycles; c++) {
      const ixm = Math.floor(this.rng.uniform() * ncycles) + 1;
      if (ixm <= this.M) this.move();
      else this.exchange();
    }
    // docs fix_gcmc.html: "Note that neighbor lists are re-built every timestep that this fix is
    // invoked, so you should not set N to be too small." Rebuild for the final atom count so the
    // run loop's force evaluation (which may decide not to reneighbor) sees the exchanged atoms.
    // As native FixGCMC::pre_exchange (fix_gcmc.cpp at b332f28): the partial-energy path only refreshes the ghosts
    // (comm->borders) after an accepted change; a rejected trial leaves the atoms and their neighbour list alone.
    if (!this.partial || this.successCount() !== before) this.sys.setupNeighbors();
  }

  /** Accepted exchanges and moves so far (any change to the atoms or their positions). */
  private successCount(): number {
    return this.ntransSucc + this.nrotSucc + this.ninsSucc + this.ndelSucc;
  }

  /** The exchange choice of a cycle (fix_gcmc.cpp: xgcmc < 0.5 is a deletion), the same for both exchange modes. */
  private exchange(): void {
    if (this.rng.uniform() < 0.5) this.deletion();
    else this.insertion();
  }

  /** The move choice of a cycle: xmcmove against the proportions normalised by their sum (fix_gcmc.cpp). */
  private move(): void {
    const total = this.patomtrans + this.pmoltrans + this.pmolrotate;
    const pa = this.patomtrans / total;
    const pm = this.pmoltrans / total;
    const x = this.rng.uniform();
    if (x < pa) this.translateAtom();
    else if (x < pa + pm) this.translateMolecule();
    else this.rotateMolecule();
  }

  private get volume(): number {
    if (this.reg) return this.reg.volume;
    const b = this.sys.state.box;
    return (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]);
  }

  /**
   * Random insertion point (fix_gcmc.cpp attempt_*_insertion): three draws of the equal stream inside the box or
   * the region's bounding box. A point outside a region is redrawn; after max_region_attempts (1000) redraws the
   * attempt gives up with no insertion, as the native loop returns once region_attempt reaches the maximum.
   */
  private drawPosition(): Vec3 | null {
    // no region: the whole box, fractional coordinates for a triclinic box (widom.ts boxInsertionPoint)
    if (!this.region) return boxInsertionPoint(this.sys, this.rng);
    const lo = this.reg!.lo;
    const hi = this.reg!.hi;
    const trial = (): Vec3 => [
      lo[0] + this.rng.uniform() * (hi[0] - lo[0]),
      lo[1] + this.rng.uniform() * (hi[1] - lo[1]),
      lo[2] + this.rng.uniform() * (hi[2] - lo[2]),
    ];
    let p = trial();
    let attempt = 0;
    while (!this.region.match(p[0], p[1], p[2])) {
      p = trial();
      if (++attempt >= MAX_REGION_ATTEMPTS) return null;
    }
    return p;
  }

  /**
   * Whether this event uses the partial-energy path (fix_gcmc.cpp energy(), b332f28): the pair energy of a trial
   * particle with the rest of the system, with no force evaluation and no neighbour rebuild. Native switches to
   * full_energy for kspace, hybrid, eam, tail and a pair style without single() (fix_gcmc.cpp init, lines 538-549);
   * init() already maps the engine's own auto cases to fullEnergy. The sum uses minimum-image distances in place of
   * native's ghost atoms: every periodic image of an atom within the largest cutoff (or overlap_cutoff) is summed, so
   * the sum is exact for any orthogonal box. It needs no neigh_modify exclude or include. Otherwise the event keeps the
   * total-energy difference.
   */
  private canPartial(): boolean {
    if (this.fullEnergy) return false;
    const s = this.sys.state;
    const pair = this.sys.ff.pair;
    if (!pair || typeof pair.single !== 'function' || pair.singleNeedsSeparation()) return false;
    if (pair.name.startsWith('hybrid')) return false;
    if (s.box.triclinic || this.sys.nb.excludes.length || this.sys.nb.includeBit) return false;
    return true;
  }

  /**
   * The periodic image shells summed for a separation (imgN): none for a periodic side at least twice the largest
   * cutoff (the minimum image is then the only image in range), else ceil(cutoff / length) shells on each side.
   * Called at every event, so the overlap check also has its images with full_energy.
   */
  private setImageShells(): void {
    const s = this.sys.state;
    const pair = this.sys.ff.pair;
    let cmax = this.overlapCutoff;
    if (pair) for (let k = 0; k < pair.cut.length; k++) if (pair.cut[k] > cmax) cmax = pair.cut[k];
    for (let d = 0; d < 3; d++) {
      const L = s.box.hi[d] - s.box.lo[d];
      this.imgN[d] = s.box.periodic[d] && 2 * cmax > L && L > 0 ? Math.ceil(cmax / L) : 0;
    }
  }

  /**
   * As native FixGCMC::energy (fix_gcmc.cpp at b332f28, lines 2312-2357): the pair energy of one trial particle of type
   * itype and charge qi at p with every other atom inside the cutoff. The particle's own index (self) is skipped, and
   * with a molecule template every atom of the same molecule is skipped ("if (imolecule == molecule[j]) continue;"), so
   * intramolecular pairs never count. An atom closer than overlap_cutoff makes the energy an overlap (native returns
   * MAXENERGYSIGNAL at once). Native passes factor_coul = factor_lj = 1, so special bonds are not applied.
   */
  private particleEnergy(p: Vec3, itype: number, qi: number, mol: number, self: number): { e: number; overlap: boolean } {
    const s = this.sys.state;
    const pair = this.sys.ff.pair!;
    const nt = s.ntypes + 1;
    const cutsq = pair.cutsq;
    const ovl2 = this.overlapCutoff * this.overlapCutoff;
    const checkOverlap = this.overlapCutoff > 0;
    const molExcl = this.molTemplate !== null;
    const x = s.x, type = s.type, q = s.q;
    let e = 0;
    for (let j = 0; j < s.n; j++) {
      if (j === self) continue;
      if (molExcl && s.molecule[j] === mol) continue;
      const jt = type[j];
      const cut2 = cutsq[itype * nt + jt];
      let overlap = false;
      let ej = 0;
      this.forEachImageSq(p[0] - x[3 * j], p[1] - x[3 * j + 1], p[2] - x[3 * j + 2], (rsq) => {
        if (checkOverlap && rsq < ovl2) { overlap = true; return; }
        if (rsq < cut2) ej += pair.single!(s.n, j, itype, jt, rsq, 1, 1, qi, q ? q[j] : 0).eng;
      });
      if (overlap) return { e: 0, overlap: true };
      e += ej;
    }
    return { e, overlap: false };
  }

  /**
   * Calls f with the squared length of the separation (dx, dy, dz) and each periodic image of it that the image shells
   * (imgN, see canPartial) cover, centred on the minimum image. With imgN all zero this is the minimum image alone.
   */
  private forEachImageSq(dx: number, dy: number, dz: number, f: (rsq: number) => void): void {
    const b = this.sys.state.box;
    const [nx, ny, nz] = this.imgN;
    const Lx = b.periodic[0] ? b.hi[0] - b.lo[0] : 0;
    const Ly = b.periodic[1] ? b.hi[1] - b.lo[1] : 0;
    const Lz = b.periodic[2] ? b.hi[2] - b.lo[2] : 0;
    const mx = Lx ? Math.round(dx / Lx) : 0, my = Ly ? Math.round(dy / Ly) : 0, mz = Lz ? Math.round(dz / Lz) : 0;
    for (let a = -nx; a <= nx; a++) {
      const ex = dx - (mx + a) * Lx;
      for (let c = -ny; c <= ny; c++) {
        const ey = dy - (my + c) * Ly;
        for (let d = -nz; d <= nz; d++) {
          const ez = dz - (mz + d) * Lz;
          f(ex * ex + ey * ey + ez * ez);
        }
      }
    }
  }

  /** particleEnergy of the stored atom i placed at p with type itype (its charge and molecule are its own). */
  private atomEnergy(i: number, p: Vec3, itype: number): { e: number; overlap: boolean } {
    const s = this.sys.state;
    return this.particleEnergy(p, itype, s.q ? s.q[i] : 0, s.molecule[i], i);
  }

  /** Sum of the particle energies of the stored atoms idx at positions xs (3 per atom); an overlap of any atom wins. */
  private groupEnergy(idx: readonly number[], xs: Float64Array): { e: number; overlap: boolean } {
    const s = this.sys.state;
    let e = 0;
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k];
      const r = this.particleEnergy([xs[3 * k], xs[3 * k + 1], xs[3 * k + 2]], s.type[i], s.q ? s.q[i] : 0, s.molecule[i], i);
      if (r.overlap) return { e: 0, overlap: true };
      e += r.e;
    }
    return { e, overlap: false };
  }

  /**
   * The positions of the molecule atoms idx rotated by rot about com (fix_gcmc.cpp attempt_molecule_rotation): the rotation
   * acts on the unwrapped coordinates, and each atom's own image flags are used to unwrap it.
   */
  private rotatedCoords(idx: readonly number[], com: Vec3, rot: number[][]): Float64Array {
    const s = this.sys.state;
    const out = new Float64Array(3 * idx.length);
    const u: number[] = [0, 0, 0];
    idx.forEach((i, k) => {
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const rx = u[0] - com[0], ry = u[1] - com[1], rz = u[2] - com[2];
      out[3 * k] = com[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
      out[3 * k + 1] = com[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
      out[3 * k + 2] = com[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
    });
    return out;
  }

  /** Writes rotated positions (see rotatedCoords) back, remapping each atom into the box with its image flags reset. */
  private writeRotated(idx: readonly number[], xs: Float64Array): void {
    const s = this.sys.state;
    idx.forEach((i, k) => {
      s.x[3 * i] = xs[3 * k]; s.x[3 * i + 1] = xs[3 * k + 1]; s.x[3 * i + 2] = xs[3 * k + 2];
      s.image[3 * i] = 0; s.image[3 * i + 1] = 0; s.image[3 * i + 2] = 0;
      this.sys.geom.remap(s.x, s.image, i);
    });
  }

  private potentialEnergy(): number {
    const a = this.sys.forces();
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp;
  }

  /**
   * The gas list (fix_gcmc.cpp update_gas_atoms_list): the group atoms inside the region, sorted by atom ID (tag).
   * The native pickers index this list, so its order is the atom-ID order whatever the storage order.
   */
  private gasAtoms(): number[] {
    const s = this.sys.state;
    const out: number[] = [];
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) === 0) continue;
      if (this.region && !this.region.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) continue;
      out.push(i);
    }
    return out.sort((a, b) => s.id[a] - s.id[b]);
  }

  /** Exchanged particles of a gas list: atoms, or distinct molecules with the mol keyword (ngas / natoms_per_molecule). */
  private nParticles(gas: readonly number[]): number {
    if (!this.molTemplate) return gas.length;
    return new Set(gas.map((i) => this.sys.state.molecule[i])).size;
  }

  /**
   * Picks one gas atom (fix_gcmc.cpp pick_random_gas_atom and pick_random_gas_molecule): one equal-stream draw,
   * index = int(ngas * u). The caller has checked that the list is not empty (native returns before the draw).
   */
  private pickGasAtom(gas: readonly number[]): number {
    return gas[Math.floor(gas.length * this.rng.uniform())];
  }

  /** Stream of the acceptance decisions: equal for full_energy and molecules, unequal for partial-energy atoms (fix_gcmc.cpp). */
  private acceptStream(): RanPark {
    return this.fullEnergy || this.molTemplate ? this.rng : this.accRng;
  }

  /** Stream of the trial displacements and of inserted velocities: unequal for atoms, equal for molecules (fix_gcmc.cpp). */
  private vecStream(): RanPark {
    return this.molTemplate ? this.rng : this.accRng;
  }

  /** Accepts when u < min(1, exp(exponent)), with one uniform from the stream (fix_gcmc.cpp compares u < exp(...)). */
  private accept(stream: RanPark, exponent: number): boolean {
    return stream.uniform() < Math.exp(Math.min(0, exponent));
  }

  /** Trial displacement: a point of the cube [-1, 1]^3 inside the unit sphere (rejection), scaled by displace. */
  private sphereDisplacement(stream: RanPark): Vec3 {
    let rx = 0, ry = 0, rz = 0, rsq = 1.1;
    while (rsq > 1.0) {
      rx = 2 * stream.uniform() - 1;
      ry = 2 * stream.uniform() - 1;
      rz = 2 * stream.uniform() - 1;
      rsq = rx * rx + ry * ry + rz * rz;
    }
    return [this.displace * rx, this.displace * ry, this.displace * rz];
  }

  private drawVelocity(mass: number, stream: RanPark): Vec3 {
    const u = this.sys.state.units;
    // kT / m is energy per mass; mvv2e turns it into the style's velocity^2
    const sd = Math.sqrt((u.boltz * this.T * this.tfacInsert) / (mass * u.mvv2e));
    const v: Vec3 = [sd * stream.gaussian(), sd * stream.gaussian(), sd * stream.gaussian()];
    // Measured with native LAMMPS (black box): each insertion starts a new polar pair, so the
    // unused second value of the last pair is dropped.
    stream.discardGaussian();
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

  private extraMask(): number {
    let m = this.groupBit;
    if (this.groupId) m |= this.sys.groups.bit(this.groupId);
    return m;
  }

  private insertion(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ninsAtt++;
    // native returns before any draw when the gas count is at the max bound
    const gas = this.gasAtoms();
    if (this.maxAtoms !== null && gas.length >= this.maxAtoms) return;
    const p = this.drawPosition();
    if (!p) return;
    const nPart = this.nParticles(gas);
    if (this.partial) { this.insertionPartial(p, nPart); return; }
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
      appendMolecule(sys, t, 0, coords, [0, 0, 0], maxMoleculeId(s) + 1, this.extraMask(), undefined, rot);
      sys.atomsChanged();
    } else {
      mass = s.massByType[this.type];
      appendAtoms(s, { x: Float64Array.from(p), type: this.type, v: new Float64Array(3), q: this.charge ?? undefined, mask: this.extraMask() });
      sys.atomsChanged();
    }
    // the trial molecule's bonds stay in the energy until it is accepted (see the header)
    this.shakeFix?.extendTopology();
    // native full_energy insertions draw their velocities before the acceptance test (attempt_*_insertion_full)
    const vel0 = this.fullEnergy ? this.drawVelocity(mass, this.vecStream()) : null;
    let dU = this.potentialEnergy() - eBefore;
    if (!this.fullEnergy) dU -= this.molTemplate ? this.intraMolEnergy : 0;
    dU -= this.intraEnergy;
    const overlap = this.overlapCutoff > 0 && this.hasOverlap(snap.n, this.overlapCutoff);
    const l3 = this.deBroglie(mass) ** 3;
    const exponent = (this.mu - dU) / kT + Math.log(this.volume / ((nPart + 1) * l3));
    if (!overlap && this.accept(this.acceptStream(), exponent)) {
      this.ninsSucc++;
      // the trial's bonds are kept; a partial-energy insertion draws its velocities after the acceptance
      const vel = vel0 ?? this.drawVelocity(mass, this.vecStream());
      for (let i = snap.n; i < s.n; i++) {
        s.v[3 * i] = vel[0]; s.v[3 * i + 1] = vel[1]; s.v[3 * i + 2] = vel[2];
      }
      this.applyGroupTypes(snap.n);
      if (this.shakeFix) this.shakePending = true;
      if (this.rigidFix) {
        const ids: number[] = [];
        for (let i = snap.n; i < s.n; i++) ids.push(s.id[i]);
        this.rigidFix.addMolecule(ids);
      }
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
    // native returns before any draw when the gas is empty or at the min bound
    const gas = this.gasAtoms();
    if (gas.length === 0) return;
    if (this.minAtoms !== null && gas.length <= this.minAtoms) return;
    // docs fix_gcmc.html: "deletion of the last remaining molecule is not allowed" with fix rigid
    if (this.rigidFix !== null && this.nParticles(gas) <= 1) return;
    const picked = this.pickGasAtom(gas);
    const idx: number[] = [];
    if (this.molTemplate) {
      const mid = s.molecule[picked];
      for (let i = 0; i < s.n; i++) if (s.molecule[i] === mid) idx.push(i);
    } else {
      idx.push(picked);
    }
    const nPart = this.nParticles(gas);
    if (this.partial) { this.deletionPartial(idx, nPart); return; }
    const eBefore = this.potentialEnergy();
    const snap = capture(s);
    const bodySnap = this.rigidFix?.snapshotBodies();
    const delIds = idx.map((i) => s.id[i]);
    let mass = 0;
    for (const i of idx) mass += s.massByType[s.type[i]];
    const flags = new Uint8Array(s.n);
    for (const i of idx) flags[i] = 1;
    deleteAtoms(s, flags);
    sys.atomsChanged();
    this.shakeFix?.rebuildClusters();
    this.rigidFix?.removeMolecule(delIds);
    let dU = this.potentialEnergy() - eBefore;
    if (!this.fullEnergy) dU -= this.molTemplate ? this.intraMolEnergy : 0;
    dU -= this.intraEnergy;
    const kT = s.units.boltz * this.T;
    const l3 = this.deBroglie(mass) ** 3;
    const exponent = -(this.mu + dU) / kT + Math.log((nPart * l3) / this.volume);
    if (this.accept(this.acceptStream(), exponent)) {
      this.ndelSucc++;
    } else {
      restore(s, snap);
      sys.atomsChanged();
      this.shakeFix?.rebuildClusters();
      if (this.rigidFix && bodySnap) this.rigidFix.restoreBodies(bodySnap);
    }
  }

  /**
   * Insertion on the partial-energy path (fix_gcmc.cpp attempt_atomic_insertion and attempt_molecule_insertion, b332f28):
   * the trial's energy is the sum of its pair energies with the atoms already in the system (native sums energy() for
   * each molecule atom before appending them, so the molecule's own atoms do not interact). Nothing is appended until
   * the trial is accepted. The draws keep the full-path order: the rotation first, the acceptance, then the velocities.
   */
  private insertionPartial(p: Vec3, nPart: number): void {
    const sys = this.sys;
    const s = sys.state;
    const kT = s.units.boltz * this.T;
    const n0 = s.n;
    let mass = 0;
    let coords: Float64Array | null = null;
    let rot: number[][] | null = null;
    let e = 0;
    let overlap = false;
    if (this.molTemplate) {
      const t = this.molTemplate;
      rot = this.randomRotation(Math.PI * 2);
      // docs fix_gcmc.rst: "The center of mass of the molecule is placed at the insertion point. The
      // orientation of the molecule is chosen at random by rotating about this point."
      const center = this.templateCenterOfMass(t);
      coords = new Float64Array(3 * t.natoms);
      for (let a = 0; a < t.natoms; a++) {
        const rx = t.x[3 * a] - center[0], ry = t.x[3 * a + 1] - center[1], rz = t.x[3 * a + 2] - center[2];
        coords[3 * a] = p[0] + rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz;
        coords[3 * a + 1] = p[1] + rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz;
        coords[3 * a + 2] = p[2] + rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz;
      }
      for (const ty of t.type) mass += s.massByType[ty];
      for (let a = 0; a < t.natoms && !overlap; a++) {
        const r = this.particleEnergy([coords[3 * a], coords[3 * a + 1], coords[3 * a + 2]], t.type[a], t.q ? t.q[a] : 0, -1, -1);
        e += r.e;
        overlap = r.overlap;
      }
    } else {
      mass = s.massByType[this.type];
      const r = this.particleEnergy(p, this.type, this.charge ?? 0, -1, -1);
      e = r.e;
      overlap = r.overlap;
    }
    const dU = e - this.intraEnergy;
    const l3 = this.deBroglie(mass) ** 3;
    const exponent = (this.mu - dU) / kT + Math.log(this.volume / ((nPart + 1) * l3));
    // native: insertion_energy < MAXENERGYTEST && random_unequal->uniform() < ... (no draw after an overlap)
    if (overlap || !this.accept(this.acceptStream(), exponent)) return;
    this.ninsSucc++;
    if (this.molTemplate) {
      appendMolecule(sys, this.molTemplate, 0, coords!, [0, 0, 0], maxMoleculeId(s) + 1, this.extraMask(), undefined, rot!);
    } else {
      appendAtoms(s, { x: Float64Array.from(p), type: this.type, v: new Float64Array(3), q: this.charge ?? undefined, mask: this.extraMask() });
    }
    sys.atomsChanged();
    // the accepted molecule's bonds are registered with fix shake at the next step (see postIntegrate)
    this.shakeFix?.extendTopology();
    const vel = this.drawVelocity(mass, this.vecStream());
    for (let i = n0; i < s.n; i++) {
      s.v[3 * i] = vel[0]; s.v[3 * i + 1] = vel[1]; s.v[3 * i + 2] = vel[2];
    }
    this.applyGroupTypes(n0);
    if (this.shakeFix) this.shakePending = true;
    if (this.rigidFix) {
      const ids: number[] = [];
      for (let i = n0; i < s.n; i++) ids.push(s.id[i]);
      this.rigidFix.addMolecule(ids);
    }
  }

  /**
   * Deletion on the partial-energy path (fix_gcmc.cpp attempt_atomic_deletion and attempt_molecule_deletion, b332f28):
   * the removed particle's energy with the rest of the system. An atom uses the fix's type (native energy(i, ngcmc_type,
   * ...)); a molecule uses each atom's own type (molecule_energy). Nothing is removed until the deletion is accepted.
   */
  private deletionPartial(idx: number[], nPart: number): void {
    const sys = this.sys;
    const s = sys.state;
    const kT = s.units.boltz * this.T;
    let e = 0;
    let mass = 0;
    for (const i of idx) {
      mass += s.massByType[s.type[i]];
      const p: Vec3 = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
      e += this.atomEnergy(i, p, this.molTemplate ? s.type[i] : this.type).e;
    }
    const dU = -e - this.intraEnergy;
    const l3 = this.deBroglie(mass) ** 3;
    const exponent = -(this.mu + dU) / kT + Math.log((nPart * l3) / this.volume);
    if (!this.accept(this.acceptStream(), exponent)) return;
    this.ndelSucc++;
    const delIds = idx.map((i) => s.id[i]);
    const flags = new Uint8Array(s.n);
    for (const i of idx) flags[i] = 1;
    deleteAtoms(s, flags);
    sys.atomsChanged();
    this.shakeFix?.rebuildClusters();
    this.rigidFix?.removeMolecule(delIds);
  }

  /** True if any atom appended after n0 lies closer than the cutoff to a pre-existing atom. */
  private hasOverlap(n0: number, cutoff: number): boolean {
    const s = this.sys.state;
    const c2 = cutoff * cutoff;
    for (let i = n0; i < s.n; i++) {
      for (let j = 0; j < n0; j++) {
        let hit = false;
        this.forEachImageSq(s.x[3 * i] - s.x[3 * j], s.x[3 * i + 1] - s.x[3 * j + 1], s.x[3 * i + 2] - s.x[3 * j + 2], (rsq) => {
          if (rsq < c2) hit = true;
        });
        if (hit) return true;
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

  /** Atom translation (fix_gcmc.cpp attempt_atomic_translation): pick, displacement, optional region redraws, acceptance. */
  private translateAtom(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ntransAtt++;
    const gas = this.gasAtoms();
    if (!gas.length) return;
    const i = this.pickGasAtom(gas);
    const old: Vec3 = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
    const stream = this.vecStream();
    let d = this.sphereDisplacement(stream);
    let nw: Vec3 = [old[0] + d[0], old[1] + d[1], old[2] + d[2]];
    if (this.region) {
      let attempt = 0;
      while (!this.region.match(nw[0], nw[1], nw[2])) {
        d = this.sphereDisplacement(stream);
        nw = [old[0] + d[0], old[1] + d[1], old[2] + d[2]];
        if (++attempt >= MAX_REGION_ATTEMPTS) break;
      }
      // docs fix_gcmc.html: a trial move that cannot be placed inside the region is rejected (and still counted)
      if (!this.region.match(nw[0], nw[1], nw[2])) { this.regionMoveRejected(); return; }
    }
    if (this.partial) {
      const kT = s.units.boltz * this.T;
      const before = this.atomEnergy(i, old, this.type).e;
      const after = this.atomEnergy(i, nw, this.type);
      // native: energy_after < MAXENERGYTEST && random_unequal->uniform() < ... (no draw after an overlap)
      if (!after.overlap && this.accept(this.acceptStream(), -(after.e - before) / kT)) {
        this.ntransSucc++;
        for (let k = 0; k < 3; k++) s.x[3 * i + k] = nw[k];
        sys.bump();
      }
      return;
    }
    if (this.overlapCutoff > 0 && this.atomEnergy(i, nw, this.type).overlap) return;
    const oldImage = this.snapshotImages([i]);
    const eBefore = this.potentialEnergy();
    for (let k = 0; k < 3; k++) s.x[3 * i + k] = nw[k];
    sys.bump();
    const dU = this.potentialEnergy() - eBefore;
    if (this.accept(this.acceptStream(), -dU / (s.units.boltz * this.T))) {
      this.ntransSucc++;
    } else {
      this.restoreTrial([i], Float64Array.from(old), oldImage);
      sys.bump();
    }
  }

  /** Molecule translation (fix_gcmc.cpp attempt_molecule_translation): the molecule of a picked gas atom moves as a whole. */
  private translateMolecule(): void {
    const sys = this.sys;
    const s = sys.state;
    this.ntransAtt++;
    const gas = this.gasAtoms();
    if (!gas.length) return;
    const mid = s.molecule[this.pickGasAtom(gas)];
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.molecule[i] === mid) idx.push(i);
    const com = this.moleculeCom(idx);
    let d = this.sphereDisplacement(this.rng);
    let nw: Vec3 = [com[0] + d[0], com[1] + d[1], com[2] + d[2]];
    if (this.region) {
      let attempt = 0;
      while (!this.region.match(nw[0], nw[1], nw[2])) {
        d = this.sphereDisplacement(this.rng);
        nw = [com[0] + d[0], com[1] + d[1], com[2] + d[2]];
        if (++attempt >= MAX_REGION_ATTEMPTS) break;
      }
      if (!this.region.match(nw[0], nw[1], nw[2])) { this.regionMoveRejected(); return; }
    }
    // native shifts by the displacement of the last draw (com_displace), which is the first one outside a region
    const shift = d;
    const old = this.snapshotCoords(idx);
    const moved = Float64Array.from(old, (v, k) => v + shift[k % 3]);
    if (this.partial) {
      const kT = s.units.boltz * this.T;
      const before = this.groupEnergy(idx, old);
      const after = this.groupEnergy(idx, moved);
      if (!after.overlap && this.accept(this.acceptStream(), -(after.e - before.e) / kT)) {
        this.ntransSucc++;
        for (let k = 0; k < idx.length; k++) for (let c = 0; c < 3; c++) s.x[3 * idx[k] + c] = moved[3 * k + c];
        sys.bump();
      }
      return;
    }
    if (this.overlapCutoff > 0 && this.groupEnergy(idx, moved).overlap) return;
    const oldImage = this.snapshotImages(idx);
    const eBefore = this.potentialEnergy();
    for (const i of idx) for (let k = 0; k < 3; k++) s.x[3 * i + k] += shift[k];
    sys.bump();
    const dU = this.potentialEnergy() - eBefore;
    if (this.accept(this.acceptStream(), -dU / (s.units.boltz * this.T))) {
      this.ntransSucc++;
    } else {
      this.restoreTrial(idx, old, oldImage);
      sys.bump();
    }
  }

  /**
   * docs.lammps.org/fix_gcmc.html: "This process is repeated up to 1000 times; if no trial position with the
   * center-of-mass inside the region is found, the move is rejected (but still counted as an attempted move)
   * and a warning is printed once." The caller has counted the attempt and changed no position.
   */
  private regionMoveRejected(): void {
    if (this.regionWarned) return;
    this.regionWarned = true;
    this.sys.warn('fix gcmc: could not place a region-restricted trial move inside the region after 1000 attempts; rejecting the move (reduce displace or enlarge the region)');
  }

  /** Molecule rotation (fix_gcmc.cpp attempt_molecule_rotation): pick, random rotation about the centre of mass, acceptance. */
  private rotateMolecule(): void {
    const sys = this.sys;
    const s = sys.state;
    this.nrotAtt++;
    const gas = this.gasAtoms();
    if (!gas.length) return;
    const mid = s.molecule[this.pickGasAtom(gas)];
    const idx: number[] = [];
    for (let i = 0; i < s.n; i++) if (s.molecule[i] === mid) idx.push(i);
    const com = this.moleculeCom(idx);
    const rot = this.randomRotation(this.maxangle);
    const old = this.snapshotCoords(idx);
    // fix_gcmc.cpp attempt_molecule_rotation: the rotation acts on the unwrapped coordinates about the unwrapped
    // centre of mass; the rotated atoms are then remapped into the box with their image flags reset first
    const nw = this.rotatedCoords(idx, com, rot);
    if (this.partial) {
      const kT = s.units.boltz * this.T;
      const before = this.groupEnergy(idx, old);
      const after = this.groupEnergy(idx, nw);
      if (!after.overlap && this.accept(this.acceptStream(), -(after.e - before.e) / kT)) {
        this.nrotSucc++;
        this.writeRotated(idx, nw);
        sys.bump();
      }
      return;
    }
    if (this.overlapCutoff > 0 && this.groupEnergy(idx, nw).overlap) return;
    const oldImage = this.snapshotImages(idx);
    const eBefore = this.potentialEnergy();
    this.writeRotated(idx, nw);
    sys.bump();
    const dU = this.potentialEnergy() - eBefore;
    if (this.accept(this.acceptStream(), -dU / (s.units.boltz * this.T))) {
      this.nrotSucc++;
    } else {
      this.restoreTrial(idx, old, oldImage);
      sys.bump();
    }
  }

  /**
   * Mass-weighted centre of an inserted molecule's atoms (docs fix_gcmc.rst: "center-of-mass"), from the unwrapped
   * positions as native's group->xcm does (fix_gcmc.cpp attempt_molecule_translation and rotation).
   */
  private moleculeCom(idx: number[]): Vec3 {
    const s = this.sys.state;
    const c: Vec3 = [0, 0, 0];
    const u: number[] = [0, 0, 0];
    let mtot = 0;
    for (const i of idx) {
      const m = s.massByType[s.type[i]];
      mtot += m;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      for (let d = 0; d < 3; d++) c[d] += m * u[d];
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

  /**
   * Puts back the positions and image flags of a rejected trial. The trial's bump lets the neighbour build wrap an
   * atom that was moved out of the box, which changes its image flags, so both must be restored (native never moves
   * a rejected trial: fix_gcmc.cpp evaluates the trial at the coordinates it would move to).
   */
  private restoreTrial(idx: readonly number[], x: Float64Array, image: Int32Array): void {
    const s = this.sys.state;
    for (let k = 0; k < idx.length; k++) {
      for (let d = 0; d < 3; d++) {
        s.x[3 * idx[k] + d] = x[3 * k + d];
        s.image[3 * idx[k] + d] = image[3 * k + d];
      }
    }
  }

  private snapshotImages(idx: number[]): Int32Array {
    const s = this.sys.state;
    const out = new Int32Array(3 * idx.length);
    idx.forEach((i, k) => { for (let d = 0; d < 3; d++) out[3 * k + d] = s.image[3 * i + d]; });
    return out;
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

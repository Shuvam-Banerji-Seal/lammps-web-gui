import { Fix } from './fix';
import { StyleError } from '../force/types';
import { appendAtoms, maxAtomId, isMolecularStyle } from '../atoms';
import type { System } from '../system';
import { insertionStream, separation, appendMolecule, type InsertionRegion } from './pour';
import { CompoundRegion, PrismRegion, Region } from '../region';
import { geometricCenter, type MoleculeTemplate } from '../molecule';
import { gplHook } from '../gplHooks';
import { latticeScale } from '../commands/args';
import { inumeric, missingArgs, numeric } from '../force/util';
import { typeWord } from '../labelmap';
import type { RanPark } from '../rng';
import { takeFixRecord } from './restartRecord';

/*
 * fix ID group-ID deposit N type M seed keyword values ... — docs.lammps.org/fix_deposit.html
 * (plans/lammps-docs/fix_deposit.rst). "Insert a single atom or molecule into
 * the simulation domain every M timesteps until N atoms or molecules have
 * been inserted." All the keywords are read: region (any region style with a bounding box, see regionExtent), var and
 * set, id max|next, global, local, near, gaussian, attempt, rate, vx, vy, vz, target, units, and mol (a molecule
 * template of one or more molecules) with molfrac, rigid, shake and orient. As native LAMMPS does (src/fix_deposit.cpp
 * FixDeposit::FixDeposit and options at b332f28) the arguments are checked when the fix is defined, in native's order
 * and with its messages; M = 0 (native divides by it and stops with a floating-point exception) is refused.
 *
 * fix_deposit.html: "The *var* and *set* keywords can be used together to provide a criterion for accepting or
 * rejecting the addition of an individual atom, based on its coordinates."; "if one (or more) is not defined, then the
 * *set* option creates an internal-style variable with the specified name."; "When an atom is about to be created, its
 * (x,y,z) coordinates become the values for any *set* variable that is defined.  The *var* variable is then evaluated.
 * If the returned value is 0.0, the atom is not created."; "If either the *near* or the *var* keyword is not satisfied,
 * a new random position within the insertion volume is chosen and another trial is made." As native LAMMPS does
 * (FixDeposit::vartest at b332f28) the test comes before the rate offset and the global or local height, and a
 * rejected trial counts as one of the Q attempts. "The locations of inserted particles are taken from uniform
 * distributed random numbers, unless the *gaussian* keyword is used. Then the individual coordinates are taken from a
 * gaussian distribution of width *sigma* centered on *xmid,ymid,zmid*"; native draws x, y and z with
 * RanPark::gaussian (the second value of a pair kept for the next draw) until the point is in the region, sigma scaled
 * by the x lattice spacing.
 *
 * Region (native src/fix_deposit.cpp at b332f28: "Fix deposit region does not support a bounding box" when bboxflag
 * is 0, and "Fix deposit region cannot be dynamic"; fix_deposit.rst: "region-ID = ID of region to use as insertion volume"
 * and "It must be defined with side = in"). The draws come from the region's extent (extent_xlo .. extent_zhi of
 * native's Region), each candidate is accepted only when region->match() holds, and a rejected candidate draws
 * again without counting as an attempt (src/fix_deposit.cpp pre_exchange: "do { coord[0] = xlo + random->uniform()
 * * (xhi-xlo); ... } while (iregion->match(coord[0],coord[1],coord[2]) == 0);"). The rate offset is added after the
 * match ("adjust vertical coord by offset"), so the match tests the un-shifted point.
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

/** 2*pi as fix_deposit.cpp's MY_2PI. */
const TWO_PI = 2 * Math.PI;

/** Candidate draws for one attempt before a region that matches nothing is reported (native would loop forever). */
const MAX_REGION_DRAWS = 100000;

/**
 * The extent reaches outside the box (native FixDeposit: "Deposition region extends outside simulation box"), the box's
 * bounding box for a triclinic box (boxlo_bound, boxhi_bound).
 */
const outsideBox = (sys: System, e: RegionExtent): boolean => {
  const b = sys.state.box;
  const lo = [b.lo[0], b.lo[1], b.lo[2]], hi = [b.hi[0], b.hi[1], b.hi[2]];
  if (b.triclinic) {
    const [xy, xz, yz] = b.tilt;
    lo[0] += Math.min(0, xy, xz, xy + xz); hi[0] += Math.max(0, xy, xz, xy + xz);
    lo[1] += Math.min(0, yz); hi[1] += Math.max(0, yz);
  }
  return e.xlo < lo[0] || e.xhi > hi[0] || e.ylo < lo[1] || e.yhi > hi[1] || e.zlo < lo[2] || e.zhi > hi[2];
};

/** The extent of a region's interior (native Region::extent_*), or null when it has no bounding box (bboxflag 0). */
export interface RegionExtent { xlo: number; xhi: number; ylo: number; yhi: number; zlo: number; zhi: number }
const fromBox = (lo: number[], hi: number[]): RegionExtent => ({ xlo: lo[0], xhi: hi[0], ylo: lo[1], yhi: hi[1], zlo: lo[2], zhi: hi[2] });
export const regionExtent = (sys: System, r: Region): RegionExtent | null => {
  // native: bboxflag is 1 only for an interior region (side in); a plane and a side-out region have none
  if (!r.interior || r.dynamic) return null;
  if (r instanceof PrismRegion) {
    // region_prism.cpp: extent_xlo = MIN(xlo, xlo + xy), then MIN(extent_xlo, extent_xlo + xz), and so on
    const [xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz] = r.values();
    let exlo = Math.min(xlo, xlo + xy); exlo = Math.min(exlo, exlo + xz);
    const eylo = Math.min(ylo, ylo + yz);
    let exhi = Math.max(xhi, xhi + xy); exhi = Math.max(exhi, exhi + xz);
    const eyhi = Math.max(yhi, yhi + yz);
    return { xlo: exlo, xhi: exhi, ylo: eylo, yhi: eyhi, zlo, zhi };
  }
  if (r instanceof CompoundRegion) {
    // region_union.cpp: bboxflag only if every member has one (the min and max of the extents);
    // region_intersect.cpp: the members with a bounding box, intersected
    const ext = r.subRegions().map((m) => regionExtent(sys, m));
    if (r.style === 'union') {
      if (ext.some((e) => e === null)) return null;
      const out = { xlo: Infinity, xhi: -Infinity, ylo: Infinity, yhi: -Infinity, zlo: Infinity, zhi: -Infinity };
      for (const e of ext as RegionExtent[]) {
        out.xlo = Math.min(out.xlo, e.xlo); out.xhi = Math.max(out.xhi, e.xhi);
        out.ylo = Math.min(out.ylo, e.ylo); out.yhi = Math.max(out.yhi, e.yhi);
        out.zlo = Math.min(out.zlo, e.zlo); out.zhi = Math.max(out.zhi, e.zhi);
      }
      return out;
    }
    const have = ext.filter((e): e is RegionExtent => e !== null);
    if (!have.length) return null;
    const out = { ...have[0] };
    for (const e of have.slice(1)) {
      out.xlo = Math.max(out.xlo, e.xlo); out.xhi = Math.min(out.xhi, e.xhi);
      out.ylo = Math.max(out.ylo, e.ylo); out.yhi = Math.min(out.yhi, e.yhi);
      out.zlo = Math.max(out.zlo, e.zlo); out.zhi = Math.min(out.zhi, e.zhi);
    }
    return out;
  }
  // block, sphere, ellipsoid, cylinder and cone: bbox() is the native extent (centre +- radius, lo/hi along the axis)
  const b = r.bbox();
  return b ? fromBox(b.lo, b.hi) : null;
};

/** The rotation of a molecule insertion (the routine is in src/engine-gpl/fix/deposit_rotation.ts). */
export interface DepositRotationHook {
  rotation(r: number[], theta: number): number[][];
}

/** The lazy HOOKS entry 'deposit' (lazyStyles.ts group gpl_deposit): the native rotation of fix_deposit.cpp. */
const nativeRotation = (r: number[], theta: number): number[][] => gplHook<DepositRotationHook>('deposit').rotation(r, theta);

/** The state fix deposit keeps in a restart file (fix_deposit.cpp write_restart). */
export interface DepositRestart { seed: number; inserted: number; nfirst: number; nextStep: number; step: number }

export class FixDeposit extends Fix {
  readonly style = 'deposit';
  readonly N: number;
  readonly type: number;
  readonly M: number;
  private readonly rng: RanPark;
  private readonly regionId: string;
  private reg: InsertionRegion | null = null;
  private region: Region | null = null;
  private readonly idNext: boolean;
  private nextId: number;
  private globalRange: [number, number] | null = null;
  /** local lo hi delta: the height range and delta^2 (deltasq, scaled by the x lattice spacing squared). */
  private localRange: [number, number, number] | null = null;
  private near = 0;
  private attempt = 10;
  private rateV = 0;
  private vx: [number, number] = [0, 0];
  private vy: [number, number] = [0, 0];
  private vz: [number, number] = [0, 0];
  private target: Vec3 | null = null;
  /** var name: the equal-style variable that accepts (non-zero) or rejects (0.0) a candidate position. */
  private varName: string | null = null;
  /** set x|y|z name: the internal-style variables that receive the candidate's coordinates. */
  private readonly setNames: (string | null)[] = [null, null, null];
  /** gaussian xmid ymid zmid sigma (scaled): positions from a normal distribution instead of the region's extent. */
  private gaussian: { mid: Vec3; sigma: number } | null = null;
  private readonly molTemplateId: string | null;
  private molTemplates: MoleculeTemplate[] = [];
  private readonly molfrac: number[] | null;
  private readonly rigidId: string | null;
  private readonly shakeId: string | null;
  private readonly orient: Vec3 | null;
  private nextMolId: number;
  private rigidFix: Fix | null = null;
  private shakeFix: Fix | null = null;
  private nfirst: number;
  private nextStep: number;
  inserted = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_deposit.cpp: "dimension == 2" inserts in the y direction (the rate offset, the lateral placement and the
    // global and local heights) with a z coordinate drawn but unused, and a molecule rotates about z
    // As native LAMMPS does (src/fix_deposit.cpp FixDeposit::FixDeposit and options at b332f28): the four required
    // values, the keywords, the variables, the type, the region, the molecules, rigid and shake, then the scaling
    const illegal = () => new StyleError('Illegal fix deposit command');
    if (args.length < 4) throw illegal();
    this.N = inumeric(args[0]);
    const ntype = inumeric(typeWord(sys.labels, 'atom', args[1]));
    this.M = inumeric(args[2]);
    const seed = inumeric(args[3]);
    if (seed <= 0) throw illegal();
    let region: string | null = null;
    let idNext = false;
    let scaleBox = false;
    let molTemplateId: string | null = null;
    let molfrac: number[] | null = null;
    let rigidId: string | null = null;
    let shakeId: string | null = null;
    let orient: Vec3 | null = null;
    let i = 4;
    const need = (k: number) => { if (i + k > args.length) throw illegal(); };
    while (i < args.length) {
      const key = args[i];
      if (key === 'region') {
        need(2);
        if (!sys.regions.has(args[i + 1])) throw new StyleError(`Region ID ${args[i + 1]} for fix deposit does not exist`);
        region = args[i + 1];
        i += 2;
      } else if (key === 'var') {
        if (i + 2 > args.length) throw missingArgs('fix deposit var');
        this.varName = args[i + 1];
        i += 2;
      } else if (key === 'set') {
        if (i + 3 > args.length) throw missingArgs('fix deposit set');
        const d = ['x', 'y', 'z'].indexOf(args[i + 1]);
        // native names the variable, not the option, in this message
        if (d < 0) throw new StyleError(`Unknown fix deposit set option ${args[i + 2]}`);
        this.setNames[d] = args[i + 2];
        i += 3;
      } else if (key === 'mol') {
        need(2);
        const sets = sys.molecules.get(args[i + 1]);
        if (!sets || !sets.length) throw new StyleError('Molecule template ID for fix deposit does not exist');
        molTemplateId = args[i + 1];
        // the default fractions: equal shares, the last cumulative value 1
        molfrac = sets.map((_, k) => (k + 1) / sets.length);
        molfrac[sets.length - 1] = 1;
        i += 2;
      } else if (key === 'molfrac') {
        if (!molTemplateId || !molfrac) throw illegal();
        const n = molfrac.length;
        need(n + 1);
        // cumulative fractions; native's sum check is void (EPSILON = 1.0e6) and the last value is set to 1
        molfrac[0] = numeric(args[i + 1]);
        for (let k = 1; k < n; k++) molfrac[k] = molfrac[k - 1] + numeric(args[i + 1 + k]);
        molfrac[n - 1] = 1;
        i += n + 1;
      } else if (key === 'rigid') { need(2); rigidId = args[i + 1]; i += 2; }
      else if (key === 'shake') { need(2); shakeId = args[i + 1]; i += 2; }
      else if (key === 'id') {
        need(2);
        if (args[i + 1] !== 'max' && args[i + 1] !== 'next') throw illegal();
        idNext = args[i + 1] === 'next';
        i += 2;
      } else if (key === 'global') {
        need(3);
        this.globalRange = [numeric(args[i + 1]), numeric(args[i + 2])];
        this.localRange = null;
        i += 3;
      } else if (key === 'local') {
        need(4);
        this.localRange = [numeric(args[i + 1]), numeric(args[i + 2]), numeric(args[i + 3])];
        this.globalRange = null;
        i += 4;
      } else if (key === 'near') { need(2); this.near = numeric(args[i + 1]); i += 2; }
      else if (key === 'attempt') { need(2); this.attempt = inumeric(args[i + 1]); i += 2; }
      else if (key === 'rate') { need(2); this.rateV = numeric(args[i + 1]); i += 2; }
      else if (key === 'vx') { need(3); this.vx = [numeric(args[i + 1]), numeric(args[i + 2])]; i += 3; }
      else if (key === 'vy') { need(3); this.vy = [numeric(args[i + 1]), numeric(args[i + 2])]; i += 3; }
      else if (key === 'vz') { need(3); this.vz = [numeric(args[i + 1]), numeric(args[i + 2])]; i += 3; }
      else if (key === 'orient') {
        need(4);
        orient = [numeric(args[i + 1]), numeric(args[i + 2]), numeric(args[i + 3])];
        if (sys.dimension === 2 && (orient[0] !== 0 || orient[1] !== 0)) throw new StyleError('Illegal fix deposit orient settings');
        if (orient[0] === 0 && orient[1] === 0 && orient[2] === 0) throw new StyleError('Illegal fix deposit orient settings');
        i += 4;
      } else if (key === 'units') {
        need(2);
        if (args[i + 1] === 'box') scaleBox = true;
        else if (args[i + 1] === 'lattice') scaleBox = false;
        else throw illegal();
        i += 2;
      } else if (key === 'gaussian') {
        need(5);
        this.gaussian = { mid: [numeric(args[i + 1]), numeric(args[i + 2]), numeric(args[i + 3])], sigma: numeric(args[i + 4]) };
        i += 5;
      } else if (key === 'target') {
        need(4);
        this.target = [numeric(args[i + 1]), numeric(args[i + 2]), numeric(args[i + 3])];
        i += 4;
      } else throw illegal();
    }
    // the variable test: var with at least one set variable, an equal-style var, and internal-style set variables
    // (created when missing)
    const anySet = this.setNames.some((n) => n !== null);
    if ((!this.varName && anySet) || (this.varName && !anySet)) throw new StyleError('Incomplete use of variables in fix deposit command');
    if (this.varName) {
      const v = sys.vars.get(this.varName);
      if (!v) throw new StyleError(`Variable ${this.varName} for fix deposit does not exist`);
      if (v.style !== 'equal' && v.style !== 'timer' && v.style !== 'internal') throw new StyleError('Variable for fix deposit is invalid style');
      for (const n of this.setNames) {
        if (n === null) continue;
        const sv = sys.vars.get(n);
        if (!sv) sys.vars.setInternal(n, 0);
        else if (sv.style !== 'internal') throw new StyleError(`Variable ${n} for fix deposit is invalid style`);
      }
    }
    if (!molTemplateId && (ntype <= 0 || ntype > sys.state.ntypes)) throw new StyleError('Invalid atom type in fix deposit command');
    if (region === null) throw new StyleError('Must specify a region in fix deposit');
    const r = sys.region(region);
    // a moving interior region has a bounding box natively (bboxflag), so it fails on the dynamic check
    if (r.dynamic && r.interior) throw new StyleError('Fix deposit region cannot be dynamic');
    const e = regionExtent(sys, r);
    if (!e) throw new StyleError('Fix deposit region does not support a bounding box');
    if (r.dynamic) throw new StyleError('Fix deposit region cannot be dynamic');
    if (outsideBox(sys, e)) throw new StyleError('Deposition region extends outside simulation box');
    if (molTemplateId) {
      for (const t of sys.molecules.get(molTemplateId)!) {
        let maxT = 0;
        for (const ty of t.type) if (ty > maxT) maxT = ty;
        if (ntype + maxT <= 0 || ntype + maxT > sys.state.ntypes) throw new StyleError('Invalid atom type in fix deposit mol command');
      }
      if (!isMolecularStyle(sys.state.atomStyle)) throw new StyleError(`fix deposit mol: atom_style ${sys.state.atomStyle} cannot store molecule IDs and bonds`);
    }
    if (rigidId && !molTemplateId) throw new StyleError('Cannot use fix deposit rigid and not molecule');
    if (shakeId && !molTemplateId) throw new StyleError('Cannot use fix deposit shake and not molecule');
    if (rigidId && shakeId) throw new StyleError('Cannot use fix deposit rigid and shake');
    // native divides by M (the insertion interval) and stops with a floating-point exception for M = 0
    if (this.M === 0) throw new StyleError('fix deposit with M = 0 is refused: native LAMMPS stops with a floating-point exception for it');
    // fix_deposit.rst: "A lattice value means the distance units are in lattice spacings." and "the
    // units choice affects all the keyword values that have units of distance or velocity"; the
    // default is units lattice.
    const sc = latticeScale(sys, scaleBox ? 'box' : 'lattice', 'deposit');
    this.near *= sc[0];
    // As native LAMMPS does (src/fix_deposit.cpp at b332f28): lo, hi and rate scale with the lattice spacing of the
    // vertical dimension (y in 2d, z in 3d), deltasq = delta * delta with the x spacing squared, the gaussian centre
    // per dimension and sigma with the x spacing ("same as in region sphere")
    const sv = sys.dimension === 2 ? sc[1] : sc[2];
    this.rateV *= sv;
    if (this.globalRange) this.globalRange = [this.globalRange[0] * sv, this.globalRange[1] * sv];
    if (this.localRange) {
      const d = this.localRange[2];
      this.localRange = [this.localRange[0] * sv, this.localRange[1] * sv, d * d * (sc[0] * sc[0])];
    }
    if (this.gaussian) {
      const g = this.gaussian;
      this.gaussian = { mid: [g.mid[0] * sc[0], g.mid[1] * sc[1], g.mid[2] * sc[2]], sigma: g.sigma * sc[0] };
    }
    if (this.target) this.target = [this.target[0] * sc[0], this.target[1] * sc[1], this.target[2] * sc[2]];
    this.vx = [this.vx[0] * sc[0], this.vx[1] * sc[0]];
    this.vy = [this.vy[0] * sc[1], this.vy[1] * sc[1]];
    this.vz = [this.vz[0] * sc[2], this.vz[1] * sc[2]];
    this.rng = insertionStream(seed);
    this.type = ntype;
    this.regionId = region;
    this.molTemplateId = molTemplateId;
    this.molfrac = molfrac;
    this.rigidId = rigidId;
    this.shakeId = shakeId;
    this.orient = orient;
    this.idNext = idNext;
    this.nextId = maxAtomId(sys.state);
    this.nextMolId = maxMoleculeId(sys.state);
    this.nfirst = sys.state.step + 1;
    this.nextStep = this.nfirst;
    this.scalarFlag = true;
    const saved = takeFixRecord(sys, id, this.style)?.state as DepositRestart | undefined;
    if (saved) this.restoreState(saved);
  }

  /**
   * fix_deposit.cpp write_restart: the generator, the number inserted, the first and next steps and the step the file was
   * written at (restart checks that it is the file's).
   */
  restartState(): DepositRestart {
    return { seed: this.rng.state(), inserted: this.inserted, nfirst: this.nfirst, nextStep: this.nextStep, step: this.sys.state.step };
  }

  /** FixDeposit::restart: the generator, the counts and steps; the step must be the file's. */
  private restoreState(r: DepositRestart): void {
    if (r.step !== this.sys.state.step) throw new StyleError(`fix ${this.id}: must not reset timestep when restarting this fix`);
    this.rng.reset(r.seed);
    this.inserted = r.inserted;
    this.nfirst = r.nfirst;
    this.nextStep = r.nextStep;
  }

  init(): void {
    // As native LAMMPS does (FixDeposit::init at b332f28): the region again (it may have been deleted since), then the
    // rigid and shake fixes and their molecule templates
    const sys = this.sys;
    if (!sys.regions.has(this.regionId)) throw new StyleError(`Region ID ${this.regionId} for fix deposit does not exist`);
    if (this.molTemplateId) this.molTemplates = sys.molecules.get(this.molTemplateId) ?? [];
    if (this.rigidId) {
      const rf = sys.fixes.find((f) => f.id === this.rigidId);
      if (!rf) throw new StyleError(`Fix deposit rigid fix ID ${this.rigidId} does not exist`);
      if (typeof (rf as { addMolecule?: unknown }).addMolecule !== 'function') {
        throw new StyleError(`fix deposit rigid: fix ${this.rigidId} (${rf.style}) is not a fix rigid/small`);
      }
      const fixMol = (rf as { moleculeTemplateId?: () => string | null }).moleculeTemplateId?.() ?? null;
      if (fixMol !== this.molTemplateId) throw new StyleError('Fix deposit and rigid fix are not using the same molecule template ID');
      this.rigidFix = rf;
    }
    if (this.shakeId) {
      const sf = sys.fixes.find((f) => f.id === this.shakeId);
      if (!sf) throw new StyleError(`Fix deposit shake fix ID ${this.shakeId} does not exist`);
      if (typeof (sf as { rebuildClusters?: unknown }).rebuildClusters !== 'function') {
        throw new StyleError(`fix deposit shake: fix ${this.shakeId} (${sf.style}) is not a fix shake`);
      }
      const fixMol = (sf as { moleculeTemplateId?: () => string | null }).moleculeTemplateId?.() ?? null;
      if (fixMol !== this.molTemplateId) throw new StyleError('Fix deposit and fix shake are not using the same molecule template ID');
      this.shakeFix = sf;
    }
    const r = sys.region(this.regionId);
    const e = regionExtent(sys, r);
    if (!e) throw new StyleError('Fix deposit region does not support a bounding box');
    this.region = r;
    this.reg = { kind: 'block', xlo: e.xlo, xhi: e.xhi, ylo: e.ylo, yhi: e.yhi, zlo: e.zlo, zhi: e.zhi, xc: 0, yc: 0, radius: 0 };
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
    const region = this.region!;
    const step = s.step;
    const shift = this.rateV * (step - this.nfirst + this.M) * s.dt;
    // fix_deposit.cpp: in 2d the rate offset moves the insertion in y (coord[1] += offset), not z
    const twod = s.dimension === 2;
    const yOff = twod ? shift : 0;
    const zOff = twod ? 0 : shift;
    const mols = this.molTemplates;
    let placed: { p: Vec3; v: Vec3; atoms: Float64Array | null; rot: number[][] | null; t: MoleculeTemplate | null } | null = null;
    for (let a = 0; a < this.attempt && !placed; a++) {
      // fix_deposit.cpp: draws in the extent (or from the gaussian) until the region matches (rejections are not
      // attempts)
      let x = 0, y = 0, z = 0;
      let draws = 0;
      const g = this.gaussian;
      do {
        if (++draws > MAX_REGION_DRAWS) throw new StyleError('fix deposit: no point inside the region could be drawn (is the region empty?)');
        if (g) {
          x = g.mid[0] + this.rng.gaussian() * g.sigma;
          y = g.mid[1] + this.rng.gaussian() * g.sigma;
          z = g.mid[2] + this.rng.gaussian() * g.sigma;
        } else {
          x = reg.xlo + this.rng.uniform() * (reg.xhi - reg.xlo);
          y = reg.ylo + this.rng.uniform() * (reg.yhi - reg.ylo);
          z = reg.zlo + this.rng.uniform() * (reg.zhi - reg.zlo);
        }
      } while (!region.match(x, y, z));
      // var and set: a rejected position uses up the attempt (FixDeposit::vartest)
      if (this.varName && !this.varTest(x, y, z)) continue;
      y += yOff;
      z += zOff;
      // global and local: the vertical coordinate lo..hi above the highest atom (nearby, for local), the point or the
      // molecule centre
      const range = this.globalRange ?? this.localRange;
      if (range) {
        const top = this.highestAtom(this.localRange ? [x, y] : null);
        const h = top + range[0] + this.rng.uniform() * (range[1] - range[0]);
        if (twod) y = h;
        else z = h;
      }
      let atoms: Float64Array | null = null;
      let rot0: number[][] | null = null;
      let t: MoleculeTemplate | null = null;
      if (mols.length) {
        // fix_deposit.cpp: one draw picks the molecule, the first whose cumulative fraction is not below it
        const sel = this.rng.uniform();
        let imol = 0;
        while (imol < mols.length - 1 && sel > this.molfrac![imol]) imol++;
        t = mols[imol];
        // fix_deposit.cpp: in 2d the rotation axis is z and only the angle is drawn; in 3d the axis is drawn (unless
        // orient is given) and then the angle
        const rot = twod
          ? nativeRotation([0, 0, 1], this.rng.uniform() * TWO_PI)
          : this.orient
            ? nativeRotation([this.orient[0], this.orient[1], this.orient[2]], this.rng.uniform() * TWO_PI)
            : (() => {
              const u1 = this.rng.uniform(), u2 = this.rng.uniform(), u3 = this.rng.uniform(), u4 = this.rng.uniform();
              return nativeRotation([u1 - 0.5, u2 - 0.5, u3 - 0.5], u4 * TWO_PI);
            })();
        atoms = this.moleculeCoords(t, geometricCenter(t), rot, [x, y, z]);
        rot0 = rot;
      }
      if (this.near > 0 && this.tooNear(atoms ?? Float64Array.of(x, y, z), t !== null)) continue;
      let v: Vec3 = [
        this.vx[0] + this.rng.uniform() * (this.vx[1] - this.vx[0]),
        this.vy[0] + this.rng.uniform() * (this.vy[1] - this.vy[0]),
        this.vz[0] + this.rng.uniform() * (this.vz[1] - this.vz[0]),
      ];
      if (this.target) v = this.aimAt([x, y, z], v);
      placed = { p: [x, y, z], v, atoms, rot: t ? rot0 : null, t };
    }
    if (!placed) {
      sys.warn('Particle deposition was unsuccessful');
      return;
    }
    const t = placed.t;
    if (t && placed.atoms) {
      const n = t.natoms;
      const ids = this.idNext ? new Int32Array(n) : undefined;
      if (ids) for (let k = 0; k < n; k++) ids[k] = ++this.nextId;
      const molId = this.idNext ? ++this.nextMolId : maxMoleculeId(s) + 1;
      const firstId = appendMolecule(sys, t, this.type, placed.atoms, placed.v, molId, this.groupBit, ids, placed.rot ?? undefined);
      // fix_deposit.cpp: a template with a Molecules section gives its atoms the current maximum molecule ID plus their
      // template molecule IDs, and the maximum grows by the template's molecule count
      if (t.mol) {
        const base = s.n - n;
        let nmol = 0;
        for (let k = 0; k < n; k++) {
          s.molecule[base + k] = molId - 1 + t.mol[k];
          if (t.mol[k] > nmol) nmol = t.mol[k];
        }
        if (this.idNext) this.nextMolId += nmol - 1;
      }
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

  /**
   * var and set (FixDeposit::vartest): the set variables take the candidate's coordinates, then the var variable decides
   * (0.0 rejects the position).
   */
  private varTest(x: number, y: number, z: number): boolean {
    const c = [x, y, z];
    for (let d = 0; d < 3; d++) {
      const n = this.setNames[d];
      if (n !== null) this.sys.vars.setInternal(n, c[d]);
    }
    return this.sys.equalVariable(this.varName!) !== 0;
  }

  /**
   * Flat 3N coordinates of the molecule rotated by `rot` about its geometric center and placed at p. As fix_deposit.cpp:
   * coords = rotmat . dx first, then the insertion point is added (the same roundoff as native).
   */
  private moleculeCoords(t: MoleculeTemplate, center: number[], rot: number[][], p: Vec3): Float64Array {
    const out = new Float64Array(3 * t.natoms);
    for (let i = 0; i < t.natoms; i++) {
      const rx = t.x[3 * i] - center[0], ry = t.x[3 * i + 1] - center[1], rz = t.x[3 * i + 2] - center[2];
      out[3 * i] = (rot[0][0] * rx + rot[0][1] * ry + rot[0][2] * rz) + p[0];
      out[3 * i + 1] = (rot[1][0] * rx + rot[1][1] * ry + rot[1][2] * rz) + p[1];
      out[3 * i + 2] = (rot[2][0] * rx + rot[2][1] * ry + rot[2][2] * rz) + p[2];
    }
    return out;
  }

  /**
   * The highest vertical coordinate (y in 2d, z in 3d) of the atoms, starting from the box bottom; with a lateral
   * point only the atoms with rsq <= deltasq (minimum image; x alone in 2d, x and y in 3d) count, as fix_deposit.cpp
   * does.
   */
  private highestAtom(lateral: [number, number] | null): number {
    const s = this.sys.state;
    const dim = s.dimension === 2 ? 1 : 2;
    const deltasq = this.localRange ? this.localRange[2] : 0;
    let max = s.box.lo[dim];
    for (let i = 0; i < s.n; i++) {
      if (lateral) {
        const [dx, dy] = separation(this.sys, lateral[0] - s.x[3 * i], lateral[1] - s.x[3 * i + 1], 0);
        const rsq = dim === 1 ? dx * dx : dx * dx + dy * dy;
        if (rsq > deltasq) continue;
      }
      if (s.x[3 * i + dim] > max) max = s.x[3 * i + dim];
    }
    return max;
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

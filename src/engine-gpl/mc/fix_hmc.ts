/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MC/fix_hmc.cpp and fix_hmc.h (class
 * FixHMC; contributing authors Charlles R. A. Abreu (Federal University of Rio de Janeiro) and Ana J. Silveira (Federal
 * University of Rio de Janeiro); copyright Sandia Corporation), with src/group.cpp Group::mass and Group::vcm and
 * src/compute_ke.cpp ComputeKE::compute_scalar, and FixRigidSmall::resample_momenta of src/RIGID/fix_rigid_small.cpp (the rigid
 * keyword). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_hmc.html (verbatim): "This fix implements the hybrid or Hamiltonian Monte Carlo (HMC) algorithm. The
 * basic idea is to use molecular dynamics (MD) to generate trial MC "moves" which are then accepted or rejected via the
 * Metropolis criterion."
 *
 * Native behaviour kept here (fix_hmc.cpp at b332f28):
 *  - two Park generators: random = RanPark(seed + me) advanced by 100 uniforms (the velocity draws), random_equal =
 *    RanPark(seed) (the Metropolis uniform, drawn only when the energy change is not negative).
 *  - setup() of every run (first_setup_complete is never set): new velocities, PE from compute pe, KE from compute ke, and
 *    the state saved.
 *  - end_of_step() every N steps: dE = (PE - PE0) + (KE - KE0); accept when dE < 0, else when uniform() <= exp(-dE/kT).
 *    Accept: save the state. Reject: restore the saved per-atom state (the global energies and virials restored; the
 *    force array keeps the rejected trial's forces for the next half-kick) and reneighbor on the next step. Velocities
 *    are drawn again after a rejection, and after an acceptance with resample yes.
 *  - random_velocities(): per owned atom in local order, v[i][j] = sqrt(KT/m) gaussian() for j < dimension (the z velocity
 *    of a 2d system is left as it is); with mom yes the group's centre-of-mass velocity (Group::vcm) is subtracted.
 *
 *  - rigid ID (fix rigid/small or rigid/nve/small): setup draws the atom velocities, then resample_momenta draws the bodies'
 *    centre-of-mass and angular velocities (per body in the native storage order: vcm_j then the body-frame omega_j for
 *    each axis, omega from the principal moments, the mom option on the bodies' mass-weighted vcm) and sets the atom
 *    velocities from them; later moves resample the bodies only. The body data travel with their owner atoms in the
 *    saved state (pack_exchange), so a rejection brings the bodies back and stores them in the saved atom order.
 *
 * Not supported (named StyleErrors): a change in the number of atoms during a move. Per-atom arrays of other fixes
 * (native restores them through their exchange callbacks) are not saved, except the rigid bodies.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import type { CustomProp } from '../../engine/types';
import { RanPark } from '../random_park';
import { exp as libmExp } from '../libm/exp';
import { matvec, omegaToAngmom, qToExyz } from '../rigid/math_extra';
import { nativeGhosts } from '../neighbor/nativeOrder';

/** The body data fix hmc reads and writes (src/engine/fix/rigid.ts, rigid/small). */
interface HmcBody {
  owner: number;
  mass: number;
  inertia: number[];
  q: number[];
  vcm: number[];
  omega: number[];
  angmom: number[];
}

/** The rigid/small fix as fix hmc uses it (src/engine/fix/rigid.ts). */
interface HmcRigid {
  readonly id: string;
  readonly style: string;
  readonly virial: Float64Array;
  bodiesInNativeOrder(): HmcBody[];
  setBodyNativeOrder(ownerIds: readonly number[]): void;
  velocitiesFromBodies(virialFactor: number): void;
  snapshotBodies(): object[];
  restoreBodies(snap: object[]): void;
}

/** Fix styles that set box_change in native LAMMPS (fix_deform.cpp, fix_nh.cpp with a barostat, ...). */
const BOX_CHANGE = /^(deform|deform\/pressure|balance|box\/relax|press\/berendsen|press\/langevin|npt\/cauchy|msst|qbmsst|bocs|append\/atoms|gemc|ipi|nphug)$|^(np[th]|rigid\/np[th]|tgnp[th])(\/|$)|\/np[th](\/|$)/;

/** The per-atom state native packs with AtomVec::pack_exchange (the arrays the engine keeps per atom). */
interface Saved {
  n: number;
  x: Float64Array;
  v: Float64Array;
  image: Int32Array;
  order: Int32Array;
  type: Int32Array;
  mask: Int32Array;
  q: Float64Array;
  rmass: Float64Array | null;
  radius: Float64Array | null;
  omega: Float64Array | null;
  angmom: Float64Array | null;
  mu: Float64Array | null;
  sp: Float64Array | null;
  shape: Float64Array | null;
  quat: Float64Array | null;
  custom: [string, CustomProp][];
  /** the rigid bodies (rigid keyword) */
  bodies: object[] | null;
}

const copy = <T extends Float64Array | Int32Array>(a: T | null): T | null => (a ? (a.slice() as T) : null);

const logical = (w: string | undefined, what: string): boolean => {
  // utils::logical: yes/no, true/false, on/off, 1/0
  if (w === 'yes' || w === 'true' || w === 'on' || w === '1') return true;
  if (w === 'no' || w === 'false' || w === 'off' || w === '0') return false;
  if (w === undefined) throw new StyleError(`Illegal ${what} command: missing argument(s)`);
  throw new StyleError(`Expected boolean parameter instead of '${w}' in input script or data file`);
};

const intArg = (w: string): number => {
  if (!/^[-+]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

const numArg = (w: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return v;
};

export class FixHmc extends Fix {
  readonly style = 'hmc';
  private readonly KT: number;
  private readonly mbeta: number;
  private readonly momFlag: boolean;
  private readonly resampleOnAccept: boolean;
  /** random = new RanPark(lmp, seed + comm->me), 100 uniforms discarded; random_equal = new RanPark(lmp, seed). */
  private readonly random: RanPark;
  private readonly randomEqual: RanPark;
  private nattempts = 0;
  private naccepts = 0;
  private deltaPE = 0;
  private deltaKE = 0;
  private PE = 0;
  private KE = 0;
  private saved: Saved | null = null;
  /** The force array of a rejected trial, put back after the restored state's energies are computed. */
  private trialForce: Float64Array | null = null;
  private nextReneighbor = -1;
  private readonly rigidId: string | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_hmc.cpp: "if (narg < 5) utils::missing_cmd_args(FLERR, "fix hmc", error)" (and arg[5] is read); the engine's
    // args start at native arg[3] (N)
    if (args.length < 2) throw new StyleError('Illegal fix hmc command: missing argument(s)');
    // with N and the seed only, native reads the missing arg[5] as NULL
    if (args.length < 3) throw new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file');
    this.nevery = intArg(args[0]);
    // native reads N without a check and takes ntimestep % nevery; N < 1 is refused here instead of dividing by zero
    if (this.nevery < 1) throw new StyleError(`Illegal fix hmc command: N must be a positive integer, got ${this.nevery}`);
    const seed = intArg(args[1]);
    const temp = numArg(args[2]);
    if (seed <= 0) throw new StyleError('Fix hmc seed must be > 0');
    if (temp <= 0) throw new StyleError('Fix hmc temperature must be > 0.0');
    const u = sys.state.units;
    this.KT = (u.boltz * temp) / u.mvv2e;
    this.mbeta = -1.0 / (u.boltz * temp);
    let mom = true, resample = false;
    for (let iarg = 3; iarg < args.length;) {
      const key = args[iarg];
      if (key === 'mom') {
        mom = logical(args[iarg + 1], 'hmc mom');
        iarg += 2;
      } else if (key === 'resample') {
        resample = logical(args[iarg + 1], 'hmc resample');
        iarg += 2;
      } else if (key === 'rigid') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal hmc rigid command: missing argument(s)');
        const rid = args[iarg + 1];
        const ifix = sys.fixes.find((f) => f.id === rid);
        if (!ifix) throw new StyleError(`Unknown rigid fix id ${rid} for fix hmc`);
        const style = (ifix as { style?: string }).style ?? '';
        if (!/^rigid\/small/.test(style) && !/^rigid\/nve\/small/.test(style)) {
          throw new StyleError(`Fix ID ${rid} for fix hmc does not point to fix rigid/small or rigid/nve/small`);
        }
        this.rigidId = rid;
        iarg += 2;
      } else {
        throw new StyleError(`Unknown fix hmc keyword ${key}`);
      }
    }
    this.momFlag = mom;
    this.resampleOnAccept = resample;
    this.random = new RanPark(seed);
    for (let i = 0; i < 100; i++) this.random.uniform();
    this.randomEqual = new RanPark(seed);
    this.scalarFlag = true;
    this.extscalar = 0;
    this.vectorFlag = true;
    this.sizeVector = 5;
    this.extvector = 0;
  }

  /** FixHMC::init: the box may not change, the integrators should be NVE, no fix after this one adds to the virial. */
  init(): void {
    const sys = this.sys;
    for (const f of sys.fixes) {
      const st = (f as { style?: string }).style ?? '';
      if (BOX_CHANGE.test(st)) throw new StyleError('Fix hmc is incompatible with fixes that change box size or shape');
    }
    for (const f of sys.fixes) {
      const st = (f as { style?: string }).style ?? '';
      if (f.timeIntegrate && !/^rigid/.test(st) && !/^nve/.test(st)) sys.warn?.(`Non NVE time integration fix ${f.id} ${st} found`);
    }
    let pastThis = false;
    for (const f of sys.fixes) {
      if (pastThis && f.virialGlobal) {
        sys.log(`Fix ${(f as { style?: string }).style ?? ''} with id ${f.id} defined after fix hmc.`);
        throw new StyleError('Fix hmc cannot precede fixes that contribute to the system pressure');
      }
      if (f === this) pastThis = true;
    }
  }

  /** The rigid/small fix of the rigid keyword (looked up again at each use, as native init does). */
  private rigid(): HmcRigid | null {
    if (!this.rigidId) return null;
    const ifix = this.sys.fixes.find((f) => f.id === this.rigidId) as unknown as HmcRigid | undefined;
    if (!ifix) throw new StyleError(`Unknown rigid fix id ${this.rigidId} for fix hmc`);
    if ((!/^rigid\/small/.test(ifix.style) && !/^rigid\/nve\/small/.test(ifix.style)) || typeof ifix.bodiesInNativeOrder !== 'function') {
      throw new StyleError(`Fix ID ${this.rigidId} for fix hmc does not point to fix rigid/small or rigid/nve/small`);
    }
    return ifix;
  }

  /**
   * The bodies native keeps in front of the live ones: restore_saved_state unpacks the saved bodies after the existing
   * ones (nlocal_body is not reset), so each rejected move leaves a copy of the body list before it. resample_momenta
   * still draws for them (with the mask of the atom at their old local index) and counts their masses in the mom sum.
   */
  private stale: { mass: number; inertia: number[]; ilocal: number }[] = [];
  /**
   * vglobal of the rigid fix: native saves the global virials (pair, bonded, kspace and the fixes', the rigid one
   * included) at the end of setup and of every end_of_step, and puts them back on a rejection; the engine recomputes the
   * force-field virials of the restored configuration and keeps the rigid fix's here.
   */
  private savedRigidVirial: Float64Array | null = null;

  /** nghost_body: the ghost images of body owners after native borders() at the current positions. */
  private nativeGhostBodyCount(owners: Set<number>): number {
    const sys = this.sys, s = sys.state;
    const order = nativeOrder(s);
    const n = order.length;
    const xn = new Float64Array(3 * n);
    for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) xn[3 * k + c] = s.x[3 * order[k] + c];
    const gh = nativeGhosts(xn, n, sys.geom, sys.nb.cutghost, s.dimension === 2);
    let count = 0;
    for (let k = n; k < gh.nall; k++) if (owners.has(s.id[order[gh.root[k]]])) count++;
    return count;
  }

  /** FixRigidSmall::resample_momenta(groupbit, mom_flag, random, KT). */
  private rigidBodyRandomVelocities(): void {
    const rigid = this.rigid()!;
    const s = this.sys.state;
    const idx = new Map<number, number>();
    for (let i = 0; i < s.n; i++) idx.set(s.id[i], i);
    const order = nativeOrder(s);
    const live = rigid.bodiesInNativeOrder();
    const bit = this.groupBit;
    // the native body array: the stale copies, then the live bodies
    type Entry = { mass: number; inertia: number[]; inGroup: boolean; body: HmcBody | null; vcm: number[]; omega: number[] };
    const entries: Entry[] = [];
    for (const st of this.stale) {
      entries.push({ mass: st.mass, inertia: st.inertia, inGroup: (s.mask[order[st.ilocal]] & bit) !== 0, body: null, vcm: [0, 0, 0], omega: [0, 0, 0] });
    }
    for (const b of live) entries.push({ mass: b.mass, inertia: b.inertia, inGroup: (s.mask[idx.get(b.owner)!] & bit) !== 0, body: b, vcm: [0, 0, 0], omega: [0, 0, 0] });
    let totalMass = 0;
    const vcm = [0.0, 0.0, 0.0];
    const wbody = [0, 0, 0];
    for (const e of entries) {
      if (!e.inGroup) continue;
      const bmass = e.mass;
      const stdev = Math.sqrt(this.KT / bmass);
      totalMass += bmass;
      for (let j = 0; j < 3; j++) {
        e.vcm[j] = stdev * this.random.gaussian();
        vcm[j] += e.vcm[j] * bmass;
        wbody[j] = e.inertia[j] > 0.0 ? Math.sqrt(this.KT / e.inertia[j]) * this.random.gaussian() : 0.0;
      }
      if (e.body) {
        const { ex, ey, ez } = qToExyz(e.body.q);
        matvec(ex, ey, ez, wbody, e.omega);
      }
    }
    if (this.momFlag && totalMass > 0.0) {
      for (let j = 0; j < 3; j++) vcm[j] /= totalMass;
      for (const e of entries) if (e.inGroup) for (let j = 0; j < 3; j++) e.vcm[j] -= vcm[j];
    }
    for (const e of entries) {
      if (!e.body || !e.inGroup) continue;
      e.body.vcm = e.vcm.slice();
      e.body.omega = e.omega.slice();
    }
    // angular momenta: native loops "for (ibody = 0; ibody < ntotal; ibody++)" with ntotal = nghost_body, so only the
    // first nghost_body entries of the body array take the angular momentum of their new omega
    const nghost = this.nativeGhostBodyCount(new Set(live.map((b) => b.owner)));
    for (let k = 0; k < Math.min(nghost, entries.length); k++) {
      const b = entries[k].body;
      if (!b) continue;
      const { ex, ey, ez } = qToExyz(b.q);
      const m = [0, 0, 0];
      omegaToAngmom(b.omega, ex, ey, ez, b.inertia, m);
      b.angmom = m;
    }
    // set_v(): with the virial of the step being accumulated (setup, and the steps whose pressure is computed), native
    // adds one half of x . f_c of the velocity change to the rigid fix's virial
    rigid.velocitiesFromBodies(0.5);
    this.sys.refreshComputes();
  }

  reneighborDue(step: number): boolean {
    return step === this.nextReneighbor;
  }

  /** compute pe (the thermo_pe compute): the potential energy of the current step. */
  private computePE(): number {
    return this.sys.compute('thermo_pe').scalarValue();
  }

  /** ComputeKE::compute_scalar over all atoms in local order, times 0.5 mvv2e. */
  private computeKE(): number {
    const s = this.sys.state;
    const v = s.v;
    let ke = 0.0;
    for (const i of nativeOrder(s)) {
      ke += massOf(s, i) * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
    }
    return ke * (0.5 * s.units.mvv2e);
  }

  /** FixHMC::setup: draw velocities, PE and KE of the start, save the state (every run). */
  setup(): void {
    // native: fix_rigid->setup first (the engine's rigid fix, defined before this one, has run its setup), then the atom
    // velocities, then with the rigid keyword the bodies' momenta
    this.randomVelocities();
    if (this.rigidId) this.rigidBodyRandomVelocities();
    this.PE = this.computePE();
    this.KE = this.computeKE();
    this.saveCurrentState();
    if (this.rigidId) this.savedRigidVirial = this.rigid()!.virial.slice();
  }

  /** FixHMC::end_of_step: the Metropolis test, then save or restore, then new velocities if needed. */
  endOfStep(): void {
    const s = this.sys.state;
    this.nattempts++;
    const newPE = this.computePE();
    const newKE = this.computeKE();
    this.deltaPE = newPE - this.PE;
    this.deltaKE = newKE - this.KE;
    const deltaE = this.deltaPE + this.deltaKE;
    let accept = deltaE < 0.0;
    if (!accept) accept = this.randomEqual.uniform() <= libmExp(this.mbeta * deltaE);
    if (accept) {
      this.naccepts++;
      this.PE = newPE;
      this.KE = newKE;
      this.saveCurrentState();
    } else {
      this.restoreSavedState();
      this.nextReneighbor = s.step + 1;
    }
    if (!accept && this.rigidId && this.savedRigidVirial) this.rigid()!.virial.set(this.savedRigidVirial);
    if (!accept || this.resampleOnAccept) {
      if (this.rigidId) this.rigidBodyRandomVelocities();
      else this.randomVelocities();
      this.KE = this.computeKE();
    }
    // the global virials saved for the next move are those at the end of this one
    if (this.rigidId) this.savedRigidVirial = this.rigid()!.virial.slice();
    if (!accept) {
      // the energies (and virial) of the restored configuration, then the trial forces back: native restores the global
      // energies and keeps atom->f of the rejected trial for the next half-kick
      const trialForce = this.trialForce!;
      this.sys.forces();
      s.f.set(trialForce);
      this.trialForce = null;
    }
  }

  /** FixHMC::save_current_state: the owned atoms as pack_exchange stores them. */
  private saveCurrentState(): void {
    const s = this.sys.state;
    const n = s.n;
    const custom: [string, CustomProp][] = [];
    for (const [name, p] of s.custom) custom.push([name, { int: p.int, cols: p.cols, data: p.data.slice() }]);
    this.saved = {
      n,
      x: s.x.slice(0, 3 * n), v: s.v.slice(0, 3 * n), image: s.image.slice(0, 3 * n), order: nativeOrder(s).slice(),
      type: s.type.slice(0, n), mask: s.mask.slice(0, n), q: s.q.slice(0, n),
      rmass: copy(s.rmass), radius: copy(s.radius), omega: copy(s.omega), angmom: copy(s.angmom), mu: copy(s.mu),
      sp: copy(s.sp), shape: copy(s.shape), quat: copy(s.quat), custom,
      bodies: this.rigidId ? this.rigid()!.snapshotBodies() : null,
    };
  }

  /**
   * FixHMC::restore_saved_state: the saved atoms back (unpack_exchange in the saved order), the fixes' pre_exchange and
   * domain->pbc(). Native also restores the saved global energies and virials; endOfStep recomputes them for the restored
   * configuration (the same configuration, so the same values) and then puts the trial forces back, as native keeps them.
   */
  private restoreSavedState(): void {
    const sys = this.sys;
    const s = sys.state;
    const sv = this.saved;
    if (!sv) return;
    if (s.n !== sv.n) throw new StyleError(`fix ${this.id} hmc: the number of atoms changed during an MC move (${sv.n} -> ${s.n}), which the browser engine does not restore`);
    this.trialForce = s.f.slice();
    const preOrder = nativeOrder(s).slice();
    s.x.set(sv.x); s.v.set(sv.v); s.image.set(sv.image);
    s.type.set(sv.type); s.mask.set(sv.mask); s.q.set(sv.q);
    if (sv.rmass && s.rmass) s.rmass.set(sv.rmass);
    if (sv.radius && s.radius) s.radius.set(sv.radius);
    if (sv.omega && s.omega) s.omega.set(sv.omega);
    if (sv.angmom && s.angmom) s.angmom.set(sv.angmom);
    if (sv.mu && s.mu) s.mu.set(sv.mu);
    if (sv.sp && s.sp) s.sp.set(sv.sp);
    if (sv.shape && s.shape) s.shape.set(sv.shape);
    if (sv.quat && s.quat) s.quat.set(sv.quat);
    for (const [name, p] of sv.custom) {
      const cur = s.custom.get(name);
      if (cur) cur.data.set(p.data);
    }
    // the atoms come back in the saved local order
    s.order = sv.order.slice();
    // the rigid bodies come back with their owner atoms (unpack_exchange appends each body in the saved atom order)
    if (sv.bodies) {
      const rigid = this.rigid()!;
      // the bodies before the restore stay in front (see stale), at the local indices their owners had
      const rank = new Map<number, number>();
      preOrder.forEach((i, r) => rank.set(s.id[i], r));
      for (const b of rigid.bodiesInNativeOrder()) this.stale.push({ mass: b.mass, inertia: b.inertia.slice(), ilocal: rank.get(b.owner)! });
      rigid.restoreBodies(sv.bodies);
      const owners = new Set(rigid.bodiesInNativeOrder().map((b) => b.owner));
      const ordered: number[] = [];
      for (const i of sv.order) if (owners.has(s.id[i])) ordered.push(s.id[i]);
      rigid.setBodyNativeOrder(ordered);
    }
    for (const f of sys.fixes) f.preExchange?.();
    sys.pbc();
    sys.bump();
  }

  /** FixHMC::random_velocities. */
  private randomVelocities(): void {
    const s = this.sys.state;
    const v = s.v;
    const dimension = s.dimension;
    const bit = this.groupBit;
    const order = nativeOrder(s);
    for (const i of order) {
      if (!(s.mask[i] & bit)) continue;
      const stdev = Math.sqrt(this.KT / massOf(s, i));
      for (let j = 0; j < dimension; j++) v[3 * i + j] = stdev * this.random.gaussian();
    }
    if (this.momFlag) {
      // group->vcm(igroup, group->mass(igroup), vcm): both sums over the owned atoms in local order
      let masstotal = 0.0;
      for (const i of order) if (s.mask[i] & bit) masstotal += massOf(s, i);
      const p = [0.0, 0.0, 0.0];
      for (const i of order) {
        if (!(s.mask[i] & bit)) continue;
        const massone = massOf(s, i);
        p[0] += v[3 * i] * massone;
        p[1] += v[3 * i + 1] * massone;
        p[2] += v[3 * i + 2] * massone;
      }
      if (masstotal > 0.0) { p[0] /= masstotal; p[1] /= masstotal; p[2] /= masstotal; }
      for (const i of order) {
        if (!(s.mask[i] & bit)) continue;
        for (let j = 0; j < dimension; j++) v[3 * i + j] -= p[j];
      }
    }
    // new velocities: the compute values are stale, the forces are not
    this.sys.refreshComputes();
  }

  computeScalar(): number {
    return this.naccepts / Math.max(1, this.nattempts);
  }

  computeVector(i: number): number {
    switch (i) {
      case 0: return this.naccepts;
      case 1: return this.nattempts;
      case 2: return this.deltaPE;
      case 3: return this.deltaKE;
      case 4: return this.deltaPE + this.deltaKE;
      default: return 0.0;
    }
  }
}

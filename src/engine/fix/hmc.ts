import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { nativeOrder, massOf } from '../atoms';
import type { CustomProp } from '../types';
import { RanPark, Rng } from '../rng';
import { kineticEnergy } from '../observables';
import { num, posInt } from './pour';

/*
 * fix ID group-ID hmc N seed T keyword values ... — docs.lammps.org/fix_hmc.html
 * (plans/lammps-docs/fix_hmc.rst). "This fix implements the hybrid or
 * Hamiltonian Monte Carlo (HMC) algorithm. The basic idea is to use molecular
 * dynamics (MD) to generate trial MC "moves" which are then accepted or
 * rejected via the Metropolis criterion." Supported keywords are rigid,
 * resample and mom; everything else is a StyleError naming it.
 *
 * Algorithm, quoted from the doc: "(1) The configuration of the system is
 * stored along with its current total energy." "(2) The system is time
 * integrated in the NVE ensemble for the specified N MD steps and the new
 * energy is calculated." Step (3) accepts with the doc's probability
 * p_acc = min(1, exp(-ΔH / k_B T)); "The idea
 * of HMC is to use a timestep large enough that total energy is not conserved.
 * The change in total energy (the Hamiltonian) is what the Metropolis
 * criterion is based on, not the change in potential energy." "(4) If
 * accepted, the new configuration becomes the starting point for the next
 * trial MC "move". If resample is yes then the velocities are resampled at
 * this point as well." "(5) If rejected, the old configuration (from N steps
 * ago) is restored and new momenta (velocities) are assigned to each particle
 * in the fix group by randomly resampling from a normal distribution at the
 * specified temperature" T. The doc's restriction: "only per-atom data is
 * restored on MC move rejection, so anything which adds or remove particles,
 * changes the box size, or has some external state not dependent on per-atom
 * data will have undefined behavior." The timestep counter is not rewound.
 * mom: "If mom = yes (default), the linear momentum of the ensemble of
 * velocities is zeroed. If mom = no, the linear momentum of the ensemble of
 * velocities is not zeroed." resample: default no. The fix must be run with
 * fix nve (or the rigid integrator) doing the MD; this fix never integrates.
 *
 * Random numbers. Native LAMMPS seeds its own Park generator from the seed
 * argument and draws its own stream for the momentum resampling. Measured with
 * native LAMMPS (black box): the velocity stream is RanPark(seed) advanced
 * past exactly 100 discarded draws (independent of the atom count and of the
 * seed; fix pour and fix deposit discard 30, fix widom and fix mol/swap none),
 * then per atom in native order, three gaussian() draws (the polar method of
 * rng.ts, v2 f first) for x, y, z. The velocity is v = g sqrt(k_B T / (mvv2e
 * m)) with k_B T = units.boltz * T; verified in lj (sigma = sqrt(T/m)) and
 * for unequal masses (m = 1 and 4, T = 2: the m = 4 component is scaled by
 * sqrt(1/4)). With mom = yes the mass-weighted centre-of-mass velocity of the
 * fix group is subtracted after scaling (measured: two equal atoms, seed
 * 12345, T = 2 give v = ±(-1.5305934230859495, 0.16398419429743805,
 * -0.65258234554116346), i.e. the raw gaussians minus their mean times
 * sqrt(2)).
 *
 * Measured with native LAMMPS (black box): the fix resamples every particle's
 * velocity at the start of a run (fix setup) before the first MC move, and the
 * very first MC move is always accepted whatever its energy change (a
 * warm-up; measured over 129 seeds at T = 2, 25 of which had ΔH > 0 up to
 * +9.6). Every later move is accepted or rejected by the Metropolis test.
 *
 * The Metropolis draw itself is NOT from the velocity stream: measured with
 * native LAMMPS (black box) on a rejecting move, the resampled velocities
 * begin exactly where the previous resample's gaussians ended, with no
 * intervening uniform (e.g. seed 12345, T = 1e-4, resample no: setup
 * gaussians at stream indices 100..105, the step-2 rejection's gaussians at
 * 106). The decision stream is a separate generator whose seed/advance could
 * not be pinned down as a black box (no fixed-offset RanPark or RanMars
 * stream reproduces the accept/reject sequence over 129 seeds); the engine
 * therefore uses its own Rng for the Metropolis uniform and this file does not
 * claim bit-parity for stochastic decisions. In the deterministic limits
 * (T -> 0, accept iff ΔH < 0; T -> infinity, always accept) the whole
 * trajectory is decided by physics and the velocity stream above, so those
 * regimes reproduce native exactly (see the w30hmc_* oracle cases).
 *
 * Output (measured with native LAMMPS, black box): the scalar is
 * accepted/attempted (the doc: "The scalar is the fraction (0-1) of attempted
 * MC moves which have been accepted."). The length-5 vector is, in order,
 * cumulative accepted, cumulative attempted, ΔPE, ΔKE, ΔH for the last trial.
 * The doc calls component 2 "cumulative number of rejected moves", but the
 * binary (2 Sep 2026) reports the cumulative number of attempted moves there:
 * at the first accepted move both are 1 (the flexible-melt log's step 50650
 * row has f[1] = 216, f[2] = 496, scalar = 216/496 = 0.43548387). The engine
 * follows the measured binary.
 *
 * Forces after a rejection. Measured with native LAMMPS (black box) on the
 * reject case (T = 1e-9, resample no, timestep 0.01), dumping x v f after
 * every step: at the step-3 rejection the positions are restored to their
 * step-2 values but f is -25.2209269560346 (the force of the rejected trial's
 * END configuration), not the force at the restored positions (-59.4713487634416,
 * which still appears at step 2). The thermo PotEng on that row is the restored
 * 0.990123149543619, i.e. the cached energy of the restored configuration, not
 * the trial energy. So the restore puts back per-atom x and v, while the force
 * array and the cached energies keep their pre-restore/trial values; the first
 * half-kick of the next MD segment therefore uses the rejected trial's force.
 * This matters: at T -> 0 the step-4 decision flips between the two choices
 * (native rejects at both step 3 and step 4, accepting only at step 5). The
 * engine reproduces it by recomputing the restored configuration's energy (so
 * the cached accum and thermo pe are the restored values) and then putting the
 * trial force array back before the next step's half-kick.
 */

type Vec = Float64Array;

/** A copy of the per-atom arrays a rejected move must restore. */
interface Snapshot {
  x: Vec;
  v: Vec;
  image: Int32Array;
  omega: Vec | null;
  mu: Vec | null;
  quat: Vec | null;
  angmom: Vec | null;
  shape: Vec | null;
  q: Vec | null;
  custom: [string, CustomProp][];
}

export class FixHmc extends Fix {
  readonly style = 'hmc';
  private readonly N: number;
  private readonly T: number;
  private readonly resample: boolean;
  private readonly mom: boolean;
  /** Velocity stream: RanPark(seed) advanced past 100 draws (measured). */
  private readonly velocityRng: RanPark;
  /** Metropolis stream: the engine's own generator (see the header). */
  private readonly decisionRng: Rng;
  private discarded = false;
  private first = true;
  private snap: Snapshot | null = null;
  private storedPE = 0;
  private storedKE = 0;
  private accepted = 0;
  private attempted = 0;
  private lastDPE = 0;
  private lastDKE = 0;
  private lastDH = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('usage: fix ID group-ID hmc N seed T keyword values ...');
    this.N = posInt(args[0], 'N');
    const seed = posInt(args[1], 'seed');
    this.T = num(args[2], 'T');
    if (!(this.T > 0)) throw new StyleError(`fix hmc: T must be > 0, got '${args[2]}'`);
    let resample = false;
    let mom = true;
    let rigidId: string | null = null;
    for (let i = 3; i < args.length;) {
      const key = args[i];
      if (key === 'resample') {
        const w = args[i + 1];
        if (w !== 'yes' && w !== 'no') throw new StyleError(`fix hmc: resample must be yes or no, got '${w ?? ''}'`);
        resample = w === 'yes';
        i += 2;
      } else if (key === 'mom') {
        const w = args[i + 1];
        if (w !== 'yes' && w !== 'no') throw new StyleError(`fix hmc: mom must be yes or no, got '${w ?? ''}'`);
        mom = w === 'yes';
        i += 2;
      } else if (key === 'rigid') {
        rigidId = args[i + 1];
        if (!rigidId) throw new StyleError('fix hmc: rigid needs a fix rigid/small ID');
        i += 2;
      } else {
        throw new StyleError(`fix hmc: unknown keyword '${key}'`);
      }
    }
    if (rigidId) {
      // docs.lammps.org/fix_hmc.html: "The rigidID value should be the ID of a
      // fix rigid/small or fix rigid/nve/small command which defines the rigid
      // bodies. Its integrator will be used during the MD timesteps."
      throw new StyleError(`fix hmc rigid ${rigidId}: the browser engine's fix rigid does not expose the per-body state reinitialisation that HMC needs after a rejected move, so rigid HMC is not supported`);
    }
    this.resample = resample;
    this.mom = mom;
    this.velocityRng = new RanPark(seed);
    this.decisionRng = new Rng(seed);
    this.nevery = this.N;
    this.scalarFlag = true;
    this.extscalar = 0;
    this.vectorFlag = true;
    this.sizeVector = 5;
    this.extvector = 0;
  }

  /** Total potential energy, exactly as compute pe sums its terms. */
  private potentialEnergy(): number {
    const a = this.sys.forces();
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp + this.sys.fixEnergy();
  }

  private kinetic(): number {
    return kineticEnergy(this.sys.state);
  }

  private snapshot(): Snapshot {
    const s = this.sys.state;
    const custom: [string, CustomProp][] = [];
    for (const [name, p] of s.custom) custom.push([name, { int: p.int, cols: p.cols, data: p.data.slice() }]);
    return {
      x: s.x.slice(), v: s.v.slice(), image: s.image.slice(),
      omega: s.omega ? s.omega.slice() : null,
      mu: s.mu ? s.mu.slice() : null,
      quat: s.quat ? s.quat.slice() : null,
      angmom: s.angmom ? s.angmom.slice() : null,
      shape: s.shape ? s.shape.slice() : null,
      q: s.propQ ? s.q.slice() : null,
      custom,
    };
  }

  private restore(snap: Snapshot): void {
    const s = this.sys.state;
    s.x.set(snap.x);
    s.v.set(snap.v);
    s.image.set(snap.image);
    if (snap.omega && s.omega) s.omega.set(snap.omega);
    if (snap.mu && s.mu) s.mu.set(snap.mu);
    if (snap.quat && s.quat) s.quat.set(snap.quat);
    if (snap.angmom && s.angmom) s.angmom.set(snap.angmom);
    if (snap.shape && s.shape) s.shape.set(snap.shape);
    if (snap.q && s.propQ) s.q.set(snap.q);
    for (const [name, p] of snap.custom) {
      const cur = s.custom.get(name);
      if (cur) cur.data.set(p.data);
    }
    this.sys.bump();
  }

  /** Resamples the fix group's velocities from the normal distribution at T (measured stream/order). */
  private resampleVelocities(): void {
    const s = this.sys.state;
    const rng = this.velocityRng;
    const c0 = Math.sqrt((s.units.boltz * this.T) / s.units.mvv2e);
    const ord = nativeOrder(s);
    for (let k = 0; k < s.n; k++) {
      const i = ord[k];
      if (!(s.mask[i] & this.groupBit)) continue;
      const c = c0 / Math.sqrt(massOf(s, i));
      s.v[3 * i] = rng.gaussian() * c;
      s.v[3 * i + 1] = rng.gaussian() * c;
      s.v[3 * i + 2] = s.dimension === 2 ? 0 : rng.gaussian() * c;
    }
    if (this.mom) this.zeroGroupMomentum();
    this.sys.bump();
  }

  /** Subtracts the mass-weighted centre-of-mass velocity of the fix group (mom = yes). */
  private zeroGroupMomentum(): void {
    const s = this.sys.state;
    const p = [0, 0, 0];
    let mtot = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      mtot += m;
      p[0] += m * s.v[3 * i];
      p[1] += m * s.v[3 * i + 1];
      p[2] += m * s.v[3 * i + 2];
    }
    if (!(mtot > 0)) return;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      s.v[3 * i] -= p[0] / mtot;
      s.v[3 * i + 1] -= p[1] / mtot;
      s.v[3 * i + 2] -= p[2] / mtot;
    }
  }

  /** Verlet::setup: resample velocities, store the configuration and its total energy. */
  setup(): void {
    if (!this.discarded) {
      for (let k = 0; k < 100; k++) this.velocityRng.uniform();
      this.discarded = true;
    }
    this.resampleVelocities();
    this.snap = this.snapshot();
    this.storedPE = this.potentialEnergy();
    this.storedKE = this.kinetic();
    this.first = true;
  }

  /** Every N steps: the Metropolis test, restore/resample, and the next stored state. */
  endOfStep(): void {
    const s = this.sys.state;
    if (this.snap === null) return;
    const pe = this.potentialEnergy();
    const ke = this.kinetic();
    this.lastDPE = pe - this.storedPE;
    this.lastDKE = ke - this.storedKE;
    this.lastDH = this.lastDPE + this.lastDKE;

    let accept: boolean;
    if (this.first) {
      accept = true;
      this.first = false;
    } else {
      const p = this.lastDH <= 0 ? 1 : Math.exp(-this.lastDH / (s.units.boltz * this.T));
      accept = this.decisionRng.uniform() < p;
    }
    this.attempted++;
    if (accept) {
      this.accepted++;
      if (this.resample) this.resampleVelocities();
      this.snap = this.snapshot();
      this.storedPE = pe;
      this.storedKE = this.kinetic();
    } else {
      // Native keeps the rejected trial's force for the first half-kick of the
      // next segment (see the header) and its cached energies are those of the
      // restored configuration. Recompute the energies at the restored x (which
      // also refreshes the neighbor lists), then put the trial force array back.
      const trialForce = s.f.slice();
      const restoredPE = this.storedPE;
      this.restore(this.snap);
      this.resampleVelocities();
      this.sys.forces();
      s.f.set(trialForce);
      this.snap = this.snapshot();
      this.storedPE = restoredPE;
      this.storedKE = this.kinetic();
    }
  }

  computeScalar(): number {
    return this.attempted > 0 ? this.accepted / this.attempted : 0;
  }

  computeVector(i: number): number {
    switch (i) {
      case 0: return this.accepted;
      case 1: return this.attempted;
      case 2: return this.lastDPE;
      case 3: return this.lastDKE;
      case 4: return this.lastDH;
      default: throw new StyleError(`fix ${this.id}: vector index ${i + 1} out of range`);
    }
  }
}

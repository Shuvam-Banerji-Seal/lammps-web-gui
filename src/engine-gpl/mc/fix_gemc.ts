/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MC/fix_gemc.cpp, src/MC/fix_gemc.h
 * and src/MC/fix_gemc_mcmoves.cpp (class FixGEMC; contributing authors Andrew Hong and Aidan Thompson, Sandia National
 * Laboratories; copyright Sandia Corporation). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_gemc.html (verbatim): "Fix gemc performs Gibbs ensemble Monte Carlo (GEMC) for phase equilibria
 * of two boxes, which are the two partitions of the run."
 *
 * Structure. The two partitions are the two boxes. The fix of partition 0 is the driver: its preExchange runs the whole
 * move sequence of one event for both boxes, reading and writing the other box's System directly (native runs the two
 * worlds in lockstep and exchanges the energy changes with MPI_Sendrecv; the run loop here runs the replicas in lockstep
 * too, run/coupled.ts). The fix of partition 1 only does its own neighbor-decision bookkeeping (next_reneighbor).
 *
 * Random streams (fix_gemc.cpp constructor, b332f28, serial worlds so me = 0):
 *  - random_proc  = RanPark(seed + 3*(world+1) + 7), per world, drawn by the moves that place or displace atoms in it;
 *  - random_world = RanPark(seed + 13*(world+1)), per world, the atom pick and the acceptance of translations;
 *  - random_universe = RanPark(seed), the same sequence in both worlds (every draw is made in lockstep), so the driver
 *    keeps one stream: the move choice, the volume step, the exchange sender and the exchange acceptance.
 *
 * Event (pre_exchange): next_reneighbor = step + N; the energy of each box is computed; then nmoves = X + V + T moves,
 * each drawn from random_universe: imove < pc_exchange (an exchange), < pc_volume (a volume change), else a translation
 * of each box. Energies come from energy_full(): a pbc wrap, a neighbour build and the thermo_pe compute.
 *
 * Exchange (fix_gemc_mcmoves.cpp attempt_atomic_exchange_full): the sender deletes a random atom of its box, which keeps
 * its neighbour interactions excluded during the energy evaluation (the exclusion group, as native's neigh_modify
 * exclude group); the receiver appends a type 1 atom at a uniform point of its box with Gaussian velocities of width
 * sigma. The acceptance uses the ideal-gas volume-per-particle factor of each box.
 *
 * Volume change (attempt_volume_change_full): the box volume ratio follows log(V1/V2) with the uniform step drawn from
 * random_universe; each box is rescaled toward its lower corner, and the move is accepted with Metropolis on the total
 * energy change of both boxes (Frenkel & Smit, 3rd Ed., Eq. (6.6.10)).
 *
 * With pair_modify tail yes, the tail correction is the one the pair style computed at its last init or reinit, as in
 * native: an insertion and an accepted deletion leave it at the old atom counts, and a rejected exchange recomputes it
 * from the counts the box returns to (attempt_atomic_exchange_full: if (force->pair->tail_flag) force->pair->reinit()).
 * The engine recomputes the tail at each force-field init and atom change unless it is held (ForceField.tailHold), so
 * each box holds it while the fix runs and refreshes it where native reinits.
 *
 * A triclinic box: the volume change scales the lengths and keeps the tilt factors (the atoms move in lamda
 * coordinates), and an inserted atom is a uniform point in lamda coordinates (domain->lamda2x), as native does.
 *
 * Not supported (named StyleErrors): molecules (native refuses them too), rigid bodies, an exchange from an empty box,
 * and nevery below 1.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { appendAtoms, hasChargeStyle, hasMolecule, massOf, nativeOrder } from '../../engine/atoms';
import { RanPark } from '../random_park';
import { exp as libmExp } from '../libm/exp';
import { log as libmLog } from '../libm/log';
import { pow as libmPow } from '../libm/pow';

/** fix_gemc_mcmoves.cpp: "static constexpr double MAXENERGYTEST = 1.0e50;". */
const MAXENERGYTEST = 1.0e50;

const intArg = (w: string): number => {
  if (!/^[-+]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

const numArg = (w: string): number => {
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return v;
};

export class FixGEMC extends Fix {
  readonly style = 'gemc';
  /** Coupled to the fix of the other partition: the session runs the two boxes in lockstep (run/coupled.ts). */
  replicaCoupled = true;

  // user inputs (fix_gemc.cpp constructor: arg[3]..arg[10])
  private readonly mcEvery: number;
  private readonly ntranslate: number;
  private readonly nexchange: number;
  private readonly nvolume: number;
  private readonly boxTemp: number;
  private readonly displace: number;
  private readonly maxDlogvolratio: number;
  private readonly seed: number;

  /** Index of this box: 0 or 1 (myworld). */
  readonly world: number;
  /** RNG streams (see the header). randomUniverse is drawn by the driver (partition 0) only. */
  readonly randomProc: RanPark;
  readonly randomWorld: RanPark;
  readonly randomUniverse: RanPark;

  // derived quantities (init)
  private beta = 0;
  private boltz = 0;
  private mvv2e = 0;
  private nmoves = 0;
  private pcExchange = 0;
  private pcVolume = 0;
  private sigma = 0;
  private massper = 0;
  private voltot = 0;
  private ntot = 0;
  private logvolratio = 0;
  private exclBit = 0;
  /** Atoms of this box in the fix group (update_gas_atoms_list). */
  natomTotal = 0;
  /** energy_stored: the potential energy of the box after the last accepted move. */
  energyStored = 0;
  /** Next step of the event (Fix::next_reneighbor). */
  private nextReneighbor: number;

  // acceptance counters of this box (native's fix->ntranslation_attempts etc.)
  ntranslationAttempts = 0;
  ntranslationSuccesses = 0;
  nexchangeAttempts = 0;
  nexchangeSuccesses = 0;
  nvolumeAttempts = 0;
  nvolumeSuccesses = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_gemc.cpp: "if (narg != 11) utils::missing_cmd_args(FLERR, "fix gemc", error);" (the engine's args start at arg[3])
    if (args.length !== 8) throw new StyleError('Illegal fix gemc command: missing argument(s)');
    const peers = sys.replicaPeers();
    // fix_gemc.cpp: "if (universe->nworlds != 2) error->universe_all(FLERR, "Must use exactly two partitions with fix gemc");"
    if (peers.length !== 2) throw new StyleError('Must use exactly two partitions with fix gemc');
    // fix_gemc.cpp: "if (molecule_flag) error->universe_all(FLERR, "Using fix gemc with molecules not (yet) supported");"
    if (hasMolecule(sys.state)) throw new StyleError('Using fix gemc with molecules not (yet) supported');
    this.world = peers.indexOf(sys);
    if (this.world < 0) throw new StyleError('fix gemc: this system is not a partition of the session');
    this.mcEvery = intArg(args[0]);
    // native reads N without a check; N < 1 would never advance next_reneighbor, so it is refused here
    if (this.mcEvery < 1) throw new StyleError(`Illegal fix gemc command: N must be a positive integer, got ${this.mcEvery}`);
    this.ntranslate = intArg(args[1]);
    this.nexchange = intArg(args[2]);
    this.nvolume = intArg(args[3]);
    this.boxTemp = numArg(args[4]);
    this.displace = numArg(args[5]);
    this.maxDlogvolratio = numArg(args[6]);
    this.seed = intArg(args[7]);
    // random_proc = RanPark(seed + 3*(myworld+1) + 7*(me+1)) with me = 0; random_world = RanPark(seed + 13*(myworld+1));
    // random_universe = RanPark(seed). RanPark refuses a seed <= 0 ("Invalid seed for Park random # generator").
    const seedOf = (v: number): number => {
      if (v <= 0) throw new StyleError('Invalid seed for Park random # generator');
      return v;
    };
    this.randomProc = new RanPark(seedOf(this.seed + 3 * (this.world + 1) + 7));
    this.randomWorld = new RanPark(seedOf(this.seed + 13 * (this.world + 1)));
    this.randomUniverse = new RanPark(seedOf(this.seed));
    this.nextReneighbor = sys.state.step + 1;
  }

  /** The fix of each partition, in partition order (the same fix id in every replica). */
  private peers(): FixGEMC[] {
    return this.sys.replicaPeers().map((s) => {
      const f = s.fix(this.id);
      if (!(f instanceof FixGEMC)) throw new StyleError(`fix ${this.id} is not a fix gemc in every partition`);
      return f;
    });
  }

  init(): void {
    const s = this.sys.state;
    // rigid bodies move with the box in native (rfix deform) and are not reproduced here
    if (this.sys.fixes.some((f) => f.style.startsWith('rigid'))) {
      throw new StyleError('fix gemc with rigid bodies is not supported by the browser engine yet');
    }
    this.sys.compute('thermo_pe');
    this.boltz = s.units.boltz;
    this.mvv2e = s.units.mvv2e;
    this.exclBit = this.exclusionBit();
    if (!this.sys.nb.excludes.some((e) => e.kind === 'group' && e.a === this.exclBit && e.b === 1)) {
      // neighbor->modify_params("exclude group <exclusion group> all")
      this.sys.nb.excludes.push({ kind: 'group', a: this.exclBit, b: 1 });
    }

    this.ntranslationAttempts = this.ntranslationSuccesses = 0;
    this.nvolumeAttempts = this.nvolumeSuccesses = 0;
    this.nexchangeAttempts = this.nexchangeSuccesses = 0;

    // total moves and cumulative probabilities (nrotate = 0 for atoms)
    this.nmoves = this.nvolume + this.nexchange + this.ntranslate;
    const dN = this.nmoves;
    const pExchange = this.nexchange / dN;
    const pVolume = this.nvolume / dN;
    const pTranslate = this.ntranslate / dN;
    const pRotate = 0 / dN;
    const pTotal = pExchange + pVolume + pTranslate + pRotate;
    this.pcExchange = pExchange / pTotal;
    this.pcVolume = (pVolume + pExchange) / pTotal;

    this.beta = 1.0 / (this.boltz * this.boxTemp);

    // initial log volume ratio (the volumes of both boxes, exchanged between the partitions)
    const peers = this.peers();
    const volI = this.boxVolume();
    const volJ = peers[1 - this.world].boxVolume();
    this.voltot = volI + volJ;
    const ratio = this.world === 0 ? volI / volJ : volJ / volI;
    this.logvolratio = libmLog(ratio);

    this.updateGasAtoms();
    const nI = this.natomTotal;
    const nJ = peers[1 - this.world].countGroup();
    this.ntot = nI + nJ;

    // massper = group->mass(igroup) / group->count(igroup); sigma = sqrt(boltz T / massper / mvv2e)
    let mass = 0.0;
    let count = 0;
    const order = nativeOrder(s);
    for (let p = 0; p < s.n; p++) {
      const i = order[p];
      if (s.mask[i] & this.groupBit) {
        mass += massOf(s, i);
        count++;
      }
    }
    this.massper = mass / count;
    this.sigma = Math.sqrt(this.boltz * this.boxTemp / this.massper / this.mvv2e);
  }

  /** The bit of the exclusion group (created as an empty group, as native's group->assign "subtract all all"). */
  private exclusionBit(): number {
    const name = `FixGEMC:gemc_exclusion_group:${this.id}`;
    this.sys.groups.create(name);
    return this.sys.groups.bit(name);
  }

  reneighborDue(step: number): boolean {
    return step === this.nextReneighbor;
  }

  /**
   * fix_gemc.cpp pre_exchange. Each partition advances its own next step; the driver (partition 0) runs the event for
   * both boxes, drawing from the shared random_universe in the order of native's loop.
   */
  preExchange(): void {
    const step = this.sys.state.step;
    if (this.nextReneighbor !== step) return;
    this.nextReneighbor = step + this.mcEvery;
    if (this.world !== 0) return;

    const fs = this.peers();
    for (const f of fs) {
      // the tail correction stays at the pair style's last init or reinit (native) while the moves change the atoms
      f.holdTail();
      f.updateGasAtoms();
      f.energyStored = f.energyFull();
    }
    for (let i = 0; i < this.nmoves; i++) {
      const imove = this.randomUniverse.uniform();
      if (imove < this.pcExchange) this.attemptExchange(fs);
      else if (imove < this.pcVolume) this.attemptVolumeChange(fs);
      else for (const f of fs) f.attemptTranslation();
    }
  }

  /** Number of atoms of this box in the fix group (no side effects). */
  countGroup(): number {
    const s = this.sys.state;
    let c = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) c++;
    return c;
  }

  /** update_gas_atoms_list: the count the moves use. */
  updateGasAtoms(): void {
    this.natomTotal = this.countGroup();
  }

  /** The box volume of this box (xprd * yprd * zprd, as native's cached bounds). */
  private boxVolume(): number {
    const b = this.sys.state.box;
    return (b.hi[0] - b.lo[0]) * (b.hi[1] - b.lo[1]) * (b.hi[2] - b.lo[2]);
  }

  /** Holds the tail correction at its current value (the pair style's last init or reinit), if it is not held yet. */
  holdTail(): void {
    const ff = this.sys.ff;
    if (ff.pair?.tail && !ff.tailHold) ff.tailHold = { etail: ff.etailV, ptail: ff.ptailV };
  }

  /** force->pair->reinit() when the pair has a tail correction: recomputed from the current atom counts, then held. */
  reinitTail(): void {
    const ff = this.sys.ff;
    if (!ff.pair?.tail) return;
    ff.tailHold = null;
    ff.updateTail(this.sys.state);
    ff.tailHold = { etail: ff.etailV, ptail: ff.ptailV };
  }

  /** compute_scalar of thermo_pe after energy_full (pbc, neighbour build, forces) on this box. */
  energyFull(): number {
    this.sys.bump();
    // modify->post_force reassigns a dynamic group (FixGroup is a post_force_group fix) before compute pe
    this.sys.assignDynamicGroups(false);
    return this.sys.compute('thermo_pe').scalarValue();
  }

  /** remap of Domain::remap for a point (orthogonal boxes; periodic dimensions wrap). */
  private remapPoint(c: number[]): void {
    const b = this.sys.state.box;
    for (let d = 0; d < 3; d++) {
      if (b.boundary[d][0] !== 'p') continue;
      const prd = b.hi[d] - b.lo[d];
      while (c[d] < b.lo[d]) c[d] += prd;
      while (c[d] >= b.hi[d]) c[d] -= prd;
      c[d] = c[d] > b.lo[d] ? c[d] : b.lo[d];
    }
  }

  /** Domain::inside: lo <= x < hi in every dimension. */
  private inside(c: number[]): boolean {
    const b = this.sys.state.box;
    for (let d = 0; d < 3; d++) if (c[d] < b.lo[d] || c[d] >= b.hi[d]) return false;
    return true;
  }

  /** Lamda-coordinate scaling of all atoms with a new upper corner (remap_all follows in the caller). */
  private setHi(hi: number[]): void {
    const b = this.sys.state.box;
    this.sys.setBox([b.lo[0], b.lo[1], b.lo[2]], hi, [b.tilt[0], b.tilt[1], b.tilt[2]], 1);
  }

  /** attempt_atomic_translation_full (fix_gemc_mcmoves.cpp), on this box. */
  attemptTranslation(): void {
    this.ntranslationAttempts += 1.0;
    if (this.natomTotal === 0) return;
    const energyBefore = this.energyStored;
    // pick_random_gas_atom: the local index is int(natom_total * uniform), serial so the index is the atom's
    const iwhich = Math.trunc(this.natomTotal * this.randomWorld.uniform());
    const i = iwhich < this.natomTotal ? nativeOrder(this.sys.state)[iwhich] : -1;
    const xtmp = [0.0, 0.0, 0.0];
    if (i >= 0) {
      let rsq = 1.1;
      let rx = 0.0, ry = 0.0, rz = 0.0;
      while (rsq > 1.0) {
        rx = 2 * this.randomProc.uniform() - 1.0;
        ry = 2 * this.randomProc.uniform() - 1.0;
        rz = 2 * this.randomProc.uniform() - 1.0;
        rsq = rx * rx + ry * ry + rz * rz;
      }
      const s = this.sys.state;
      const coord = [s.x[3 * i] + this.displace * rx, s.x[3 * i + 1] + this.displace * ry, s.x[3 * i + 2] + this.displace * rz];
      if (!this.insideNonperiodic(coord)) throw new StyleError('Fix gcmc put atom outside box');
      xtmp[0] = s.x[3 * i]; xtmp[1] = s.x[3 * i + 1]; xtmp[2] = s.x[3 * i + 2];
      s.x[3 * i] = coord[0]; s.x[3 * i + 1] = coord[1]; s.x[3 * i + 2] = coord[2];
    }
    const energyAfter = this.energyFull();
    if (energyAfter < MAXENERGYTEST && this.randomWorld.uniform() < libmExp(this.beta * (energyBefore - energyAfter))) {
      this.energyStored = energyAfter;
      this.ntranslationSuccesses += 1.0;
    } else {
      // restore the position of the atom and remap as native does (the neighbour lists follow in the run loop)
      if (i >= 0) {
        const s = this.sys.state;
        s.x[3 * i] = xtmp[0]; s.x[3 * i + 1] = xtmp[1]; s.x[3 * i + 2] = xtmp[2];
      }
      this.energyStored = energyBefore;
      this.sys.pbc();
      this.sys.bump();
    }
    this.updateGasAtoms();
  }

  /** inside_nonperiodic: only the non-periodic dimensions are checked. */
  private insideNonperiodic(c: number[]): boolean {
    const b = this.sys.state.box;
    // Domain::inside_nonperiodic: a triclinic box compares the lamda coordinates with [0, 1]
    const p = b.triclinic ? (() => { const l = [0, 0, 0]; this.sys.geom.toLamda(c[0], c[1], c[2], l); return l; })() : c;
    for (let d = 0; d < 3; d++) {
      if (b.boundary[d][0] === 'p') continue;
      const lo = b.triclinic ? 0 : b.lo[d], hi = b.triclinic ? 1 : b.hi[d];
      // half-open, as native: x < lo or x >= hi is outside
      if (p[d] < lo || p[d] >= hi) return false;
    }
    return true;
  }

  /**
   * attempt_volume_change_full, for both boxes (the driver). The step dlogvolratio and the acceptance uniform come from
   * random_universe; each box is rescaled toward its lower corner and its energy is evaluated; the energy changes are
   * summed across the boxes.
   */
  private attemptVolumeChange(fs: FixGEMC[]): void {
    for (const f of fs) f.nvolumeAttempts += 1.0;
    const dlogvolratio = fs[0].maxDlogvolratio * (2.0 * fs[0].randomUniverse.uniform() - 1.0);
    const dimension = this.sys.state.dimension;
    const hiOld: number[][] = [];
    const eAfter: number[] = [];
    const idU: number[] = [];
    const eBefore: number[] = fs.map((f) => f.energyStored);
    fs.forEach((f, w) => {
      const b = f.sys.state.box;
      // fvolume = vnew / vold, equal and opposite on the two boxes (log V1/V2 = logvolratio)
      const L = f.logvolratio;
      const fvolume = f.world === 0
        ? (1.0 + libmExp(-L)) / (1.0 + libmExp(-(L + dlogvolratio)))
        : (1.0 + libmExp(L)) / (1.0 + libmExp(L + dlogvolratio));
      const scaleLength = libmPow(fvolume, 1.0 / dimension);
      hiOld[w] = [b.hi[0], b.hi[1], b.hi[2]];
      // xhi_tmp = xlo + (xhi - xlo) * scale_length; the lower corner is kept
      const hiNew = [0, 1, 2].map((d) => b.lo[d] + (b.hi[d] - b.lo[d]) * scaleLength);
      f.setHi(hiNew);
      f.sys.pbc();
      // Frenkel & Smit, 3rd Ed. (2023), p. 221, Eq. (6.6.10): change in potential due to the volume change
      const dUVolume = (f.sys.state.n + 1) * f.boltz * f.boxTemp * libmLog(fvolume);
      eAfter[w] = f.energyFull();
      idU[w] = eAfter[w] - eBefore[w] - dUVolume;
    });
    const beta = fs[0].beta;
    const prob = [0, 1].map((w) => {
      const dU = idU[w] + idU[1 - w];
      const e = libmExp(-beta * dU);
      return e < 1.0 ? e : 1.0;
    });
    const rf = fs[0].randomUniverse.uniform();
    fs.forEach((f, w) => {
      if (prob[w] < rf || eAfter[w] >= MAXENERGYTEST) {
        // volume change rejected: the old box and positions come back, then remap (neighbour lists follow)
        f.setHi(hiOld[w]);
        f.sys.pbc();
        f.sys.bump();
      } else {
        f.nvolumeSuccesses += 1.0;
        f.logvolratio += dlogvolratio;
        f.energyStored = eAfter[w];
      }
    });
    for (const f of fs) f.updateGasAtoms();
  }

  /**
   * attempt_atomic_exchange_full (fix_gemc_mcmoves.cpp), for both boxes (the driver). The sender is chosen by the shared
   * uniform (drand > 0.5: partition 0 sends); the sender removes a random atom from its box and the receiver appends one
   * at a uniform point with Gaussian velocities.
   */
  private attemptExchange(fs: FixGEMC[]): void {
    for (const f of fs) f.nexchangeAttempts += 1;
    const drand = fs[0].randomUniverse.uniform();
    const sw = drand > 0.5 ? 0 : 1;
    const snd = fs[sw];
    const rcv = fs[1 - sw];

    // the sender picks an atom (only the sender's random_world is drawn) and hides it from the energy
    if (snd.natomTotal === 0) throw new StyleError('fix gemc: an exchange from an empty box is not reproduced');
    const iwhich = Math.trunc(snd.natomTotal * snd.randomWorld.uniform());
    const iatom = nativeOrder(snd.sys.state)[iwhich];
    const tmpMask = snd.sys.state.mask[iatom];
    const charged = hasChargeStyle(snd.sys.state.atomStyle);
    let qTmp = 0;
    snd.sys.state.mask[iatom] = snd.exclBit;
    if (charged) {
      qTmp = snd.sys.state.q[iatom];
      snd.sys.state.q[iatom] = 0.0;
    }

    // the receiver samples a point of its box (three draws of its random_proc) and appends a type 1 atom
    {
      const b = rcv.sys.state.box;
      const coord = [0, 0, 0];
      if (b.triclinic) {
        // a uniform point in lamda coordinates (1.0 is folded to 0.0), then domain->lamda2x
        const lamda = [rcv.randomProc.uniform(), rcv.randomProc.uniform(), rcv.randomProc.uniform()];
        for (let d = 0; d < 3; d++) if (lamda[d] === 1.0) lamda[d] = 0.0;
        const lx = b.hi[0] - b.lo[0], ly = b.hi[1] - b.lo[1], lz = b.hi[2] - b.lo[2];
        const [xy, xz, yz] = b.tilt;
        coord[0] = lx * lamda[0] + xy * lamda[1] + xz * lamda[2] + b.lo[0];
        coord[1] = ly * lamda[1] + yz * lamda[2] + b.lo[1];
        coord[2] = lz * lamda[2] + b.lo[2];
      } else {
        coord[0] = b.lo[0] + rcv.randomProc.uniform() * (b.hi[0] - b.lo[0]);
        coord[1] = b.lo[1] + rcv.randomProc.uniform() * (b.hi[1] - b.lo[1]);
        coord[2] = b.lo[2] + rcv.randomProc.uniform() * (b.hi[2] - b.lo[2]);
        rcv.remapPoint(coord);
        if (!rcv.inside(coord)) throw new StyleError('Fix gemc put atom outside box');
      }
      appendAtoms(rcv.sys.state, { x: Float64Array.from(coord), type: 1, v: new Float64Array(3), mask: rcv.groupBit });
      const s = rcv.sys.state;
      const m = s.n - 1;
      s.v[3 * m] = rcv.randomProc.gaussian() * rcv.sigma;
      s.v[3 * m + 1] = rcv.randomProc.gaussian() * rcv.sigma;
      s.v[3 * m + 2] = rcv.randomProc.gaussian() * rcv.sigma;
      // the insertion does not reinit the pair style: the tail correction keeps the old atom counts (held)
      rcv.sys.atomsChanged();
    }

    for (const f of fs) f.updateGasAtoms();
    const eBefore = fs.map((f) => f.energyStored);
    const eAfter = fs.map((f) => f.energyFull());

    // Frenkel & Smit, 3rd Ed. (2023), p. 221, Eq. (6.6.11): prob = (V/N, receiver, new) / (V/N, sender, old) * exp(-beta dU)
    const idU = fs.map((f, w) => {
      let v = eAfter[w] - eBefore[w];
      const volume = f.boxVolume();
      const logVN = w === sw ? -libmLog(volume / f.natomTotal) : libmLog(volume / f.natomTotal);
      v += -f.boxTemp * f.boltz * logVN;
      return v;
    });
    const beta = snd.beta;
    const prob = [0, 1].map((w) => {
      const allDU = idU[w] + idU[1 - w];
      const e = libmExp(-beta * allDU);
      return e < 1.0 ? e : 1.0;
    });
    const u = fs[0].randomUniverse.uniform();
    const success = [0, 1].map((w) => prob[w] > u && eAfter[w] < MAXENERGYTEST);

    fs.forEach((f, w) => {
      if (w === sw) {
        if (success[w]) {
          f.nexchangeSuccesses += 1.0;
          const flags = new Uint8Array(f.sys.state.n);
          flags[iatom] = 1;
          // avec->copy(nlocal-1, iatom) and nlocal--: the last atom takes the deleted atom's slot (no pair reinit, so the
          // tail correction keeps the old atom counts: held)
          f.sys.deleteAtoms(flags);
          f.energyStored = eAfter[w];
        } else {
          f.sys.state.mask[iatom] = tmpMask;
          if (charged) f.sys.state.q[iatom] = qTmp;
          // force->pair->reinit(): the tail correction of the current atom counts
          f.reinitTail();
        }
      } else if (success[w]) {
        f.nexchangeSuccesses += 1.0;
        f.energyStored = eAfter[w];
      } else {
        // the appended atom is the last one: remove it again (natoms--, nlocal--), then force->pair->reinit()
        const flags = new Uint8Array(f.sys.state.n);
        flags[f.sys.state.n - 1] = 1;
        f.sys.deleteAtoms(flags);
        f.reinitTail();
        f.energyStored = eBefore[w];
      }
    });
    for (const f of fs) f.updateGasAtoms();
  }
}

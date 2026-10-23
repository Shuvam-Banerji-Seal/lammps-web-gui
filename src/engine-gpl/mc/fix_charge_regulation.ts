/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MC/fix_charge_regulation.cpp and src/MC/fix_charge_regulation.h at commit b332f28 (Tine Curk and
 * Jiaxing Yuan; the LAMMPS developers). Copyright (2003) Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * fix charge/regulation: reaction-ensemble Monte Carlo of acid/base protonation and of salt ion insertion and
 * deletion, run between MD steps (docs.lammps.org/fix_charge_regulation.html). The browser engine supports the
 * box-wide moves (rxd 0), pmcmoves, onlysalt (multivalent ion pairs), tag, group, the thermostat temperature
 * (tempfixid or temp), and the seeded native random streams; the other keywords are refused by name (see the
 * constructor). Each inserted ion takes the largest ID + 1 when it is inserted, as native insert_particle calls
 * Atom::tag_extend(); so native's assign_tags at the end of an event (tag yes) finds no ion without an ID, and the
 * tag keyword changes nothing.
 *
 * Native structure kept here: two Park-Miller streams seeded with the fix's seed (random_equal drives the move
 * choice, positions and acceptances; random_unequal the velocities of inserted ions); a trial inserts or changes
 * charges, takes the potential energy from a full evaluation, and accepts with the native acceptance rule
 * (the acceptance draw is taken only when the trial energy is below the MAXENERGYTEST bound, as native's &&).
 * Particles are picked in native storage order (SimState.order), and a deleted particle's slot is taken by the
 * last atom of the list (AtomVec::copy(nlocal-1, m)); a rejected insertion removes the last atom. Charges of
 * candidates being deleted are set to zero and their group bit replaced by the exclusion group bit, which
 * neigh_modify exclude group removes from the pair interactions (native init: "exclude group <excl> all").
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { inumeric, numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { ALL_GROUP_BIT, appendAtoms, deleteAtoms, nativeOrder } from '../../engine/atoms';
import { RanPark } from '../random_park';
import { takeFixRecord } from '../../engine/fix/restartRecord';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

/** Avogadro's constant times the reference concentration (N_A * mol / liter) [nm^-3]. */
const NA_RHO0 = 0.602214;
/** Energy above which a trial is rejected without a random draw (native MAXENERGYTEST). */
const MAXENERGYTEST = 1.0e50;
const SMALL = 0.0000001;
const THIRD = 1.0 / 3.0;

/** The thermostat fix tempfixid names (native reads its t_target). */
interface TargetTemperature { targetTemperature(): number }

const isTarget = (f: unknown): f is TargetTemperature => typeof (f as Partial<TargetTemperature>)?.targetTemperature === 'function';

// As native LAMMPS does (src/MC/fix_charge_regulation.cpp at b332f28): every number is read by utils::numeric and every
// integer (types, nevery, nmc, seed, salt charges) by utils::inumeric, which give the native input-parser messages.
const num = (w: string | undefined, _what: string): number => numeric(w);

const int = (w: string | undefined, _what: string): number => inumeric(w);

/** utils::logical: yes/no, true/false, on/off, 1/0. */
const logical = (w: string): boolean => {
  if (w === 'yes' || w === 'true' || w === 'on' || w === '1') return true;
  if (w === 'no' || w === 'false' || w === 'off' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w}' in input script or data file`);
};

/** The state fix charge/regulation keeps in a restart file (fix_charge_regulation.cpp write_restart). */
export interface ChargeRegulationRestart { seed: number; seedUnequal: number; nextEvent: number; step: number; counters: number[] }

export class FixChargeRegulation extends Fix {
  readonly style = 'charge/regulation';
  private readonly cationType: number;
  private readonly anionType: number;
  private acidType = -1;
  private baseType = -1;
  private pH = 7.0;
  private pKa = 100;
  private pKb = 100;
  private pKs = 14.0;
  private pIp = 5;
  private pIm = 5;
  private mcEvery = 100;
  private nmc = 100;
  private readonly pmcmoves = [THIRD, THIRD, THIRD];
  private lunit = 0.71;
  private reservoirT = 1.0;
  private tempFix: string | null = null;
  private seed = 0;
  private readonly rngEq: RanPark;
  private readonly rngUneq: RanPark;
  private nextEvent: number;
  private excl = 0;
  private readonly exclName: string;
  private ready = false;
  private energyStored = 0;
  private nacidAtt = 0; private nacidSucc = 0;
  private nbaseAtt = 0; private nbaseSucc = 0;
  private nsaltAtt = 0; private nsaltSucc = 0;
  private nacidNeutral = 0; private nacidCharged = 0;
  private nbaseNeutral = 0; private nbaseCharged = 0;
  private ncation = 0; private nanion = 0;
  private npartXrd = 0;
  private volumeRx = 0;
  private c10pH = 0; private c10pKa = 0; private c10pKb = 0; private c10pOH = 0; private c10pIplus = 0; private c10pIminus = 0;
  private beta = 0;
  /** onlysalt: only ion insertion and deletion, with the cation and anion charges (salt_charge) and their ratio. */
  private onlySalt = false;
  private readonly saltCharge = [1, -1];
  private saltRatio = 1;
  private addTags = false;
  private readonly groupNames: string[] = [];
  /** The mask of inserted ions (native groupbitall: all, the fix group and the group keywords). */
  private groupBitAll = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (sys.dimension !== 3) throw new StyleError('Cannot use fix charge/regulation in a 2d simulation');
    if (args.length < 2) throw new StyleError('Illegal fix charge/regulation command: usage fix ID group charge/regulation cation-type anion-type keyword values ...');
    this.vectorFlag = true;
    this.sizeVector = 8;
    this.globalFreq = 1;
    this.extvector = 0;
    this.cationType = int(args[0], 'cation type');
    this.anionType = int(args[1], 'anion type');
    for (let i = 2; i < args.length;) {
      const key = args[i];
      const v = (k: number) => {
        // As native LAMMPS does (src/MC/fix_charge_regulation.cpp FixChargeRegulation::options at b332f28): a keyword without its value is "Illegal fix charge/regulation command"
        if (args[i + k] === undefined) throw new StyleError('Illegal fix charge/regulation command');
        return args[i + k];
      };
      switch (key) {
        case 'acid_type': this.acidType = int(v(1), 'acid_type'); i += 2; break;
        case 'base_type': this.baseType = int(v(1), 'base_type'); i += 2; break;
        case 'pH':
          if (v(1).startsWith('v_')) throw new StyleError('fix charge/regulation: pH from an equal-style variable (v_name) is not supported by the browser engine');
          this.pH = num(v(1), 'pH'); i += 2; break;
        case 'pIp': this.pIp = num(v(1), 'pIp'); i += 2; break;
        case 'pIm': this.pIm = num(v(1), 'pIm'); i += 2; break;
        case 'pKa': this.pKa = num(v(1), 'pKa'); i += 2; break;
        case 'pKb': this.pKb = num(v(1), 'pKb'); i += 2; break;
        case 'pKs': this.pKs = num(v(1), 'pKs'); i += 2; break;
        case 'temp': this.reservoirT = num(v(1), 'temp'); i += 2; break;
        case 'tempfixid': this.tempFix = v(1); i += 2; break;
        case 'lunit_nm': this.lunit = num(v(1), 'lunit_nm'); i += 2; break;
        case 'nevery': this.mcEvery = int(v(1), 'nevery'); i += 2; break;
        case 'nmc': this.nmc = int(v(1), 'nmc'); i += 2; break;
        case 'seed': this.seed = int(v(1), 'seed'); i += 2; break;
        case 'pmcmoves':
          this.pmcmoves[0] = num(v(1), 'pmcmoves'); this.pmcmoves[1] = num(v(2), 'pmcmoves'); this.pmcmoves[2] = num(v(3), 'pmcmoves');
          i += 4; break;
        // tag yes: native's assign_tags numbers ions without an ID, and every inserted ion already has one (see the header)
        case 'tag': this.addTags = logical(v(1)); i += 2; break;
        case 'onlysalt':
          this.onlySalt = logical(v(1));
          i += 2;
          if (this.onlySalt) {
            if (args[i + 1] === undefined) throw new StyleError('Illegal fix charge/regulation command');
            this.saltCharge[0] = int(args[i], 'onlysalt cation charge');
            this.saltCharge[1] = int(args[i + 1], 'onlysalt anion charge');
            i += 2;
          }
          break;
        case 'group': this.groupNames.push(v(1)); i += 2; break;
        case 'rxd': {
          const rd = num(v(1), 'rxd');
          if (rd > 0) throw new StyleError('fix charge/regulation rxd > 0 (reaction distance) is not supported by the browser engine');
          i += 2; break;
        }
        default:
          // As native LAMMPS does (src/MC/fix_charge_regulation.cpp FixChargeRegulation::options at b332f28): "Illegal fix charge/regulation command".
          throw new StyleError('Illegal fix charge/regulation command');
      }
    }
    if (this.mcEvery <= 0 || this.nmc < 0 || this.lunit < 0 || this.reservoirT < 0 || this.cationType <= 0 || this.anionType <= 0
      || this.saltCharge[0] <= 0 || this.saltCharge[1] >= 0) {
      throw new StyleError('Illegal fix charge/regulation command');
    }
    if (!(this.seed > 0)) throw new StyleError('Illegal fix charge/regulation command: Seed value (positive integer) must be provided ');
    if (this.saltCharge[1] % this.saltCharge[0] !== 0 && this.saltCharge[0] % this.saltCharge[1] !== 0) {
      throw new StyleError('Illegal fix charge/regulation command, multivalent cation/anion charges are allowed, but must be divisible, e.g. (3,-1) is fine, but (3,-2) is not implemented');
    }
    if (this.pmcmoves.some((p) => p < 0)) throw new StyleError('Illegal fix charge/regulation command');
    if (this.acidType < 0) this.pmcmoves[0] = 0;
    if (this.baseType < 0) this.pmcmoves[1] = 0;
    const psum = this.pmcmoves[0] + this.pmcmoves[1] + this.pmcmoves[2];
    if (psum <= 0) throw new StyleError('Illegal fix charge/regulation command');
    this.pmcmoves[0] /= psum;
    this.pmcmoves[1] /= psum;
    this.pmcmoves[2] /= psum;
    this.rngEq = new RanPark(this.seed);
    this.rngUneq = new RanPark(this.seed);
    this.nextEvent = sys.hasBox ? sys.state.step + 1 : 1;
    this.exclName = `FixChargeRegulation:exclusion_group:${id}`;
    const saved = takeFixRecord(sys, id, this.style)?.state as ChargeRegulationRestart | undefined;
    if (saved) this.restoreState(saved);
  }

  /**
   * fix_charge_regulation.cpp write_restart: the two generators, the six attempt and success counts (native order: acid,
   * base, salt), the next event step and the step the file was written at.
   */
  restartState(): ChargeRegulationRestart {
    return {
      seed: this.rngEq.state(), seedUnequal: this.rngUneq.state(), nextEvent: this.nextEvent, step: this.sys.state.step,
      counters: [this.nacidAtt, this.nacidSucc, this.nbaseAtt, this.nbaseSucc, this.nsaltAtt, this.nsaltSucc],
    };
  }

  /** FixChargeRegulation::restart: the generators, the next event and the counts; the step must be the file's. */
  private restoreState(r: ChargeRegulationRestart): void {
    if (r.step !== this.sys.state.step) throw new StyleError(`fix ${this.id}: must not reset timestep when restarting fix charge/regulation`);
    this.rngEq.reset(r.seed);
    this.rngUneq.reset(r.seedUnequal);
    this.nextEvent = r.nextEvent;
    [this.nacidAtt, this.nacidSucc, this.nbaseAtt, this.nbaseSucc, this.nsaltAtt, this.nsaltSucc] = r.counters;
  }

  init(): void {
    const sys = this.sys;
    const s = sys.state;
    if (!s.massByType || !s.massByType.length) throw new StyleError('Fix charge/regulation requires per atom type masses');
    if (s.box.triclinic) throw new StyleError('fix charge/regulation with a triclinic box is not supported by the browser engine');
    if (this.cationType > s.ntypes || this.anionType > s.ntypes) throw new StyleError('fix charge/regulation: cation or anion type is larger than ntypes');
    if (this.tempFix !== null) {
      const f = sys.fix(this.tempFix) as unknown;
      if (!isTarget(f)) throw new StyleError(`Fix id ${this.tempFix} does not control temperature`);
    }
    for (let i = 0; i < s.n; i++) {
      if ((s.type[i] === this.cationType || s.type[i] === this.anionType) && s.molecule[i]) {
        throw new StyleError('fix charge/regulation cannot exchange individual atoms (ions) belonging to a molecule');
      }
    }
    if (!this.excl) {
      // native: group->assign("<name> subtract all all"); neighbor->modify_params("exclude group <name> all")
      const k = sys.groups.create(this.exclName);
      this.excl = k === 31 ? -0x80000000 : 1 << k;
      sys.nb.excludes.push({ kind: 'group', a: this.excl, b: ALL_GROUP_BIT });
    }
    this.groupBitAll = ALL_GROUP_BIT | this.groupBit;
    for (const g of this.groupNames) {
      if (sys.groups.find(g) < 0) throw new StyleError('Could not find fix charge/regulation group ID');
      this.groupBitAll |= sys.groups.bit(g);
    }
    this.ready = true;
  }

  private get target(): number {
    if (this.tempFix !== null) return (this.sys.fix(this.tempFix) as unknown as TargetTemperature).targetTemperature();
    return this.reservoirT;
  }

  postIntegrate(): void {
    if (this.sys.state.step !== this.nextEvent) return;
    this.event();
    this.nextEvent += this.mcEvery;
  }

  /** One Monte Carlo event (native pre_exchange): nmc trial moves, then the neighbour lists for the new atoms. */
  private event(): void {
    if (!this.ready) this.init();
    const s = this.sys.state;
    const lx = s.box.hi[0] - s.box.lo[0], ly = s.box.hi[1] - s.box.lo[1], lz = s.box.hi[2] - s.box.lo[2];
    const cube = this.lunit * this.lunit * this.lunit;
    this.volumeRx = lx * ly * lz * cube * NA_RHO0;
    const kT = s.units.boltz * this.target;
    this.beta = 1.0 / kT;
    this.c10pH = libmPow(10.0, -this.pH);
    this.c10pKa = libmPow(10.0, -this.pKa);
    this.c10pKb = libmPow(10.0, -this.pKb);
    this.c10pOH = libmPow(10.0, -this.pKs + this.pH);
    this.c10pIplus = libmPow(10.0, -this.pIp);
    this.c10pIminus = libmPow(10.0, -this.pIm);

    // native: the kspace G-ewald, kmax and mesh are the ones of the run's init while the charges change
    const kspace = this.sys.ff.kspace;
    if (kspace) kspace.paramsFrozen = true;
    this.energyStored = this.energyFull();
    this.nacidNeutral = this.countParticles(this.acidType, 0);
    this.nacidCharged = this.countParticles(this.acidType, -1);
    this.nbaseNeutral = this.countParticles(this.baseType, 0);
    this.nbaseCharged = this.countParticles(this.baseType, 1);
    this.ncation = this.countParticles(this.cationType, this.saltCharge[0]);
    this.nanion = this.countParticles(this.anionType, this.saltCharge[1]);

    if (!this.onlySalt) {
      const pm = this.pmcmoves;
      for (let i = 0; i < this.nmc; i++) {
        const r = this.rngEq.uniform();
        if (r < pm[0] / 2) { this.forwardAcid(); this.nacidAtt++; }
        else if (r < pm[0]) { this.backwardAcid(); this.nacidAtt++; }
        else if (r < pm[0] + pm[1] / 2) { this.forwardBase(); this.nbaseAtt++; }
        else if (r < pm[0] + pm[1]) { this.backwardBase(); this.nbaseAtt++; }
        else if (r < pm[0] + pm[1] + pm[2] / 2) { this.forwardIons(); this.nsaltAtt++; }
        else { this.backwardIons(); this.nsaltAtt++; }
      }
    } else {
      // only ion insertion and deletion; the larger charge is balanced by saltRatio ions of the other kind
      // (C++ integer division of the two charges)
      const [qc, qa] = this.saltCharge;
      this.saltRatio = qc >= -qa ? Math.trunc(-qc / qa) : Math.trunc(-qa / qc);
      for (let i = 0; i < this.nmc; i++) {
        if (this.rngEq.uniform() < 0.5) this.forwardIonsMultival();
        else this.backwardIonsMultival();
        this.nsaltAtt++;
      }
    }
    this.sys.atomsChanged();
    this.sys.ff.reinitTail(this.sys.state);
    this.sys.setupNeighbors();
    if (kspace) kspace.paramsFrozen = false;
  }

  /** Native get_random_particle with rd = 0: a uniform pick among matching particles, in native storage order. */
  private pick(ptype: number, charge: number): number {
    const s = this.sys.state;
    const cand: number[] = [];
    for (const i of nativeOrder(s)) {
      if (s.type[i] === ptype && Math.abs(s.q[i] - charge) < SMALL && s.mask[i] !== this.excl) cand.push(i);
    }
    this.npartXrd = cand.length;
    if (cand.length > 0) {
      const k = Math.floor(this.rngEq.uniform() * cand.length);
      return cand[k];
    }
    return -1;
  }

  /** Native particle_number(ptype, charge). */
  private countParticles(ptype: number, charge: number): number {
    const s = this.sys.state;
    let c = 0;
    for (let i = 0; i < s.n; i++) {
      if (s.type[i] === ptype && Math.abs(s.q[i] - charge) < SMALL && s.mask[i] !== this.excl) c++;
    }
    return c;
  }

  /** Native insert_particle with rd = 0: a particle in the box with Gaussian velocity; returns its index. */
  private insert(ptype: number, charge: number): number {
    const s = this.sys.state;
    const lo = s.box.lo, hi = s.box.hi;
    const c = new Float64Array(3);
    c[0] = lo[0] + this.rngEq.uniform() * (hi[0] - lo[0]);
    c[1] = lo[1] + this.rngEq.uniform() * (hi[1] - lo[1]);
    c[2] = lo[2] + this.rngEq.uniform() * (hi[2] - lo[2]);
    const n0 = s.n;
    appendAtoms(s, { x: c, type: ptype, mask: this.groupBitAll });
    const sigma = Math.sqrt(s.units.boltz * this.target / s.massByType[ptype] / s.units.mvv2e);
    s.v[3 * n0] = this.rngUneq.gaussian() * sigma;
    s.v[3 * n0 + 1] = this.rngUneq.gaussian() * sigma;
    s.v[3 * n0 + 2] = this.rngUneq.gaussian() * sigma;
    s.q[n0] = charge;
    return n0;
  }

  /** Deletes the atom at engine index i (native avec->copy(nlocal-1, i) and nlocal--). */
  private removeAt(i: number): void {
    const s = this.sys.state;
    const flags = new Uint8Array(s.n);
    flags[i] = 1;
    deleteAtoms(s, flags);
  }

  /** Deletes the atom with the given ID (its index may have moved). */
  private removeId(id: number): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) if (s.id[i] === id) { this.removeAt(i); return; }
  }

  /** Deletes the last k atoms of the native list (a rejected insertion). */
  private popLast(k: number): void {
    const s = this.sys.state;
    const ord = nativeOrder(s);
    const flags = new Uint8Array(s.n);
    for (let p = s.n - k; p < s.n; p++) flags[ord[p]] = 1;
    deleteAtoms(s, flags);
  }

  /**
   * Full potential energy of the current configuration (native energy_full). Native's energy_full also calls
   * modify->post_force, so every post-force fix (fix langevin among them) runs on each trial: its random numbers
   * advance once per energy evaluation. Measured with native LAMMPS (black box): the langevin velocities after
   * one Monte Carlo event differ from a stream that only the MD steps draw from.
   */
  private energyFull(): number {
    const sys = this.sys;
    sys.atomsChanged();
    // native reinits the pair (force->pair->reinit()) with the tail at each insertion and deletion trial; the tail is
    // recounted at each energy evaluation here (an accepted deletion's stale count until the next reinit is not kept)
    sys.ff.reinitTail(sys.state);
    // energy_full: modify->pre_force, the forces, modify->pre_reverse, modify->post_force (a dynamic group's
    // post_force_group fix first, then every post_force fix), and compute pe with the fix_modify energy contributions
    for (const f of sys.fixes) f.preForce?.();
    const a = sys.forces();
    for (const f of sys.fixes) f.preReverse?.();
    sys.assignDynamicGroups(false);
    for (const f of sys.fixes) f.postForce?.();
    return a.evdwl + a.ecoul + a.elong + a.ebond + a.eangle + a.edihed + a.eimp + sys.fixEnergy();
  }

  /** Native acceptance: energy_after < MAXENERGYTEST && random_equal->uniform() < ratio * exp(beta * (before - after)). */
  private accept(ratio: number, after: number): boolean {
    if (!(after < MAXENERGYTEST)) return false;
    return this.rngEq.uniform() < ratio * libmExp(this.beta * (this.energyStored - after));
  }

  private forwardAcid(): void {
    const s = this.sys.state;
    const before = this.energyStored;
    const m1 = this.pick(this.acidType, 0);
    if (this.nacidNeutral > 0) {
      if (m1 >= 0) s.q[m1] = -1;
      const npartXrd2 = this.ncation;
      this.insert(this.cationType, 1);
      const factor = this.nacidNeutral * this.volumeRx * this.c10pKa * this.c10pIplus
        / (this.c10pH * (1 + this.nacidCharged) * (1 + npartXrd2));
      const after = this.energyFull();
      if (this.accept(factor, after)) {
        this.energyStored = after;
        this.nacidSucc++;
        this.ncation++;
        this.nacidCharged++;
        this.nacidNeutral--;
      } else {
        this.energyStored = before;
        this.popLast(1);
        if (m1 >= 0) s.q[m1] = 0;
      }
    }
  }

  private backwardAcid(): void {
    const s = this.sys.state;
    const before = this.energyStored;
    const m1 = this.pick(this.acidType, -1);
    if (this.nacidCharged > 0) {
      if (m1 >= 0) s.q[m1] = 0;
      const m2 = this.pick(this.cationType, 1);
      if (this.npartXrd > 0) {
        let maskTmp = 0;
        if (m2 >= 0) { s.q[m2] = 0; maskTmp = s.mask[m2]; s.mask[m2] = this.excl; }
        const factor = (1 + this.nacidNeutral) * this.volumeRx * this.c10pKa * this.c10pIplus
          / (this.c10pH * this.nacidCharged * this.npartXrd);
        const after = this.energyFull();
        if (this.accept(1.0 / factor, after)) {
          this.nacidSucc++;
          this.energyStored = after;
          this.nacidCharged--;
          this.nacidNeutral++;
          this.ncation--;
          if (m2 >= 0) this.removeId(this.idOf(m2));
        } else {
          this.energyStored = before;
          if (m1 >= 0) s.q[m1] = -1;
          if (m2 >= 0) { s.q[m2] = 1; s.mask[m2] = maskTmp; }
        }
      } else if (m1 >= 0) {
        s.q[m1] = -1;
      }
    }
  }

  private forwardBase(): void {
    const s = this.sys.state;
    const before = this.energyStored;
    const m1 = this.pick(this.baseType, 0);
    if (this.nbaseNeutral > 0) {
      if (m1 >= 0) s.q[m1] = 1;
      const npartXrd2 = this.nanion;
      this.insert(this.anionType, -1);
      const factor = this.nbaseNeutral * this.volumeRx * this.c10pKb * this.c10pIminus
        / (this.c10pOH * (1 + this.nbaseCharged) * (1 + npartXrd2));
      const after = this.energyFull();
      if (this.accept(factor, after)) {
        this.energyStored = after;
        this.nbaseSucc++;
        this.nbaseCharged++;
        this.nbaseNeutral--;
        this.nanion++;
      } else {
        this.energyStored = before;
        this.popLast(1);
        if (m1 >= 0) s.q[m1] = 0;
      }
    }
  }

  private backwardBase(): void {
    const s = this.sys.state;
    const before = this.energyStored;
    const m1 = this.pick(this.baseType, 1);
    if (this.nbaseCharged > 0) {
      if (m1 >= 0) s.q[m1] = 0;
      const m2 = this.pick(this.anionType, -1);
      if (this.npartXrd > 0) {
        let maskTmp = 0;
        if (m2 >= 0) { s.q[m2] = 0; maskTmp = s.mask[m2]; s.mask[m2] = this.excl; }
        const factor = (1 + this.nbaseNeutral) * this.volumeRx * this.c10pKb * this.c10pIminus
          / (this.c10pOH * this.nbaseCharged * this.npartXrd);
        const after = this.energyFull();
        if (this.accept(1.0 / factor, after)) {
          this.nbaseSucc++;
          this.energyStored = after;
          this.nbaseCharged--;
          this.nbaseNeutral++;
          this.nanion--;
          if (m2 >= 0) this.removeId(this.idOf(m2));
        } else {
          this.energyStored = before;
          if (m1 >= 0) s.q[m1] = 1;
          if (m2 >= 0) { s.q[m2] = -1; s.mask[m2] = maskTmp; }
        }
      } else if (m1 >= 0) {
        s.q[m1] = 1;
      }
    }
  }

  private forwardIons(): void {
    const before = this.energyStored;
    const factor = this.volumeRx * this.volumeRx * this.c10pIplus * this.c10pIminus / ((1 + this.ncation) * (1 + this.nanion));
    this.insert(this.cationType, 1);
    this.insert(this.anionType, -1);
    const after = this.energyFull();
    if (this.accept(factor, after)) {
      this.energyStored = after;
      this.nsaltSucc++;
      this.ncation++;
      this.nanion++;
    } else {
      this.energyStored = before;
      this.popLast(2);
    }
  }

  private backwardIons(): void {
    const s = this.sys.state;
    const before = this.energyStored;
    const m1 = this.pick(this.cationType, 1);
    if (this.ncation > 0) {
      const m2 = this.pick(this.anionType, -1);
      if (this.nanion > 0) {
        let mask1 = 0, mask2 = 0;
        if (m1 >= 0) { s.q[m1] = 0; mask1 = s.mask[m1]; s.mask[m1] = this.excl; }
        if (m2 >= 0) { s.q[m2] = 0; mask2 = s.mask[m2]; s.mask[m2] = this.excl; }
        const factor = this.volumeRx * this.volumeRx * this.c10pIplus * this.c10pIminus / (this.ncation * this.nanion);
        const after = this.energyFull();
        if (this.accept(1.0 / factor, after)) {
          this.energyStored = after;
          this.nsaltSucc++;
          this.ncation--;
          this.nanion--;
          // native deletes the larger list position first (ions must be deleted in order)
          const ids = [m1, m2].filter((i) => i >= 0).map((i) => this.idOf(i));
          if (m1 >= 0 && m2 >= 0) {
            const ord = nativeOrder(s);
            const pos1 = ord.indexOf(m1);
            const pos2 = ord.indexOf(m2);
            if (pos1 > pos2) { this.removeId(ids[0]); this.removeId(ids[1]); } else { this.removeId(ids[1]); this.removeId(ids[0]); }
          } else {
            for (const id of ids) this.removeId(id);
          }
        } else {
          this.energyStored = before;
          if (m1 >= 0) { s.q[m1] = 1; s.mask[m1] = mask1; }
          if (m2 >= 0) { s.q[m2] = -1; s.mask[m2] = mask2; }
        }
      }
    }
  }

  /** Native forward_ions_multival: one ion of the larger charge and saltRatio ions of the other kind. */
  private forwardIonsMultival(): void {
    const before = this.energyStored;
    const [qc, qa] = this.saltCharge;
    const k = 1 + this.saltRatio;
    let factor = 1;
    if (qc <= -qa) {
      // one anion and saltRatio cations
      this.insert(this.anionType, qa);
      factor *= this.volumeRx * this.c10pIminus / (1 + this.nanion);
      for (let i = 0; i < this.saltRatio; i++) {
        this.insert(this.cationType, qc);
        factor *= this.volumeRx * this.c10pIplus / (1 + this.ncation + i);
      }
    } else {
      this.insert(this.cationType, qc);
      factor *= this.volumeRx * this.c10pIplus / (1 + this.ncation);
      for (let i = 0; i < this.saltRatio; i++) {
        this.insert(this.anionType, qa);
        factor *= this.volumeRx * this.c10pIminus / (1 + this.nanion + i);
      }
    }
    const after = this.energyFull();
    if (this.accept(factor, after)) {
      this.energyStored = after;
      this.nsaltSucc++;
      if (qc <= -qa) { this.ncation += this.saltRatio; this.nanion++; } else { this.nanion += this.saltRatio; this.ncation++; }
    } else {
      this.energyStored = before;
      this.popLast(k);
    }
  }

  /** Native backward_ions_multival: deletes one ion of the larger charge and saltRatio ions of the other kind. */
  private backwardIonsMultival(): void {
    const s = this.sys.state;
    const before = this.energyStored;
    const [qc, qa] = this.saltCharge;
    const k = 1 + this.saltRatio;
    const mm: number[] = [];
    const qq: number[] = [];
    const maskTmp: number[] = [];
    let factor = 1;
    const take = (m: number) => {
      mm.push(m);
      if (m >= 0) { qq.push(s.q[m]); s.q[m] = 0; maskTmp.push(s.mask[m]); s.mask[m] = this.excl; } else { qq.push(0); maskTmp.push(0); }
    };
    const inconsistent = () => new StyleError('fix charge/regulation salt count inconsistent');
    if (qc <= -qa) {
      if (this.ncation < this.saltRatio || this.nanion < 1) return;
      take(this.pick(this.anionType, qa));
      if (this.npartXrd !== this.nanion) throw inconsistent();
      factor *= this.volumeRx * this.c10pIminus / this.nanion;
      for (let i = 0; i < this.saltRatio; i++) {
        const m = this.pick(this.cationType, qc);
        if (this.npartXrd !== this.ncation - i) throw inconsistent();
        factor *= this.volumeRx * this.c10pIplus / (this.ncation - i);
        take(m);
      }
    } else {
      if (this.nanion < this.saltRatio || this.ncation < 1) return;
      take(this.pick(this.cationType, qc));
      if (this.npartXrd !== this.ncation) throw inconsistent();
      factor *= this.volumeRx * this.c10pIplus / this.ncation;
      for (let i = 0; i < this.saltRatio; i++) {
        const m = this.pick(this.anionType, qa);
        if (this.npartXrd !== this.nanion - i) throw inconsistent();
        take(m);
        factor *= this.volumeRx * this.c10pIminus / (this.nanion - i);
      }
    }
    const after = this.energyFull();
    if (this.accept(1.0 / factor, after)) {
      this.energyStored = after;
      // native deletes them in order of decreasing storage position (a deletion moves the last atom into its slot)
      const ord = nativeOrder(s);
      const byPos = mm.filter((m) => m >= 0).map((m) => ({ id: this.idOf(m), pos: ord.indexOf(m) })).sort((a, b) => b.pos - a.pos);
      for (const { id } of byPos) this.removeId(id);
      this.nsaltSucc++;
      if (qc <= -qa) { this.ncation -= this.saltRatio; this.nanion--; } else { this.nanion -= this.saltRatio; this.ncation--; }
    } else {
      this.energyStored = before;
      for (let i = 0; i < k; i++) {
        const m = mm[i];
        if (m >= 0) { s.q[m] = qq[i]; s.mask[m] = maskTmp[i]; }
      }
    }
  }

  private idOf(i: number): number { return this.sys.state.id[i]; }

  /** fix charge/regulation f_ID[1..8] (native compute_vector): attempts, successes, then particle counts. */
  computeVector(n: number): number {
    switch (n) {
      case 0: return this.nacidAtt + this.nbaseAtt + this.nsaltAtt;
      case 1: return this.nacidSucc + this.nbaseSucc + this.nsaltSucc;
      case 2: return this.countParticles(this.acidType, 0);
      case 3: return this.countParticles(this.acidType, -1);
      case 4: return this.countParticles(this.baseType, 0);
      case 5: return this.countParticles(this.baseType, 1);
      case 6: return this.countParticles(this.cationType, this.saltCharge[0]);
      case 7: return this.countParticles(this.anionType, this.saltCharge[1]);
      default: return 0;
    }
  }
}

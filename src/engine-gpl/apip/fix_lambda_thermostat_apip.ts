/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0:
 * src/APIP/fix_lambda_thermostat_apip.cpp (fix lambda_thermostat/apip: the arguments, init, local_neighbour_list,
 * post_force with calculate_energy_change, end_of_step with apply_thermostat, compute_vector and reset_dt), at commit
 * b332f28. Author named in the source: David Immel (FZ Juelich), with the LAMMPS Sandia copyright notice (GNU General
 * Public License). The random stream is the libstdc++ one of ./stdrandom.ts. This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/fix_lambda_thermostat_apip.html. When lambda of an atom changes, the energy difference of the
 * two potentials and the work of the changed forces are corrected by rescaling the velocities of the atom and of up to
 * N_rescaling local neighbours (shuffled with std::shuffle and std::mt19937) relative to their centre of mass. As
 * native, the atoms are visited in the order of the native atom list and the neighbours come from the full neighbour
 * list in native order, restricted to owned atoms of the group. Per-atom output (store_atomic_forces): native writes
 * the energy change of atom i to column 4 of a four-column array, that is column 0 of atom i + 1, and the engine keeps
 * that layout.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import {
  APIP_E_FAST, APIP_E_PRECISE, APIP_F_CONST_LAMBDA, APIP_F_DYN_LAMBDA, APIP_LAMBDA, APIP_LAMBDA_CONST, massOf, nativeOrder,
} from '../../engine/atoms';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { Mt19937, stdShuffle } from './stdrandom';

export class FixLambdaThermostatAPIP extends Fix {
  readonly style = 'lambda_thermostat/apip';
  private dtf = 0;
  private updateStats = false;
  private readonly rescalingN: number = 200;
  private readonly peratomFreq: number = -1;
  private readonly random: Mt19937;
  private energyChangeAtom = new Float64Array(0);
  private energyChangePot = 0;
  private energyChangeKin = 0;
  private reduceflag = false;
  private readonly outvec = new Float64Array(6);
  private sumEnergyChange = 0;
  private sumEnergyViolation = 0;
  private nEnergyViolation = 0;
  private nEnergyDifferences = 0;
  private readonly natOrder = new NativeFullOrder();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extvector = 0;
    let seed = 42, rescaling = 200, peratom = false, freq = -1;
    const int = (w: string | undefined): number => {
      if (w === undefined || w === '') throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
      if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
      return parseInt(w, 10);
    };
    for (let k = 0; k < args.length; k++) {
      const w = args[k];
      if (w === 'seed') {
        if (k + 1 >= args.length) throw new StyleError('fix lambda_thermostat/apip: seed requires one argument');
        seed = int(args[k + 1]);
        if (seed <= 0) throw new StyleError('fix lambda_thermostat/apip seed <= 0');
        k++;
      } else if (w === 'store_atomic_forces') {
        if (k + 1 >= args.length) throw new StyleError('fix lambda_thermostat/apip: store_atomic_forces requires one argument');
        peratom = true;
        freq = int(args[k + 1]);
        if (freq < 1) throw new StyleError('fix lambda_thermostat/apip: frequency of store_atomic_forces < 1');
        k++;
      } else if (w === 'N_rescaling') {
        if (k + 1 >= args.length) throw new StyleError('fix lambda_thermostat/apip: mode number requires one argument');
        rescaling = int(args[k + 1]);
        k++;
      } else {
        throw new StyleError(`fix lambda_thermostat/apip: unknown argument ${w}`);
      }
    }
    const c = sys.state.custom;
    if (!c.has(APIP_E_FAST)) throw new StyleError('fix lambda_thermostat/apip requires atomic style with e_simple.');
    if (!c.has(APIP_E_PRECISE)) throw new StyleError('fix lambda_thermostat/apip requires atomic style with e_complex.');
    if (!c.has(APIP_LAMBDA_CONST)) throw new StyleError('fix lambda_thermostat/apip requires atomic style with lambda_const.');
    if (!c.has(APIP_LAMBDA)) throw new StyleError('fix lambda_thermostat/apip requires atomic style with lambda.');
    if (!c.has(APIP_F_CONST_LAMBDA)) throw new StyleError('fix lambda_thermostat/apip requires atomic style with f_const_lambda.');
    if (!c.has(APIP_F_DYN_LAMBDA)) throw new StyleError('fix lambda_thermostat/apip requires atomic style with f_dyn_lambda.');
    if (rescaling <= 1) throw new StyleError('fix lambda_thermostat/apip: rescaling_N_neighbours <= 1');
    this.rescalingN = rescaling;
    this.random = new Mt19937(seed);
    if (peratom) {
      this.peratomFlag = true;
      this.peratomFreq = freq;
      this.sizePeratomCols = 4;
      this.arrayAtom = new Float64Array(sys.state.n * 4);
    }
  }

  init(): void {
    const s = this.sys.state;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
    if (this.sys.fixes.filter((f) => f.style === 'lambda_thermostat/apip').length > 1) {
      throw new StyleError('fix lambda_thermostat/apip: more than one fix lambda_thermostat/apip');
    }
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  /** setup(): native has none (the engine default would call postForce). */
  setup(): void {}

  /**
   * The native order of the full list is that of the build: native bins the atoms at the positions of the
   * reneighboring step, so the order is taken right after each build (setup and post_neighbor), not at end_of_step.
   */
  private refreshOrder(): void {
    const sys = this.sys;
    if (sys.nb.full) this.natOrder.neighbors(sys.state, sys.nb, sys.geom, sys.nb.full);
  }

  setupPreForce(): void {
    this.refreshOrder();
  }

  postNeighbor(): void {
    this.refreshOrder();
  }

  postForce(): void {
    this.initPeratomStats();
    this.calculateEnergyChange();
  }

  endOfStep(): void {
    this.applyThermostat();
  }

  /** The per-atom array with one spare row (the native writes past column 3, see the header). */
  private stats(): Float64Array {
    const n = this.sys.state.n;
    if (this.arrayAtom.length < (n + 1) * 4) {
      const a = new Float64Array((n + 1) * 4);
      a.set(this.arrayAtom.subarray(0, Math.min(this.arrayAtom.length, a.length)));
      this.arrayAtom = a;
    }
    return this.arrayAtom;
  }

  private initPeratomStats(): void {
    if (!this.peratomFlag || this.sys.state.step % this.peratomFreq !== 0) {
      this.updateStats = false;
      return;
    }
    this.updateStats = true;
    const out = this.stats();
    const n = this.sys.state.n;
    for (let i = 0; i < n; i++) out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = out[4 * i + 3] = 0;
  }

  /** calculate_energy_change: the potential energy change of changed lambdas and the kinetic work of the lambda forces. */
  private calculateEnergyChange(): void {
    const s = this.sys.state;
    const c = s.custom;
    const eSimple = c.get(APIP_E_FAST)!.data, eComplex = c.get(APIP_E_PRECISE)!.data;
    const lc = c.get(APIP_LAMBDA_CONST)!.data, lam = c.get(APIP_LAMBDA)!.data;
    const fc = c.get(APIP_F_CONST_LAMBDA)!.data, fd = c.get(APIP_F_DYN_LAMBDA)!.data;
    const n = s.n;
    const mvv2e = s.units.mvv2e;
    if (this.energyChangeAtom.length < n) this.energyChangeAtom = new Float64Array(n);
    const ec = this.energyChangeAtom;
    this.energyChangePot = 0;
    this.energyChangeKin = 0;
    for (let i = 0; i < n; i++) ec[i] = 0;
    const gb = this.groupBit;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & gb) || lc[i] === lam[i]) continue;
      if (eSimple[i] === 0 || eComplex[i] === 0) {
        throw new StyleError(`lambda = ${lam[i]} != ${lc[i]} = lambda_const and e_simple = ${eSimple[i]} and e_complex = ${eComplex[i]}`);
      }
      const ch = (lc[i] - lam[i]) * eSimple[i] + (lam[i] - lc[i]) * eComplex[i];
      ec[i] += ch;
      this.energyChangePot += ch;
    }
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & gb)) continue;
      const m = massOf(s, i);
      const dtfm = this.dtf / m;
      const v = s.v;
      // the six terms summed left to right, as the native expression
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        const o = 3 * i + k;
        sum = k === 0 ? v[o] * 2 * dtfm * (fc[o] - fd[o]) : sum + v[o] * 2 * dtfm * (fc[o] - fd[o]);
        sum += dtfm * dtfm * (fc[o] * fc[o] - fd[o] * fd[o]);
      }
      const ch = sum * m * mvv2e * 0.5;
      ec[i] += ch;
      this.energyChangeKin += ch;
    }
    if (this.updateStats) {
      const out = this.stats();
      for (let i = 0; i < n; i++) out[4 * i + 4] = ec[i];
    }
  }

  /** apply_thermostat: rescale the velocities of each changed atom and its shuffled local neighbours. */
  private applyThermostat(): void {
    const sys = this.sys, s = sys.state, nb = sys.nb;
    const list = nb.full;
    if (!list) throw new StyleError('fix lambda_thermostat/apip needs a full neighbor list (an APIP pair style)');
    const nbrs = this.natOrder.neighbors(s, nb, sys.geom, list);
    const v = s.v, x = s.x;
    const lc = s.custom.get(APIP_LAMBDA_CONST)!.data, lam = s.custom.get(APIP_LAMBDA)!.data;
    const nlocal = nb.nlocal;
    const gb = this.groupBit;
    const mvv2e = s.units.mvv2e;
    const ec = this.energyChangeAtom;
    const out = this.updateStats ? this.stats() : null;
    let jl = new Int32Array(64);
    this.reduceflag = true;
    this.sumEnergyChange = 0;
    this.nEnergyDifferences = 0;
    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & gb)) continue;
      if (ec[i] === 0) continue;
      this.nEnergyDifferences++;
      const mi = massOf(s, i);
      // local_neighbour_list: owned neighbours of the group, in list order
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      if (jl.length < k1 - k0) jl = new Int32Array(k1 - k0);
      let jnum = 0;
      for (let k = k0; k < k1; k++) {
        const j = nbrs[k] & NEIGHMASK;
        if (j < nlocal && (s.mask[j] & gb)) jl[jnum++] = j;
      }
      if (jnum === 0) {
        throw new StyleError(`fix lambda_thermostat/apip: thermostating required for particle with no local particles in neighbour list particle: ${x[3 * i]} ${x[3 * i + 1]} ${x[3 * i + 2]} groupbit ${s.mask[i] & gb}`);
      }
      let eRemain = ec[i];
      stdShuffle(jl, jnum, this.random);
      const nNgh = Math.min(this.rescalingN, jnum);
      if (nNgh < 2) throw new StyleError(`fix lambda_thermostat/apip: rescaling not possible for local ngh list size ${jnum}`);
      let mcm = mi;
      const vcm = [mi * v[3 * i], mi * v[3 * i + 1], mi * v[3 * i + 2]];
      for (let jj = 0; jj < nNgh; jj++) {
        const j = jl[jj];
        const mj = massOf(s, j);
        mcm += mj;
        vcm[0] += mj * v[3 * j];
        vcm[1] += mj * v[3 * j + 1];
        vcm[2] += mj * v[3 * j + 2];
      }
      vcm[0] /= mcm;
      vcm[1] /= mcm;
      vcm[2] /= mcm;
      let r0 = v[3 * i] - vcm[0], r1 = v[3 * i + 1] - vcm[1], r2 = v[3 * i + 2] - vcm[2];
      let krel = mi * (r0 * r0 + r1 * r1 + r2 * r2);
      for (let jj = 0; jj < nNgh; jj++) {
        const j = jl[jj];
        const mj = massOf(s, j);
        r0 = v[3 * j] - vcm[0];
        r1 = v[3 * j + 1] - vcm[1];
        r2 = v[3 * j + 2] - vcm[2];
        krel += mj * (r0 * r0 + r1 * r1 + r2 * r2);
      }
      krel *= mvv2e / 2.0;
      let radicand = eRemain / krel + 1;
      if (radicand < 0) {
        this.sumEnergyViolation += (-eRemain - krel);
        this.nEnergyViolation++;
        eRemain += krel;
        radicand = 0;
      }
      const beta = Math.sqrt(radicand) - 1;
      r0 = beta * (v[3 * i] - vcm[0]);
      r1 = beta * (v[3 * i + 1] - vcm[1]);
      r2 = beta * (v[3 * i + 2] - vcm[2]);
      v[3 * i] += r0;
      v[3 * i + 1] += r1;
      v[3 * i + 2] += r2;
      if (out) {
        out[4 * i] += r0 * mi / this.dtf;
        out[4 * i + 1] += r1 * mi / this.dtf;
        out[4 * i + 2] += r2 * mi / this.dtf;
      }
      for (let jj = 0; jj < nNgh; jj++) {
        const j = jl[jj];
        r0 = beta * (v[3 * j] - vcm[0]);
        r1 = beta * (v[3 * j + 1] - vcm[1]);
        r2 = beta * (v[3 * j + 2] - vcm[2]);
        v[3 * j] += r0;
        v[3 * j + 1] += r1;
        v[3 * j + 2] += r2;
        if (out) {
          const mj = massOf(s, j);
          out[4 * j] += r0 * mj / this.dtf;
          out[4 * j + 1] += r1 * mj / this.dtf;
          out[4 * j + 2] += r2 * mj / this.dtf;
        }
      }
      this.sumEnergyChange += Math.abs(eRemain);
    }
    for (const i of nativeOrder(s)) if (s.mask[i] & gb) lc[i] = lam[i];
  }

  computeVector(i: number): number {
    if (this.reduceflag) {
      this.outvec[0] = this.nEnergyDifferences;
      this.outvec[1] = this.energyChangePot;
      this.outvec[2] = this.energyChangeKin;
      this.outvec[3] = this.sumEnergyChange;
      this.outvec[4] = this.sumEnergyViolation;
      this.outvec[5] = this.nEnergyViolation;
      this.reduceflag = false;
    }
    return i < 6 ? this.outvec[i] : 0;
  }
}

import { Fix } from './fix';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import { nativeOrder } from '../atoms';
import type { System } from '../system';

/*
 * fix hyper/global — bond-boost global hyperdynamics, implemented from docs.lammps.org/fix_hyper_global.html
 * (plans/lammps-docs-live/fix_hyper_global.rst) and the two papers it cites (Voter 2013; Miron and
 * Fichthorn 2003). Quoted sentences are copied character for character from that page:
 *
 *   "Bonds are defined between each pair of atoms *ij*, whose :math:`R^0_{ij}`
 *   distance is less than *cutbond*, when the system is in a quenched state
 *   (minimum) energy."
 *   "A bond is only formed if one or both of the *ij* atoms are in
 *   the specified group."
 *   "On that step, :math:`V_{ij}` is added as a bias potential
 *   to only the single bond with strain :math:`E^{max}`, call it
 *   :math:`V^{max}_{ij}`."
 *   "The time boost factor for the system is given each timestep I by"
 *   "The elapsed time :math:`t_{hyper}` for a GHD simulation running for *N*
 *   timesteps is simply"
 *   "This fix computes a global scalar and global vector of length 12,
 *   which can be accessed by various :doc:`output commands
 *   <Howto_output>`."
 *   "This fix is not invoked during
 *   :doc:`energy minimization <minimize>`."
 *
 * The strain of a bond is E = (R - R0)/R0 (R0 from the quenched state where the bond list was made). The bias of
 * the bond with the largest |E| is V = Vmax (1 - (E/q)^2) for |E| < q (qfactor q), else 0. Its force on the two
 * atoms is 2 Vmax E / (q^2 R0) along the bond (the derivative of V with respect to R, F = -dV/dE divided by R0),
 * applied only when |E| < q. The boost is B = exp(V / (kB Tequil)) and the hyper time adds B dt on each dynamics
 * step. The bias is not evaluated during minimization (the hyper command's quenches).
 *
 * Output (f_ID, the global scalar and vector of the docs): the scalar is V on this step; the vector is
 * [boost on this step, max |E| on this step, ID of the first atom of the max-strain bond, ID of the second,
 * average bonds/atom on this step, fraction of steps of this run with zero bias, fraction of steps of this run
 * with negative strain, max drift of any atom during this run, max bond length during this run, cumulative hyper
 * time, cumulative count of event steps, cumulative count of atoms in events].
 *
 * Measured with native LAMMPS (black box, 2d square lattice with an adatom, see tests/oracle/w49hy_*): the
 * fractions divide by the current timestep minus the step at which the hyper run started, the timestep counter
 * including the iterations of the quenches; the boost is exp(V/(kB T)) with kB = 1 in lj units.
 * refused: fix_modify energy and virial (the bias energy is not added to pe).
 */
export class FixHyperGlobal extends Fix {
  readonly style = 'hyper/global';
  readonly cutbond: number;
  readonly qfactor: number;
  readonly vmax: number;
  readonly tequil: number;
  private readonly beta: number;

  /** the bond list: atom indices (i, j) and the quenched distance R0 */
  private bi = new Int32Array(0);
  private bj = new Int32Array(0);
  private r0 = new Float64Array(0);
  private nbonds = 0;
  /** values of the last bias evaluation */
  private curV = 0;
  private curBoost = 0;
  private curStrain = 0;
  private curId1 = 0;
  private curId2 = 0;
  private curAvgBonds = 0;
  /** statistics of the current hyper run */
  private runStart = 0;
  private zeroSteps = 0;
  private negSteps = 0;
  private maxDrift = 0;
  private maxBond = 0;
  /** cumulative since the fix was defined */
  private hyperTime = 0;
  private eventSteps = 0;
  private eventAtoms = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4) throw new StyleError(`fix ${id} hyper/global: expected cutbond qfactor Vmax Tequil, got ${args.length} arguments`);
    this.cutbond = parseNum(args[0], `fix ${id} hyper/global cutbond`);
    this.qfactor = parseNum(args[1], `fix ${id} hyper/global qfactor`);
    this.vmax = parseNum(args[2], `fix ${id} hyper/global Vmax`);
    this.tequil = parseNum(args[3], `fix ${id} hyper/global Tequil`);
    if (!(this.cutbond > 0)) throw new StyleError(`fix ${id} hyper/global: cutbond must be > 0`);
    if (!(this.qfactor > 0)) throw new StyleError(`fix ${id} hyper/global: qfactor must be > 0`);
    if (!(this.vmax >= 0)) throw new StyleError(`fix ${id} hyper/global: Vmax must be >= 0`);
    if (!(this.tequil > 0)) throw new StyleError(`fix ${id} hyper/global: Tequil must be > 0`);
    this.beta = 1 / (sys.units.boltz * this.tequil);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 12;
    this.extscalar = 0;
    this.extvector = 0;
    this.timeIntegrate = false;
  }

  /** Makes the bond list from the current (quenched) positions: pairs closer than cutbond, one of them in the group. */
  buildBonds(): void {
    const s = this.sys.state;
    const d = [0, 0, 0];
    const bi: number[] = [];
    const bj: number[] = [];
    const r0: number[] = [];
    // each pair is listed with its atom that comes first in the native local order (the sorted atom order), so
    // the IDs reported for the max-strain bond follow the same order as native
    const order = nativeOrder(s);
    const rank = new Int32Array(s.n);
    for (let k = 0; k < s.n; k++) rank[order[k]] = k;
    for (let a = 0; a < s.n; a++) {
      for (let b = a + 1; b < s.n; b++) {
        const i = rank[a] < rank[b] ? a : b;
        const j = i === a ? b : a;
        if (!(this.inGroup(i) || this.inGroup(j))) continue;
        for (let c = 0; c < 3; c++) d[c] = s.x[3 * j + c] - s.x[3 * i + c];
        this.sys.geom.minimumImage(d);
        const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (r < this.cutbond) { bi.push(i); bj.push(j); r0.push(r); }
      }
    }
    this.nbonds = bi.length;
    this.bi = Int32Array.from(bi);
    this.bj = Int32Array.from(bj);
    this.r0 = Float64Array.from(r0);
  }

  /** The hyper command's quenches and dynamics: the bias fix has no quench-specific state (fix/hyper/local does). */
  setQuench(): void {}

  /** Cumulative quantities of the hyper command's final stats (hyper.html "Cummulative quantities for fix hyper"). */
  summary(): { hyperTime: number; events: number; eventAtoms: number } {
    return { hyperTime: this.computeVector(9), events: this.computeVector(10), eventAtoms: this.computeVector(11) };
  }

  /** A hyper run starts at this timestep: the per-run statistics are reset (fix_hyper_global.html). */
  beginRun(step: number): void {
    this.runStart = step;
    this.zeroSteps = 0;
    this.negSteps = 0;
    this.maxDrift = 0;
    this.maxBond = 0;
  }

  /** An event was detected by the hyper command: nflag atoms moved, the largest move was drift. */
  recordEvent(nflag: number, drift: number): void {
    this.eventSteps++;
    this.eventAtoms += nflag;
    if (drift > this.maxDrift) this.maxDrift = drift;
  }

  /** A bond list reset without an event: the largest move of the quench is still reported as the drift. */
  recordDrift(drift: number): void {
    if (drift > this.maxDrift) this.maxDrift = drift;
  }

  /** Forces and values for the current positions. record = a dynamics step (counts and the hyper time advance). */
  private bias(record: boolean): void {
    const s = this.sys.state;
    const q = this.qfactor;
    this.curV = 0;
    this.curBoost = 1;
    this.curStrain = 0;
    this.curId1 = 0;
    this.curId2 = 0;
    this.curAvgBonds = 0;
    if (this.nbonds === 0) {
      if (record) this.countStep(0, false);
      return;
    }
    const d = [0, 0, 0];
    let emax = -1, best = -1, ebest = 0, rbest = 0;
    let nAvgCount = 0;
    for (let k = 0; k < this.nbonds; k++) {
      const i = this.bi[k], j = this.bj[k];
      for (let c = 0; c < 3; c++) d[c] = s.x[3 * i + c] - s.x[3 * j + c];
      this.sys.geom.minimumImage(d);
      const r = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (r > this.maxBond) this.maxBond = r;
      const e = (r - this.r0[k]) / this.r0[k];
      if (Math.abs(e) > emax) { emax = Math.abs(e); best = k; ebest = e; rbest = r; }
      nAvgCount++;
    }
    this.curAvgBonds = nAvgCount / this.groupCount();
    const i = this.bi[best], j = this.bj[best];
    this.curStrain = emax;
    this.curId1 = s.id[i];
    this.curId2 = s.id[j];
    if (emax < q) {
      const v = this.vmax * (1 - (ebest / q) * (ebest / q));
      this.curV = v;
      this.curBoost = Math.exp(this.beta * v);
      // force along the bond (minimum image unit vector from j to i), magnitude 2 Vmax E / (q^2 R0)
      const c = (2 * this.vmax * ebest) / (q * q * this.r0[best]);
      for (let k = 0; k < 3; k++) d[k] = s.x[3 * i + k] - s.x[3 * j + k];
      this.sys.geom.minimumImage(d);
      const uf = c / rbest;
      for (let k = 0; k < 3; k++) {
        s.f[3 * i + k] += uf * d[k];
        s.f[3 * j + k] -= uf * d[k];
      }
    }
    if (record) this.countStep(this.curV, ebest < 0);
  }

  private countStep(v: number, negative: boolean): void {
    if (v === 0) this.zeroSteps++;
    if (negative) this.negSteps++;
    this.hyperTime += this.curBoost * this.sys.state.dt;
  }

  private groupCount(): number {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.inGroup(i)) n++;
    return n;
  }

  /** Dynamics step: the bias force is added after the pair forces (the fix's post_force). */
  postForce(): void {
    this.bias(true);
  }

  /** Setup of a run and restored states: forces and values without counting a step. */
  setup(): void {
    this.bias(false);
  }

  computeScalar(): number {
    return this.curV;
  }

  computeVector(i: number): number {
    const steps = this.sys.state.step - this.runStart;
    switch (i) {
      case 0: return this.curBoost;
      case 1: return this.curStrain;
      case 2: return this.curId1;
      case 3: return this.curId2;
      case 4: return this.curAvgBonds;
      case 5: return steps > 0 ? this.zeroSteps / steps : 0;
      case 6: return steps > 0 ? this.negSteps / steps : 0;
      case 7: return this.maxDrift;
      case 8: return this.maxBond;
      case 9: return this.hyperTime;
      case 10: return this.eventSteps;
      case 11: return this.eventAtoms;
      default: throw new StyleError(`fix ${this.id} hyper/global vector index ${i + 1} is out of range`);
    }
  }

  modify(key: string, values: string[]): number {
    if (key === 'energy' || key === 'virial') throw new StyleError(`fix ${this.id} (hyper/global) does not support fix_modify ${key} in the browser engine`);
    return super.modify(key, values);
  }
}

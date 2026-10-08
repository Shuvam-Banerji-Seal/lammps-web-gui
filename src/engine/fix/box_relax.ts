import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { ownCompute, removeCompute } from './util';
import type { Compute } from '../compute/compute';

/*
 * fix box/relax (Haiku wave 10). docs.lammps.org/fix_box_relax.html
 *
 * Objective, as the doc states it: "E = U + P_t \left(V-V_0 \right) + E_{strain}"
 * "where *U* is the system potential energy," P_t the desired hydrostatic
 * pressure, V and V_0 the system and reference volumes (the reference cell is
 * the box at the start of the minimization). "nreset": "A value of
 * *nstep* means that every *nstep* minimization steps, the reference dimensions
 * are set to those of the current simulation domain" (implemented from this
 * sentence, not measured).
 *
 * The fix's scalar is "the pressure-volume energy, plus the strain energy, if it
 * exists": it is the f_ID value in thermo. "The energy values reported at the end
 * of a minimization run under" Minimization stats "include this energy".
 *
 * Measured with native LAMMPS (black box), 108-atom fcc LJ box, lj units:
 *   - the printed f_ID is per atom: N f_1 = Ph (V - V0) + S, where Ph is the mean
 *     of the specified diagonal targets (x 1.0 alone: Ph = 1.0; x 1.0 y 0.0 z 0.0:
 *     Ph = 1/3). Fits of N f_1 hold to 1e-13 (diagonal) and 1e-14 (one xy, xz or
 *     yz target).
 *   - diagonal strain: S = V0 * sum_d (P_d - Ph) * eta_d with eta_d = (l_d^2/l0_d^2 - 1)/2
 *     (zero for iso and for all-zero targets).
 *   - tilt strain: a target P_xy adds V0 * P_xy * (xy/l0_x) * (ly/l0_y); xz pairs
 *     with lz and yz with lz (coefficient 1). Cubic reference boxes only were tested.
 *   - the minimizer energy is pe + f_ID per atom (lj units) in the stats.
 *   - the box change per iteration: the largest relative change of a linear
 *     dimension equals vmax: vmax 0.001 gives lx -0.001 on the first iteration,
 *     vmax 1e-5 gives -1e-5, default 0.0001 gives -1e-4.
 *   - the force test uses the atoms and the box DOF together: the initial
 *     Force two-norm of an empty-atom box (perfect fcc at density 1.0, iso 0.0)
 *     is 1099.5, the box gradient 3 * V0 * 3.39.
 *   - aniso (x 1.0 y 0.5 z 0.0) converged at force tolerance with Pyy = -0.0146,
 *     which is the stationary stress of this objective: sigma_dd = Ph + (P_d - Ph)
 *     (V0/V)(l_d/l0_d)^2 (the doc's "only applies when the box dimensions are equal
 *     to those of the reference dimensions" in the first order).
 *   - iso (couple xyz) converged to P = 0.99999999999992 with ftol 1e-10.
 *   - tri 0.5 converged (all six components 0.5, shear 0). A tri run with only a
 *     shear target (xy 1.0) stopped with linesearch alpha is zero (not checked
 *     against the stationary relation; not used as an oracle case).
 *
 * Gradient: the exact derivative of the objective above, using the virial of
 * compute pressure. The doc: "are assumed to only be the virial component of the
 * pressure (the non-kinetic portion)", so the kinetic part is excluded. Stationary relation (the doc):
 * "This equation only applies when the box dimensions are equal to those of the
 * reference dimensions." Coupled dimensions use the group average of the stress:
 * "the instantaneous stress will be computed as an average of the corresponding
 * diagonal components".
 *
 * Not supported (StyleError): dilate partial (not implemented). scaleyz/scalexz/
 * scalexy are accepted; their effect on the path is not measured.
 */

/** Box degrees of freedom as the minimizer sees them (see run/min.ts). */
export interface BoxDof {
  readonly dofCount: number;
  readonly vmax: number;
  readonly nreset: number;
  /** Records the box at the start of an iteration (the remap is relative to it). */
  begin(): void;
  /** Current DOF values: dimensionless strains (diagonal) and tilt/l0 (tilt). */
  dof(out: Float64Array): void;
  /** Sets the box for DOF values u and writes the remapped positions of x0 into xOut. */
  trial(u: Float64Array, x0: Float64Array, xOut: Float64Array): void;
  /** Energy of the box terms at the current state; force[k] = -dE/du_k. */
  evaluate(force: Float64Array): number;
  /** Every nreset iterations: the current box becomes the reference. */
  resetReference(): void;
}

type Mat3 = number[][];

const inv3 = (h: Mat3): Mat3 => {
  const [[a, b, c], [d, e, f], [g, k, m]] = h;
  const det = a * (e * m - f * k) - b * (d * m - f * g) + c * (d * k - e * g);
  if (!Number.isFinite(det) || det === 0) throw new StyleError(`fix box/relax: singular box ${JSON.stringify(h)}`);
  return [
    [(e * m - f * k) / det, (c * k - b * m) / det, (b * f - c * e) / det],
    [(f * g - d * m) / det, (a * m - c * g) / det, (c * d - a * f) / det],
    [(d * k - e * g) / det, (b * g - a * k) / det, (a * e - b * d) / det],
  ];
};

const mul3 = (a: Mat3, b: Mat3): Mat3 => a.map((row) => [0, 1, 2].map((j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j]));

/** Upper-triangular box matrix h = [[lx xy xz],[0 ly yz],[0 0 lz]] (the engine's geometry convention). */
const hMatrix = (l: number[], t: number[]): Mat3 => [[l[0], t[0], t[1]], [0, l[1], t[2]], [0, 0, l[2]]];

const COUPLE_DIMS: Record<string, number[]> = { none: [], xyz: [0, 1, 2], xy: [0, 1], yz: [1, 2], xz: [0, 2] };
/** Tilt index in the box tilt array [xy, xz, yz]; the (i, j) pair of the tilt's strain term. */
const TILT_PAIR = [[0, 1], [0, 2], [1, 2]];
const TILT_NAME = ['xy', 'xz', 'yz'];
const DIAG_NAME = ['x', 'y', 'z'];

export class FixBoxRelax extends Fix {
  readonly style = 'box/relax';
  /** Target pressure per diagonal dim (x, y, z) and per tilt (xy, xz, yz); null = not controlled. */
  private pDiag: (number | null)[] = [null, null, null];
  private pTilt: (number | null)[] = [null, null, null];
  private couple = 'none';
  private vmaxValue = 0.0001;
  private nresetValue = 0;
  private fixedPoint: number[] | null = null;
  private pc: Compute;
  // derived at minSetup
  private groups: number[][] = [];         // diagonal DOF groups (dims)
  private tiltDofs: number[] = [];         // tilt indices with a target
  private ph = 0;
  private lRef = [1, 1, 1];
  private tRef = [0, 0, 0];
  private vRef = 1;
  private pivot: number[] = [0, 0, 0];
  private startLo: number[] = [0, 0, 0];
  private startH: Mat3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  private startL: number[] = [1, 1, 1];
  private startT: number[] = [0, 0, 0];
  private ready = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length === 0) throw new StyleError('fix box/relax needs at least one keyword (iso, aniso, tri, x, y, z, xy, xz, yz, ...)');
    let any = false;
    const need = (k: number, what: string): string => {
      const v = args[k];
      if (v === undefined) throw new StyleError(`fix box/relax: keyword '${what}' needs a value`);
      return v;
    };
    const num = (w: string, what: string): number => {
      const v = Number(w);
      if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`fix box/relax: ${what} must be a number, got '${w}'`);
      return v;
    };
    const yesno = (w: string, what: string): void => {
      if (w !== 'yes' && w !== 'no') throw new StyleError(`fix box/relax: ${what} must be yes or no, got '${w}'`);
    };
    for (let k = 0; k < args.length;) {
      const key = args[k];
      switch (key) {
        case 'iso': {
          const p = num(need(k + 1, key), 'iso Ptarget');
          this.pDiag = [p, p, p];
          this.couple = 'xyz';
          k += 2; any = true; break;
        }
        case 'aniso': {
          const p = num(need(k + 1, key), 'aniso Ptarget');
          this.pDiag = [p, p, p];
          this.couple = 'none';
          k += 2; any = true; break;
        }
        case 'tri': {
          const p = num(need(k + 1, key), 'tri Ptarget');
          this.pDiag = [p, p, p];
          this.pTilt = [0, 0, 0];
          this.couple = 'none';
          k += 2; any = true; break;
        }
        case 'x': case 'y': case 'z': {
          const d = ['x', 'y', 'z'].indexOf(key);
          this.pDiag[d] = num(need(k + 1, key), `${key} Ptarget`);
          k += 2; any = true; break;
        }
        case 'xy': case 'xz': case 'yz': {
          const t = TILT_NAME.indexOf(key);
          this.pTilt[t] = num(need(k + 1, key), `${key} Ptarget`);
          k += 2; any = true; break;
        }
        case 'couple': {
          const v = need(k + 1, key);
          if (!(v in COUPLE_DIMS)) throw new StyleError(`fix box/relax: couple must be none, xyz, xy, yz or xz, got '${v}'`);
          this.couple = v;
          k += 2; break;
        }
        case 'nreset': {
          const v = num(need(k + 1, key), 'nreset');
          if (!Number.isInteger(v) || v < 0) throw new StyleError('fix box/relax: nreset must be an integer >= 0');
          this.nresetValue = v;
          k += 2; break;
        }
        case 'vmax': {
          const v = num(need(k + 1, key), 'vmax');
          if (!(v > 0)) throw new StyleError('fix box/relax: vmax must be greater than 0.0');
          this.vmaxValue = v;
          k += 2; break;
        }
        case 'dilate': {
          const v = need(k + 1, key);
          if (v === 'partial') throw new StyleError('fix box/relax: dilate partial is not supported by the browser engine');
          if (v !== 'all') throw new StyleError(`fix box/relax: dilate must be all or partial, got '${v}'`);
          k += 2; break;
        }
        case 'scaleyz': case 'scalexz': case 'scalexy': {
          yesno(need(k + 1, key), key);
          k += 2; break;
        }
        case 'fixedpoint': {
          const vals = args.slice(k + 1, k + 4);
          if (vals.length < 3) throw new StyleError('fix box/relax: fixedpoint needs three values x y z');
          this.fixedPoint = vals.map((w, i) => num(w, `fixedpoint ${'xyz'[i]}`));
          k += 4; break;
        }
        default:
          throw new StyleError(`fix box/relax: unknown or unsupported keyword '${key}'`);
      }
    }
    if (!any) throw new StyleError('fix box/relax: no pressure keyword (iso, aniso, tri, x, y, z, xy, xz, yz) given');
    if (this.couple !== 'none') {
      const dims = COUPLE_DIMS[this.couple];
      for (const d of dims) if (this.pDiag[d] === null) throw new StyleError(`fix box/relax: couple ${this.couple} needs a ${DIAG_NAME[d]} target`);
      const vals = dims.map((d) => this.pDiag[d]);
      if (vals.some((v) => v !== vals[0])) throw new StyleError(`fix box/relax: couple ${this.couple} needs identical Ptarget values for the coupled dimensions`);
    }
    this.pc = ownCompute(sys, `${id}_press`, 'all', 'pressure', ['NULL', 'virial']);
    this.scalarFlag = true;
  }

  get vmax(): number { return this.vmaxValue; }

  /** Box DOF interface for the minimizer; valid after minSetup. */
  get boxDof(): BoxDof {
    return {
      dofCount: this.groups.length + this.tiltDofs.length,
      vmax: this.vmaxValue,
      nreset: this.nresetValue,
      begin: () => this.beginIteration(),
      dof: (out) => this.readDof(out),
      trial: (u, x0, xOut) => this.applyTrial(u, x0, xOut),
      evaluate: (force) => this.evaluateBox(force),
      resetReference: () => this.setReference(),
    };
  }

  minSetup(): void {
    const s = this.sys.state;
    const b = s.box;
    if (s.dimension === 2 && (this.pDiag[2] !== null)) throw new StyleError('fix box/relax: z is not available for 2d simulations');
    if (this.pTilt.some((t) => t !== null) && !b.triclinic) throw new StyleError('fix box/relax: xy, xz, yz need a triclinic box (change_box all triclinic)');
    for (let d = 0; d < 3; d++) {
      if (this.pDiag[d] !== null && !b.periodic[d]) throw new StyleError(`fix box/relax: ${DIAG_NAME[d]} is not periodic`);
    }
    if (this.pDiag.every((p) => p === null) && this.pTilt.every((t) => t === null)) throw new StyleError('fix box/relax: nothing to relax');
    const coupled = COUPLE_DIMS[this.couple].filter((d) => !(s.dimension === 2 && d === 2));
    const diagDims = [0, 1, 2].filter((d) => this.pDiag[d] !== null && !(s.dimension === 2 && d === 2));
    this.groups = [];
    if (coupled.length) this.groups.push(coupled);
    for (const d of diagDims) if (!coupled.includes(d)) this.groups.push([d]);
    this.tiltDofs = [0, 1, 2].filter((t) => this.pTilt[t] !== null);
    const specified = diagDims.map((d) => this.pDiag[d] as number);
    this.ph = specified.length ? specified.reduce((a, c) => a + c, 0) / specified.length : 0;
    this.setReference();
    // default fixed point: the centre of the (possibly tilted) box, lo + 0.5 (a + b + c)
    const [lx, ly, lz] = this.currentLengths();
    const [xy, xz, yz] = b.tilt;
    this.pivot = this.fixedPoint ? [...this.fixedPoint] : [b.lo[0] + 0.5 * (lx + xy + xz), b.lo[1] + 0.5 * (ly + yz), b.lo[2] + 0.5 * lz];
    this.ready = true;
  }

  /** Reference cell: the current box (minimize start, or nreset). */
  private setReference(): void {
    const s = this.sys.state;
    this.lRef = this.currentLengths();
    this.tRef = [...s.box.tilt];
    this.vRef = this.lRef[0] * this.lRef[1] * this.lRef[2];
  }

  private currentLengths(): number[] {
    const b = this.sys.state.box;
    return [b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]];
  }

  private pv2e(): number { return 1 / this.sys.state.units.nktv2p; }

  private beginIteration(): void {
    const b = this.sys.state.box;
    this.startLo = [...b.lo];
    this.startL = this.currentLengths();
    this.startT = [...b.tilt];
    this.startH = hMatrix(this.startL, this.startT);
  }

  /** Current DOF values: group strain e = l/l0 - 1 (mean over the group), tilt (t - t0)/l0_i. */
  private readDof(out: Float64Array): void {
    const l = this.currentLengths();
    const t = this.sys.state.box.tilt;
    let k = 0;
    for (const g of this.groups) {
      let e = 0;
      for (const d of g) e += l[d] / this.lRef[d] - 1;
      out[k++] = e / g.length;
    }
    for (const ti of this.tiltDofs) out[k++] = (t[ti] - this.tRef[ti]) / this.lRef[TILT_PAIR[ti][0]];
  }

  /** Sets the box for DOF values u (relative to the reference) and remaps x0 affinely about the fixed point. */
  private applyTrial(u: Float64Array, x0: Float64Array, xOut: Float64Array): void {
    const s = this.sys.state;
    const newL = [...this.startL];
    const newT = [...this.startT];
    let k = 0;
    for (const g of this.groups) {
      const e = u[k++];
      for (const d of g) newL[d] = this.lRef[d] * (1 + e);
    }
    for (const ti of this.tiltDofs) {
      newT[ti] = this.tRef[ti] + u[k++] * this.lRef[TILT_PAIR[ti][0]];
    }
    const hNew = hMatrix(newL, newT);
    const F = mul3(hNew, inv3(this.startH));
    const p = this.pivot;
    const lo = [0, 1, 2].map((a) => p[a] + F[a][0] * (this.startLo[0] - p[0]) + F[a][1] * (this.startLo[1] - p[1]) + F[a][2] * (this.startLo[2] - p[2]));
    const hi = [lo[0] + newL[0], lo[1] + newL[1], lo[2] + newL[2]];
    this.sys.setBox(lo, hi, newT, 0, this);
    const n = s.n;
    for (let i = 0; i < n; i++) {
      const x = x0[3 * i] - p[0], y = x0[3 * i + 1] - p[1], z = x0[3 * i + 2] - p[2];
      xOut[3 * i] = p[0] + F[0][0] * x + F[0][1] * y + F[0][2] * z;
      xOut[3 * i + 1] = p[1] + F[1][0] * x + F[1][1] * y + F[1][2] * z;
      xOut[3 * i + 2] = p[2] + F[2][0] * x + F[2][1] * y + F[2][2] * z;
    }
  }

  /** Box energy and force at the current state (virial from compute pressure). */
  private evaluateBox(force: Float64Array): number {
    const energy = this.boxEnergy();
    this.sys.refreshComputes();
    const v = this.pc.vectorValues();
    const l = this.currentLengths();
    const t = [...this.sys.state.box.tilt];
    const vol = l[0] * l[1] * l[2];
    const pv2e = this.pv2e();
    // stress in pressure units; coupled dims use the group average (doc: "the instantaneous stress will be computed as an average of the corresponding diagonal components")
    const sig: Mat3 = [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]];
    for (const g of this.groups) {
      if (g.length < 2) continue;
      const avg = g.reduce((a, d) => a + sig[d][d], 0) / g.length;
      for (const d of g) sig[d][d] = avg;
    }
    const hinv = inv3(hMatrix(l, t));
    // M_ab = sum_c sig_ac * hinv_bc ; dU/dh_ab = -pv2e * vol * M_ab
    const M: Mat3 = sig.map((row) => [0, 1, 2].map((b) => row[0] * hinv[b][0] + row[1] * hinv[b][1] + row[2] * hinv[b][2]));
    const dEdl = [0, 1, 2].map((d) => {
      let r = -pv2e * vol * M[d][d];
      if (this.pDiag[d] !== null) {
        r += pv2e * this.ph * vol / l[d];
        r += pv2e * this.vRef * ((this.pDiag[d] as number) - this.ph) * l[d] / (this.lRef[d] * this.lRef[d]);
      }
      return r;
    });
    for (const ti of this.tiltDofs) {
      const [i, j] = TILT_PAIR[ti];
      dEdl[j] += pv2e * this.vRef * (this.pTilt[ti] as number) * (t[ti] - this.tRef[ti]) / (this.lRef[i] * this.lRef[j]);
    }
    let k = 0;
    for (const g of this.groups) {
      let dE = 0;
      for (const d of g) dE += this.lRef[d] * dEdl[d];
      force[k++] = -dE;
    }
    for (const ti of this.tiltDofs) {
      const [i, j] = TILT_PAIR[ti];
      const dEdt = -pv2e * vol * M[i][j] + pv2e * this.vRef * (this.pTilt[ti] as number) * l[j] / (this.lRef[i] * this.lRef[j]);
      force[k++] = -this.lRef[i] * dEdt;
    }
    return energy;
  }

  /** The box part of the objective: pv2e * [Ph (V - V0) + V0 sum (P_d - Ph) eta_d + V0 sum P_ij (t_ij - t0)/l0_i * l_j/l0_j]. */
  boxEnergy(): number {
    if (!this.ready) return 0;
    const l = this.currentLengths();
    const t = this.sys.state.box.tilt;
    const vol = l[0] * l[1] * l[2];
    let e = this.ph * (vol - this.vRef);
    for (let d = 0; d < 3; d++) {
      const p = this.pDiag[d];
      if (p === null) continue;
      const r = l[d] / this.lRef[d];
      e += this.vRef * (p - this.ph) * (r * r - 1) / 2;
    }
    for (const ti of this.tiltDofs) {
      const [i, j] = TILT_PAIR[ti];
      e += this.vRef * (this.pTilt[ti] as number) * ((t[ti] - this.tRef[ti]) / this.lRef[i]) * (l[j] / this.lRef[j]);
    }
    return e * this.pv2e();
  }

  /** f_ID: "the pressure-volume energy, plus the strain energy" per atom (see the header). */
  computeScalar(): number {
    if (!this.ready) return 0;
    return this.boxEnergy() / this.sys.state.n;
  }

  destroy(): void { removeCompute(this.sys, `${this.id}_press`); }
}

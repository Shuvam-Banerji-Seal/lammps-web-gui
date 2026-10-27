/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/pair_pace_apip.cpp,
 * pair_pace_fast_apip.cpp and pair_pace_precise_apip.cpp (pair_style pace/apip, pace/precise/apip and pace/fast/apip:
 * settings, init_style, setup, compute with the lambda weights, check_abort_condition and compute_factor_lambda), at
 * commit b332f28. Authors named in the sources: Yury Lysogorskiy and co-authors (pair_style pace) and David Immel (FZ
 * Juelich) for APIP, with the LAMMPS Sandia copyright notice (GNU General Public License). The ACE evaluator is the
 * port in ../pace. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_pace_apip.html. The precise styles (pace/apip, pace/precise/apip) weight the ACE energy and
 * forces of atom i with 1 - lambda_i and skip atoms with lambda_i = 1; pace/fast/apip weights them with lambda_i and
 * skips lambda_i = 0. Each computed atom sets the COMPLEX (precise) or SIMPLE (fast) bit of lambda_required, a skipped
 * atom the NO_ bit.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { PairPace } from '../pace/pair_pace';
import { COMPLEX, NO_COMPLEX, NO_SIMPLE, SIMPLE, apipFields, apipModes, checkApipAtomStyle, reverseAdd3 } from './common';

export class PairPaceAPIP extends PairPace {
  override readonly name: string = 'pace/apip';
  /** fast potential (pace/fast/apip): weight lambda; otherwise precise: weight 1 - lambda. */
  protected readonly fast: boolean = false;
  private lambdaThermostat = false;
  private lambdaLa = false;
  /** The extract name of the time per atom: pace/fast/apip for the fast potential, pace/apip for the precise ones. */
  protected get timeName(): string {
    return this.fast ? 'pace/fast/apip:time_per_atom' : 'pace/apip:time_per_atom';
  }

  /** Wall time and counted atoms since the last time_per_atom extraction, and the total wall time (seconds). */
  private timeWallAccumulated = 0;
  private nAccumulated = 0;
  private wallTotal = 0;

  /** pair->extract("<style>:time_per_atom") through calculate_time_per_atom: the mean, then a reset (-1 without atoms). */
  apipTimePerAtom(name: string): number | null {
    if (name !== this.timeName) return null;
    const t = this.nAccumulated > 0 ? this.timeWallAccumulated / this.nAccumulated : -1;
    this.timeWallAccumulated = 0;
    this.nAccumulated = 0;
    return t;
  }

  apipWallTotal(): number {
    return this.wallTotal;
  }

  private addTiming(t0: number, n: number): void {
    const dt = (performance.now() - t0) / 1000;
    this.timeWallAccumulated += dt;
    this.wallTotal += dt;
    this.nAccumulated += n;
  }

  /** PairPACEAPIP::settings: [recursive|product] [chunksize N], at most 3 words (native messages). */
  override settings(args: string[], ctx: StyleContext): void {
    if (args.length > 3) throw new StyleError('Illegal pair_style pace command: missing argument(s)');
    if (ctx.s && ctx.s.units.style !== 'metal') throw new StyleError("ACE potentials require 'metal' units");
    this.recursive = true;
    let iarg = 0;
    while (iarg < args.length) {
      const w = args[iarg];
      if (w === 'recursive') {
        this.recursive = true;
        iarg += 1;
      } else if (w === 'product') {
        this.recursive = false;
        iarg += 1;
      } else if (w === 'chunksize') {
        const v = args[iarg + 1];
        if (v === undefined || !/^[+-]?\d+$/.test(v)) throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
        this.chunksize = parseInt(v, 10);
        iarg += 2;
      } else {
        throw new StyleError(`Unknown pair_style pace keyword: ${w}`);
      }
    }
  }

  override initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    checkApipAtomStyle(ctx, this.name);
    const m = apipModes(ctx, this.name);
    this.lambdaThermostat = m.thermostat;
    this.lambdaLa = m.la;
  }

  /** compute_factor_lambda: lambda for the fast potential, 1 - lambda for the precise one. */
  protected factorLambda(lambda: number): number {
    return this.fast ? lambda : 1 - lambda;
  }

  /** check_abort_condition: true (and the NO_ bit) when the weight of atom i is zero. */
  private abort(lambda: Float64Array, lambdaConst: Float64Array | null, req: Float64Array, i: number): boolean {
    const off = this.fast ? 0 : 1;
    const thermo = this.lambdaThermostat;
    if (lambda[i] === off && (!thermo || (thermo && lambdaConst![i] === off))) {
      req[i] = (req[i] | (this.fast ? NO_SIMPLE : NO_COMPLEX));
      return true;
    }
    req[i] = (req[i] | (this.fast ? SIMPLE : COMPLEX));
    return false;
  }

  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    if (!this.ace) throw new StyleError('All pair coeffs are not set');
    const ace = this.ace;
    const t0 = performance.now();
    let nComputations = 0;
    const ap = apipFields(pc);
    const thermo = this.lambdaThermostat;
    const eRef = thermo || this.lambdaLa ? (this.fast ? ap.eFast : ap.ePrecise) : null;
    const fDyn = thermo ? new Float64Array(3 * pc.nall) : null;
    const fConst = thermo ? new Float64Array(3 * pc.nall) : null;
    const x = pc.x, f = pc.f, type = pc.type;
    const nbrs = this.natOrder.neighbors(pc.s, pc.nb, pc.geom, list);
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      if (this.abort(ap.lambda, ap.lambdaConst, ap.lambdaRequired, i)) continue;
      nComputations++;
      const factorI = this.factorLambda(ap.lambda[i]);
      const factorConstI = thermo ? this.factorLambda(ap.lambdaConst![i]) : 0;
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      const jl = new Int32Array(jnum);
      for (let jj = 0; jj < jnum; jj++) jl[jj] = nbrs[k0 + jj] & NEIGHMASK;
      ace.computeAtom(i, x, type, jnum, jl, 0);
      const nf = ace.neighboursForces;
      // fix adapt's scale[itype][itype] on the neighbour forces and the atomic energy (src/APIP/pair_pace_apip.cpp)
      const sc = this.scale[type[i] * (this.ntypes + 1) + type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let jj = 0; jj < jnum; jj++) {
        const j = jl[jj];
        const fx = sc * nf[3 * jj], fy = sc * nf[3 * jj + 1], fz = sc * nf[3 * jj + 2];
        const wx = factorI * fx, wy = factorI * fy, wz = factorI * fz;
        f[3 * i] += wx; f[3 * i + 1] += wy; f[3 * i + 2] += wz;
        f[3 * j] -= wx; f[3 * j + 1] -= wy; f[3 * j + 2] -= wz;
        if (fDyn && fConst) {
          fDyn[3 * i] += wx; fDyn[3 * i + 1] += wy; fDyn[3 * i + 2] += wz;
          fDyn[3 * j] -= wx; fDyn[3 * j + 1] -= wy; fDyn[3 * j + 2] -= wz;
          const cx = factorConstI * fx, cy = factorConstI * fy, cz = factorConstI * fz;
          fConst[3 * i] += cx; fConst[3 * i + 1] += cy; fConst[3 * i + 2] += cz;
          fConst[3 * j] -= cx; fConst[3 * j + 1] -= cy; fConst[3 * j + 2] -= cz;
        }
        if (pc.vatom) {
          // ev_tally_xyz with the weighted force and the separation x_i - x_j, half to each atom
          const dx = -(x[3 * j] - xtmp), dy = -(x[3 * j + 1] - ytmp), dz = -(x[3 * j + 2] - ztmp);
          const v = [dx * wx, dy * wy, dz * wz, dx * wy, dx * wz, dy * wz];
          const va = pc.vatom;
          for (let c = 0; c < 6; c++) {
            va[6 * i + c] += 0.5 * v[c];
            va[6 * j + c] += 0.5 * v[c];
          }
        }
      }
      // the reference energy for the thermostat, and ev_tally_full with 2 x the weighted atomic energy
      if (eRef) eRef[i] = sc * ace.eAtom;
      const evdwl = factorI * sc * ace.eAtom;
      pc.acc.evdwl += evdwl;
      if (pc.eatom) pc.eatom[i] += evdwl;
    }
    if (fDyn && fConst) {
      reverseAdd3(pc, fDyn, ap.fDynLambda!);
      reverseAdd3(pc, fConst, ap.fConstLambda!);
    }
    this.addTiming(t0, nComputations);
  }
}

/** pair_style pace/precise/apip: the precise ACE potential (the same weights as pace/apip). */
export class PairPacePreciseAPIP extends PairPaceAPIP {
  override readonly name: string = 'pace/precise/apip';
}

/** pair_style pace/fast/apip: the fast ACE potential, weighted with lambda. */
export class PairPaceFastAPIP extends PairPaceAPIP {
  override readonly name: string = 'pace/fast/apip';
  protected override readonly fast: boolean = true;
}

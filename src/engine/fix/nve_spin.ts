import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { findSpinPair, type PairSpinExchange } from '../force/pair/spin_exchange';
import { massOf, nativeOrder } from '../atoms';
import { FixPrecessionSpin } from './precession_spin';

/*
 * fix ID group-ID nve/spin keyword values — docs.lammps.org/fix_nve_spin.html:
 * "Perform a symplectic integration for the spin or spin-lattice system." and
 * the lattice keyword takes moving (integrate both spin and atomic degrees of
 * freedom) or frozen (integrate spins on a fixed lattice). "By default a spin-lattice
 * integration is performed (lattice = moving)." "To use the spin algorithm, it is
 * necessary to define a map with the atom_modify command."
 *
 * The spin update is the Suzuki-Trotter sequence of Tranchida et al. (J. Comput. Phys.
 * 372, 406 (2018), arXiv 1801.10233, section 3): e^{Ls dt/2} is a forward sweep over the
 * spins, each updated with dt/4 immediately after its precession vector is computed,
 * followed by a backward sweep with dt/4. The single-spin propagator is the Omelyan
 * one (same paper, section 4): s_i(t+dt) = { s_i(t) + dt (w_i x s_i) + dt^2/4 [ 2 w_i (w_i . s_i) - |w_i|^2 s_i ] } / (1 + dt^2/4 |w_i|^2),
 * with w_i the precession vector.
 *
 * Measured with native LAMMPS (black box, two atoms, lattice frozen, dt 0.001, one step):
 * the spins equal two forward and backward sweeps of dt/4 each (four sweeps in all, the
 * order fwd, bwd, fwd, bwd) to about 1e-11, with the precession vector from
 * SPIN_HBAR = 6.5821191e-4 eV ps. After the step, compute property/atom fmx fmy fmz
 * shows the precession vector of the new spins.
 *
 * Frozen lattice: this fix applies the first half of the spin sequence in initial_integrate
 * and the second half in final_integrate (the forces of the step are not used). Moving
 * lattice is not implemented yet and is refused.
 */

const omelyan = (s: Float64Array, i: number, w: number[], dt: number): void => {
  const sx = s[4 * i], sy = s[4 * i + 1], sz = s[4 * i + 2];
  const w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  const ws = w[0] * sx + w[1] * sy + w[2] * sz;
  // w x s
  const cx = w[1] * sz - w[2] * sy, cy = w[2] * sx - w[0] * sz, cz = w[0] * sy - w[1] * sx;
  const a = dt * dt / 4;
  const den = 1 + a * w2;
  s[4 * i] = (sx + dt * cx + a * (2 * w[0] * ws - w2 * sx)) / den;
  s[4 * i + 1] = (sy + dt * cy + a * (2 * w[1] * ws - w2 * sy)) / den;
  s[4 * i + 2] = (sz + dt * cz + a * (2 * w[2] * ws - w2 * sz)) / den;
};

/** The spin pair style of a force field (a sub-style of pair hybrid), or null. */

export class FixNVESpin extends Fix {
  readonly style: string = 'nve/spin';
  readonly integrates = true;
  private lattice: 'moving' | 'frozen' = 'moving';
  private pair: PairSpinExchange | null = null;
  private w = [0, 0, 0];
  private wbuf = new Float64Array(3);
  private fields: FixPrecessionSpin[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!sys.state.sp) throw new StyleError('fix nve/spin requires atom_style spin');
    for (let k = 0; k < args.length;) {
      if (args[k] === 'lattice') {
        const v = args[k + 1];
        if (v !== 'moving' && v !== 'frozen') throw new StyleError(`fix nve/spin: lattice must be moving or frozen, got '${v ?? ''}'`);
        this.lattice = v;
        k += 2;
      } else throw new StyleError(`Illegal fix nve/spin keyword: ${args[k]}`);
    }
    this.timeIntegrate = true;
  }

  init(): void {
    this.pair = findSpinPair(this.sys.ff.pair);
    this.fields = this.sys.fixes.filter((f): f is FixPrecessionSpin => f instanceof FixPrecessionSpin);
    this.sys.state.spinLattice = this.lattice === 'moving';
  }

  /**
   * Precession vector of owned atom i with the current spins: the spin pair style (Pair.omegaOf, when there is
   * one) plus the precession/spin fixes of the system (fix/precession_spin.ts). Native LAMMPS runs nve/spin
   * without a spin pair style too (measured: a zeeman-only spin rotates).
   */
  private omega(i: number): void {
    const nb = this.sys.nb;
    const sp = this.sys.state.sp!;
    this.w[0] = 0; this.w[1] = 0; this.w[2] = 0;
    if (this.pair) {
      this.pair.omegaOf(i, nb, sp, nb.typeall, nb.xall, nb.full!, this.wbuf);
      this.w[0] = this.wbuf[0]; this.w[1] = this.wbuf[1]; this.w[2] = this.wbuf[2];
    }
    for (const f of this.fields) {
      f.omegaOf(i, sp, this.wbuf);
      this.w[0] += this.wbuf[0]; this.w[1] += this.wbuf[1]; this.w[2] += this.wbuf[2];
    }
  }

  /** One sweep over the owned atoms of the group in the given order, each spin updated with dt. */
  private sweep(dt: number, reverse: boolean): void {
    const s = this.sys.state;
    const sp = s.sp!;
    const bit = this.groupBit;
    const mask = s.mask;
    // sweep in the native local atom order (SimState.order, which sortAtoms rewrites)
    const ord = nativeOrder(s);
    for (let m = 0; m < s.n; m++) {
      const i = ord[reverse ? s.n - 1 - m : m];
      if (!(mask[i] & bit)) continue;
      this.omega(i);
      omelyan(sp, i, this.w, dt);
    }
  }

  /** Precession vectors of all atoms with the current spins (compute property/atom fmx ... shows them). */
  private refreshFm(): void {
    const s = this.sys.state;
    const fm = s.fm!;
    for (let i = 0; i < s.n; i++) {
      this.omega(i);
      fm[3 * i] = this.w[0]; fm[3 * i + 1] = this.w[1]; fm[3 * i + 2] = this.w[2];
    }
  }

  /**
   * Moving lattice, the sequence measured with native LAMMPS (black box, two atoms, one step, lattice moving):
   * v += dt/2 F/m; spin forward and backward sweeps (dt/4 each); x += dt v; [neighbor list and forces at the new
   * positions; the second pair of sweeps runs in preForce, after the ghosts are updated]; v += dt/2 F/m. The
   * positions, velocities and forces after the step, and the spins, agree with the native values to about 1e-9.
   */
  initialIntegrate(): void {
    const s = this.sys.state;
    const dt = s.dt;
    if (this.lattice === 'moving') {
      const h = 0.5 * dt * s.units.ftm2v;
      const { x, v, f, mask } = s;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        const c = h / massOf(s, i);
        v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
      }
      this.sweep(dt / 4, false);
      this.sweep(dt / 4, true);
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        x[3 * i] += dt * v[3 * i]; x[3 * i + 1] += dt * v[3 * i + 1]; x[3 * i + 2] += dt * v[3 * i + 2];
      }
      return;
    }
    this.sweep(dt / 4, false);
    this.sweep(dt / 4, true);
  }

  /** Second half of the spin step, after the neighbor list and the ghosts match the drifted positions. */
  preForce(): void {
    if (this.lattice !== 'moving') return;
    const dt = this.sys.state.dt;
    this.sweep(dt / 4, false);
    this.sweep(dt / 4, true);
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const dt = s.dt;
    if (this.lattice === 'moving') {
      const h = 0.5 * dt * s.units.ftm2v;
      const { v, f, mask } = s;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        const c = h / massOf(s, i);
        v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
      }
    } else {
      this.sweep(dt / 4, false);
      this.sweep(dt / 4, true);
    }
    this.refreshFm();
  }
}

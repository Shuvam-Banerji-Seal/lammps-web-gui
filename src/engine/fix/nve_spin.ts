import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { findSpinPairs } from '../force/pair/spin_exchange';
import type { SpinPairStyle } from '../force/spin_book';
import { FixLangevinSpin } from './langevin_spin';
import { massOf, nativeOrder } from '../atoms';
import { FixPrecessionSpin } from './precession_spin';
import { FixSetForceSpin } from './setforce_spin';

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
 * SPIN_HBAR (hplanck / 2 pi in metal units). After the step, compute property/atom fmx fmy fmz
 * shows the precession vector of the new spins.
 *
 * Frozen lattice: all four sweeps run in initial_integrate (native order; see initialIntegrate), so the
 * energy of the step is the energy of the end-of-step spins. Moving lattice: the first half of the spin
 * sequence in initialIntegrate, the second half in preForce (after the ghosts follow the drifted positions).
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
  private pairs: SpinPairStyle[] = [];
  private w = [0, 0, 0];
  private wlist: number[] = [0, 0, 0];
  private wbuf = new Float64Array(3);
  private fields: FixPrecessionSpin[] = [];
  private damping: FixLangevinSpin[] = [];
  /** The last fix setforce/spin (native locksetforcespin = get_fix_by_style("^setforce/spin").back()). */
  private setforce: FixSetForceSpin | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/SPIN/fix_nve_spin.cpp at b332f28): the keywords are read before the atom style is
    // checked, and lattice also takes the earlier yes/no values ("for now, (yes,no) still works"): no, off, false or
    // frozen is a frozen lattice, yes, on, true or moving a moving one; anything else is "Illegal fix/nve/spin command".
    for (let k = 0; k < args.length;) {
      if (args[k] === 'lattice') {
        if (k + 2 > args.length) throw new StyleError('Illegal fix/nve/spin command');
        const v = args[k + 1];
        if (v === 'no' || v === 'off' || v === 'false' || v === 'frozen') this.lattice = 'frozen';
        else if (v === 'yes' || v === 'on' || v === 'true' || v === 'moving') this.lattice = 'moving';
        else throw new StyleError('Illegal fix/nve/spin command');
        k += 2;
      } else throw new StyleError('Illegal fix/nve/spin command');
    }
    if (!sys.state.sp) throw new StyleError('fix nve/spin requires atom_style spin');
    this.timeIntegrate = true;
  }

  init(): void {
    this.pairs = findSpinPairs(this.sys.ff.pair);
    this.fields = this.sys.fixes.filter((f): f is FixPrecessionSpin => f instanceof FixPrecessionSpin);
    this.damping = this.sys.fixes.filter((f): f is FixLangevinSpin => f instanceof FixLangevinSpin);
    const sf = this.sys.fixes.filter((f): f is FixSetForceSpin => f instanceof FixSetForceSpin);
    this.setforce = sf.length ? sf[sf.length - 1] : null;
    this.sys.state.spinLattice = this.lattice === 'moving';
  }

  /**
   * Precession vector of owned atom i with the current spins: the spin pair style (Pair.omegaOf, when there is
   * one) plus the precession/spin fixes of the system (fix/precession_spin.ts). Native LAMMPS runs nve/spin
   * without a spin pair style too (measured: a zeeman-only spin rotates).
   */
  private omega(i: number, damped: boolean, output = false): void {
    const nb = this.sys.nb;
    const sp = this.sys.state.sp!;
    this.w[0] = 0; this.w[1] = 0; this.w[2] = 0;
    for (const p of this.pairs) {
      // the sweep accumulates the pair terms into one vector in pair order (native compute_single_pair on fmi), which a
      // pair style may reset (Pair.singlePair); the output vector of a pair style may differ (Pair.outputOf)
      if (!output && p.singlePair) {
        p.singlePair(i, nb, sp, nb.typeall, nb.xall, nb.full!, this.w);
        continue;
      }
      const f = output && p.outputOf ? p.outputOf.bind(p) : p.omegaOf.bind(p);
      f(i, nb, sp, nb.typeall, nb.xall, nb.full!, this.wbuf);
      this.w[0] += this.wbuf[0]; this.w[1] += this.wbuf[1]; this.w[2] += this.wbuf[2];
    }
    for (const f of this.fields) {
      f.omegaOf(i, sp, this.wbuf);
      this.w[0] += this.wbuf[0]; this.w[1] += this.wbuf[1]; this.w[2] += this.wbuf[2];
    }
    // fix langevin/spin (transverse damping, tdamp = lambda, eta = 0): ds/dt = w x s + lambda s x (w x s) is the
    // rotation with w' = w - lambda (w x s), since (w' x s) = w x s - lambda (w x s) x s = w x s + lambda s x (w x s).
    // Measured with native LAMMPS (black box, one spin, field along z, 20 steps of 0.001): the damped spin equals this
    // ODE (no 1/(1+lambda^2) factor) to 7e-7 for lambda 0.1 and 1e-5 for lambda 1. The precession vector (fm) is undamped.
    if (damped) {
      const w = this.wlist;
      w[0] = this.w[0]; w[1] = this.w[1]; w[2] = this.w[2];
      for (const d of this.damping) d.torque(i, w, sp);
      this.w[0] = w[0]; this.w[1] = w[1]; this.w[2] = w[2];
    }
    // As native LAMMPS does (src/SPIN/fix_nve_spin.cpp ComputeInteractionsSpin at b332f28): fix setforce/spin is applied
    // last, after the pair, precession and damping terms, in the sweeps as in the precession vector fm
    this.setforce?.single(i, this.w);
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
      this.omega(i, true);
      omelyan(sp, i, this.w, dt);
    }
  }

  /** Precession vectors of all atoms with the current spins (compute property/atom fmx ... shows them). */
  private refreshFm(): void {
    const s = this.sys.state;
    const fm = s.fm!;
    for (let i = 0; i < s.n; i++) {
      this.omega(i, false, true);
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
    // frozen lattice: all four sweeps run here (native initial_integrate: the spin sweeps before and after the
    // position update are both in it). Measured with native LAMMPS (black box, bcc iron with pair spin/neel, 128
    // spins, lattice frozen, dt 1e-4, thermo every step): the pair energy printed after step 1 is the energy of the
    // spins at the end of that step (its change per step is the full-step change); evaluating it after two sweeps
    // gave half of that change.
    this.sweep(dt / 4, false);
    this.sweep(dt / 4, true);
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
    }
    this.refreshFm();
  }
}

import { FixNH } from './nh';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { ramp, removeCompute } from './util';
import type { FixDeform } from './deform';
import { ComputeTempDeform } from '../compute/temp_deform';
import { massOf } from '../atoms';

/*
 * fix nvt/sllod — docs.lammps.org/fix_nvt_sllod.html. Sentences quoted below
 * are verbatim from plans/lammps-docs/fix_nvt_sllod.rst (checked with grep -F).
 *
 *   |This thermostat is used for a simulation box that is changing size
 *   |and/or shape, for example in a non-equilibrium MD (NEMD) simulation.
 *   |The size/shape change is induced by use of the :doc:`fix deform
 *   |<fix_deform>` command, so each point in the simulation box can be
 *   |thought of as having a "streaming" velocity.
 *
 *   |This fix computes a temperature each timestep.  To do this, the fix
 *   |creates its own compute of style "temp/deform", as if this command had
 *   |been issued:
 *
 * psllod keyword: "*psllod* value = *no* or *yes* = use SLLOD or p-SLLOD variant,
 * respectively" (no is the default).
 *
 * Restrictions: "To use fix nvt/sllod, fix deform should NOT remap atom
 * positions, because fix nvt/sllod adjusts the atom positions and velocities"
 * (the doc line is wrapped after "To use"). "LAMMPS will give an error if this
 * setting is not consistent." The engine errors when fix deform does not remap
 * velocities (remap v).
 *
 * Default: "Same as :doc:`fix nvt <fix_nh>`, except *tchain* = 1, psllod = *no*."
 *
 * The thermostat chain, the bias removal (fix_modify temp) and drag come from
 * FixNH (fix nvt, docs.lammps.org/fix_nh.html).
 *
 * Measured with native LAMMPS (black box; cases tests/oracle/w7sllod_*.in):
 * - The velocities present when the run starts are thermal: the box streaming
 *   velocity at each atom is added to them at run setup (setup()).
 * - SLLOD term (psllod no): the lab velocity gets +dt G v_s per step, with
 *   G_ij = d v_s,i / d r_j the box velocity gradient and v_s the streaming
 *   velocity. It is zero for pure shear (xy, xz) and not zero for a diagonal
 *   rate (x erate or x trate).
 * - psllod yes: the term uses w_i = v_s,i - G_ii x_i (the diagonal, dilation,
 *   part of the streaming velocity is dropped): +dt G w per step. Zero for pure
 *   shear, so psllod matches psllod no there.
 * - The term is added to the lab velocity just before the thermostat scales the
 *   thermal velocity, in both thermostat calls of a step (half a timestep each).
 * - Native warns for N = 1 (fix deform should update positions every step) and
 *   for x erate (trate recommended). The engine's fix deform has no N = 0, so
 *   the oracle cases set neigh_modify every 1 delay 0 check no.
 *
 * Known residual (not fixed): with a diagonal rate, a ~1e-10 per-step velocity
 * difference is amplified by the thermostat to ~4e-7 relative in temperature
 * after 200 steps. w7sllod_diag and w7sllod_psllod therefore use rel = 1e-5;
 * the pure-shear cases match at the default tolerance.
 */

const findDeform = (sys: System): FixDeform => {
  const f = sys.fixes.find((x) => x.style === 'deform');
  if (!f) throw new StyleError('fix nvt/sllod requires a fix deform command (none defined)');
  return f as FixDeform;
};

/**
 * Streaming velocity of atom i at the position native LAMMPS would hold.
 * Native keeps atoms that left the box until its next neighbor rebuild; the
 * engine wraps them earlier, and fix deform keeps the undone wrap shift in
 * accShift until its remap-v step. The stored position is the engine position
 * plus that shift.
 */
const streamAtNative = (sys: System, i: number, out: number[]): void => {
  const df = findDeform(sys);
  const s = sys.state;
  const sh = (df as unknown as { accShift?: Float64Array }).accShift;
  const x0 = s.x[3 * i], x1 = s.x[3 * i + 1], x2 = s.x[3 * i + 2];
  const pending = sh !== undefined && sh.length >= 3 * s.n && (sh[3 * i] !== 0 || sh[3 * i + 1] !== 0 || sh[3 * i + 2] !== 0);
  if (pending) {
    s.x[3 * i] += sh![3 * i]; s.x[3 * i + 1] += sh![3 * i + 1]; s.x[3 * i + 2] += sh![3 * i + 2];
  }
  df.vstream(i, out);
  s.x[3 * i] = x0; s.x[3 * i + 1] = x1; s.x[3 * i + 2] = x2;
};

/**
 * compute temp/deform as the fix uses it: the stock formulas, with the
 * streaming velocity at the native-equivalent position.
 */
class SllodTempDeform extends ComputeTempDeform {
  private sllodBiasOn = new Uint8Array(0);
  private sllodSaved = new Float64Array(0);

  constructor(sys: System, id: string, group: string) {
    super(sys, id, group, []);
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const vs = [0, 0, 0];
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      streamAtNative(this.sys, i, vs);
      const m = massOf(s, i);
      const dx = s.v[3 * i] - vs[0], dy = s.v[3 * i + 1] - vs[1], dz = s.v[3 * i + 2] - vs[2];
      t += m * (dx * dx + dy * dy + dz * dz);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const vs = [0, 0, 0];
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      streamAtNative(this.sys, i, vs);
      const m = massOf(s, i);
      const dx = s.v[3 * i] - vs[0], dy = s.v[3 * i + 1] - vs[1], dz = s.v[3 * i + 2] - vs[2];
      t[0] += m * dx * dx; t[1] += m * dy * dy; t[2] += m * dz * dz;
      t[3] += m * dx * dy; t[4] += m * dx * dz; t[5] += m * dy * dz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  /** Bias = streaming velocity, removed while the thermostat acts on the thermal velocity. */
  removeBias(i: number): void {
    const s = this.sys.state;
    if (this.sllodBiasOn.length < s.n) {
      const on = new Uint8Array(s.n); on.set(this.sllodBiasOn); this.sllodBiasOn = on;
      const sv = new Float64Array(3 * s.n); sv.set(this.sllodSaved); this.sllodSaved = sv;
    }
    if (this.sllodBiasOn[i] || !(s.mask[i] & this.groupBit)) return;
    this.sllodBiasOn[i] = 1;
    const vs = [0, 0, 0];
    streamAtNative(this.sys, i, vs);
    this.sllodSaved[3 * i] = vs[0]; this.sllodSaved[3 * i + 1] = vs[1]; this.sllodSaved[3 * i + 2] = vs[2];
    s.v[3 * i] -= vs[0]; s.v[3 * i + 1] -= vs[1]; s.v[3 * i + 2] -= vs[2];
  }

  restoreBias(i: number): void {
    if (!this.sllodBiasOn[i]) return;
    this.sllodBiasOn[i] = 0;
    const s = this.sys.state;
    s.v[3 * i] += this.sllodSaved[3 * i];
    s.v[3 * i + 1] += this.sllodSaved[3 * i + 1];
    s.v[3 * i + 2] += this.sllodSaved[3 * i + 2];
  }
}

/** Private members of FixNH that the SLLOD integrator composes (fix nh.ts). */
interface NHInternals {
  tstat: boolean;
  tStart: number;
  tStop: number;
  tTarget: number;
  nhcTemp(): void;
  kick(): void;
  drift(): void;
}

export class FixNVTSllod extends FixNH {
  readonly psllod: boolean;
  /** Positions for the SLLOD term while a thermostat call runs (null: the current positions). */
  private kickPos: Float64Array | null = null;
  /** Native positions after this step's drift, before the neighbor wrap. */
  private xSnap: Float64Array | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const rest: string[] = [];
    let psllod = false;
    for (let k = 0; k < args.length;) {
      if (args[k] === 'psllod') {
        const v = args[k + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`fix nvt/sllod psllod must be yes or no (got '${v ?? ''}')`);
        psllod = v === 'yes';
        k += 2;
      } else {
        rest.push(args[k]);
        k += 1;
      }
    }
    // Default: "Same as :doc:`fix nvt <fix_nh>`, except *tchain* = 1, psllod = *no*." (fix_nvt_sllod.rst)
    if (!rest.includes('tchain')) rest.push('tchain', '1');
    super(sys, id, group, rest, 'nvt');
    this.psllod = psllod;
    // FixNH's chain calls this.scaleVelocities(factor): the SLLOD term goes in
    // just before the thermal velocity is scaled.
    const base = (FixNH.prototype as unknown as { scaleVelocities: (f: number | number[]) => void }).scaleVelocities;
    (this as unknown as { scaleVelocities: (f: number | number[]) => void }).scaleVelocities = (factor) => {
      this.sllodKick(0.5 * sys.state.dt);
      base.call(this, factor);
    };
    // the fix's own temperature is compute temp/deform, not the temp FixNH made
    removeCompute(sys, `${id}_temp`);
    sys.computes.push(new SllodTempDeform(sys, `${id}_temp`, group));
  }

  /** fix deform may be defined after this fix; the checks run at run init. */
  init(): void {
    this.checkDeform();
    super.init();
  }

  /** fix deform must exist and remap atom velocities (not positions). */
  private checkDeform(): void {
    const mode = (findDeform(this.sys) as unknown as { remapMode: string }).remapMode;
    if (mode !== 'v') {
      throw new StyleError(`fix nvt/sllod needs fix deform with remap v (fix deform remap is ${mode})`);
    }
  }

  /** The box streaming velocity has been added (once per fix instance, see setup). */
  private profileAdded = false;

  /**
   * At the first run of this fix the velocities present are thermal: add the box streaming
   * velocity. Measured with native LAMMPS (black box): later runs leave the velocities alone
   * (after run 50 with xy erate 0.2, two more run 0 keep temp/deform at 0.154967), while an
   * unfix and a new fix nvt/sllod adds the profile again at its first run (temp/deform becomes
   * the previous plain temperature, 0.320004).
   */
  setup(): void {
    if (!this.profileAdded) {
      this.profileAdded = true;
      const df = findDeform(this.sys);
      const s = this.sys.state;
      const vs = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        df.vstream(i, vs);
        s.v[3 * i] += vs[0]; s.v[3 * i + 1] += vs[1]; s.v[3 * i + 2] += vs[2];
      }
    }
    super.setup();
  }

  /** Velocity gradient G[i][j] = d v_s,i / d r_j of the box (affine; probed at atom 0). */
  private velocityGradient(): number[] {
    const df = findDeform(this.sys);
    const s = this.sys.state;
    const g = new Array(9).fill(0);
    if (s.n === 0) return g;
    const x0 = [s.x[0], s.x[1], s.x[2]];
    const base = [0, 0, 0], probe = [0, 0, 0];
    const set = (p: number[]) => { s.x[0] = p[0]; s.x[1] = p[1]; s.x[2] = p[2]; };
    set([0, 0, 0]);
    df.vstream(0, base);
    for (let j = 0; j < 3; j++) {
      const p = [0, 0, 0];
      p[j] = 1;
      set(p);
      df.vstream(0, probe);
      for (let i = 0; i < 3; i++) g[3 * i + j] = probe[i] - base[i];
    }
    set(x0);
    return g;
  }

  /**
   * SLLOD term on the lab velocity: v_i += dt sum_j G_ij w_j, with w = v_s for
   * psllod no, and w_j = v_s,j - G_jj x_j for psllod yes.
   */
  private sllodKick(dt: number): void {
    const s = this.sys.state;
    const df = findDeform(this.sys);
    const g = this.velocityGradient();
    const vs = [0, 0, 0];
    const pos = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const snap = this.kickPos;
      if (snap) {
        const x0 = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
        s.x[3 * i] = snap[3 * i]; s.x[3 * i + 1] = snap[3 * i + 1]; s.x[3 * i + 2] = snap[3 * i + 2];
        df.vstream(i, vs);
        s.x[3 * i] = x0[0]; s.x[3 * i + 1] = x0[1]; s.x[3 * i + 2] = x0[2];
        pos[0] = snap[3 * i]; pos[1] = snap[3 * i + 1]; pos[2] = snap[3 * i + 2];
      } else {
        streamAtNative(this.sys, i, vs);
        pos[0] = s.x[3 * i]; pos[1] = s.x[3 * i + 1]; pos[2] = s.x[3 * i + 2];
      }
      if (this.psllod) for (let a = 0; a < 3; a++) vs[a] -= g[4 * a] * pos[a];
      for (let a = 0; a < 3; a++) {
        s.v[3 * i + a] += dt * (g[3 * a] * vs[0] + g[3 * a + 1] * vs[1] + g[3 * a + 2] * vs[2]);
      }
    }
  }

  /** Velocity Verlet: thermostat chain, half kick, drift; the SLLOD term rides on the chain's scaling. */
  initialIntegrate(): void {
    const nh = this as unknown as NHInternals;
    const s = this.sys.state;
    if (nh.tstat) {
      nh.tTarget = ramp(this.sys, nh.tStart, nh.tStop);
      this.kickPos = null;
      nh.nhcTemp();
    }
    nh.kick();
    nh.drift();
    // native positions after this step's drift, before its neighbor wrap
    const sh = (findDeform(this.sys) as unknown as { accShift?: Float64Array }).accShift;
    this.xSnap = new Float64Array(3 * s.n);
    for (let i = 0; i < 3 * s.n; i++) this.xSnap[i] = s.x[i] + (sh && sh.length >= 3 * s.n ? sh[i] : 0);
  }

  finalIntegrate(): void {
    const nh = this as unknown as NHInternals;
    nh.kick();
    if (nh.tstat) {
      this.kickPos = this.xSnap;
      nh.nhcTemp();
    }
    this.kickPos = null;
  }
}

/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/fix_nvt_sllod.cpp (class
 * FixNVTSllod, fix nvt/sllod), the reversible integrator's nve_x (sllodNveX below). Contributing author (from the
 * source): Pieter in 't Veld (SNL). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The Nose-Hoover chain is the core FixNH (src/engine/fix/nh.ts) and the temperature the core compute temp/deform,
 * reused by import. Registered for the core through src/engine/lazyStyles.ts (group gpl_sllod).
 */
import { FixNH } from '../../engine/fix/nh';
import { StyleError } from '../../engine/force/types';
import { logical } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { ramp, removeCompute } from '../../engine/fix/util';
import { isDeformFix, type FixDeform } from '../../engine/fix/deform';
import { ComputeTempDeform } from '../../engine/compute/temp_deform';
import { massOf } from '../../engine/atoms';
import { exp as libmExp } from '../libm/exp';

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
 * The SLLOD update is native's reversible integrator (src/fix_nvt_sllod.cpp nve_x at b332f28, the default
 * integrator): the peculiar velocity takes the flow half steps, the positions the box-flow half steps about the box
 * centre, and the streaming velocity of the new positions is added back (sllodNveX). The psllod yes term is the
 * diagonal (dilation) part of native's psllod branch. The thermostat scales the peculiar velocity through FixNH's
 * per-atom bias removal (fix_nh.cpp nh_v_temp), with no SLLOD term of its own. The engine's cases
 * (tests/oracle/w7sllod_*.in) agree with native to about 1e-12 for shear, dilation and psllod.
 * - Native warns for N = 1 (fix deform should update positions every step) and for x erate (trate recommended).
 *   The w7sllod cases use N = 1 with neigh_modify every 1 delay 0 check no; the w39nemd cases use N = 0 (see
 *   fix/deform.ts) with the default neighbor settings.
 */

const findDeform = (sys: System): FixDeform => {
  const f = sys.fixes.find((x) => isDeformFix(x));
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
  nhcTemp(initial?: boolean): void;
  kick(): void;
}

export class FixNVTSllod extends FixNH {
  readonly psllod: boolean;

  constructor(sys: System, id: string, group: string, args: string[]) {
    const rest: string[] = [];
    let psllod = false;
    let peculiar = false;
    let kick = true;
    for (let k = 0; k < args.length;) {
      if (args[k] === 'psllod') {
        const v = args[k + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`fix nvt/sllod psllod must be yes or no (got '${v ?? ''}')`);
        psllod = v === 'yes';
        k += 2;
      } else if (args[k] === 'peculiar') {
        const v = args[k + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`fix nvt/sllod peculiar must be yes or no (got '${v ?? ''}')`);
        peculiar = v === 'yes';
        k += 2;
      } else if (args[k] === 'kick') {
        const v = args[k + 1];
        if (v !== 'yes' && v !== 'no') throw new StyleError(`fix nvt/sllod kick must be yes or no (got '${v ?? ''}')`);
        kick = v === 'yes';
        k += 2;
      } else if (args[k] === 'integrator') {
        const v = args[k + 1];
        // fix_nvt_sllod.rst: "*integrator* value = *reversible* or *legacy*"; only reversible is implemented
        if (v === 'legacy') throw new StyleError('fix nvt/sllod integrator legacy is not supported by the browser engine (only integrator reversible, the default, is implemented)');
        if (v !== 'reversible') throw new StyleError(`fix nvt/sllod integrator must be reversible or legacy (got '${v ?? ''}')`);
        k += 2;
      } else {
        rest.push(args[k]);
        k += 1;
      }
    }
    // Default: "Same as :doc:`fix nvt <fix_nh>`, except *tchain* = 1, psllod = *no*." (fix_nvt_sllod.rst)
    if (!rest.includes('tchain')) rest.push('tchain', '1');
    // fix_nvt_sllod.rst: "*peculiar* = *yes*" stores the velocity relative to the flow; not implemented here
    if (peculiar) throw new StyleError('fix nvt/sllod peculiar yes is not supported by the browser engine (only peculiar no, the default, is implemented)');
    super(sys, id, group, rest, 'nvt', 'nvt/sllod');
    this.psllod = psllod;
    this.kickPending = kick;
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

  /**
   * A kick (superimposing the box streaming velocity) is queued: at the first run (kick yes,
   * the default for peculiar no), or by fix_modify kick yes. The kick is applied once per queue.
   */
  private kickPending = true;

  /**
   * fix_modify kick: "*kick* value = *yes* or *no*" (fix_modify.html). Queues a kick for the
   * next run. fix_modify.html: "The *kick* keyword can only be used with fix nvt/sllod".
   */
  modify(key: string, values: string[]): number {
    if (key !== 'kick') return super.modify(key, values);
    // FixNVTSllod::modify_param: "Illegal fix_modify command" without a value, else utils::logical
    if (values.length < 1) throw new StyleError('Illegal fix_modify command');
    if (logical(values[0])) this.kickPending = true;
    return 1;
  }

  /**
   * kick yes (fix_nvt_sllod.rst): "The kick will only be applied ONCE." The run setup adds the
   * box streaming velocity to the velocities present, so they are taken as thermal.
   * Measured with native LAMMPS (black box; tests/oracle/w39nemd_kick*.in): kick no leaves the
   * velocities alone at the first run (temp/deform subtracts the streaming velocity from them),
   * kick yes adds it. A later fix_modify kick yes adds the streaming velocity of the current
   * fix deform on top of the lab velocities without first removing the old one (after the
   * rate changed from +0.05 to -0.05 with unfix/fix deform the lab velocities carry both).
   * The queued kick is applied with run 0 too.
   */
  setup(): void {
    if (this.kickPending) {
      this.kickPending = false;
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

  /**
   * Velocity Verlet of native fix nvt/sllod's reversible integrator (fix_nvt_sllod.cpp at b332f28): the thermostat
   * chain (FixNH::nh_v_temp, the bias removed around each scaling), the half kick (FixNH::nve_v), then nve_x, which
   * carries the SLLOD update. Native nve_x is the only place the streaming flow enters the positions and velocities.
   */
  initialIntegrate(): void {
    const nh = this as unknown as NHInternals;
    if (nh.tstat) {
      nh.tTarget = ramp(this.sys, nh.tStart, nh.tStop);
      nh.nhcTemp(true);
    }
    nh.kick();
    this.sllodNveX();
  }

  /**
   * As native LAMMPS does (src/fix_nvt_sllod.cpp nve_x at b332f28, reversible integrator, peculiar no): the peculiar
   * velocity v = v_lab - v_stream (the streaming velocity at the start of the step, from fix deform's h_rate and
   * h_ratelo, domain.cpp x2lamda) takes the SLLOD half steps, the positions take the box-flow half steps about the
   * box centre (xfac = exp(grad_u dt/2)), then the drift dt v, the second half steps, and the streaming velocity of
   * the new positions (apply_deform_bias_all, with the lo boundary propagated by dt) is added back to v.
   * grad_u = h_rate h_inv (math_extra.h multiply_shape_shape). psllod yes adds the diagonal (dilation) terms of
   * native's psllod branch. The positions are the native (unwrapped) ones: the engine's wrap shift (accShift, see
   * streamAtNative) is added before and removed after.
   */
  private sllodNveX(): void {
    const s = this.sys.state;
    const df = findDeform(this.sys);
    const { hRate, hRateLo } = df.nativeRates();
    const g = this.sys.geom;
    const lo = [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
    const hi = [s.box.hi[0], s.box.hi[1], s.box.hi[2]];
    const { lx, ly, lz, xy, xz, yz } = g;
    // domain.cpp set_global_box: h_inv of h = [[lx xy xz], [0 ly yz], [0 0 lz]]
    const hinv = [1 / lx, 1 / ly, 1 / lz, -yz / (ly * lz), (yz * xy - ly * xz) / (lx * ly * lz), -xy / (lx * ly)];
    // math_extra.h multiply_shape_shape(h_rate, h_inv) = grad_u
    const gu = [
      hRate[0] * hinv[0],
      hRate[1] * hinv[1],
      hRate[2] * hinv[2],
      hRate[1] * hinv[3] + hRate[3] * hinv[2],
      hRate[0] * hinv[4] + hRate[5] * hinv[3] + hRate[4] * hinv[2],
      hRate[0] * hinv[5] + hRate[5] * hinv[1],
    ];
    const dtv = s.dt;
    const dthalf = 0.5 * dtv;
    const xfac = [0, 1, 2].map((k) => libmExp(gu[k] * dthalf));
    const vfac = [0, 1, 2].map((k) => libmExp(-gu[k] * dthalf));
    // fix_nvt_sllod.cpp: xmid is the box centre; the lo boundary is propagated by dt for the streaming velocity
    const xmid = [0, 1, 2].map((k) => 0.5 * (hi[k] + lo[k]));
    const ylo2 = xmid[1] + (lo[1] - xmid[1]) * xfac[1] * xfac[1];
    const zlo2 = xmid[2] + (lo[2] - xmid[2]) * xfac[2] * xfac[2];
    const yloNext = xmid[1] + (lo[1] - xmid[1]) * libmExp(gu[1] * dtv);
    const zloNext = xmid[2] + (lo[2] - xmid[2]) * libmExp(gu[2] * dtv);
    const sh = (df as unknown as { accShift?: Float64Array }).accShift;
    const shifted = sh !== undefined && sh.length >= 3 * s.n;
    const x = [0, 0, 0];
    const v = s.v, mask = s.mask, pos = s.x;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const sx = shifted ? sh![3 * i] : 0, sy = shifted ? sh![3 * i + 1] : 0, sz = shifted ? sh![3 * i + 2] : 0;
      x[0] = pos[3 * i] + sx; x[1] = pos[3 * i + 1] + sy; x[2] = pos[3 * i + 2] + sz;
      // remove the streaming velocity at the start of the step (compute temp/deform's remove_deform_bias)
      const dx = x[0] - lo[0], dy = x[1] - lo[1], dz = x[2] - lo[2];
      const l0 = hinv[0] * dx + hinv[5] * dy + hinv[4] * dz;
      const l1 = hinv[1] * dy + hinv[3] * dz;
      const l2 = hinv[2] * dz;
      v[3 * i] -= hRate[0] * l0 + hRate[5] * l1 + hRate[4] * l2 + hRateLo[0];
      v[3 * i + 1] -= hRate[1] * l1 + hRate[3] * l2 + hRateLo[1];
      v[3 * i + 2] -= hRate[2] * l2 + hRateLo[2];
      let v0 = v[3 * i], v1 = v[3 * i + 1], v2 = v[3 * i + 2];
      // first half SLLOD update
      if (this.psllod) {
        v2 -= dthalf * gu[2] * gu[2] * x[2];
        v1 -= dthalf * gu[3] * v2 + dthalf * gu[1] * gu[1] * x[1];
        v0 -= dthalf * (gu[5] * v1 + gu[4] * v2) + dthalf * gu[0] * gu[0] * x[0];
      } else {
        v1 -= dthalf * gu[3] * v2;
        v0 -= dthalf * (gu[5] * v1 + gu[4] * v2);
      }
      v0 *= vfac[0]; v1 *= vfac[1]; v2 *= vfac[2];
      // position half steps about the box centre, then the drift
      x[1] += dthalf * gu[3] * (x[2] - lo[2]);
      x[0] += dthalf * (gu[5] * (x[1] - lo[1]) + gu[4] * (x[2] - lo[2]));
      x[0] = xmid[0] + (x[0] - xmid[0]) * xfac[0];
      x[1] = xmid[1] + (x[1] - xmid[1]) * xfac[1];
      x[2] = xmid[2] + (x[2] - xmid[2]) * xfac[2];
      x[0] += dtv * v0; x[1] += dtv * v1; x[2] += dtv * v2;
      // second half of the position update
      x[0] = xmid[0] + (x[0] - xmid[0]) * xfac[0];
      x[1] = xmid[1] + (x[1] - xmid[1]) * xfac[1];
      x[2] = xmid[2] + (x[2] - xmid[2]) * xfac[2];
      x[0] += dthalf * (gu[5] * (x[1] - ylo2) + gu[4] * (x[2] - zlo2));
      x[1] += dthalf * gu[3] * (x[2] - zlo2);
      // second half SLLOD velocity step
      v0 *= vfac[0]; v1 *= vfac[1]; v2 *= vfac[2];
      if (this.psllod) {
        v0 -= dthalf * (gu[5] * v1 + gu[4] * v2) + dthalf * gu[0] * gu[0] * x[0];
        v1 -= dthalf * gu[3] * v2 + dthalf * gu[1] * gu[1] * x[1];
        v2 -= dthalf * gu[2] * gu[2] * x[2];
      } else {
        v0 -= dthalf * (gu[5] * v1 + gu[4] * v2);
        v1 -= dthalf * gu[3] * v2;
      }
      // the streaming velocity at the new positions (apply_deform_bias)
      v0 += (x[0] - xmid[0]) * gu[0] + (x[1] - yloNext) * gu[5] + (x[2] - zloNext) * gu[4];
      v1 += (x[1] - xmid[1]) * gu[1] + (x[2] - zloNext) * gu[3];
      v2 += (x[2] - xmid[2]) * gu[2];
      v[3 * i] = v0; v[3 * i + 1] = v1; v[3 * i + 2] = v2;
      pos[3 * i] = x[0] - sx; pos[3 * i + 1] = x[1] - sy; pos[3 * i + 2] = x[2] - sz;
    }
  }
}

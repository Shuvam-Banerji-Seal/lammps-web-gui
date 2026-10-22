/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MACHDYN/fix_smd_setvel.cpp and fix_smd_setvel.h (FixSMDSetVel: fix smd/setvel), src/MACHDYN/
 * fix_smd_integrate_tlsph.cpp (FixSMDIntegrateTlsph: fix smd/integrate_tlsph) and src/MACHDYN/fix_smd_adjust_dt.cpp
 * (FixSMDTlsphDtReset: fix smd/adjust_dt) at commit b332f28. MACHDYN author (from the sources): Georg C. Ganzenmueller,
 * Fraunhofer EMI. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix smd/setvel (docs.lammps.org/fix_smd_setvel.html): the velocity components of the group are set to the constant or
 * equal-style variable values in the post-force step, and the force those components had is kept (the vector output
 * f_ID[k] is the sum of the forces of the group in that component, before it was zeroed).
 * fix smd/integrate_tlsph (docs.lammps.org/fix_smd_integrate_tlsph.html): velocity Verlet with the estimated velocity
 * vest for the pair style; the internal energy is advanced in the final integration step.
 * fix smd/adjust_dt (docs.lammps.org/fix_smd_adjust_dt.html): "This fix inquires the minimum stable time increment
 * across all particles contained in the group for which this fix is defined. An additional safety factor s_fact is
 * applied to the time increment."
 */
import { StyleError, type Pair } from '../../engine/force/types';
import { numeric } from '../../engine/force/util';
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import { SMD_DESPH, SMD_ESPH, SMD_VEST } from '../../engine/smd';
import { PairSMDTlsph } from './pair_smd_tlsph';
import { PairSMDHertz } from './pair_smd_hertz';
import { PairSMDTriSurface } from './pair_smd_tri_surface';
import { PairSMDUlsph } from './pair_smd_ulsph';
import { PairHybrid } from '../../engine/force/pair/hybrid';

/** Lower bound used by the native code for the stable time step (fix_smd_adjust_dt.cpp: BIG). */
const BIG = 1.0e20;

const isVarName = (w: string): boolean => w.startsWith('v_');

type Component = { kind: 'none' } | { kind: 'constant'; value: number } | { kind: 'equal'; name: string };

const parseComponent = (w: string, what: string): Component => {
  if (isVarName(w)) return { kind: 'equal', name: w.slice(2) };
  if (w === 'NULL') return { kind: 'none' };
  const v = Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`Illegal fix smd/setvel ${what} value '${w}'`);
  return { kind: 'constant', value: v };
};

/** fix ID group smd/setvel vx vy vz [region ID] (docs.lammps.org/fix_smd_setvel.html). */
export class FixSMDSetVel extends Fix {
  readonly style = 'smd/setvel';
  private comps: [Component, Component, Component];
  private foriginal: [number, number, number] = [0, 0, 0];
  vectorFlag = true;
  sizeVector = 3;
  extvector = 1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix setvelocity command');
    this.comps = [parseComponent(args[0], 'x'), parseComponent(args[1], 'y'), parseComponent(args[2], 'z')];
    if (args.length > 3) throw new StyleError(`fix smd/setvel keyword '${args[3]}' is not supported by the browser engine yet`);
    for (const c of this.comps) {
      if (c.kind === 'equal' && !sys.vars.has(c.name)) throw new StyleError(`Variable name for fix setvelocity does not exist: ${c.name}`);
    }
  }

  postForce(): void {
    const s = this.sys.state;
    const fo: [number, number, number] = [0, 0, 0];
    const vest = s.custom.get(SMD_VEST)!.data;
    const vals: [number, number, number] = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      const c = this.comps[d];
      if (c.kind === 'constant') vals[d] = c.value;
      else if (c.kind === 'equal') vals[d] = this.sys.equalVariable(c.name);
    }
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      for (let d = 0; d < 3; d++) fo[d] += s.f[3 * i + d];
      for (let d = 0; d < 3; d++) {
        if (this.comps[d].kind === 'none') continue;
        s.v[3 * i + d] = vals[d];
        vest[3 * i + d] = vals[d];
        s.f[3 * i + d] = 0;
      }
    }
    this.foriginal = fo;
  }

  computeVector(i: number): number {
    return this.foriginal[i];
  }
}

/** fix ID group smd/integrate_tlsph (no keyword is supported here except the defaults). */
export class FixSMDIntegrateTlsph extends Fix {
  readonly style = 'smd/integrate_tlsph';
  timeIntegrate = true;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // the native keywords xsph (not available) and limit_velocity (not ported) are refused here
    // fix_smd_integrate_tlsph.cpp: xsph stops the run, limit_velocity takes a number, any other word is unknown
    for (const w of args) {
      if (w === 'xsph') throw new StyleError('XSPH is currently not available');
      if (w === 'limit_velocity') throw new StyleError(`fix smd/integrate_tlsph keyword '${w}' is not supported by the browser engine yet`);
      throw new StyleError(`Unknown fix smd/integrate_tlsph keyword: ${w}`);
    }
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const dtv = s.dt;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const vest = s.custom.get(SMD_VEST)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const dtfm = dtf / s.rmass![i];
      for (let d = 0; d < 3; d++) s.v[3 * i + d] += dtfm * s.f[3 * i + d];
      for (let d = 0; d < 3; d++) vest[3 * i + d] = s.v[3 * i + d] + dtfm * s.f[3 * i + d];
      for (let d = 0; d < 3; d++) s.x[3 * i + d] += dtv * s.v[3 * i + d];
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const dtv = s.dt;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const esph = s.custom.get(SMD_ESPH)!.data;
    const desph = s.custom.get(SMD_DESPH)!.data;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      const dtfm = dtf / s.rmass![i];
      for (let d = 0; d < 3; d++) s.v[3 * i + d] += dtfm * s.f[3 * i + d];
      esph[i] += dtv * desph[i];
    }
  }
}

/** fix ID group smd/adjust_dt s_fact (docs.lammps.org/fix_smd_adjust_dt.html). */
export class FixSMDAdjustDt extends Fix {
  readonly style = 'smd/adjust_dt';
  scalarFlag = true;
  extscalar = 0;
  private safety: number;
  private tElapsed = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Illegal fix smd/adjust_dt command');
    this.safety = numeric(args[0]);
  }

  /** Native setup: the stable time step is computed before the first step (the time step is 1e-16 at step 0). */
  setup(): void { this.endOfStep(); }

  initialIntegrate(): void {
    this.tElapsed += this.sys.state.dt;
  }

  /** The minimum stable time increment over the particle styles (here the SMD tlsph pair), times the safety factor. */
  endOfStep(): void {
    const s = this.sys.state;
    // fix_smd_adjust_dt.cpp: the minimum over the SMD styles' stable time increments (tlsph/ulsph dtCFL, tri_surface and
    // hertz contacts)
    const stable: number[] = [];
    const visit = (p: Pair | null | undefined): void => {
      if (p instanceof PairHybrid) for (const sub of p.subs) visit(sub.style);
      else if (p instanceof PairSMDTlsph || p instanceof PairSMDUlsph) stable.push(p.dtCFL);
      else if (p instanceof PairSMDHertz || p instanceof PairSMDTriSurface) stable.push(p.stableTimeIncrement);
    };
    visit(this.sys.ff.pair);
    if (!stable.length) throw new StyleError('fix smd/adjust_dt failed to access a valid dtCFL');
    let dtmin = BIG;
    for (const v of stable) dtmin = Math.min(dtmin, v);
    dtmin *= this.safety;
    let dt = dtmin;
    if (s.step === 0) dt = 1.0e-16;
    // update->dt is replaced; the simulation time keeps its origin (thermo 'time' is step times the current dt)
    s.dt = dt;
    for (const f of this.sys.fixes) f.resetDt?.();
  }

  computeScalar(): number {
    return this.tElapsed;
  }
}

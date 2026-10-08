import { Fix } from './fix';
import { FixShake } from './shake';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Region } from '../region';
import { massOf } from '../atoms';
import { num } from '../commands/args';

/*
 * fix heat and fix ehex (the HEX and eHEX non-equilibrium heat exchange
 * algorithms). Implemented from docs.lammps.org/fix_heat.html and
 * docs.lammps.org/fix_ehex.html only; the numerical coefficients of the
 * eHEX coordinate correction were measured with native LAMMPS (black box).
 *
 * fix heat — docs.lammps.org/fix_heat.html, Syntax:
 *   fix ID group-ID heat N eflux keyword value ...
 *   "Add non-translational kinetic energy (heat) to a group of atoms in a
 *   manner that conserves their aggregate momentum."
 *   "If the *region* keyword is used, the atom must be in both the group and
 *   the specified geometric region in order to have energy added or
 *   subtracted to it. If not specified, then the atoms in the group are
 *   affected wherever they may move to."
 *   "Heat addition/subtraction is performed every N timesteps."
 *   "If *eflux* is a numeric constant or equal-style variable which evaluates
 *   to a scalar value, then *eflux* determines the change in aggregate energy
 *   of the entire group of atoms per unit time"
 *   "This fix does not change the coordinates of its atoms; it only scales
 *   their velocities."
 *   "This fix computes a global scalar ... This scalar is the most recent
 *   value by which velocities were scaled. The scalar value calculated by
 *   this fix is "intensive"."
 *   "If heat is subtracted from the system too aggressively so that the
 *   group's kinetic energy would go to zero ... then LAMMPS will halt with an
 *   error message."
 *
 * eHEX — docs.lammps.org/fix_ehex.html: "This fix implements the asymmetric
 * version of the enhanced heat exchange algorithm (Wirnsberger). The eHEX
 * algorithm is an extension of the heat exchange algorithm (Ikeshoji) and
 * adds an additional coordinate integration to account for higher-order
 * truncation terms in the operator splitting."
 * "The thermostatting force does not affect the center of mass velocities of
 * the individual reservoirs and the entire simulation box."
 * "This fix will default to fix_heat (HEX algorithm) if the keyword *hex* is
 * specified."
 *
 * The velocity step (both styles), with the group G and the region members
 * R = G ∩ region (all of G without region), taken at the end of a step
 * whose number is a multiple of N:
 *   K     = sum 1/2 m_i (v_i - v_cm)^2 over R, v_cm the mass-weighted COM of R
 *   eps   = N F dt / K                     (K before the scaling)
 *   v_i  <- v_cm + sqrt(1 + eps) (v_i - v_cm)
 * Measured with native LAMMPS (black box), 108-atom LJ fluid, pair_style zero:
 * the scalar equals sqrt(1 + N F dt / K) with K from the velocities before the
 * scaling; total KE rises by exactly N F dt per application; the region
 * membership and the COM are those of the atoms at the application step
 * (region re-evaluated each application); no scaling on step 0 of a run.
 *
 * eHEX coordinate correction (keyword hex omits it). Measured with native
 * LAMMPS (black box): after the velocity step, each atom of R is moved by
 *   dx_i = -(dt / 96) eps^2 (v_i - v_cm)     (velocities before the scaling)
 * with eps as above. Checked for dt = 0.001, 0.002, F = 2..200, N = 1 and 2:
 * the ratio dx / (dt^3 alpha^2 (v - v_cm)), alpha = eps/(2 dt), is -1/24 to
 * the 1e-4 level, and the velocities are identical to the HEX run. The
 * correction accumulates in positions over the run while velocities do not
 * change, so it is a pure position shift each application.
 *
 * Compatible with SHAKE/RATTLE (docs.lammps.org/fix_ehex.html, "Compatibility
 * with SHAKE and RATTLE (rigid molecules)"): "If either of these constraining
 * algorithms is specified in the input script and the keyword *constrain* is
 * set, the bond distances will be corrected a second time at the end of the
 * integration step." Measured with native LAMMPS (black box) on rigid water:
 * the correction is applied by the shake/rattle fix as an additional force at
 * the end of the step; positions and velocities on that step are unchanged.
 *
 * Not implemented (StyleError): com (cluster rescaling; constrain alone scales
 * individual atoms), atom-style variables for eflux.
 */

const USAGE_HEAT = 'usage: fix ID group-ID heat N eflux [region region-ID]';
const USAGE_EHEX = 'usage: fix ID group-ID ehex N F [region region-ID] [constrain] [com] [hex]';

abstract class HeatBase extends Fix {
  protected regionId: string | null = null;
  protected regionObj: Region | null = null;
  scalarFlag = true;
  extscalar = 0;
  /** The most recent velocity scale factor (1 until the first application). */
  protected scale = 1;
  /** Flux per unit time F (number or equal-style variable name). */
  protected flux: { value: number } | { variable: string };

  constructor(sys: System, id: string, group: string, args: string[], usage: string, fluxWord: string | undefined, name: string) {
    super(sys, id, group, args);
    const n = Number(args[0]);
    if (!Number.isInteger(n) || n < 1) throw new StyleError(`fix ${id} (${name}): N must be a positive integer, got '${args[0] ?? ''}' (${usage})`);
    this.nevery = n;
    if (fluxWord === undefined) throw new StyleError(`fix ${id} (${name}): missing eflux (${usage})`);
    if (fluxWord.startsWith('v_')) {
      const name = fluxWord.slice(2);
      const v = sys.vars.get(name);
      if (!v) throw new StyleError(`fix ${id} (${name}): variable ${name} does not exist`);
      if (v.style === 'atom' || v.style === 'atomfile' || v.style === 'vector') {
        throw new StyleError(`fix ${id} (${name}): atom-style variable eflux is not implemented in this engine (use an equal-style variable)`);
      }
      this.flux = { variable: name };
    } else {
      this.flux = { value: num(fluxWord, 'eflux') };
    }
  }

  /** Keywords after the flux; called by each subclass constructor once its fields exist. */
  protected abstract parseKeywords(words: string[], usage: string): void;

  protected parseRegion(words: string[], k: number, usage: string): number {
    const val = words[k + 1];
    if (val === undefined) throw new StyleError(`fix ${this.id} (${this.style}): keyword region needs a region-ID (${usage})`);
    this.sys.region(val); // throws StyleError if it does not exist
    this.regionId = val;
    this.regionObj = this.sys.region(val);
    return 2;
  }

  computeScalar(): number { return this.scale; }

  endOfStep(): void {
    this.applyFlux();
  }

  private fluxValue(): number {
    return 'value' in this.flux ? this.flux.value : this.sys.equalVariable(this.flux.variable);
  }

  /** The group (and region) members that take part in the step. */
  protected members(): number[] {
    const s = this.sys.state;
    const out: number[] = [];
    const reg = this.regionObj;
    // Region membership uses the periodic image inside the box. Measured with native
    // LAMMPS (black box): an atom that has just left the box (z = -0.003 on the
    // step) counts as inside a region that reaches the upper box face.
    const scratchX = new Float64Array(3);
    const scratchImage = new Int32Array(3);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (reg) {
        scratchX[0] = s.x[3 * i]; scratchX[1] = s.x[3 * i + 1]; scratchX[2] = s.x[3 * i + 2];
        this.sys.geom.remap(scratchX, scratchImage, 0);
        if (!reg.match(scratchX[0], scratchX[1], scratchX[2])) continue;
      }
      out.push(i);
    }
    return out;
  }

  /** The velocity step; eHEX also corrects positions (see the header). */
  protected applyFlux(): void {
    const s = this.sys.state;
    const idx = this.members();
    if (!idx.length) return;
    let mtot = 0, px = 0, py = 0, pz = 0;
    for (const i of idx) {
      const m = massOf(s, i);
      mtot += m;
      px += m * s.v[3 * i]; py += m * s.v[3 * i + 1]; pz += m * s.v[3 * i + 2];
    }
    const vc = [px / mtot, py / mtot, pz / mtot];
    let k2 = 0; // twice the kinetic energy about the COM velocity, in mass*velocity^2
    for (const i of idx) {
      const m = massOf(s, i);
      const dx = s.v[3 * i] - vc[0], dy = s.v[3 * i + 1] - vc[1], dz = s.v[3 * i + 2] - vc[2];
      k2 += m * (dx * dx + dy * dy + dz * dz);
    }
    const ke = 0.5 * s.units.mvv2e * k2;
    const energy = this.nevery * this.fluxValue() * s.dt; // energy added this application
    if (!(ke > 0)) throw new StyleError(`fix ${this.id} (${this.style}): kinetic energy of the group is zero`);
    const eps = energy / (ke);
    if (1 + eps < 0 || Number.isNaN(eps)) throw new StyleError(`fix ${this.id} (${this.style}): kinetic energy went negative`);
    const s2 = Math.sqrt(1 + eps);
    // dK/dt from the forces on the members: P = sum_i f_i . (v_i - v_cm)
    let power = 0;
    for (const i of idx) {
      for (let c = 0; c < 3; c++) power += s.f[3 * i + c] * (s.v[3 * i + c] - vc[c]);
    }
    this.correctPositions(idx, vc, eps, ke, power);
    for (const i of idx) {
      for (let c = 0; c < 3; c++) s.v[3 * i + c] = vc[c] + s2 * (s.v[3 * i + c] - vc[c]);
    }
    this.scale = s2;
  }

  /** eHEX position correction; the HEX fix has none. */
  protected correctPositions(_idx: number[], _vc: number[], _eps: number, _ke: number, _power: number): void { /* HEX */ }
}

/** fix heat (HEX) — velocity scaling only. */
export class FixHeat extends HeatBase {
  readonly style = 'heat';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, USAGE_HEAT, args[1], 'heat');
    this.parseKeywords(args.slice(2), USAGE_HEAT);
  }

  protected parseKeywords(words: string[], usage: string): void {
    for (let k = 0; k < words.length;) {
      const w = words[k];
      if (w === 'region') { k += this.parseRegion(words, k, usage); continue; }
      throw new StyleError(`fix ${this.id} (heat): unknown keyword '${w}' (only region is documented) (${usage})`);
    }
  }
}

/** fix ehex (eHEX); the hex keyword selects the HEX algorithm (no correction). */
export class FixEhex extends HeatBase {
  readonly style = 'ehex';
  private hex = false;
  private constrain = false;
  private shakeFix: FixShake | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, USAGE_EHEX, args[1], 'ehex');
    this.parseKeywords(args.slice(2), USAGE_EHEX);
  }

  init(): void {
    if (!this.constrain) return;
    const shake = this.sys.fixes.find((f) => f instanceof FixShake);
    if (!shake) throw new StyleError(`fix ${this.id} (ehex): keyword constrain requires a fix shake or rattle; without one it is not implemented`);
    this.shakeFix = shake;
  }

  protected parseKeywords(words: string[], usage: string): void {
    let com = false;
    for (let k = 0; k < words.length;) {
      const w = words[k];
      if (w === 'region') { k += this.parseRegion(words, k, usage); continue; }
      if (w === 'constrain') { this.constrain = true; k++; continue; }
      if (w === 'com') { com = true; k++; continue; }
      if (w === 'hex') { this.hex = true; k++; continue; }
      throw new StyleError(`fix ${this.id} (ehex): unknown keyword '${w}' (${usage})`);
    }
    if (com && !this.constrain) throw new StyleError(`fix ${this.id} (ehex): You can only use the keyword 'com' together with the keyword 'constrain'`);
    if (com) throw new StyleError(`fix ${this.id} (ehex): keyword com is not implemented in this engine; constrain without com is supported`);
  }

  endOfStep(): void {
    super.endOfStep();
    // docs.lammps.org/fix_ehex.html: constrain re-applies SHAKE/RATTLE after the
    // thermostat rescaling, which otherwise introduces velocity components along
    // the fixed bonds.
    this.shakeFix?.applyConstraint();
  }

  protected correctPositions(idx: number[], vc: number[], eps: number, ke: number, power: number): void {
    if (this.hex) return;
    const s = this.sys.state;
    const dt = s.dt;
    // Measured with native LAMMPS (black box), 108-atom LJ fluid, one step, nevery 1,
    // with and without pair forces, F = 5, 20, 80 (fit residual 1e-4 relative):
    //   dx_i = -(dt/96) eps^2 r_i + (dt^2 eps/12) [ a_i - (P/K) r_i ]
    // r_i = v_i - v_cm (before the scaling), a_i = ftm2v f_i / m_i, P = sum_j f_j . r_j,
    // K = ke (before the scaling). The first term is the pure-thermostat correction;
    // the bracket is the force on the member minus its part that changes K.
    const ftm2v = s.units.ftm2v;
    const ca = -(dt * eps * eps) / 96;
    const cb = (dt * dt * eps) / 12;
    const cp = power / ke;
    for (const i of idx) {
      const m = massOf(s, i);
      for (let c = 0; c < 3; c++) {
        const r = s.v[3 * i + c] - vc[c];
        const a = ftm2v * s.f[3 * i + c] / m;
        s.x[3 * i + c] += ca * r + cb * (a - cp * r);
      }
    }
  }
}

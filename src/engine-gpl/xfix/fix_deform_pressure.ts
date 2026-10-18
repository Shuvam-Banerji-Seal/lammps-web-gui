/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_deform_pressure.cpp and .h at commit b332f28 (class FixDeformPressure, fix deform/pressure;
 * contributing author Joel Clemmer, SNL). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Pressure-controlled box deformation (docs.lammps.org/fix_deform_pressure.html). Each x, y or z dimension given as
 * "pressure" or "pressure/mean" has its box length changed every N steps by a proportional control on the pressure
 * deviation: the engineering strain rate is pgain * (P - Ptarget) (optionally normalized by |Ptarget| and capped by
 * max/rate), the box length rate is that strain rate times the current length, and the box is expanded or compressed
 * about its mid point from its start-of-run bounds by the cumulative shift. The "couple" keyword averages the
 * diagonal pressure components of the coupled dimensions and drives them identically.
 *
 * Scope of this browser port: x, y and z dimensions with pressure or pressure/mean control, couple, max/rate,
 * normalize/pressure, v_ variable targets, and remap x or none on an orthogonal box. The tilt (xy, xz, yz) pressure
 * control, the box keyword, the volume styles, vol/balance/p, remap v, and the non-pressure deform styles (which the
 * native fix mixes in through its FixDeform parent) are refused with a StyleError naming the option. The native
 * per-step bookkeeping of the volume options (prior pressures and rates) is not kept because no volume option is
 * supported.
 *
 * The box update follows native FixDeform::update_domain (fix_deform.cpp at b332f28): atoms of the group are mapped
 * to lamda coordinates of the old box and back to the new one (remap x), and only the controlled dimensions are reset.
 * The start-of-run bounds are taken at init, as the single-run case of native FixDeform::init.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { int, num, yesno } from '../../engine/commands/args';
import { ownCompute, parseNumOrVar, removeCompute, valueOf, type NumOrVar } from '../../engine/fix/util';

/** MathConst::THIRD as native fix_deform_pressure.cpp uses it (the same rounding). */
const THIRD = 1.0 / 3.0;

type PStyle = 'pressure' | 'pmean';

/** One controlled length dimension (native set[i] with style PRESSURE or PMEAN, and set_extra[i]). */
interface PDim {
  style: PStyle;
  target: NumOrVar;
  pgain: number;
  coupled: boolean;
}

type Couple = 'none' | 'xyz' | 'xy' | 'yz' | 'xz';
const COUPLES: readonly Couple[] = ['none', 'xyz', 'xy', 'yz', 'xz'];

/** The diagonal dimensions (0 x, 1 y, 2 z) that a couple keyword ties together (native NOCOUPLE/XYZ/XY/YZ/XZ). */
const COUPLED_DIMS: Record<Couple, number[]> = { none: [], xyz: [0, 1, 2], xy: [0, 1], yz: [1, 2], xz: [0, 2] };

const sameTarget = (a: NumOrVar, b: NumOrVar): boolean => {
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  return typeof a !== 'number' && typeof b !== 'number';
};

export class FixDeformPressure extends Fix {
  readonly style = 'deform/pressure';
  /** The controlled x, y, z dimensions (null: the dimension is not deformed by this fix). */
  private readonly dims: (PDim | null)[] = [null, null, null];
  private pcouple: Couple = 'none';
  private maxRate = 0;
  private normalize = false;
  private remapMode: 'x' | 'none' = 'x';
  /** Box bounds at the start of the run (native set[i].lo_start / hi_start) and the cumulative shift. */
  private readonly loStart: number[] = [0, 0, 0];
  private readonly hiStart: number[] = [0, 0, 0];
  private readonly cumShift: number[] = [0, 0, 0];
  private captured = false;
  private readonly ownTempId: string;
  private readonly ownPressId: string;
  private tempId: string;
  private pressId: string;
  private pressCompute!: Compute;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.ownTempId = `${id}_temp`;
    this.ownPressId = `${id}_press`;
    this.tempId = this.ownTempId;
    this.pressId = this.ownPressId;
    const nevery = int(args[0], 'fix deform/pressure N');
    // fix_deform.cpp constructor: "Fix {} Nevery must be >= 0"
    if (nevery < 0) throw new StyleError('Fix deform/pressure Nevery must be >= 0');
    this.nevery = nevery;

    const need = (k: number, what: string): string => {
      const w = args[k];
      if (w === undefined) throw new StyleError(`fix deform/pressure: missing value for ${what}`);
      return w;
    };

    for (let k = 1; k < args.length;) {
      const w = args[k];
      if (w === 'x' || w === 'y' || w === 'z') {
        const st = args[k + 1];
        if (st !== 'pressure' && st !== 'pressure/mean') {
          throw new StyleError(`fix deform/pressure: ${w} style '${st ?? ''}' is not supported by the browser engine (only pressure and pressure/mean are; the other deform styles need the core fix deform)`);
        }
        // docs: "*pressure* values = target gain"
        const target = parseNumOrVar(need(k + 2, `${w} ${st} target`), 'target pressure');
        const pgain = num(need(k + 3, `${w} ${st} gain`), `fix deform/pressure ${w} ${st} gain`);
        this.dims['xyz'.indexOf(w)] = { style: st === 'pressure' ? 'pressure' : 'pmean', target, pgain, coupled: false };
        k += 4;
      } else if (w === 'xy' || w === 'xz' || w === 'yz') {
        throw new StyleError(`fix deform/pressure: tilt parameter ${w} is not supported by the browser engine (tilt pressure control and triclinic boxes are refused)`);
      } else if (w === 'box') {
        throw new StyleError('fix deform/pressure: the box keyword is not supported by the browser engine (the volume and box pressure styles are refused)');
      } else if (w === 'couple') {
        const v = need(k + 1, 'couple');
        if (!COUPLES.includes(v as Couple)) throw new StyleError(`Illegal fix deform/pressure couple command: ${v}`);
        this.pcouple = v as Couple;
        k += 2;
      } else if (w === 'max/rate') {
        const rate = num(need(k + 1, 'max/rate'), 'fix deform/pressure max/rate');
        if (rate <= 0.0) throw new StyleError('Maximum strain rate must be a positive, non-zero value');
        this.maxRate = rate;
        k += 2;
      } else if (w === 'normalize/pressure') {
        this.normalize = yesno(need(k + 1, 'normalize/pressure'), 'fix deform/pressure normalize/pressure');
        k += 2;
      } else if (w === 'vol/balance/p') {
        if (yesno(need(k + 1, 'vol/balance/p'), 'fix deform/pressure vol/balance/p')) {
          throw new StyleError('fix deform/pressure: vol/balance/p yes is not supported by the browser engine (it needs two constant-volume dimensions)');
        }
        k += 2;
      } else if (w === 'remap') {
        const v = need(k + 1, 'remap');
        if (v === 'v') throw new StyleError('fix deform/pressure: remap v is not supported by the browser engine (remap x and remap none are)');
        if (v !== 'x' && v !== 'none') throw new StyleError(`fix deform: remap must be x, v or none (got '${v}')`);
        this.remapMode = v;
        k += 2;
      } else if (w === 'flip') {
        // flip only acts on triclinic boxes, which this port refuses at init
        yesno(need(k + 1, 'flip'), 'fix deform flip');
        k += 2;
      } else if (w === 'units') {
        const v = need(k + 1, 'units');
        if (v !== 'lattice' && v !== 'box') throw new StyleError(`fix deform: units must be lattice or box (got '${v}')`);
        // no distance parameter is given in this port (pressure controls are in pressure units), so units has no effect
        k += 2;
      } else {
        throw new StyleError(`Illegal fix deform/pressure command: ${w}`);
      }
    }

    if (this.dims.every((d) => d === null)) {
      throw new StyleError('fix deform/pressure: at least one x, y or z pressure or pressure/mean parameter is required');
    }

    if (this.pcouple !== 'none') {
      if (sys.dimension === 2 && (this.pcouple === 'xyz' || this.pcouple === 'xz' || this.pcouple === 'yz')) {
        throw new StyleError('Cannot couple Z dimension in fix deform/pressure in 2D');
      }
      const idx = COUPLED_DIMS[this.pcouple];
      // Check coupled styles and find the reference (the last pressure-controlled coupled dimension)
      let j = -1;
      for (const i of idx) {
        const di = this.dims[i];
        if (!di) continue;
        if (di.style !== 'pressure') throw new StyleError('Cannot couple non-pressure-controlled dimensions');
        j = i;
      }
      if (j === -1) throw new StyleError('Must specify deformation style for at least one coupled dimension');
      const ref = this.dims[j] as PDim;
      for (const i of idx) {
        const di = this.dims[i];
        if (!di) {
          // native: "Copy coupling information if dimension style is undefined"
          this.dims[i] = { style: 'pressure', target: ref.target, pgain: ref.pgain, coupled: true };
          continue;
        }
        if (di === ref) continue;
        if (ref.style !== di.style) throw new StyleError('Cannot couple dimensions with different control options');
        if (di.pgain !== ref.pgain || !sameTarget(di.target, ref.target)) {
          throw new StyleError('Coupled dimensions must have identical gain parameters');
        }
        if (typeof ref.target !== 'number' && typeof di.target !== 'number' && ref.target.variable !== di.target.variable) {
          throw new StyleError('Coupled dimensions must have the same target pressure');
        }
        di.coupled = true;
      }
      ref.coupled = true;
    }

    for (const d of this.dims) {
      if (d && !(d.pgain > 0.0)) throw new StyleError('Illegal fix deform/pressure gain constant, must be positive');
    }

    ownCompute(sys, this.tempId, 'all', 'temp', []);
    this.pressCompute = ownCompute(sys, this.pressId, 'all', 'pressure', [this.tempId]);
  }

  init(): void {
    const sys = this.sys;
    const s = sys.state;
    if (s.box.triclinic) throw new StyleError('fix deform/pressure: triclinic boxes are not supported by the browser engine');
    // native FixDeform::init (single run): the start values are the box at the start of the run
    for (let d = 0; d < 3; d++) {
      this.loStart[d] = s.box.lo[d];
      this.hiStart[d] = s.box.hi[d];
      this.cumShift[d] = 0.0;
    }
    this.captured = true;
    this.pressCompute = sys.compute(this.pressId);
  }

  destroy(): void {
    if (this.tempId === this.ownTempId) removeCompute(this.sys, this.tempId);
    if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
  }

  /** native FixDeformPressure::setup only schedules the virial; nothing to do here. */
  setup(): void {}

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.tempFlag) throw new StyleError(`fix_modify temp: compute ${values[0]} does not compute a temperature`);
      if (this.tempId === this.ownTempId) removeCompute(this.sys, this.tempId);
      this.tempId = values[0];
      // the fix's own pressure compute takes its temperature from the fix temperature
      if (this.pressId === this.ownPressId) this.pressCompute = ownCompute(this.sys, this.pressId, 'all', 'pressure', [this.tempId]);
      return 1;
    }
    if (key === 'press') {
      const c = this.sys.compute(values[0] ?? '');
      if (!c.pressFlag) throw new StyleError(`fix_modify press: compute ${values[0]} does not compute a pressure`);
      if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
      this.pressId = values[0];
      this.pressCompute = c;
      return 1;
    }
    return super.modify(key, values);
  }

  /** Native end_of_step every N steps (Modify::end_of_step gates on ntimestep % nevery). */
  endOfStep(): void {
    if (this.nevery === 0 || this.sys.state.step % this.nevery !== 0) return;
    this.update();
  }

  /** N = 0: native registers post_integrate, so the box is updated every step. */
  postIntegrate(): void {
    if (this.nevery === 0 && this.captured) this.update();
  }

  /** native FixDeformPressure::update_box with the pressure branch; the strain and volume branches do not apply here. */
  private update(): void {
    const sys = this.sys;
    const s = sys.state;
    const g = sys.geom;
    const dt = s.dt;
    sys.refreshComputes();
    const tensor = Float64Array.from(this.pressCompute.vectorValues());
    const scalar = this.pressCompute.scalarValue();

    // native apply_pressure: variable targets are evaluated first
    const target = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      const di = this.dims[i];
      if (di) target[i] = valueOf(sys, di.target);
    }

    // Find current (possibly coupled) pressure for X, Y, Z
    const p = [0, 0, 0];
    if (this.pcouple === 'xyz') {
      const ave = THIRD * (tensor[0] + tensor[1] + tensor[2]);
      p[0] = p[1] = p[2] = ave;
    } else if (this.pcouple === 'xy') {
      const ave = 0.5 * (tensor[0] + tensor[1]);
      p[0] = p[1] = ave;
      p[2] = tensor[2];
    } else if (this.pcouple === 'yz') {
      const ave = 0.5 * (tensor[1] + tensor[2]);
      p[1] = p[2] = ave;
      p[0] = tensor[0];
    } else if (this.pcouple === 'xz') {
      const ave = 0.5 * (tensor[0] + tensor[2]);
      p[0] = p[2] = ave;
      p[1] = tensor[1];
    } else {
      for (let i = 0; i < 3; i++) {
        const di = this.dims[i];
        if (!di) continue;
        p[i] = di.style === 'pressure' ? tensor[i] : scalar;
      }
    }

    const prd = [g.lx, g.ly, g.lz];
    const lo = [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
    const hi = [s.box.hi[0], s.box.hi[1], s.box.hi[2]];
    for (let i = 0; i < 3; i++) {
      const di = this.dims[i];
      if (!di) continue;
      // proportional control gives an engineering strain rate (1/time)
      let strain = di.pgain * (p[i] - target[i]);
      if (this.normalize) {
        if (target[i] === 0) {
          if (this.maxRate === 0) {
            throw new StyleError('Cannot normalize pressure deviation by zero pressure target without defining a max rate');
          } else if (strain !== 0.0) {
            strain = (this.maxRate * strain) / Math.abs(strain);
          }
        } else strain /= Math.abs(target[i]);
      }
      if (this.maxRate !== 0) {
        if (Math.abs(strain) > this.maxRate) strain = (this.maxRate * strain) / Math.abs(strain);
      }
      // domain->h_rate is the rate of change of the box length: strain rate times the current length
      const hRate = strain * prd[i];
      const shift = dt * hRate;
      this.cumShift[i] += shift;
      lo[i] = this.loStart[i] - 0.5 * this.cumShift[i];
      hi[i] = this.hiStart[i] + 0.5 * this.cumShift[i];
    }

    // native update_domain: only the controlled dimensions are reset; remap x maps the group's atoms through lamda
    // coordinates of the old box (rigid bodies follow through deformBodies, as in the core fix deform)
    const remapBit = this.remapMode === 'x' ? this.groupBit : 0;
    const tilt = [s.box.tilt[0], s.box.tilt[1], s.box.tilt[2]];
    if (remapBit) for (const f of sys.fixes) f.deformBodies?.(0);
    sys.setBox(lo, hi, tilt, remapBit, this);
    if (remapBit) for (const f of sys.fixes) f.deformBodies?.(1);
    // thermo output and dumps after end_of_step keep this step's force evaluation (as the core fix deform does)
    if (this.nevery > 0) sys.forcesCurrent();
  }
}

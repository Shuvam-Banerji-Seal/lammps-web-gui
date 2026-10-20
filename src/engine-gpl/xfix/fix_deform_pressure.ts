/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_deform_pressure.cpp and .h at commit b332f28 (class FixDeformPressure, fix deform/pressure;
 * contributing author Joel Clemmer, SNL). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Pressure-controlled box deformation (docs.lammps.org/fix_deform_pressure.html), a child of the core fix deform
 * (src/engine/fix/deform.ts) as native FixDeformPressure is of FixDeform: the parent parses and applies the strain
 * styles (final, delta, scale, vel, erate, trate, wiggle, variable, volume) and the keywords remap, flip and units,
 * changes the box (remap x, v or none; tilt flips), and gives the box rates to compute temp/deform and fix nvt/sllod.
 * This child adds, in native order within each box update (FixDeformPressure::update_box):
 *
 * - x, y, z "pressure" or "pressure/mean" and xy, xz, yz "pressure" (apply_pressure): the engineering strain rate is
 *   pgain (P - Ptarget), normalized by |Ptarget| (normalize/pressure) and capped (max/rate); a length changes by that
 *   rate times its current length about its start mid point, a tilt by that rate times the length it is measured
 *   against (z for yz and xz, y for xy). couple averages the coupled diagonal pressures;
 * - the volume dimensions (apply_volume), with vol/balance/p for two volume dimensions: their strain rates follow the
 *   linearized pressure change so that the two pressures stay equal;
 * - "box volume" (the box scaled to its start volume) and "box pressure" (all three lengths by one volume rate from
 *   the scalar pressure) (apply_box);
 * - the pressures and strain rates of this update, kept for the next vol/balance/p update.
 *
 * The pressure is that of the fix's own computes ID_temp (temp, group all) and ID_press (pressure), replaced with
 * fix_modify temp and press. The box is not changed at the run setup (native setup() only requests the virial).
 * The core fix deform has erate/rescale only for a lone tilt, so it refuses erate/rescale mixed with other
 * parameters here too; erate/rescale for x, y or z is an error as in native.
 */
import { FixDeform, type DeformTargets } from '../../engine/fix/deform';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import { num, yesno } from '../../engine/commands/args';
import { ownCompute, parseNumOrVar, removeCompute, valueOf, type NumOrVar } from '../../engine/fix/util';
import { pow } from '../libm/pow';

/** MathConst::THIRD as native uses it (the same rounding). */
const THIRD = 1.0 / 3.0;

/** native fix_deform.cpp child_styles, child_parameters and child_options of deform/pressure, with their word counts. */
const CHILD = {
  name: 'deform/pressure',
  styles: { pressure: 4, 'pressure/mean': 4, 'erate/rescale': 3, volume: 2 },
  params: ['box'],
  options: { couple: 2, 'max/rate': 2, 'normalize/pressure': 2, 'vol/balance/p': 2 },
};

type PStyle = 'none' | 'pressure' | 'pmean';
type Couple = 'none' | 'xyz' | 'xy' | 'yz' | 'xz';

/** native SetExtra: the pressure control of x, y, z, yz, xz, xy (Voigt order 0-5) and of the box (6). */
interface SetExtra {
  target: NumOrVar;
  ptarget: number;
  pgain: number;
  coupled: boolean;
  cumulativeShift: number;
  cumulativeVshift: number[];
  priorRate: number;
  priorPressure: number;
  saved: boolean;
}

/** The core's tilt index (xy 0, xz 1, yz 2) of a Voigt index 3 (yz), 4 (xz), 5 (xy). */
const tiltOf = (i: number): number => 5 - i;

export class FixDeformPressure extends FixDeform {
  readonly style: string = 'deform/pressure';
  /** The pressure style of x, y, z, yz, xz, xy (native set[i].style PRESSURE or PMEAN for this child). */
  private pstyle: PStyle[] | undefined;
  private readonly extra: SetExtra[];
  private boxStyle: 'none' | 'volume' | 'pressure' = 'none';
  private pcouple: Couple = 'none';
  private maxStrainRate = 0.0;
  private volBalance = false;
  private normalizePressure = false;
  private readonly pressureFlag: boolean;
  /** Box volume at the start of the run (native set_box.vol_start). */
  private boxVolStart = 0;
  /** The pressure tensor and scalar of the current update. */
  private tensor: number[] = [0, 0, 0, 0, 0, 0];
  private scalar = 0;
  private readonly ownTempId: string;
  private readonly ownPressId: string;
  private tempId: string;
  private pressId: string;
  private pressCompute: Compute | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, CHILD);
    const pstyle: PStyle[] = ['none', 'none', 'none', 'none', 'none', 'none'];
    this.pstyle = pstyle;
    this.extra = Array.from({ length: 7 }, () => ({
      target: 0, ptarget: 0, pgain: 0, coupled: false, cumulativeShift: 0, cumulativeVshift: [0, 0, 0],
      priorRate: 0, priorPressure: 0, saved: false,
    }));
    const box = sys.state.box;
    const leftover = this.leftover;
    const setTarget = (index: number, w: string, gain: string): void => {
      this.extra[index].target = parseNumOrVar(w, 'fix deform/pressure target pressure');
      if (typeof this.extra[index].target === 'number') this.extra[index].ptarget = this.extra[index].target as number;
      this.extra[index].pgain = num(gain, 'fix deform/pressure gain');
    };
    // parse only parameter/style arguments specific to this child class
    let i = 0;
    while (i < leftover.length) {
      const iarg = leftover[i];
      const w = args[iarg];
      if (w === 'x' || w === 'y' || w === 'z') {
        const index = 'xyz'.indexOf(w);
        const st = args[iarg + 1];
        if (st === 'pressure' || st === 'pressure/mean') {
          pstyle[index] = st === 'pressure' ? 'pressure' : 'pmean';
          setTarget(index, args[iarg + 2], args[iarg + 3]);
          i += 4;
        } else throw new StyleError(`Illegal fix deform/pressure command argument: ${st}`);
      } else if (w === 'xy' || w === 'xz' || w === 'yz') {
        // fix_deform.cpp: "Fix {} tilt factors require triclinic box" (checked before the style)
        if (!box.triclinic) throw new StyleError('Fix deform/pressure tilt factors require triclinic box');
        const index = w === 'xy' ? 5 : w === 'xz' ? 4 : 3;
        const st = args[iarg + 1];
        if (st === 'pressure') {
          pstyle[index] = 'pressure';
          setTarget(index, args[iarg + 2], args[iarg + 3]);
          i += 4;
        } else throw new StyleError(`Illegal fix deform/pressure command: ${st}`);
      } else if (w === 'box') {
        const st = args[iarg + 1];
        if (st === 'volume') {
          this.boxStyle = 'volume';
          i += 2;
        } else if (st === 'pressure') {
          this.boxStyle = 'pressure';
          setTarget(6, args[iarg + 2], args[iarg + 3]);
          i += 4;
        } else throw new StyleError(`Illegal fix deform/pressure command argument: ${st}`);
      } else break;
    }
    // read options from end of input line
    while (i < leftover.length) {
      const iarg = leftover[i];
      const w = args[iarg], v = args[iarg + 1];
      if (w === 'couple') {
        if (v !== 'xyz' && v !== 'xy' && v !== 'yz' && v !== 'xz' && v !== 'none') throw new StyleError(`Illegal fix deform/pressure couple command: ${v}`);
        this.pcouple = v;
      } else if (w === 'max/rate') {
        this.maxStrainRate = num(v, 'fix deform/pressure max/rate');
        if (this.maxStrainRate <= 0.0) throw new StyleError('Maximum strain rate must be a positive, non-zero value');
      } else if (w === 'normalize/pressure') {
        this.normalizePressure = yesno(v, 'fix deform/pressure normalize/pressure');
      } else if (w === 'vol/balance/p') {
        this.volBalance = yesno(v, 'fix deform/pressure vol/balance/p');
      } else throw new StyleError(`Illegal fix deform/pressure command: ${w}`);
      i += 2;
    }

    // no tensile deformation of the box on shrink-wrapped dims, no tilt deformation on a shrink-wrapped 2nd dim
    const shrink = (d: number): boolean =>
      box.boundary[d][0] === 's' || box.boundary[d][0] === 'm' || box.boundary[d][1] === 's' || box.boundary[d][1] === 'm';
    for (let d = 0; d < 3; d++) {
      if (this.boxStyle !== 'none' && shrink(d)) throw new StyleError('Cannot use fix deform/pressure on a shrink-wrapped boundary');
    }
    if ((this.controlsTilt(2) || this.controlsTilt(1)) && shrink(2)) throw new StyleError('Cannot use fix deform/pressure tilt on a shrink-wrapped 2nd dim');
    if (this.controlsTilt(0) && shrink(1)) throw new StyleError('Cannot use fix deform/pressure tilt on a shrink-wrapped 2nd dim');

    // populate coupled pressure controls
    if (this.pcouple !== 'none') {
      if (sys.dimension === 2 && (this.pcouple === 'xyz' || this.pcouple === 'xz' || this.pcouple === 'yz')) {
        throw new StyleError('Cannot couple Z dimension in fix deform/pressure in 2D');
      }
      const coupled = [this.pcouple !== 'yz', this.pcouple !== 'xz', this.pcouple !== 'xy'];
      // check coupled styles and find the reference (the last pressure-controlled coupled dimension)
      let j = -1;
      for (let d = 0; d < 3; d++) {
        if (!coupled[d]) continue;
        this.extra[d].coupled = true;
        const st = this.lengthStyle(d);
        if (st !== 'pressure' && st !== 'none') throw new StyleError('Cannot couple non-pressure-controlled dimensions');
        if (pstyle[d] === 'pressure') j = d;
      }
      if (j === -1) throw new StyleError('Must specify deformation style for at least one coupled dimension');
      const ref = this.extra[j];
      for (let d = 0; d < 3; d++) {
        if (!coupled[d]) continue;
        const ed = this.extra[d];
        if (pstyle[d] === 'none') {
          // copy coupling information if dimension style is undefined
          pstyle[d] = 'pressure';
          ed.pgain = ref.pgain;
          ed.target = ref.target;
          ed.ptarget = ref.ptarget;
        } else {
          const varI = typeof ed.target !== 'number', varJ = typeof ref.target !== 'number';
          if (ed.pgain !== ref.pgain || varI !== varJ || ed.ptarget !== ref.ptarget) throw new StyleError('Coupled dimensions must have identical gain parameters');
          if (varJ && (ed.target as { variable: string }).variable !== (ref.target as { variable: string }).variable) {
            throw new StyleError('Coupled dimensions must have the same target pressure');
          }
        }
      }
    }

    // for VOLUME, set up the links to the other dims (the pressure-controlled dims are deformed ones)
    this.volumeLinks = this.linkVolumes((d) => {
      const st = this.lengthStyle(d);
      return st === 'none' ? 'none' : st === 'volume' ? 'volume' : 'other';
    }, 'deform/pressure');

    // if vol/balance/p used, must have 2 free dimensions
    if (this.volBalance) {
      for (const vl of this.volumeLinks) {
        if (vl.kind !== 'two-from-one') throw new StyleError('Two dimensions must maintain constant volume to use the vol/balance/p option');
      }
    }

    let pressureFlag = false;
    for (let k = 0; k < 6; k++) {
      if (pstyle[k] !== 'none') {
        pressureFlag = true;
        if (this.extra[k].pgain <= 0.0) throw new StyleError('Illegal fix deform/pressure gain constant, must be positive');
      }
      if (this.extra[k].coupled) pressureFlag = true;
    }
    if (this.boxStyle === 'pressure') pressureFlag = true;
    if (this.volBalance) pressureFlag = true;
    this.pressureFlag = pressureFlag;

    // check conflict between constant volume/pressure
    if (this.volumeLinks.length > 0 && pstyle.some((st) => st === 'pmean')) {
      throw new StyleError('Cannot use fix deform/pressure to assign constant volume and pressure');
    }
    // check conflicts between x, y, z styles and box
    if (this.boxStyle !== 'none') {
      for (let d = 0; d < 3; d++) {
        const st = this.lengthStyle(d);
        if (st === 'final' || st === 'delta' || st === 'scale' || st === 'pmean' || st === 'variable' || st === 'volume') {
          throw new StyleError('Cannot use fix deform/pressure box parameter with x, y, or z styles other than vel, erate, trate, pressure, and wiggle');
        }
      }
    }
    // check pressure used for max rate and normalize error flag
    if (!pressureFlag && this.maxStrainRate !== 0) throw new StyleError('Can only assign a maximum strain rate using pressure-controlled dimensions');
    if (!pressureFlag && this.normalizePressure) throw new StyleError('Can only normalize error using pressure-controlled dimensions');

    this.ownTempId = `${id}_temp`;
    this.ownPressId = `${id}_press`;
    this.tempId = this.ownTempId;
    this.pressId = this.ownPressId;
    // the computes of group all: pressure is always global, and so is its kinetic part
    if (pressureFlag) {
      ownCompute(sys, this.tempId, 'all', 'temp', []);
      this.pressCompute = ownCompute(sys, this.pressId, 'all', 'pressure', [this.tempId]);
    }
  }

  /** The style of length d: the parent's strain style, this child's pressure style, or none. */
  private lengthStyle(d: number): string {
    const sp = this.specs.find((x) => !x.tilt && x.dim === d);
    if (sp) return sp.style;
    return this.pstyle![d];
  }

  /** native dimflag[0..2]: the parent's styles, this child's pressure styles, and every length with the box keyword. */
  controlsLength(d: number): boolean {
    if (super.controlsLength(d)) return true;
    // the parent asks while it parses, before this child's styles exist
    return this.pstyle !== undefined && (this.pstyle[d] !== 'none' || this.boxStyle !== 'none');
  }

  /** native dimflag[3..5]: the parent's tilt styles and this child's tilt pressure styles. */
  controlsTilt(t: number): boolean {
    if (super.controlsTilt(t)) return true;
    return this.pstyle !== undefined && this.pstyle[5 - t] !== 'none';
  }

  init(): void {
    super.init();
    const g = this.sys.geom;
    this.boxVolStart = g.lx * g.ly * g.lz;
    // reset cumulative counters to match resetting "start" variables in parent
    for (const e of this.extra) {
      e.cumulativeShift = 0.0;
      e.cumulativeVshift[0] = 0.0; e.cumulativeVshift[1] = 0.0; e.cumulativeVshift[2] = 0.0;
    }
    // check optional variables for PRESSURE or PMEAN style
    for (const e of this.extra) {
      if (typeof e.target === 'number') continue;
      const name = e.target.variable;
      const v = this.sys.vars.get(name);
      if (!v) throw new StyleError(`Variable name ${name} for fix deform/pressure does not exist`);
      if (v.style !== 'equal' && v.style !== 'internal' && v.style !== 'timer') throw new StyleError(`Variable ${name} for fix deform/pressure is invalid style`);
    }
    if (this.pressureFlag) {
      const t = this.sys.computes.find((c) => c.id === this.tempId);
      if (!t) throw new StyleError(`Temperature compute ID ${this.tempId} for fix deform/pressure does not exist`);
      if (!t.tempFlag) throw new StyleError(`Compute ID ${this.tempId} for fix deform/pressure does not compute a temperature`);
      const p = this.sys.computes.find((c) => c.id === this.pressId);
      if (!p) throw new StyleError(`Pressure compute ID ${this.pressId} for fix deform/pressure does not exist`);
      if (!p.pressFlag) throw new StyleError(`Compute ID ${this.pressId} for fix deform/pressure does not compute pressure`);
      this.pressCompute = p;
    }
    // a flipping yz pressure tilt cannot be followed with a defined xy unless xz is erate/rescale
    const xzRescale = this.specs.some((sp) => sp.tilt && sp.dim === 1 && sp.rescale);
    if (this.controlsTilt(2) && this.controlsTilt(0) && !xzRescale && this.flip && this.pstyle![3] === 'pressure') {
      throw new StyleError('Fix deform/pressure cannot use yz pressure with box flips and a defined xy unless xz is erate/rescale');
    }
  }

  destroy(): void {
    if (!this.pressureFlag) return;
    if (this.tempId === this.ownTempId) removeCompute(this.sys, this.tempId);
    if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
  }

  modify(key: string, values: string[]): number {
    if (key === 'temp') {
      if (values.length < 1) throw new StyleError('Illegal fix_modify command');
      const c = this.sys.computes.find((x) => x.id === values[0]);
      if (!c) throw new StyleError(`Could not find fix_modify temperature compute ID: ${values[0]}`);
      if (!c.tempFlag) throw new StyleError(`Fix_modify temperature compute ${values[0]} does not compute temperature`);
      if (this.tempId === this.ownTempId) removeCompute(this.sys, this.tempId);
      this.tempId = values[0];
      // reset the temperature of the pressure compute to the new one
      if (this.pressId === this.ownPressId) this.pressCompute = ownCompute(this.sys, this.pressId, 'all', 'pressure', [this.tempId]);
      return 1;
    }
    if (key === 'press') {
      if (values.length < 1) throw new StyleError('Illegal fix_modify command');
      const c = this.sys.computes.find((x) => x.id === values[0]);
      if (!c) throw new StyleError(`Could not find fix_modify pressure compute ID: ${values[0]}`);
      if (!c.pressFlag) throw new StyleError(`Fix_modify pressure compute ${values[0]} does not compute pressure`);
      if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
      this.pressId = values[0];
      this.pressCompute = c;
      return 1;
    }
    return super.modify(key, values);
  }

  /** A box update (not the rates at init, nor the parent's update at the run setup). */
  private updating(tg: DeformTargets): boolean {
    return tg.apply && !tg.setup;
  }

  /** update_box: the pressures of this update, the saved values of the first one, and apply_pressure. */
  protected controlTargets(tg: DeformTargets): void {
    if (!this.updating(tg) || !this.pressureFlag) return;
    this.sys.refreshComputes();
    const pc = this.pressCompute!;
    const v = pc.vectorValues();
    this.tensor = [v[0], v[1], v[2], v[3], v[4], v[5]];
    this.scalar = pc.scalarValue();
    for (let d = 0; d < 3; d++) {
      const e = this.extra[d];
      if (!e.saved) {
        e.saved = true;
        e.priorRate = 0.0;
        e.priorPressure = this.tensor[d];
      }
    }
    this.applyPressure(tg);
  }

  /** The proportional control of one target: the strain rate, normalized and capped as native. */
  private strainRate(e: SetExtra, pcurrent: number, zeroMessage: string): number {
    let rate = e.pgain * (pcurrent - e.ptarget);
    if (this.normalizePressure) {
      if (e.ptarget === 0) {
        if (this.maxStrainRate === 0) throw new StyleError(zeroMessage);
        else if (rate !== 0.0) rate = (this.maxStrainRate * rate) / Math.abs(rate);
      } else rate /= Math.abs(e.ptarget);
    }
    if (this.maxStrainRate !== 0) {
      if (Math.abs(rate) > this.maxStrainRate) rate = (this.maxStrainRate * rate) / Math.abs(rate);
    }
    return rate;
  }

  /** The current box lengths (native domain->prd). */
  private prd(): number[] {
    const b = this.sys.state.box;
    return [b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]];
  }

  /** apply_pressure: the pressure-controlled lengths and tilts. */
  private applyPressure(tg: DeformTargets): void {
    const sys = this.sys;
    const dt = sys.state.dt;
    const pstyle = this.pstyle!;
    const prd = this.prd();
    // if variable pressure, calculate current target
    for (let i = 0; i < 6; i++) {
      const e = this.extra[i];
      if (pstyle[i] !== 'none' && typeof e.target !== 'number') e.ptarget = valueOf(sys, e.target);
    }
    // current (possibly coupled/hydrostatic) pressure for x, y, z
    const tensor = this.tensor, scalar = this.scalar;
    const pCurrent = [0.0, 0.0, 0.0];
    if (this.pcouple === 'xyz') {
      const ave = THIRD * (tensor[0] + tensor[1] + tensor[2]);
      pCurrent[0] = pCurrent[1] = pCurrent[2] = ave;
    } else if (this.pcouple === 'xy') {
      const ave = 0.5 * (tensor[0] + tensor[1]);
      pCurrent[0] = pCurrent[1] = ave;
      pCurrent[2] = tensor[2];
    } else if (this.pcouple === 'yz') {
      const ave = 0.5 * (tensor[1] + tensor[2]);
      pCurrent[1] = pCurrent[2] = ave;
      pCurrent[0] = tensor[0];
    } else if (this.pcouple === 'xz') {
      const ave = 0.5 * (tensor[0] + tensor[2]);
      pCurrent[0] = pCurrent[2] = ave;
      pCurrent[1] = tensor[1];
    } else {
      for (let i = 0; i < 3; i++) {
        if (pstyle[i] === 'pressure') pCurrent[i] = tensor[i];
        else if (pstyle[i] === 'pmean') pCurrent[i] = scalar;
      }
    }
    for (let i = 0; i < 3; i++) {
      if (pstyle[i] === 'none') continue;
      const e = this.extra[i];
      // proportional control gives an engineering strain rate (1/time)
      const rate = this.strainRate(e, pCurrent[i], 'Cannot normalize pressure deviation by zero pressure target without defining a max rate');
      // the length rate is the strain rate times the current length
      tg.lenD[i] = rate * prd[i];
      tg.loD[i] = -0.5 * tg.lenD[i];
      const shift = dt * tg.lenD[i];
      e.cumulativeShift += shift;
      tg.lo[i] = this.lo0[i] - 0.5 * e.cumulativeShift;
      tg.hi[i] = this.hi0[i] + 0.5 * e.cumulativeShift;
      tg.len[i] = tg.hi[i] - tg.lo[i];
    }
    for (let i = 3; i < 6; i++) {
      if (pstyle[i] !== 'pressure') continue;
      const e = this.extra[i];
      // the tilt is measured against z (yz, xz) or y (xy); the stress is yz (tensor 5), xz (4) or xy (3)
      const L = i === 5 ? prd[1] : prd[2];
      const pcurrent = i === 3 ? tensor[5] : i === 4 ? tensor[4] : tensor[3];
      const rate = this.strainRate(e, pcurrent, 'Cannot normalize error for zero pressure without defining a max rate');
      // the tilt rate is the shear strain rate times that length
      const t = tiltOf(i);
      tg.tiltD[t] = rate * L;
      e.cumulativeShift += dt * tg.tiltD[t];
      tg.tilt[t] = this.tilt0[t] + e.cumulativeShift;
    }
  }

  /**
   * apply_volume of this child: as the parent's, except that vol/balance/p (two volume dimensions) takes the strain
   * rates that keep the two pressures equal to linear order from the last update's pressures and rates.
   */
  protected applyVolume(tg: DeformTargets): void {
    if (!this.updating(tg) || !this.volBalance) {
      super.applyVolume(tg);
      return;
    }
    const dt = this.sys.state.dt;
    const prd = this.prd();
    let e1 = 0.0, e2 = 0.0;
    let linkedPressure = false;
    for (const vl of this.volumeLinks) {
      const i = vl.dim;
      const fixed = vl.fixed, dynamic1 = vl.dynamic1;
      const v0 = this.vol0;
      let shift: number;
      // vol/balance/p takes two-from-one links only (checked in the constructor)
      const e1i = this.extra[i].priorRate;
      const e2i = this.extra[fixed].priorRate;
      const L1i = prd[i];
      const L2i = prd[fixed];
      const L3i = prd[dynamic1];
      const L3 = tg.hi[dynamic1] - tg.lo[dynamic1];
      const Vi = L1i * L2i * L3i;
      const V = L3 * L1i * L2i;
      const e3 = (L3 / L3i - 1.0) / dt;
      const p1 = this.tensor[i];
      const p2 = this.tensor[fixed];
      const p1i = this.extra[i].priorPressure;
      const p2i = this.extra[fixed].priorPressure;
      if (e3 === 0) {
        e1 = 0.0;
        e2 = 0.0;
        shift = 0.5 * L1i;
      } else if (e1i === 0 || e2i === 0 || (p2 === p2i && p1 === p1i)) {
        // no prior strain or no change in pressure (initial step): scale shift by relative box lengths
        shift = 0.5 * Math.sqrt(v0 * L1i / L3 / L2i);
      } else if (!linkedPressure) {
        // the first strain rate from the stress expanded to linear order, p1(t+dt) = p2(t+dt); the second keeps the volume
        const denominator = p2 - p2i + e2i * ((p1 - p1i) / e1i);
        if (denominator !== 0.0 && e1i !== 0.0) {
          e1 = (((p2 - p2i) * (Vi - V) / (V * dt)) - e2i * (p1 - p2)) / denominator;
        } else {
          e1 = e2i;
        }
        e2 = (Vi - V * (1 + e1 * dt)) / (V * (1 + e1 * dt) * dt);
        // if the strain rate exceeds the limit in either dimension, cap it at the maximum compatible rate
        if (this.maxStrainRate !== 0) {
          if (Math.abs(e1) > this.maxStrainRate || Math.abs(e2) > this.maxStrainRate) {
            if (Math.abs(e1) > Math.abs(e2)) [e1, e2] = this.adjustLinkedRates(e1, e2, e3, Vi, V, dt);
            else [e2, e1] = this.adjustLinkedRates(e2, e1, e3, Vi, V, dt);
          }
        }
        shift = 0.5 * L1i * (1.0 + e1 * dt);
        linkedPressure = true;
      } else {
        // already calculated value of e2
        shift = 0.5 * L1i * (1.0 + e2 * dt);
      }
      // (2 shift - prd) / dt is d(box length)/dt
      tg.lenD[i] = (2.0 * shift - prd[i]) / dt;
      tg.loD[i] = -0.5 * tg.lenD[i];
      tg.lo[i] = 0.5 * (this.lo0[i] + this.hi0[i]) - shift;
      tg.hi[i] = 0.5 * (this.lo0[i] + this.hi0[i]) + shift;
      tg.len[i] = tg.hi[i] - tg.lo[i];
    }
  }

  /** adjust_linked_rates: rescale the volume-preserving strain rates to enforce max/rate; returns [larger, smaller]. */
  private adjustLinkedRates(eLarger: number, _eSmaller: number, e3: number, Vi: number, V: number, dt: number): [number, number] {
    const m = this.maxStrainRate;
    const eLimPositive = (Vi - V * (1 + m * dt)) / (V * (1 + m * dt) * dt);
    const eLimNegative = (Vi - V * (1 - m * dt)) / (V * (1 - m * dt) * dt);
    if (eLarger * e3 >= 0) {
      // same sign as primary strain rate, cap third dimension
      if (eLarger > 0.0) return [eLimNegative, -m];
      return [eLimPositive, m];
    }
    // opposite sign, set to maxrate
    if (eLarger > 0.0) return [m, eLimPositive];
    return [-m, eLimNegative];
  }

  /** apply_box, then the pressures and strain rates saved for the next update. */
  protected boxTargets(tg: DeformTargets): void {
    if (!this.updating(tg)) return;
    const sys = this.sys;
    const dt = sys.state.dt;
    const prd = this.prd();
    if (this.boxStyle === 'volume') {
      const v0 = this.boxVolStart;
      let v = 1.0;
      for (let i = 0; i < 3; i++) v *= tg.hi[i] - tg.lo[i];
      const scale = pow(v0 / v, THIRD);
      for (let i = 0; i < 3; i++) {
        const shift = 0.5 * (tg.hi[i] - tg.lo[i]) * scale;
        tg.lo[i] = 0.5 * (this.lo0[i] + this.hi0[i]) - shift;
        tg.hi[i] = 0.5 * (this.lo0[i] + this.hi0[i]) + shift;
        // d(box length)/dt, not a strain rate
        tg.lenD[i] = (tg.hi[i] - tg.lo[i] - prd[i]) / dt;
        tg.loD[i] = -0.5 * tg.lenD[i];
        tg.len[i] = tg.hi[i] - tg.lo[i];
      }
    } else if (this.boxStyle === 'pressure') {
      const e = this.extra[6];
      // if variable pressure, calculate current target
      if (typeof e.target !== 'number') e.ptarget = valueOf(sys, e.target);
      const vRate = this.strainRate(e, this.scalar, 'Cannot normalize pressure deviation by zero pressure target without defining a max rate');
      for (let i = 0; i < 3; i++) {
        const shift = (tg.hi[i] - tg.lo[i]) * dt * vRate;
        e.cumulativeVshift[i] += shift;
        if (this.lengthStyle(i) === 'none') {
          // overwrite default targets of current length
          tg.lo[i] = this.lo0[i];
          tg.hi[i] = this.hi0[i];
        }
        tg.lo[i] -= 0.5 * e.cumulativeVshift[i];
        tg.hi[i] += 0.5 * e.cumulativeVshift[i];
        tg.lenD[i] = (tg.hi[i] - tg.lo[i] - prd[i]) / dt;
        tg.loD[i] = -0.5 * tg.lenD[i];
        tg.len[i] = tg.hi[i] - tg.lo[i];
      }
    }
    // save pressure/strain rate
    if (this.pressureFlag) {
      for (let i = 0; i < 3; i++) {
        this.extra[i].priorPressure = this.tensor[i];
        this.extra[i].priorRate = ((tg.hi[i] - tg.lo[i]) / prd[i] - 1.0) / dt;
      }
    }
  }
}

import { Fix, fixModifyCompute } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Compute } from '../compute/compute';
import { ComputePressure } from '../compute/pressure';
import { ownCompute, parseNumOrVar, ramp, removeCompute, valueOf, type NumOrVar } from './util';

/*
 * fix ID group press/berendsen keyword values — docs.lammps.org/fix_press_berendsen.html: "Reset the pressure of
 * the system by using a Berendsen barostat (Berendsen), which rescales the system volume and (optionally) the atoms
 * coordinates within the simulation box every timestep." Keywords: "iso Pstart Pstop Pdamp" is "the same as
 * specifying these 4 keywords: x Pstart Pstop Pdamp, y ..., z ..., couple xyz"; "aniso Pstart Pstop Pdamp" is the
 * same with "couple none"; "couple = none or xyz or xy or yz or xz"; "modulus value = bulk modulus of system
 * (pressure units)"; "dilate value = all or partial"; the defaults are "dilate = all, modulus = 10.0".
 * "Couple xyz can be used for a 2d simulation; the z dimension is simply ignored."
 * "Any dimension being adjusted by this fix must be periodic." "The fix_modify temp and press options are supported."
 *
 * The rule of the scaling (Berendsen et al., J Chem Phys 81, 3684 (1984)): each step the box length in a
 * barostatted dimension is multiplied by mu = [1 - (dt/Pdamp) (P_target - P) / B]^(1/3), with B the bulk modulus.
 * Measured with native LAMMPS (black box): the exponent is 1/3 for every dimension (iso, aniso and x-only alike);
 * the pressure is the coupled average; the scaling uses the pressure at the end of the step and happens after the
 * step, so the thermo output of that step shows the scaled box; the box is scaled about its centre and dilated
 * atoms keep their fractional position.
 */

/** couple keyword -> the diagonal dimensions (0 x, 1 y, 2 z) that are averaged and scaled together. */
const COUPLE: Record<string, number[]> = { none: [], xyz: [0, 1, 2], xy: [0, 1], yz: [1, 2], xz: [0, 2] };

export class FixPressBerendsen extends Fix {
  readonly style = 'press/berendsen';
  /** per dimension: target start/stop, damping; null when the dimension is not barostatted */
  private readonly pStart: (NumOrVar | null)[] = [null, null, null];
  private readonly pStop: number[] = [0, 0, 0];
  private readonly pDamp: number[] = [0, 0, 0];
  private coupled: number[] = [];
  private modulus = 10.0;
  private dilateBit: number;
  private readonly ownTempId: string;
  private readonly ownPressId: string;
  private tempId: string;
  private pressId: string;
  private pressCompute!: Compute;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.dilateBit = sys.groupBit('all');
    this.ownTempId = `${id}_temp`;
    this.ownPressId = `${id}_press`;
    this.tempId = this.ownTempId;
    this.pressId = this.ownPressId;
    let couple = 'none';
    const setDim = (d: number, k: number): void => {
      const start = args[k + 1], stop = args[k + 2], damp = args[k + 3];
      if (start === undefined || stop === undefined || damp === undefined) throw new StyleError(`fix ${id} press/berendsen ${args[k]} needs Pstart Pstop Pdamp`);
      this.pStart[d] = parseNumOrVar(start, 'Pstart');
      this.pStop[d] = Number(stop);
      this.pDamp[d] = Number(damp);
      if (!Number.isFinite(this.pStop[d]) || !(this.pDamp[d] > 0)) throw new StyleError(`fix ${id} press/berendsen: Pstop must be a number and Pdamp > 0`);
    };
    for (let k = 0; k < args.length;) {
      const key = args[k];
      switch (key) {
        case 'iso': case 'aniso':
          for (const d of [0, 1, 2]) setDim(d, k);
          couple = key === 'iso' ? 'xyz' : 'none';
          k += 4;
          break;
        case 'x': case 'y': case 'z':
          setDim(key === 'x' ? 0 : key === 'y' ? 1 : 2, k);
          k += 4;
          break;
        case 'couple':
          if (!(args[k + 1] in COUPLE)) throw new StyleError(`fix ${id} press/berendsen couple must be none, xyz, xy, yz or xz`);
          couple = args[k + 1];
          k += 2;
          break;
        case 'modulus':
          this.modulus = Number(args[k + 1]);
          if (!(this.modulus > 0)) throw new StyleError(`fix ${id} press/berendsen modulus must be > 0`);
          k += 2;
          break;
        case 'dilate':
          if (args[k + 1] === 'all') this.dilateBit = sys.groupBit('all');
          else if (args[k + 1] === 'partial') this.dilateBit = sys.groupBit(group);
          else throw new StyleError(`fix ${id} press/berendsen dilate must be all or partial`);
          k += 2;
          break;
        default:
          throw new StyleError(`fix ${id} press/berendsen: unknown keyword '${key}'`);
      }
    }
    if (this.pStart.every((p) => p === null)) throw new StyleError(`fix ${id} press/berendsen: no barostatted dimension (iso, aniso, x, y or z)`);
    // a 2d simulation has no z dimension: it is ignored (also by couple xyz)
    if (sys.dimension === 2) this.pStart[2] = null;
    this.coupled = COUPLE[couple].filter((d) => sys.dimension === 3 || d !== 2);
    for (const d of this.coupled) {
      if (this.pStart[d] === null) throw new StyleError(`fix ${id} press/berendsen: couple ${couple} needs a pressure for every coupled dimension`);
      if (this.pStart[d] !== this.pStart[this.coupled[0]] || this.pStop[d] !== this.pStop[this.coupled[0]] || this.pDamp[d] !== this.pDamp[this.coupled[0]]) {
        throw new StyleError(`fix ${id} press/berendsen: Pstart, Pstop and Pdamp must be identical for coupled dimensions`);
      }
    }
    ownCompute(sys, this.tempId, group, 'temp', []);
    this.pressCompute = ownCompute(sys, this.pressId, 'all', 'pressure', [this.tempId]);
  }

  init(): void {
    const sys = this.sys;
    const s = sys.state;
    // As native LAMMPS does (FixPressBerendsen::init, src/fix_press_berendsen.cpp at b332f28)
    if (s.box.triclinic) throw new StyleError('Cannot use fix press/berendsen with triclinic box');
    for (let d = 0; d < 3; d++) {
      if (this.pStart[d] !== null && !s.box.periodic[d]) throw new StyleError(`fix ${this.id} press/berendsen: a barostatted dimension must be periodic`);
    }
    this.pressCompute = sys.compute(this.pressId);
  }

  destroy(): void {
    if (this.tempId === this.ownTempId) removeCompute(this.sys, this.tempId);
    if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
  }

  modify(key: string, values: string[]): number {
    // the messages of native's FixPressBerendsen::modify_param (src/fix_press_berendsen.cpp at b332f28; its
    // temperature message has no placeholder for the ID)
    if (key === 'temp') {
      fixModifyCompute(this.sys, values[0], 'temp', 'Could not find fix_modify temperature compute ID: ', 'Fix_modify temperature compute {} does not compute temperature');
      if (this.tempId === this.ownTempId) removeCompute(this.sys, this.tempId);
      this.tempId = values[0];
      // the fix's own pressure compute takes its temperature from the fix temperature
      if (this.pressId === this.ownPressId) this.pressCompute = ownCompute(this.sys, this.pressId, 'all', 'pressure', [this.tempId]);
      return 1;
    }
    if (key === 'press') {
      const c = fixModifyCompute(this.sys, values[0], 'press', 'Could not find fix_modify pressure compute ID: {}', 'Fix_modify pressure compute {} does not compute pressure');
      if (this.pressId === this.ownPressId) removeCompute(this.sys, this.pressId);
      this.pressId = values[0];
      this.pressCompute = c;
      return 1;
    }
    return super.modify(key, values);
  }

  endOfStep(): void {
    const sys = this.sys;
    const s = sys.state;
    const b = s.box;
    const g = sys.geom;
    sys.refreshComputes();
    // fix_press_berendsen.cpp couple(): an ISO barostat (couple xyz, or xy in 2d) reads compute_scalar(), which MSM's
    // kspace_modify pressure/scalar yes allows; the diagonal mean of the tensor is the same scalar
    const iso = this.coupled.length === this.sys.dimension && this.pressCompute instanceof ComputePressure;
    const pv = iso ? (this.pressCompute as ComputePressure).isoVectorValues() : this.pressCompute.vectorValues();
    const pDiag = [pv[0], pv[1], pv[2]];
    let pAvg = 0;
    for (const d of this.coupled) pAvg += pDiag[d];
    if (this.coupled.length) pAvg /= this.coupled.length;
    // the scale of each barostatted dimension (coupled ones share the averaged pressure)
    const mu = [1, 1, 1];
    for (let d = 0; d < 3; d++) {
      const start = this.pStart[d];
      if (start === null) continue;
      const coupledHere = this.coupled.includes(d);
      const p = coupledHere ? pAvg : pDiag[d];
      const target = ramp(sys, typeof start === 'number' ? start : valueOf(sys, start), this.pStop[d]);
      mu[d] = Math.pow(1 - (s.dt / this.pDamp[d]) * ((target - p) / this.modulus), 1 / 3);
    }
    if (!mu.some((m, d) => this.pStart[d] !== null && m !== 1)) return;

    // dilated atoms keep their fractional coordinates; the box is scaled about its centre
    const lam = new Float64Array(3 * s.n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.dilateBit)) continue;
      g.toLamda(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2], tmp);
      lam[3 * i] = tmp[0]; lam[3 * i + 1] = tmp[1]; lam[3 * i + 2] = tmp[2];
    }
    for (let d = 0; d < 3; d++) {
      if (this.pStart[d] === null || mu[d] === 1) continue;
      const c = 0.5 * (b.lo[d] + b.hi[d]);
      const half = 0.5 * (b.hi[d] - b.lo[d]) * mu[d];
      b.lo[d] = c - half;
      b.hi[d] = c + half;
    }
    b.minLo = [...b.lo]; b.minHi = [...b.hi];
    g.update();
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.dilateBit)) continue;
      g.fromLamda(lam[3 * i], lam[3 * i + 1], lam[3 * i + 2], tmp);
      s.x[3 * i] = tmp[0]; s.x[3 * i + 1] = tmp[1]; s.x[3 * i + 2] = tmp[2];
    }
    for (const f of sys.fixes) if (f !== this) f.boxChanged?.();
    sys.refreshComputes();
  }
}

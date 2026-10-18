/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_nonaffine_displacement.cpp and src/EXTRA-FIX/fix_nonaffine_displacement.h at commit b332f28
 * (class FixNonaffineDisplacement: the constructor, post_constructor, init, setup, post_force, integrate_velocity,
 * save_reference_state; contributing authors Joel Clemmer (SNL) and Ishan Srivastava (LBNL)). Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Non-affine displacement of the atoms relative to a reference state (docs.lammps.org/fix_nonaffine_displacement.html),
 * the "integrated" style: the displacement accumulates dt * v of the group atoms from the reference step on, and the
 * per-atom output (3 columns) is that displacement, refreshed every nevery steps. The reference state is "fixed" at a
 * timestep, "update" every N steps, or "offset" (saved at the step that precedes a multiple of nevery).
 *
 * Refused: the d2min style (it needs the neighbor pairs with their periodic images, the reverse communication of the
 * per-atom sums, and the cutoff styles type, radius and custom), so the core shows the reason instead of a wrong answer.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';
import type { CustomProp } from '../../engine/types';

function inumeric(w: string | undefined): number {
  if (w === undefined || !/^[+-]?\d+$/.test(w.trim())) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return Number(w);
}

function numeric(w: string | undefined): number {
  const x = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(x)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return x;
}

type RefStyle = 'fixed' | 'update' | 'offset';

export class FixNonaffineDisplacement extends Fix {
  readonly style = 'nonaffine/displacement';
  peratomFlag = true;
  sizePeratomCols = 3;
  private readonly refStyle: RefStyle;
  private readonly referenceTimestep: number;
  private readonly updateTimestep: number;
  private readonly offsetTimestep: number;
  private referenceSaved = false;
  private readonly key: string;
  private dtv = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // native narg counts "fix ID group nonaffine/displacement": narg < 4 is fewer than one argument
    if (args.length < 1) throw new StyleError('Illegal fix nonaffine/displacement command: missing arguments');
    this.nevery = inumeric(args[0]);
    if (this.nevery <= 0) throw new StyleError(`Illegal nevery value ${this.nevery} in fix nonaffine/displacement`);
    let iarg = 1;
    if (args[iarg] === 'integrated') {
      iarg += 1;
    } else if (args[iarg] === 'd2min') {
      throw new StyleError('fix nonaffine/displacement d2min is not supported by this engine yet (the integrated style is)');
    } else {
      throw new StyleError(`Illegal nonaffine displacement style ${args[iarg] ?? ''} in fix nonaffine/displacement`);
    }
    if (iarg + 2 > args.length) throw new StyleError('Illegal fix nonaffine/displacement command: missing arguments');
    let refStyle: RefStyle;
    let refTs = -1, updTs = -1, offTs = -1;
    if (args[iarg] === 'fixed') {
      refStyle = 'fixed';
      refTs = inumeric(args[iarg + 1]);
      if (refTs < 0) throw new StyleError(`Illegal reference timestep ${args[iarg + 1]} in fix nonaffine/displacement`);
    } else if (args[iarg] === 'update') {
      refStyle = 'update';
      updTs = inumeric(args[iarg + 1]);
      if (updTs <= 0) throw new StyleError(`Illegal update timestep ${args[iarg + 1]} in fix nonaffine/displacement`);
    } else if (args[iarg] === 'offset') {
      refStyle = 'offset';
      offTs = inumeric(args[iarg + 1]);
      if (offTs <= 0 || offTs > this.nevery) throw new StyleError(`Illegal offset timestep ${args[iarg + 1]} in fix nonaffine/displacement`);
    } else {
      throw new StyleError(`Illegal reference style ${args[iarg]} in fix nonaffine/displacement`);
    }
    this.refStyle = refStyle;
    this.referenceTimestep = refTs;
    this.updateTimestep = updTs;
    this.offsetTimestep = offTs;
    iarg += 2;
    while (iarg < args.length) {
      if (args[iarg] === 'z/min') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal fix nonaffine/displacement command: missing arguments');
        if (inumeric(args[iarg + 1]) < 0) throw new StyleError('Minimum coordination must be positive');
        iarg += 2;
      } else if (args[iarg] === 'intensive/d2min') {
        throw new StyleError('Can only specify extensive/intensive d2min with style d2min');
      } else {
        throw new StyleError(`Illegal keyword ${args[iarg]} in fix nonaffine/displacement`);
      }
    }
    this.key = `#nad:${id}`;
    this.ensureStore();
  }

  /** The stored reference-relative displacement, 3 values per atom (the native STORE/ATOM fix id_fix). */
  private ensureStore(): Float64Array {
    const s = this.sys.state;
    let c: CustomProp | undefined = s.custom.get(this.key);
    if (!c) {
      c = { int: false, cols: 3, data: new Float64Array(3 * s.n) };
      s.custom.set(this.key, c);
    }
    return c.data;
  }

  init(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    if (!this.referenceSaved && this.refStyle === 'fixed' && s.step > this.referenceTimestep) {
      throw new StyleError('Initial timestep exceeds that of the reference state in fix nonaffine/displacement');
    }
  }

  setup(): void { this.postForceStep(true); }

  postForce(): void { this.postForceStep(false); }

  /** post_force (setupflag: the setup call of the run, where the displacement is zero). */
  private postForceStep(setupFlag: boolean): void {
    const s = this.sys.state;
    const n = s.n;
    const store = this.ensureStore();
    if (this.arrayAtom.length !== 3 * n) this.arrayAtom = new Float64Array(3 * n);
    if (this.referenceSaved && !setupFlag) {
      this.integrateVelocity();
      if (s.step % this.nevery === 0) {
        for (let i = 0; i < n; i++) for (let m = 0; m < 3; m++) this.arrayAtom[3 * i + m] = store[3 * i + m];
      }
    } else {
      this.arrayAtom.fill(0);
    }
    if (this.refStyle === 'fixed' && s.step === this.referenceTimestep) this.saveReferenceState();
    if (this.refStyle === 'update' && s.step % this.updateTimestep === 0) this.saveReferenceState();
    if (this.refStyle === 'offset' && (s.step + this.offsetTimestep) % this.nevery === 0) this.saveReferenceState();
  }

  /** integrate_velocity: the displacement of each group atom grows by dt v. */
  private integrateVelocity(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    const store = this.ensureStore();
    const { v, mask } = s;
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      for (let m = 0; m < 3; m++) store[3 * i + m] += this.dtv * v[3 * i + m];
    }
  }

  /** save_reference_state (integrated style): the displacement restarts at zero for the group atoms. */
  private saveReferenceState(): void {
    const s = this.sys.state;
    const store = this.ensureStore();
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) for (let m = 0; m < 3; m++) store[3 * i + m] = 0.0;
    }
    this.referenceSaved = true;
  }
}

/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/UEF/fix_nh_uef.cpp and src/UEF/fix_nh_uef.h, src/UEF/fix_nvt_uef.cpp and src/UEF/fix_npt_uef.cpp at
 * commit b332f28 (FixNHUef: the SLLOD-type velocity and position updates of the uniform extensional flow, the
 * rotation into and out of the upper-triangular frame around each integration step, the box update, the lattice
 * reduction of pre_exchange; FixNVTUef and FixNPTUef only check their thermostat and barostat keywords).
 * Copyright (2003) Sandia Corporation (GNU General Public License); contributing author David Nicholson (MIT).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The thermostat chain, the barostat and the rest of fix nh are the engine's FixNH (src/engine/fix/nh.ts), which
 * LAMMPS also runs as the base of this fix (FixNH::initial_integrate and final_integrate). The hooks that native
 * FixNHUef overrides (nve_v, nve_x, remap) are shadowed on the instance here, as fix nvt/sphere does.
 *
 * Documentation: docs.lammps.org/fix_nh_uef.html. "fix ID group-ID nvt/uef keyword values" and "fix ID group-ID
 * npt/uef keyword values": the keywords erate, strain and ext are the UEF ones; the others are the fix nvt/npt
 * keywords (docs.lammps.org/fix_nh.html). Sentences quoted: "The erate keyword ... specifies the strain rates
 * in the x and y directions" (erate values are the rates in x and y; the z rate is -erate_x - erate_y).
 */
import { FixNH, type NHRestart } from '../../engine/fix/nh';
import { StyleError } from '../../engine/force/types';
import { numeric } from '../../engine/force/util';
import type { System } from '../../engine/system';
import type { Compute } from '../../engine/compute/compute';
import type { Fix } from '../../engine/fix/fix';
import { ownCompute } from '../../engine/fix/util';
import { massOf } from '../../engine/atoms';
import { ComputeTempUef } from './compute_temp_uef';
import { ComputePressureUef } from './compute_pressure_uef';
import { UEFBox, zeroRot, type M3 } from './uef_utils';
import { exp as libmExp } from '../libm/exp';

/**
 * Keyword value counts of fix nvt/npt (docs.lammps.org/fix_nh.html) that the UEF fix passes on to FixNH. Keywords
 * that FixNH takes but that the UEF integrator does not use (fixedpoint, dilate, scalexy/yz/xz, nreset > 0) are
 * refused with a StyleError rather than ignored.
 */
const NH_VALUES: Record<string, number> = {
  temp: 3, iso: 3, aniso: 3, tri: 3, x: 3, y: 3, z: 3, xy: 3, yz: 3, xz: 3,
  couple: 1, tchain: 1, pchain: 1, tloop: 1, ploop: 1, mtk: 1, drag: 1, ptemp: 1, flip: 1, update: 1,
  nreset: 1, dilate: 1, scalexy: 1, scaleyz: 1, scalexz: 1, fixedpoint: 3, disc: 0,
};
const PRESSURE_KEYS = ['iso', 'aniso', 'tri', 'x', 'y', 'z', 'xy', 'yz', 'xz'];
const TILT_KEYS = ['tri', 'xy', 'yz', 'xz'];
const REFUSED_KEYS = ['fixedpoint', 'dilate', 'scalexy', 'scaleyz', 'scalexz'];

export type UefVariant = 'nvt' | 'npt';

interface UefArgs {
  /** the arguments FixNH parses (erate, strain and ext removed; tchain 1 added as the default of this fix) */
  rest: string[];
  erate: [number, number] | null;
  strain: [number, number];
  ext: [boolean, boolean, boolean];
  hasTemp: boolean;
  hasPress: boolean;
  tilt: boolean;
}

/** Splits the UEF keywords from the fix nh keywords (fix_nh_uef.cpp constructor). */
const parseUefArgs = (args: string[], name: string): UefArgs => {
  const rest: string[] = [];
  let erate: [number, number] | null = null;
  const strain: [number, number] = [0, 0];
  const ext: [boolean, boolean, boolean] = [true, true, true];
  let hasTemp = false;
  let hasPress = false;
  let tilt = false;
  let tchainGiven = false;
  // fix_nh_uef.cpp: its own keywords say "Illegal fix nvt/npt/uef command"; FixNH names the others
  const illegal = (): StyleError => new StyleError('Illegal fix nvt/npt/uef command');
  let k = 0;
  while (k < args.length) {
    const key = args[k];
    if (key === 'erate') {
      if (k + 3 > args.length) throw illegal();
      erate = [numeric(args[k + 1]), numeric(args[k + 2])];
      k += 3;
    } else if (key === 'strain') {
      if (k + 3 > args.length) throw illegal();
      strain[0] = numeric(args[k + 1]);
      strain[1] = numeric(args[k + 2]);
      k += 3;
    } else if (key === 'ext') {
      if (k + 2 > args.length) throw illegal();
      switch (args[k + 1]) {
        case 'x': ext[1] = ext[2] = false; break;
        case 'y': ext[0] = ext[2] = false; break;
        case 'z': ext[0] = ext[1] = false; break;
        case 'xy': ext[2] = false; break;
        case 'xz': ext[1] = false; break;
        case 'yz': ext[0] = false; break;
        case 'xyz': break;
        default: throw illegal();
      }
      k += 2;
    } else if (key in NH_VALUES) {
      if (REFUSED_KEYS.includes(key)) {
        throw new StyleError(`fix ${name} keyword '${key}' is not supported by the browser engine (the UEF integrator does not use it)`);
      }
      if (key === 'nreset' && Number(args[k + 1]) > 0) {
        throw new StyleError(`fix ${name} keyword 'nreset' with a value above 0 is not supported by the browser engine`);
      }
      if (key === 'temp') hasTemp = true;
      if (PRESSURE_KEYS.includes(key)) hasPress = true;
      if (TILT_KEYS.includes(key)) tilt = true;
      if (key === 'tchain') tchainGiven = true;
      const n = NH_VALUES[key];
      for (let j = 0; j <= n; j++) rest.push(args[k + j] ?? '');
      k += n + 1;
    } else {
      throw new StyleError(`Unknown fix ${name} keyword: ${key}`);
    }
  }
  // "if (mtchain_default_flag) mtchain = 1;": the thermostat chain defaults to 1 for this fix
  if (!tchainGiven) rest.push('tchain', '1');
  return { rest, erate, strain, ext, hasTemp, hasPress, tilt };
};

/** Float comparison of fix_nh_uef.cpp nearly_equal (kept with its quirks). */
const nearlyEqual = (a: number, b: number, epsilon: number): boolean => {
  const absa = Math.abs(a);
  const absb = Math.abs(b);
  const diff = Math.abs(a - b);
  if (a === b) return true;
  if (absa + absb < epsilon) return diff < epsilon * epsilon;
  return diff / (absa + absb) < epsilon;
};

/** The FixNH members this fix reads or shadows (private in fix/nh.ts). */
interface NHInternals {
  tstat: boolean;
  pstat: boolean;
  couple: string;
  pFlag: boolean[];
  pStart: Float64Array;
  pStop: Float64Array;
  omegaDot: Float64Array;
  pCurrent: Float64Array;
  tempId: string;
  tempCompute: Compute;
  pressId: string | null;
  pressCompute: unknown;
  kick: () => void;
  drift: () => void;
  remap: () => void;
  computePressCurrent: (stage: 'setup' | 'init' | 'final') => void;
}

export class FixNHUef extends FixNH {
  readonly style: string;
  private readonly erate: [number, number];
  private readonly strain: [number, number];
  private readonly extFlags: [boolean, boolean, boolean];
  private readonly uefbox = new UEFBox();
  /** the rotation matrix Q between the lab frame and the upper-triangular frame (member rot of the native fix) */
  private readonly rot: M3 = zeroRot();
  private readonly tempUef: ComputeTempUef;
  private readonly pressUef: ComputePressureUef;

  constructor(sys: System, id: string, group: string, args: string[], variant: UefVariant) {
    const name = `${variant}/uef`;
    const parsed = parseUefArgs(args, name);
    // fix_nh_uef.cpp constructor: erate, then the triclinic box (its messages name nvt/npt/uef)
    if (!parsed.erate) throw new StyleError('Keyword erate must be set for fix nvt/npt/uef command');
    if (sys.dimension !== 3) throw new StyleError(`fix ${name} requires a 3d simulation`);
    if (!sys.state.box.triclinic) throw new StyleError('Simulation box must be triclinic for fix/nvt/npt/uef');
    if (!parsed.hasTemp) throw new StyleError(`Temperature control must be used with fix ${name}`);
    if (variant === 'nvt' && parsed.hasPress) throw new StyleError(`Pressure control can't be used with fix ${name}`);
    if (variant === 'npt' && !parsed.hasPress) throw new StyleError(`Pressure control must be used with fix ${name}`);
    if (parsed.tilt) throw new StyleError('Only normal stresses can be controlled with fix/nvt/npt/uef');
    // FixNH's messages name the style as invoked (nvt/uef, npt/uef)
    super(sys, id, group, parsed.rest, variant, name);
    this.style = name;
    this.erate = parsed.erate;
    this.strain = parsed.strain;
    this.extFlags = parsed.ext;

    const nh = this as unknown as NHInternals;
    if (nh.pstat) {
      // pstyle ANISO: "The ext keyword may only be used with iso pressure control", and the controlled stresses must
      // share their start, stop and strain rate (fix_nh_uef.cpp constructor, tol 1e-6)
      const isoStyle = nh.couple === 'xyz';
      if (!isoStyle && !(this.extFlags[0] && this.extFlags[1] && this.extFlags[2])) {
        throw new StyleError('The ext keyword may only be used with iso pressure control');
      }
      if (!isoStyle) {
        const tmp = [this.erate[0], this.erate[1], -this.erate[0] - this.erate[1]];
        const tol = 1e-6;
        for (let k = 0; k < 3; k++) {
          for (let j = 0; j < 3; j++) {
            if (!nh.pFlag[k] || !nh.pFlag[j]) continue;
            if (!nearlyEqual(nh.pStart[k], nh.pStart[j], tol) || !nearlyEqual(nh.pStop[k], nh.pStop[j], tol)) {
              throw new StyleError('All controlled stresses must have the same value in fix/nvt/npt/uef');
            }
            // (native compares the two rates twice; once is the same test)
            if (!nearlyEqual(tmp[k], tmp[j], tol)) {
              throw new StyleError('Dimensions with controlled stresses must have same strain rate in fix/nvt/npt/uef');
            }
          }
        }
      }
    }

    // "compute ID_temp all temp/uef" and "compute ID_press all pressure/uef ID_temp" (both on all atoms, as native)
    this.tempUef = ownCompute(sys, nh.tempId, 'all', 'temp/uef', []) as ComputeTempUef;
    this.tempUef.setSource(this);
    nh.tempCompute = this.tempUef;
    const pressId = `${id}_press`;
    this.pressUef = ownCompute(sys, pressId, 'all', 'pressure/uef', [nh.tempId]) as ComputePressureUef;
    this.pressUef.setSource(this);
    this.pressUef.extFlags = this.extFlags;
    nh.pressId = pressId;
    nh.pressCompute = this.pressUef;

    this.uefbox.setStrain(this.strain[0], this.strain[1]);
    this.uefbox.getRot(this.rot);
    this.installHooks(nh);
  }

  /** The box in the upper-triangular frame for the current volume xprd * yprd * zprd (fix nvt/npt/uef get_box). */
  getBox(out: M3): void {
    this.uefbox.getBox(out, this.sys.geom.volume(3));
  }

  /** The current rotation matrix Q (fix nvt/npt/uef get_rot). */
  getRot(out: M3): void {
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) out[k][j] = this.rot[k][j];
  }

  /**
   * The FixNH hooks that native FixNHUef overrides (nve_v, nve_x, remap) and the barostat's pressure. The FixNH
   * methods call them on `this`, so they are shadowed on the instance (fix/nh_sphere.ts does the same).
   */
  private installHooks(nh: NHInternals): void {
    // nve_v: SLLOD velocity update, v -> exp(-e dt/2) v, v + f/m dt, exp(-e dt/2) v
    nh.kick = () => {
      const s = this.sys.state;
      const dtf = 0.5 * s.dt * s.units.ftm2v;
      const ex = this.erate[0] * dtf / 2;
      const ey = this.erate[1] * dtf / 2;
      const ez = -ex - ey;
      const e0 = libmExp(-ex), e1 = libmExp(-ey), e2 = libmExp(-ez);
      const { v, f, mask } = s;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        const dtfm = dtf / massOf(s, i);
        v[3 * i] *= e0; v[3 * i + 1] *= e1; v[3 * i + 2] *= e2;
        v[3 * i] += dtfm * f[3 * i];
        v[3 * i + 1] += dtfm * f[3 * i + 1];
        v[3 * i + 2] += dtfm * f[3 * i + 2];
        v[3 * i] *= e0; v[3 * i + 1] *= e1; v[3 * i + 2] *= e2;
      }
    };

    // nve_x: position update with the strain rate and the barostat rate, then the box (strain) step
    nh.drift = () => {
      const s = this.sys.state;
      const dtv = s.dt;
      const om = nh.omegaDot;
      const ex = this.erate[0] * dtv;
      this.strain[0] += ex;
      const e0 = libmExp((ex + om[0] * dtv) / 2);
      const ey = this.erate[1] * dtv;
      this.strain[1] += ey;
      const e1 = libmExp((ey + om[1] * dtv) / 2);
      const ez = -ex - ey;
      const e2 = libmExp((ez + om[2] * dtv) / 2);
      const { x, v, mask } = s;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        x[3 * i] *= e0; x[3 * i + 1] *= e1; x[3 * i + 2] *= e2;
        x[3 * i] += dtv * v[3 * i];
        x[3 * i + 1] += dtv * v[3 * i + 1];
        x[3 * i + 2] += dtv * v[3 * i + 2];
        x[3 * i] *= e0; x[3 * i + 1] *= e1; x[3 * i + 2] *= e2;
      }
      this.uefbox.stepDeform(ex, ey);
      this.setBoxFromUef(this.sys.geom.volume(3));
    };

    // remap: the box only (native remap does not move the atoms), volume change by the barostat rate
    nh.remap = () => {
      const s = this.sys.state;
      const om = nh.omegaDot;
      const dto = 0.5 * s.dt;
      let vol = this.sys.geom.volume(3);
      const domega = dto * (om[0] + om[1] + om[2]) / 3;
      const ex = dto * om[0] - domega;
      const ey = dto * om[1] - domega;
      this.uefbox.stepDeform(ex, ey);
      this.strain[0] += ex;
      this.strain[1] += ey;
      vol = vol * libmExp(3 * domega);
      this.setBoxFromUef(vol);
    };

    // the barostat's pressure, as the native fix calls it (fix_nh.cpp initial/final integrate and setup). ISO couples
    // the pressure scalar (the mean over the ext directions, compute_pressure_uef.cpp compute_scalar) and does not
    // compute the tensor; the anisotropic case computes the temperature tensor first, then the pressure tensor.
    const base = nh.computePressCurrent;
    nh.computePressCurrent = (stage) => {
      this.sys.refreshComputes();
      if (nh.couple === 'xyz') {
        const p = this.pressUef.scalarValue();
        nh.pCurrent[0] = p; nh.pCurrent[1] = p; nh.pCurrent[2] = p;
        return;
      }
      this.tempUef.invokeVector();
      base.call(this, stage);
    };
  }

  /** The box from the UEF basis for volume vol: the lengths on the diagonal, the tilts from the off-diagonals. */
  private setBoxFromUef(vol: number): void {
    const s = this.sys.state;
    const b = s.box;
    const box = zeroRot();
    this.uefbox.getBox(box, vol);
    b.hi[0] = b.lo[0] + box[0][0];
    b.hi[1] = b.lo[1] + box[1][1];
    b.hi[2] = b.lo[2] + box[2][2];
    b.tilt[0] = box[0][1];
    b.tilt[1] = box[0][2];
    b.tilt[2] = box[1][2];
    b.minLo = [...b.lo];
    b.minHi = [...b.hi];
    this.sys.geom.update();
    this.uefbox.getRot(this.rot);
    for (const f of this.sys.fixes as Fix[]) if (f !== this) f.boxChanged?.();
  }

  /** Rotates atoms of the group from the upper-triangular frame into the lab frame (inv_rotate_x/v/f). */
  private invRotate(r: M3, what: 'x' | 'v' | 'f'): void {
    const s = this.sys.state;
    const arr = what === 'x' ? s.x : what === 'v' ? s.v : s.f;
    const { mask } = s;
    const lo = s.box.lo;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      let a = arr[3 * i], b = arr[3 * i + 1], c = arr[3 * i + 2];
      if (what === 'x') { a -= lo[0]; b -= lo[1]; c -= lo[2]; }
      const n0 = r[0][0] * a + r[1][0] * b + r[2][0] * c;
      const n1 = r[0][1] * a + r[1][1] * b + r[2][1] * c;
      const n2 = r[0][2] * a + r[1][2] * b + r[2][2] * c;
      arr[3 * i] = n0; arr[3 * i + 1] = n1; arr[3 * i + 2] = n2;
    }
  }

  /** Rotates atoms of the group from the lab frame back into the upper-triangular frame (rotate_x/v/f). */
  private rotate(r: M3, what: 'x' | 'v' | 'f'): void {
    const s = this.sys.state;
    const arr = what === 'x' ? s.x : what === 'v' ? s.v : s.f;
    const { mask } = s;
    const lo = s.box.lo;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const a = arr[3 * i], b = arr[3 * i + 1], c = arr[3 * i + 2];
      const n0 = r[0][0] * a + r[0][1] * b + r[0][2] * c;
      const n1 = r[1][0] * a + r[1][1] * b + r[1][2] * c;
      const n2 = r[2][0] * a + r[2][1] * b + r[2][2] * c;
      if (what === 'x') {
        arr[3 * i] = n0 + lo[0]; arr[3 * i + 1] = n1 + lo[1]; arr[3 * i + 2] = n2 + lo[2];
      } else {
        arr[3 * i] = n0; arr[3 * i + 1] = n1; arr[3 * i + 2] = n2;
      }
    }
  }

  setup(): void {
    const g = this.sys.geom;
    const box = zeroRot();
    this.uefbox.getBox(box, g.volume(3));
    const tol = 1e-4;
    const isok = nearlyEqual(g.lx, box[0][0], tol) && nearlyEqual(g.ly, box[1][1], tol)
      && nearlyEqual(g.lz, box[2][2], tol) && nearlyEqual(g.xy, box[0][1], tol)
      && nearlyEqual(g.yz, box[1][2], tol) && nearlyEqual(g.xz, box[0][2], tol);
    if (!isok) throw new StyleError('Initial box is not close enough to the expected uef box');
    this.uefbox.getRot(this.rot);
    this.tempUef.yesRot();
    this.pressUef.inFix = true;
    this.pressUef.updateRot();
    super.setup();
  }

  initialIntegrate(): void {
    this.invRotate(this.rot, 'x');
    this.invRotate(this.rot, 'v');
    this.invRotate(this.rot, 'f');
    this.tempUef.noRot();
    super.initialIntegrate();
    this.rotate(this.rot, 'x');
    this.rotate(this.rot, 'v');
    this.rotate(this.rot, 'f');
    this.tempUef.yesRot();
  }

  finalIntegrate(): void {
    // update rot here since it must directly follow the virial calculation
    this.pressUef.updateRot();
    this.invRotate(this.rot, 'v');
    this.invRotate(this.rot, 'f');
    this.tempUef.noRot();
    super.finalIntegrate();
    this.rotate(this.rot, 'v');
    this.rotate(this.rot, 'f');
    this.tempUef.yesRot();
  }

  init(): void {
    super.init();
    // "Can't use another fix which changes box shape": a deform, a box relaxation or a tilted barostat
    for (const f of this.sys.fixes as Fix[]) {
      if (f === this) continue;
      const tiltedBarostat = f instanceof FixNH && (f as unknown as NHInternals).pFlag.slice(3).some((b) => b);
      if (f.style === 'deform' || f.style === 'box/relax' || tiltedBarostat) {
        throw new StyleError(`Can't use another fix which changes box shape with fix ${this.style}`);
      }
    }
  }

  /**
   * pre_exchange: when the lattice reduction changes the cell, go to the lab frame, set the reduced box, rotate to
   * its frame and map the image flags by the inverse change of basis. The atoms are wrapped by the run loop's
   * periodic step that follows (native domain->remap). Runs on neighbor-rebuild steps, as native.
   */
  preExchange(): void {
    if (!this.uefbox.reduce()) return;
    this.invRotate(this.rot, 'x');
    this.invRotate(this.rot, 'v');
    this.invRotate(this.rot, 'f');
    this.setBoxFromUef(this.sys.geom.volume(3));
    this.rotate(this.rot, 'v');
    this.rotate(this.rot, 'x');
    this.rotate(this.rot, 'f');
    const ri = zeroRot();
    this.uefbox.getInverseCob(ri);
    const s = this.sys.state;
    const img = s.image;
    for (let i = 0; i < s.n; i++) {
      const a = img[3 * i], b = img[3 * i + 1], c = img[3 * i + 2];
      img[3 * i] = ri[0][0] * a + ri[0][1] * b + ri[0][2] * c;
      img[3 * i + 1] = ri[1][0] * a + ri[1][1] * b + ri[1][2] * c;
      img[3 * i + 2] = ri[2][0] * a + ri[2][1] * b + ri[2][2] * c;
    }
  }

  /** post_run: the cell is reduced at the end of a run, so a later run starts from the reduced box. */
  postRun(): void {
    this.preExchange();
    this.sys.pbc();
  }

  /** Restart files are not written for this fix (the strains and the rotation are not in the FixNH restart data). */
  restartState(): NHRestart {
    throw new StyleError(`write_restart with fix ${this.style} is not supported by the browser engine`);
  }
}

/** Factories for the registry (fix nvt/uef, fix npt/uef). */
export const newFixNVTUef = (sys: System, id: string, group: string, args: string[]): FixNHUef =>
  new FixNHUef(sys, id, group, args, 'nvt');
export const newFixNPTUef = (sys: System, id: string, group: string, args: string[]): FixNHUef =>
  new FixNHUef(sys, id, group, args, 'npt');
